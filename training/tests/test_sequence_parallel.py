"""CPU/GLOO checks of spatial layout, autograd, and overlapping FSDP scaling."""

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from wf_training.utils import sequence_parallel as sp


def _sample(index):
    values = torch.arange(2 * 3 * 8 * 3, dtype=torch.float64).reshape(2, 24, 3)
    return (values * 0.03 + index * 0.23).sin()


def _objective(output, index):
    target = torch.cos(torch.arange(output.numel(), dtype=output.dtype).reshape_as(output) * 0.07)
    return (output.tanh() - target - index * 0.01).square().mean()


def _check_gradient_average_and_accumulation(microbatches):
    weight_init = torch.arange(3 * 5, dtype=torch.float64).reshape(3, 5) * 0.015
    actual_weight = weight_init.clone().requires_grad_()
    reference_weight = weight_init.clone().requires_grad_()
    actual_optimizer = torch.optim.Adam([actual_weight], lr=0.003)
    reference_optimizer = torch.optim.Adam([reference_weight], lr=0.003)
    for microbatch in range(microbatches):
        sample_index = microbatch * 2 + sp.get_data_parallel_rank()
        local_input = sp.split_spatial(_sample(sample_index), num_frames=3)
        actual_output = sp.gather_spatial(local_input @ actual_weight, num_frames=3)
        (_objective(actual_output, sample_index) / microbatches).backward()
        # FSDP reduces each microbatch on WORLD, without no_sync(). Previously
        # accumulated gradients are already equal on all ranks in this toy.
        dist.all_reduce(actual_weight.grad)
        actual_weight.grad.div_(dist.get_world_size())
    reference_loss = sum(
        _objective(_sample(index) @ reference_weight, index)
        for index in range(2 * microbatches)
    ) / (2 * microbatches)
    reference_loss.backward()
    torch.testing.assert_close(actual_weight.grad, reference_weight.grad, rtol=1e-12, atol=1e-12)
    actual_optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(actual_weight, reference_weight, rtol=1e-12, atol=1e-12)
    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            actual_optimizer.state[actual_weight][key],
            reference_optimizer.state[reference_weight][key], rtol=1e-12, atol=1e-12,
        )


def _distributed_checks(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=8,
        timeout=timedelta(seconds=90),
    )
    try:
        sp.initialize_sequence_parallel(4)
        sp.initialize_sequence_parallel(4)
        assert sp.get_sp_world_size() == 4
        assert sp.get_sp_rank() == rank % 4
        assert sp.get_sp_src_rank() == rank // 4 * 4
        assert sp.get_data_parallel_rank() == rank // 4
        assert sp.get_data_parallel_world_size() == 2
        assert dist.get_process_group_ranks(sp.get_sp_group()) == list(range(rank // 4 * 4, rank // 4 * 4 + 4))
        local_rank = sp.get_sp_rank()
        value = torch.tensor([rank], dtype=torch.int64)
        assert sp.sp_broadcast(value) is value
        assert value.item() == sp.get_sp_src_rank()
        transposed_noise = (torch.arange(12, dtype=torch.float64).reshape(3, 4) + rank).T
        assert not transposed_noise.is_contiguous()
        broadcast_noise = sp.sp_broadcast(transposed_noise)
        assert broadcast_noise.is_contiguous()
        torch.testing.assert_close(
            broadcast_noise,
            (torch.arange(12, dtype=torch.float64).reshape(3, 4) + sp.get_sp_src_rank()).T,
            rtol=0, atol=0,
        )

        # Nontrivial batch, frame, and channel axes expose an accidental
        # rank-major flat concat. Different DP groups must remain isolated.
        full = torch.arange(2 * 3 * 8 * 3, dtype=torch.float64).reshape(2, 24, 3)
        full += 1000 * sp.get_data_parallel_rank()
        local = sp.split_spatial(full, num_frames=3)
        expected_local = full.reshape(2, 3, 8, 3)[:, :, local_rank * 2:(local_rank + 1) * 2].reshape(2, 6, 3)
        torch.testing.assert_close(local, expected_local, rtol=0, atol=0)
        torch.testing.assert_close(sp.gather_spatial(local, 3), full, rtol=0, atol=0)

        # Each rank contributes a different loss coefficient. SUM backward
        # must add 1+2+3+4, rather than select or average one contribution.
        local_grad_input = local.clone().requires_grad_()
        spatial_weights = (full / 100 + 1).cos()
        gathered = sp.gather_spatial(local_grad_input, 3)
        ((gathered * spatial_weights).sum() * (local_rank + 1)).backward()
        torch.testing.assert_close(
            local_grad_input.grad, sp.split_spatial(spatial_weights, 3) * 10,
            rtol=1e-12, atol=1e-12,
        )

        # Heads are contiguous shards; tokens remain [frame, spatial].
        full_qkv = torch.arange(2 * 3 * 8 * 8 * 2, dtype=torch.float64).reshape(2, 3, 8, 8, 2)
        full_qkv += 10000 * sp.get_data_parallel_rank()
        spatial_shard = full_qkv[:, :, local_rank * 2:(local_rank + 1) * 2].reshape(2, 6, 8, 2)
        head_shard = full_qkv[:, :, :, local_rank * 2:(local_rank + 1) * 2].reshape(2, 24, 2, 2)
        exchanged = sp.sequence_to_head(spatial_shard, 3)
        torch.testing.assert_close(exchanged, head_shard, rtol=0, atol=0)
        torch.testing.assert_close(sp.head_to_sequence(exchanged, 3), spatial_shard, rtol=0, atol=0)
        torch.testing.assert_close(sp.sequence_to_head(sp.head_to_sequence(head_shard, 3), 3), head_shard, rtol=0, atol=0)

        # Different destination heads carry different gradient weights,
        # checking the reverse all-to-all itself, beyond an identity roundtrip.
        exchange_input = spatial_shard.clone().requires_grad_()
        full_weights = (full_qkv / 500).sin()
        local_head_weights = full_weights[:, :, :, local_rank * 2:(local_rank + 1) * 2].reshape(2, 24, 2, 2)
        (sp.sequence_to_head(exchange_input, 3) * local_head_weights).sum().backward()
        torch.testing.assert_close(
            exchange_input.grad,
            full_weights[:, :, local_rank * 2:(local_rank + 1) * 2].reshape_as(exchange_input),
            rtol=0, atol=0,
        )
        inverse_input = head_shard.clone().requires_grad_()
        local_spatial_weights = full_weights[:, :, local_rank * 2:(local_rank + 1) * 2].reshape(2, 6, 8, 2)
        (sp.head_to_sequence(inverse_input, 3) * local_spatial_weights).sum().backward()
        torch.testing.assert_close(inverse_input.grad, local_head_weights, rtol=0, atol=0)

        # Ordinary split backward has no hidden group reduction.
        full_input = full.clone().requires_grad_()
        sp.split_spatial(full_input, 3).sum().backward()
        expected_grad = torch.zeros_like(full).reshape(2, 3, 8, 3)
        expected_grad[:, :, local_rank * 2:(local_rank + 1) * 2] = 1
        torch.testing.assert_close(full_input.grad, expected_grad.reshape_as(full), rtol=0, atol=0)

        _check_gradient_average_and_accumulation(microbatches=1)
        _check_gradient_average_and_accumulation(microbatches=4)
        for callback in (
            lambda: sp.split_spatial(torch.zeros(1, 18, 2), 3),
            lambda: sp.sequence_to_head(torch.zeros(1, 6, 3, 2), 3),
            lambda: sp.head_to_sequence(torch.zeros(1, 18, 2, 2), 3),
            lambda: sp.initialize_sequence_parallel(3),
            lambda: sp.gather_spatial(torch.zeros(1, 6, 2, 3), 3),
            lambda: sp.split_spatial(torch.zeros(1, 6, 2), 0),
            lambda: sp.split_spatial(torch.zeros(1, 6, 2), -1),
            lambda: sp.split_spatial(torch.zeros(1, 6, 2), True),
            lambda: sp.split_spatial(torch.zeros(1, 6, 2), 2.0),
            lambda: sp.split_spatial(torch.zeros(1, 6, 2), 4),
        ):
            try:
                callback()
            except ValueError:
                pass
            else:
                raise AssertionError("invalid SP shape/topology was accepted")
        dist.barrier()
    finally:
        dist.destroy_process_group()


class SequenceParallelTest(unittest.TestCase):
    def test_sp1_is_an_identity_without_distributed_initialization(self):
        sp.initialize_sequence_parallel(1)
        self.assertEqual(sp.get_sp_world_size(), 1)
        self.assertEqual(sp.get_sp_rank(), 0)
        self.assertEqual(sp.get_sp_src_rank(), 0)
        self.assertEqual(sp.get_data_parallel_rank(), 0)
        self.assertEqual(sp.get_data_parallel_world_size(), 1)
        self.assertIsNone(sp.get_sp_group())
        value = torch.randn(2, 6, 8, requires_grad=True)
        self.assertIs(sp.split_spatial(value, 3), value)
        self.assertIs(sp.gather_spatial(value, 3), value)
        self.assertIs(sp.sp_broadcast(value), value)
        heads = value.reshape(2, 6, 2, 4)
        self.assertIs(sp.sequence_to_head(heads, 3), heads)
        self.assertIs(sp.head_to_sequence(heads, 3), heads)
        # The existing SP1 teacher may pad the flat sequence independently
        # of the frame count. Identity helpers must not reject that path.
        padded = torch.zeros(1, 128, 4, 2)
        self.assertIs(sp.sequence_to_head(padded, 6), padded)
        self.assertIs(sp.head_to_sequence(padded, 6), padded)

    def test_invalid_shapes_and_uninitialized_sp_fail_early(self):
        with self.assertRaisesRegex(RuntimeError, "initialize torch.distributed"):
            sp.initialize_sequence_parallel(4)
        for size in (0, -1, True, 2.0):
            with self.subTest(size=size), self.assertRaises(ValueError):
                sp.initialize_sequence_parallel(size)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "GLOO is unavailable")
    def test_gloo_sp4_layout_gradients_and_world_average(self):
        with tempfile.TemporaryDirectory(prefix="wf-sp4-test-") as temporary:
            # Fork avoids eight independent cold imports on the shared NFS.
            # All work is CPU-only, with no inherited distributed/GPU state.
            mp.start_processes(
                _distributed_checks, args=(str(Path(temporary) / "rendezvous"),),
                nprocs=8, start_method="fork", join=True,
            )


if __name__ == "__main__":
    unittest.main()
