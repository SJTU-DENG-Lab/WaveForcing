"""CPU/GLOO numerical and communication checks for the fused QKV exchange."""

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from wf_training.utils import sequence_parallel as sp


def _noncontiguous_copy(value, requires_grad=False):
    storage = value.new_empty((*value.shape[:-1], value.shape[-1] * 2))
    result = storage[..., ::2]
    result.copy_(value)
    result.requires_grad_(requires_grad)
    assert not result.is_contiguous()
    return result


def _check_exchange(size, rank, mode):
    batch, frames, spatial, heads, width = 2, 3, 16, 16, 3
    local_rank = rank % size
    local_spatial, local_heads = spatial // size, heads // size
    shape = (batch, frames, spatial, heads, width)
    tags = torch.arange(batch * frames * spatial * heads * width,
                        dtype=torch.float64).reshape(shape)
    full = [tags + index * 1_000_000 + (rank // size) * 10_000_000
            for index in range(3)]
    weights = [tags + 13 + index * 100_000 for index in range(3)]
    mixed_dtype = mode == 'mixed_dtype'
    if mixed_dtype:
        dtypes = (torch.float32, torch.float32, torch.bfloat16)
        full = [value.to(dtype) for value, dtype in zip(full, dtypes)]
        weights = [value.to(dtype) for value, dtype in zip(weights, dtypes)]

    def spatial_slice(value):
        return value[:, :, local_rank * local_spatial:(local_rank + 1) * local_spatial].reshape(
            batch, frames * local_spatial, heads, width)

    def head_slice(value):
        return value[:, :, :, local_rank * local_heads:(local_rank + 1) * local_heads].reshape(
            batch, frames * spatial, local_heads, width)

    required = (True, False, True) if mode == 'partial_grad' else (True, True, True)
    inputs = [_noncontiguous_copy(spatial_slice(value), requires_grad=needed)
              for value, needed in zip(full, required)]
    references = [_noncontiguous_copy(value.detach(), requires_grad=needed)
                  for value, needed in zip(inputs, required)]
    output_grads = [_noncontiguous_copy(head_slice(value)) for value in weights]
    original_collective = dist.all_to_all_single
    calls = []

    def counted_collective(output, input, *args, **kwargs):
        calls.append((input.numel() * input.element_size(),
                      output.numel() * output.element_size()))
        return original_collective(output, input, *args, **kwargs)

    with patch.object(dist, 'all_to_all_single', side_effect=counted_collective):
        expected = tuple(sp.sequence_to_head(value, frames) for value in references)
        reference_forward_calls = list(calls)
        calls.clear()
        actual = sp.sequence_to_head_qkv(*inputs, frames)
        fused_forward_calls = list(calls)
        calls.clear()

        assert len(reference_forward_calls) == 3, reference_forward_calls
        assert len(fused_forward_calls) == (3 if mixed_dtype else 1), fused_forward_calls
        assert tuple(map(sum, zip(*fused_forward_calls))) == tuple(map(sum, zip(*reference_forward_calls)))
        for result, reference, full_value in zip(actual, expected, full):
            assert result.dtype == full_value.dtype
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
            torch.testing.assert_close(result, head_slice(full_value), rtol=0, atol=0)

        # Q-only deliberately leaves two Function outputs unused. The partial
        # case uses Q and V gradients while K's input does not require grad.
        used = (0,) if mode == 'unused_outputs' else tuple(
            index for index, needed in enumerate(required) if needed)
        torch.autograd.backward([expected[index] for index in used],
                                [output_grads[index] for index in used])
        reference_backward_calls = list(calls)
        calls.clear()
        actual_used = (0,) if mode == 'unused_outputs' else tuple(
            index for index, value in enumerate(actual) if value.requires_grad)
        torch.autograd.backward([actual[index] for index in actual_used],
                                [output_grads[index] for index in actual_used])
        fused_backward_calls = list(calls)
        assert len(reference_backward_calls) == len(used), reference_backward_calls
        assert len(fused_backward_calls) == (3 if mixed_dtype else 1), fused_backward_calls
        assert tuple(map(sum, zip(*fused_backward_calls))) == tuple(map(sum, zip(*fused_forward_calls)))
        if mode in ('all_grad', 'mixed_dtype'):
            assert tuple(map(sum, zip(*fused_backward_calls))) == tuple(map(sum, zip(*reference_backward_calls)))

    for index, (value, reference, needed) in enumerate(zip(inputs, references, required)):
        if not needed:
            assert value.grad is None
            assert reference.grad is None
            continue
        # Preserve disconnected-input semantics: an explicit zero would let
        # Adam update moments and apply weight decay to an otherwise unused K/V.
        assert (value.grad is None) == (reference.grad is None)
        if reference.grad is None:
            assert index not in used
            continue
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
        torch.testing.assert_close(value.grad, spatial_slice(weights[index]), rtol=0, atol=0)


def _check_higher_order_derivatives(size, rank, q_only):
    batch, frames, spatial, heads, width = 2, 2, 8, 8, 2
    local_spatial = spatial // size
    local_rank = rank % size
    tags = torch.arange(batch * frames * spatial * heads * width,
                        dtype=torch.float64).reshape(batch, frames, spatial, heads, width)
    values = [(tags * 0.003 + index * 0.2 + (rank // size) * 0.07)
              [:, :, local_rank * local_spatial:(local_rank + 1) * local_spatial]
              .reshape(batch, frames * local_spatial, heads, width)
              for index in range(3)]
    inputs = [_noncontiguous_copy(value, requires_grad=True) for value in values]
    references = [_noncontiguous_copy(value, requires_grad=True) for value in values]
    actual = sp.sequence_to_head_qkv(*inputs, frames)
    expected = tuple(sp.sequence_to_head(value, frames) for value in references)
    used = (0,) if q_only else (0, 1, 2)

    def derivatives(outputs, leaves):
        loss = sum(outputs[index].sin().sum() for index in used)
        first = torch.autograd.grad(loss, leaves, create_graph=True, allow_unused=True)
        second = torch.autograd.grad(sum(value.sum() for value in first if value is not None),
                                     leaves, allow_unused=True)
        return first, second

    actual_derivatives = derivatives(actual, inputs)
    expected_derivatives = derivatives(expected, references)
    for order, (actual_values, expected_values) in enumerate(
            zip(actual_derivatives, expected_derivatives), start=1):
        for index, (value, reference) in enumerate(zip(actual_values, expected_values)):
            assert (value is None) == (reference is None)
            if index not in used:
                assert value is None
                continue
            torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-12)
            analytic = values[index].cos() if order == 1 else -values[index].sin()
            torch.testing.assert_close(value, analytic, rtol=1e-12, atol=1e-12)


def _distributed_checks(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=8,
        timeout=timedelta(seconds=90),
    )
    try:
        for size in (2, 4, 8):
            own_group = None
            for first_rank in range(0, 8, size):
                group = dist.new_group(list(range(first_rank, first_rank + size)))
                if first_rank <= rank < first_rank + size:
                    own_group = group
            with patch.multiple(sp, _SP_GROUP=own_group, _SP_SIZE=size,
                                _SP_RANK=rank % size, _SP_SRC_RANK=rank - rank % size):
                for mode in ('all_grad', 'unused_outputs', 'partial_grad', 'mixed_dtype'):
                    _check_exchange(size, rank, mode)
                for q_only in (False, True):
                    _check_higher_order_derivatives(size, rank, q_only)
            dist.barrier()
            dist.destroy_process_group(own_group)
    finally:
        dist.destroy_process_group()


class QKVExchangeTests(unittest.TestCase):
    def test_sp1_returns_original_tensors_without_communication_or_padding_checks(self):
        # Flat padding is allowed in SP1 even when it does not form full frames.
        values = tuple(_noncontiguous_copy(torch.randn(2, 128, 4, 3), requires_grad=True)
                       for _ in range(3))
        with patch.object(sp, '_SP_SIZE', 1), patch.object(dist, 'all_to_all_single') as collective:
            actual = sp.sequence_to_head_qkv(*values, num_frames=6)
            collective.assert_not_called()
        for result, value in zip(actual, values):
            self.assertIs(result, value)
        sum(result.sum() for result in actual).backward()
        for value in values:
            torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)

    def test_invalid_inputs_fail_before_communication(self):
        value = torch.zeros(2, 6, 8, 3)
        bad_inputs = [
            (value[..., 0], value[..., 0], value[..., 0], 3),
            (value, value[:, :3], value, 3),
            (value, value, value[:, :, :4], 3),
            (value, value, torch.empty_like(value, device='meta'), 3),
            (value[:, :, :3], value[:, :, :3], value[:, :, :3], 3),
            *[(value, value, value, frames) for frames in (0, -1, True, 2.5, 4)],
        ]
        with patch.object(sp, '_SP_SIZE', 4), patch.object(dist, 'all_to_all_single') as collective:
            for q, k, v, frames in bad_inputs:
                with self.subTest(shapes=(q.shape, k.shape, v.shape), frames=frames,
                                  dtype=k.dtype, device=v.device):
                    with self.assertRaises(ValueError):
                        sp.sequence_to_head_qkv(q, k, v, frames)
            collective.assert_not_called()

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'GLOO is unavailable')
    def test_gloo_sp2_sp4_sp8_layout_gradients_and_collective_counts(self):
        with tempfile.TemporaryDirectory(prefix='wf-qkv-exchange-') as temporary:
            mp.start_processes(
                _distributed_checks, args=(str(Path(temporary) / 'rendezvous'),),
                nprocs=8, start_method='fork', join=True,
            )


if __name__ == '__main__':
    unittest.main()
