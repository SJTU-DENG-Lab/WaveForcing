"""CPU/GLOO checks for explicit HSDP placement, synchronization, and export."""
from copy import deepcopy
from datetime import timedelta
from functools import partial
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from wf_training.utils import distributed, parallel_topology as topology, sequence_parallel as sp
from wf_training.utils.ema import ShardedEMA


def _config(**changes):
    values = dict(world_size=4, gpus_per_node=2, num_nodes=2,
                  sharding_strategy="hybrid_full", fsdp_shard_size=2,
                  fsdp_replica_size=2, fsdp_group_layout="contiguous_nodes_v1",
                  sequence_parallel_size=2)
    values.update(changes)
    return SimpleNamespace(**values)


def _records():
    layout = {key: value for key, value in vars(_config()).items()
              if key != "sequence_parallel_size"}
    return [dict(rank=rank, env_rank=rank, world_size=4, env_world=4,
                 local_rank=rank % 2, local_world_size=2, hostname=f"node{rank // 2}",
                 layout=dict(layout), sp_size=2) for rank in range(4)]


class TopologyValidationTests(unittest.TestCase):
    def tearDown(self):
        topology.reset_fsdp_topology()

    def test_placement_schema_and_invalid_rank_node_sp_layouts(self):
        records = _records()
        layout, hosts = topology._validate_records(records)
        self.assertEqual(layout["fsdp_shard_size"], 2)
        self.assertEqual(hosts, ("node0", "node1"))
        changes = (
            (1, "env_rank", 0, "RANK/WORLD_SIZE"),
            (1, "env_world", 8, "RANK/WORLD_SIZE"),
            (1, "local_rank", 0, "LOCAL_RANK/LOCAL_WORLD_SIZE"),
            (1, "local_world_size", 4, "LOCAL_RANK/LOCAL_WORLD_SIZE"),
            (1, "hostname", "other", "spans multiple hostnames"),
            (1, "sp_size", 1, "configuration differs"),
        )
        for rank, key, value, message in changes:
            broken = deepcopy(records)
            broken[rank][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                topology._validate_records(broken)
        for record in records:
            record["sp_size"] = 4
        with self.assertRaisesRegex(ValueError, "SP must divide"):
            topology._validate_records(records)
        records = _records()
        records[2]["hostname"] = records[3]["hostname"] = "node0"
        with self.assertRaisesRegex(ValueError, "same hostname"):
            topology._validate_records(records)

    def test_local_configuration_errors_are_reported_for_collective_validation(self):
        cases = (
            (_config(gpus_per_node=1), "at least two ranks"),
            (_config(fsdp_shard_size=4), "fsdp_shard_size disagrees"),
            (_config(sequence_parallel_size=True), "positive integer"),
        )
        with mock.patch.object(dist, "get_rank", return_value=0), \
                mock.patch.object(dist, "get_world_size", return_value=4):
            for config, message in cases:
                with self.subTest(message=message):
                    record = topology._local_record(config)
                    self.assertIn(message, record["error"])
                    records = _records()
                    records[0] = record
                    with self.assertRaisesRegex(ValueError, message):
                        topology._validate_records(records)

    def test_groups_are_explicit_cached_and_independent_of_stage_sp(self):
        records = _records()

        def gather(output, record):
            output[:] = deepcopy(records)

        with mock.patch.object(dist, "is_initialized", return_value=True), \
                mock.patch.object(dist, "get_rank", return_value=3), \
                mock.patch.object(dist, "get_world_size", return_value=4), \
                mock.patch.object(dist, "all_gather_object", side_effect=gather), \
                mock.patch.object(dist, "new_group", side_effect=lambda ranks: tuple(ranks)) as create:
            first = topology.initialize_fsdp_topology(_config())
            self.assertEqual(first.shard_ranks, (2, 3))
            self.assertEqual(first.replica_ranks, (1, 3))
            self.assertFalse(first.export_replica)
            self.assertEqual([call.kwargs["ranks"] for call in create.call_args_list],
                             [[0, 1], [2, 3], [0, 2], [1, 3]])
            self.assertEqual(first.metadata(True)["fsdp_shard_rank"], 1)
            self.assertEqual(first.metadata(True)["fsdp_replica_rank"], 1)
            for record in records:
                record["sp_size"] = 1
            self.assertIs(topology.initialize_fsdp_topology(_config(sequence_parallel_size=1)), first)
            self.assertEqual(create.call_count, 4)
            for record in records:
                record["layout"].update(sharding_strategy="full", fsdp_shard_size=4,
                                         fsdp_replica_size=1)
            with self.assertRaisesRegex(RuntimeError, "cannot change"):
                topology.initialize_fsdp_topology(_config(
                    sharding_strategy="full", fsdp_shard_size=4, fsdp_replica_size=1))
        with mock.patch.object(dist, "is_initialized", return_value=False):
            self.assertIsNone(topology.get_fsdp_topology())

    def test_single_node_full_fsdp_reuses_world_without_extra_groups(self):
        records = _records()[:2]
        for record in records:
            record.update(world_size=2, env_world=2)
            record["layout"].update(world_size=2, num_nodes=1, fsdp_replica_size=1,
                                     sharding_strategy="full")
        with mock.patch.object(dist, "is_initialized", return_value=True), \
                mock.patch.object(dist, "get_rank", return_value=0), \
                mock.patch.object(dist, "get_world_size", return_value=2), \
                mock.patch.object(dist, "all_gather_object", side_effect=lambda out, record: out.__setitem__(slice(None), records)), \
                mock.patch.object(dist, "new_group") as create:
            result = topology.initialize_fsdp_topology(_config(
                world_size=2, num_nodes=1, fsdp_replica_size=1, sharding_strategy="full"))
            self.assertIs(result.shard_group, dist.group.WORLD)
            self.assertTrue(result.export_replica)
            create.assert_not_called()

    def test_launch_selects_cuda_before_nccl_and_reuses_existing_world(self):
        environment = dict(RANK="3", WORLD_SIZE="4", LOCAL_RANK="1",
                           MASTER_ADDR="2001:db8::1", MASTER_PORT="12345")
        calls = []
        with mock.patch.dict("os.environ", environment), \
                mock.patch.object(dist, "is_initialized", return_value=False), \
                mock.patch.object(torch.cuda, "set_device", side_effect=lambda rank: calls.append(("device", rank))), \
                mock.patch.object(dist, "init_process_group", side_effect=lambda **kw: calls.append(("init", kw))):
            distributed.launch_distributed_job()
        self.assertEqual(calls[0], ("device", 1))
        self.assertEqual(calls[1][1]["init_method"], "tcp://[2001:db8::1]:12345")
        with mock.patch.dict("os.environ", environment), \
                mock.patch.object(dist, "is_initialized", return_value=True), \
                mock.patch.object(dist, "get_rank", return_value=3), \
                mock.patch.object(dist, "get_world_size", return_value=4), \
                mock.patch.object(dist, "get_backend", return_value="nccl"), \
                mock.patch.object(torch.cuda, "current_device", return_value=1), \
                mock.patch.object(dist, "init_process_group") as initialize:
            distributed.launch_distributed_job()
            initialize.assert_not_called()
            with mock.patch.object(torch.cuda, "current_device", return_value=0), \
                    self.assertRaisesRegex(ValueError, "CUDA device"):
                distributed.launch_distributed_job()

    def test_wrapper_passes_both_groups_and_preserves_nccl_environment(self):
        groups = SimpleNamespace(sharding_strategy="hybrid_full",
                                 shard_group=object(), replica_group=object())
        module = nn.Linear(2, 2)
        with mock.patch.object(distributed, "get_fsdp_topology", return_value=groups), \
                mock.patch.object(distributed, "sync_module_buffers_from_rank0") as buffers, \
                mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(distributed, "FSDP", return_value=mock.sentinel.wrapper) as wrap, \
                mock.patch.dict("os.environ", {"NCCL_CROSS_NIC": "0"}):
            result = distributed.fsdp_wrap(module, sharding_strategy="hybrid_full",
                                           sync_module_states=True)
            import os
            self.assertEqual(os.environ["NCCL_CROSS_NIC"], "0")
        self.assertIs(result, mock.sentinel.wrapper)
        self.assertEqual(wrap.call_args.kwargs["process_group"],
                         (groups.shard_group, groups.replica_group))
        buffers.assert_called_once_with(module, torch.device("cuda", 0))
        with mock.patch.object(distributed, "get_fsdp_topology", return_value=None), \
                self.assertRaisesRegex(RuntimeError, "explicit HSDP topology"):
            distributed.fsdp_wrap(module, sharding_strategy="hybrid_full")

    def test_export_rejects_nested_fsdp_with_different_groups(self):
        groups = SimpleNamespace(hybrid=True, export_replica=False,
                                 shard_group=object(), replica_group=object())
        bad = SimpleNamespace(sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                              process_group=object(), _inter_node_pg=groups.replica_group)
        with mock.patch.object(distributed, "get_fsdp_topology", return_value=groups), \
                mock.patch.object(FSDP, "fsdp_modules", return_value=[bad]), \
                self.assertRaisesRegex(ValueError, "same explicit HSDP groups"):
            distributed.fsdp_state_dict(mock.Mock())

    def test_buffer_schema_mismatch_fails_before_tensor_broadcast(self):
        module = nn.Module()
        module.register_buffer("scale", torch.ones(1))

        def gather(output, record):
            output[:] = [record, ([], None)]

        with mock.patch.object(dist, "get_rank", return_value=0), \
                mock.patch.object(dist, "get_world_size", return_value=2), \
                mock.patch.object(dist, "all_gather_object", side_effect=gather), \
                mock.patch.object(dist, "broadcast") as broadcast:
            with self.assertRaisesRegex(ValueError, "buffer names, shapes, or dtypes"):
                distributed.sync_module_buffers_from_rank0(module, torch.device("cpu"))
            broadcast.assert_not_called()


def _sample(index):
    return torch.arange(24, dtype=torch.float32).reshape(1, 8, 3).sin() + index * 0.15


def _loss(value, index):
    return (value - 0.3 - index * 0.1).square().mean()


def _gloo_checks(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}",
                            rank=rank, world_size=4, timeout=timedelta(seconds=90))
    try:
        environment = {"RANK": str(rank), "WORLD_SIZE": "4", "LOCAL_RANK": str(rank % 2),
                       "LOCAL_WORLD_SIZE": "2"}
        with mock.patch.dict("os.environ", environment), \
                mock.patch.object(topology.socket, "gethostname", return_value=f"node{rank // 2}"):
            topo = topology.initialize_fsdp_topology(_config())
            assert topology.initialize_fsdp_topology(_config()) is topo
        assert dist.get_process_group_ranks(topo.shard_group) == list(topo.shard_ranks)
        assert dist.get_process_group_ranks(topo.replica_group) == list(topo.replica_ranks)
        sp.initialize_sequence_parallel(2)

        # Include both checkpoint-valued and nonpersistent buffers. The latter
        # intentionally carries Torch's old marker to prove it is not skipped.
        module = nn.Module()
        module.register_buffer("scale", torch.tensor([7.0 + rank]))
        module.register_buffer("meta_scale", torch.tensor([13.0]) if rank == 0
                               else torch.empty(1, device="meta"))
        module.child = nn.Module()
        module.child.register_buffer("cached", torch.tensor([11 + rank]), persistent=False)
        from torch.distributed.fsdp._init_utils import FSDP_SYNCED
        setattr(module.child.cached, FSDP_SYNCED, True)
        distributed.sync_module_buffers_from_rank0(module, torch.device("cpu"))
        torch.testing.assert_close(module.scale, torch.tensor([7.0]), rtol=0, atol=0)
        torch.testing.assert_close(module.meta_scale, torch.tensor([13.0]), rtol=0, atol=0)
        torch.testing.assert_close(module.child.cached, torch.tensor([11]), rtol=0, atol=0)

        # A real nested CPU FSDP HYBRID_SHARD run checks the process-group math
        # and selective FULL_STATE_DICT path, not CUDA/NCCL performance.
        torch.manual_seed(37)
        reference = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
        initial = deepcopy(reference.state_dict())
        actual = FSDP(deepcopy(reference),
                      process_group=(topo.shard_group, topo.replica_group),
                      sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                      auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=1),
                      use_orig_params=True, device_id=torch.device("cpu"))
        actual_optimizer = torch.optim.Adam(actual.parameters(), lr=0.003)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.003)
        ema = ShardedEMA(actual, decay=0.8)

        def export_expected(expected):
            with mock.patch.object(actual, "state_dict", wraps=actual.state_dict) as export:
                state = distributed.fsdp_state_dict(actual)
                assert export.call_count == int(topo.export_replica)
            if rank == 0:
                assert state.keys() == expected.keys()
                for key, value in state.items():
                    torch.testing.assert_close(value, expected[key], rtol=2e-6, atol=2e-8)
            else:
                assert not state
            dist.barrier()

        # Export before the first forward also covers FSDP lazy-init ordering.
        export_expected(initial)
        for micro in range(2):
            index = 2 * micro + rank // 2
            local = sp.split_spatial(_sample(index), num_frames=2)
            output = sp.gather_spatial(actual(local), num_frames=2)
            (_loss(output, index) / 2).backward()
        sum(_loss(reference(_sample(index)), index) for index in range(4)).div(4).backward()
        norm = actual.clip_grad_norm_(0.3)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
        torch.testing.assert_close(norm, expected_norm, rtol=2e-6, atol=2e-8)
        actual_optimizer.step()
        reference_optimizer.step()
        ema.update(actual)
        export_expected(reference.state_dict())
        expected_ema = {key: value.float() * 0.8 + reference.state_dict()[key].float() * 0.2
                        for key, value in initial.items()}
        with ema.applied_to(actual):
            export_expected(expected_ema)
        export_expected(reference.state_dict())
    finally:
        dist.destroy_process_group()
        topology.reset_fsdp_topology()


class HSDPGlooTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "GLOO is unavailable")
    def test_hsdp_sp_gradients_buffers_clip_ema_and_selective_export(self):
        with tempfile.TemporaryDirectory(prefix="wf-hsdp-runtime-") as temporary:
            mp.start_processes(_gloo_checks, args=(str(Path(temporary) / "rendezvous"),),
                               nprocs=4, start_method="fork", join=True)


if __name__ == "__main__":
    unittest.main()
