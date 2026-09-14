"""Actual CPU HSDP teardown regression for forward-only parameter views."""
from datetime import timedelta
from functools import partial
import gc
import json
from pathlib import Path
import tempfile
import unittest
import weakref

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from wf_training.utils.distributed import clear_completed_fsdp_saved_views


def _teardown_case(groups, *, frozen, backward, final_forward, cleanup):
    torch.manual_seed(17)
    module = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
    module.requires_grad_(not frozen)
    model = FSDP(
        module, process_group=groups, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        device_id=torch.device("cpu"), use_orig_params=True,
        auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=64),
    )
    module = None
    if backward:
        model(torch.ones(2, 32)).sum().backward()
        model.zero_grad(set_to_none=True)
    if final_forward:
        with torch.no_grad():
            model(torch.ones(2, 32))
    references = [weakref.ref(state._handle.flat_param) for state in FSDP.fsdp_modules(model)
                  if state._handle is not None]
    dist.barrier()
    cleared = clear_completed_fsdp_saved_views(model) if cleanup else None
    model = None
    gc.collect()
    survivors = sum(reference() is not None for reference in references)
    # Release intentional control-case cycles before the next case.
    for reference in references:
        if reference() is not None and reference()._tensors is not None:
            reference()._tensors[:] = [None] * len(reference()._tensors)
    gc.collect()
    remaining = sum(reference() is not None for reference in references)
    dist.barrier()
    return dict(frozen=frozen, backward=backward, final_forward=final_forward,
                cleanup=cleanup, cleared=cleared, survivors=survivors,
                remaining_after_control_cleanup=remaining)


def _teardown_worker(rank, temporary):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{temporary}/rendezvous", rank=rank, world_size=4,
        timeout=timedelta(seconds=90),
    )
    shards = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
    replicas = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
    groups = (shards[rank // 2], replicas[rank % 2])
    try:
        cases = []
        for frozen, backward, final_forward in (
            (False, False, True), (True, False, True),
            (False, True, False), (False, True, True),
        ):
            for cleanup in (False, True):
                cases.append(_teardown_case(
                    groups, frozen=frozen, backward=backward,
                    final_forward=final_forward, cleanup=cleanup,
                ))
        Path(temporary, f"rank{rank}.json").write_text(json.dumps(cases))
    finally:
        dist.destroy_process_group()


class FSDPTeardownTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "GLOO is unavailable")
    def test_completed_forward_views_are_released_on_every_replica(self):
        with tempfile.TemporaryDirectory(prefix="wf-fsdp-teardown-") as temporary:
            mp.spawn(_teardown_worker, args=(temporary,), nprocs=4, join=True)
            for rank in range(4):
                for case in json.loads(Path(temporary, f"rank{rank}.json").read_text()):
                    with self.subTest(rank=rank, case=case):
                        expected_retention = case["final_forward"] and not case["cleanup"]
                        self.assertEqual(case["survivors"] > 0, expected_retention)
                        self.assertEqual(case["remaining_after_control_cleanup"], 0)
                        if case["cleanup"] and case["final_forward"]:
                            self.assertGreater(case["cleared"]["cleared_saved_views"], 0)


if __name__ == "__main__":
    unittest.main()
