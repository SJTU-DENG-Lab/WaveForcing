"""WaveRT owns torch.distributed; sglang's parallel_state is a degree-1 shim.

Why: sglang's initialize_model_parallel builds ~7 GroupCoordinators, each of
which (for world_size>1 groups like _WORLD/_DP) spins up a pynccl communicator +
srt custom all-reduce (pinned/IPC buffers) + a gloo CPU group.  With wp_size
processes on one node these cross-process shared resources contend badly: the
SAME DiT forward measured 135ms single-GPU but ~900ms inside the 6-proc run,
independent of attention/comm/fp32 (confirmed via WAVE_NOAG).

So WaveRT initializes ONE bare NCCL process group (which doubles as the wavefront
comm) and points sglang's degree getters at a tiny identity coordinator.  No
GroupCoordinator, no pynccl, no custom all-reduce, no gloo groups.

Shim surface (from a full read of the RF runtime): the getters actually hit at
all-degree-1 are get_world_group/get_world_rank/get_world_size, get_cfg_group,
get_sp_group (+ ulysses/ring attrs) / get_sp_world_size, and optionally get_tp_group.
We set the module globals _WORLD/_CFG/_SP/_TP (NOT monkeypatch the getters, since
call sites bound the getter names at import time; the real getters read the
globals).  _DP/_PP/_VAE_DECODE/_DIT/_VAE stay None so model_parallel_is_initialized()
stays False, which keeps the graceful size-1 fallbacks in denoising/vae/encoders active.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


class _ShimGroup:
    """Degree-1 stand-in for sglang's GroupCoordinator: no subgroups, no comms."""

    def __init__(self) -> None:
        self.world_size = 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.rank_in_group = 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.ranks = [self.rank]
        # SP coordinator attrs read unconditionally by mrope / ulysses attention
        self.ulysses_world_size = 1
        self.ulysses_rank = 0
        self.ring_world_size = 1
        self.ring_rank = 0
        # only touched in >1 / CFG / main-rank branches (not reached at degree 1);
        # expose the bare WORLD pg defensively
        self.cpu_group = dist.group.WORLD if dist.is_initialized() else None
        self.device_group = dist.group.WORLD if dist.is_initialized() else None
        self.ulysses_group = self.device_group
        self.ring_group = self.device_group
        self.device = torch.device("cuda", self.local_rank)
        self.first_rank = self.rank
        self.last_rank = self.rank
        self.is_first_rank = True
        self.is_last_rank = True

    # degree-1 collectives are identity / no-ops
    def all_reduce(self, input_, *a, **k):
        return input_

    def all_gather(self, input_, *a, **k):
        return input_

    def all_to_all_4D(self, input_, *a, **k):
        return input_

    def gather(self, input_, *a, **k):
        return input_

    def broadcast(self, input_, *a, **k):
        return input_

    def broadcast_object(self, obj=None, *a, **k):
        return obj

    def barrier(self, *a, **k):
        return None


def init_wave_dist(
    world_size: int, rank: int, local_rank: int, timeout_s: int = 180
) -> None:
    """Bring up WaveRT's own bare NCCL PG and install the sglang degree-1 shim.
    MASTER_ADDR/PORT must already be in the environment (set by the launcher)."""
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=timeout_s),
        )
    _install_sgl_shim()


def _install_sgl_shim() -> None:
    import sglang.multimodal_gen.runtime.distributed.parallel_state as ps

    g = _ShimGroup()
    ps._WORLD = g
    ps._CFG = g
    ps._SP = g
    ps._TP = g  # harmless: model_parallel_is_initialized() still False (_DP/_PP None)
    ps._VAE_DECODE = g  # avoid the VAE loader's "decode group not initialized" fallback
    # mirror into srt's world group in case any srt-side code reads it
    try:
        import sglang.srt.distributed.parallel_state as srt_ps

        srt_ps._WORLD = g
    except Exception:
        pass
    return g


def ensure_causal_transformer_config(model_path: str) -> None:
    """Pre-patch transformer/config.json to the causal variant ONCE (single
    process, before spawn).  WanRollingForcingPipeline._patch_transformer_config
    otherwise does a per-rank read-modify-write + restore of this SHARED file;
    with wp_size ranks that races (a rank reads a half-written/empty file ->
    JSONDecodeError, or loads a restored non-causal config), which crashes one
    rank and hangs the rest at the next collective.  We patch once here and
    no-op the per-rank patch (see backend)."""
    import json
    import os

    from sglang.multimodal_gen.configs.pipeline_configs.wan import (
        RollingForcingWanT2V480PConfig,
    )

    cpath = os.path.join(model_path, "transformer", "config.json")
    if not os.path.isfile(cpath):
        return
    with open(cpath, encoding="utf-8") as f:
        c = json.load(f)
    arch = RollingForcingWanT2V480PConfig().dit_config.arch_config
    c["_class_name"] = "CausalWanTransformer3DModel"
    c["local_attn_size"] = arch.local_attn_size
    c["sink_size"] = arch.sink_size
    c["num_frames_per_block"] = arch.num_frames_per_block
    c["sliding_window_num_frames"] = arch.sliding_window_num_frames
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)


def disable_rf_config_patch() -> None:
    """No-op WanRollingForcingPipeline's per-rank config patch/restore (the file
    is already causal via ensure_causal_transformer_config) to avoid the race."""
    from sglang.multimodal_gen.runtime.pipelines.wan_rolling_forcing_pipeline import (
        WanRollingForcingPipeline,
    )

    WanRollingForcingPipeline._patch_transformer_config = lambda self, sa: None
    WanRollingForcingPipeline._restore_transformer_config = lambda self, cp: None


def patch_group_coordinator_lightweight() -> None:
    """Force sglang GroupCoordinators to skip the heavy device communicator +
    srt custom all-reduce.  Keeps sglang's (stable) group structure via the normal
    initialize_model_parallel, but drops the per-rank pynccl comm + custom-AR
    IPC/pinned buffers that world_size>1 groups (_WORLD/_DP) build -- the suspected
    source of the multi-process contention (135ms single-GPU vs ~900ms in 6 procs).
    The wavefront does its own raw dist collectives, so it never needs these."""
    from sglang.multimodal_gen.runtime.distributed import group_coordinator as gc

    if getattr(gc.GroupCoordinator, "_wave_lightweight", False):
        return
    orig_init = gc.GroupCoordinator.__init__

    def light_init(self, *args, **kwargs):
        # use_device_communicator is the 4th positional, use_srt_custom_allreduce
        # the 5th; force both off whether passed positionally or by keyword.
        args = list(args)
        if len(args) >= 4:
            args[3] = False
        else:
            kwargs["use_device_communicator"] = False
        if len(args) >= 5:
            args[4] = False
        else:
            kwargs["use_srt_custom_allreduce"] = False
        return orig_init(self, *args, **kwargs)

    gc.GroupCoordinator.__init__ = light_init
    gc.GroupCoordinator._wave_lightweight = True


