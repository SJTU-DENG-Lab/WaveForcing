"""Single-GPU profiler for ONE joint-window DiT forward.

Reproduces the wavefront's steady-state forward cost on a single GPU (no NCCL):
build a ~37440-token joint KV context via the exchange hook and run the DiT
forward under torch.profiler.  Prints the top CUDA ops so we can see exactly
where the ~1s/forward goes (fp32 norms? ffn? rope? attention?).

Run:
  cd .
  LD_LIBRARY_PATH=".venv/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH" \
    CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/wrt_profile_forward.py
"""

import os
import time

import torch


def main() -> None:
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29717")

    from wave_rt.config import WaveConfig
    from wave_rt.runtime import backend as be_mod
    from wave_rt.runtime.backend import WaveBackend

    cfg = WaveConfig(rf_step=0, wp_size=1, num_frames=3, height=480, width=832, seed=0)
    be = WaveBackend(cfg, rank=0)
    be.init()

    npb = cfg.num_frames_per_block
    noise = be.full_noise_bcthw[:, :, 0:npb].contiguous()

    # exchange that fabricates an 8-chunk (~37440 token) joint context, like the
    # steady wavefront tick (anchor+working+live+5 inflight), WITHOUT collectives.
    N_CTX_CHUNKS = 8

    def exch(*, roped_key, value, **kw):
        K = torch.cat([roped_key] * N_CTX_CHUNKS, dim=1)
        V = torch.cat([value] * N_CTX_CHUNKS, dim=1)
        return K, V

    # warmup
    for _ in range(3):
        be_mod.set_exchange_fn(exch)
        be.forward_chunk(noise, float(be.dsl[0].item()), start_frame=0)
        be_mod.set_exchange_fn(None)
    torch.cuda.synchronize()

    # timed (no profiler) for a clean wall number
    be_mod.set_exchange_fn(exch)
    t = time.perf_counter()
    for _ in range(5):
        be.forward_chunk(noise, float(be.dsl[0].item()), start_frame=0)
    torch.cuda.synchronize()
    print(
        f"[prof] mean forward (ctx={N_CTX_CHUNKS}x{npb * 1560}) = "
        f"{(time.perf_counter() - t) / 5 * 1000:.1f} ms",
        flush=True,
    )

    # profiler
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            be.forward_chunk(noise, float(be.dsl[0].item()), start_frame=0)
        torch.cuda.synchronize()
    be_mod.set_exchange_fn(None)

    print("\n===== TOP OPS BY CUDA TIME =====", flush=True)
    print(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True
    )
    be.shutdown()


if __name__ == "__main__":
    main()
