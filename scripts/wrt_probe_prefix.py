"""Single-GPU probe: sgl RF single-chunk forward latency vs KV-context length,
across attention backends (torch_sdpa / fa_custom / sage).

This is the sgl analogue of naive's "T2" dummy test (docs/05 §3: 48.7ms@1chunk ->
154ms@8chunk, +15ms/chunk).  It isolates, on ONE GPU (no NCCL, no cross-job
contention), how the sgl DiT forward scales with the joint KV window, and whether
routing the wavefront attention through flash-attn / SageAttention instead of the
driver-safe torch_sdpa default collapses the cost.

We drive backend.forward_chunk with a synthetic exchange closure that returns a
KV context of exactly K chunks (K-1 "prefix" + current), tiled from the current
layer's own roped K/V -- so klen = K * tokens_per_chunk is controlled precisely
and the sgl attention-sink cache path is bypassed (same code path the wavefront
uses).  Attention backend is toggled at runtime via wave_rt.backend's _WAVE_STATE.

Run:
  cd /path/to/WaveParallel
  LD_LIBRARY_PATH=".venv/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH" \
    CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/wrt_probe_prefix.py
"""

import argparse
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="1,2,3,4,5,6,7,8",
                    help="KV context sizes in chunks (K=1 -> current only)")
    ap.add_argument("--backends", type=str, default="torch_sdpa,fa_custom,sa")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29731")

    from wave_rt.config import WaveConfig
    from wave_rt import backend as be_mod
    from wave_rt.backend import WaveBackend

    cfg = WaveConfig(rf_step=0, wp_size=1, num_frames=3, height=480, width=832, seed=0)
    be = WaveBackend(cfg, rank=0)
    t0 = time.perf_counter()
    be.init()
    print(f"[probe] backend.init OK in {time.perf_counter()-t0:.1f}s "
          f"({be.num_layers} layers, klen unit = {be.num_token_per_frame * cfg.num_frames_per_block} tok/chunk)",
          flush=True)

    npb = cfg.num_frames_per_block
    noise = be.full_noise_bcthw[:, :, 0:npb].contiguous()
    step0 = float(be.dsl[0].item())

    Ks = [int(x) for x in args.chunks.split(",") if x.strip()]
    backends = [b for b in args.backends.split(",") if b.strip()]

    def set_backend(name: str) -> bool:
        """Switch the wavefront attention kernel at runtime; return False if unavailable."""
        try:
            if name == "fa_custom":
                be_mod.enable_fa_custom()
            elif name == "sa":
                be_mod.enable_sage()
            else:  # torch_sdpa
                be_mod._WAVE_STATE["attn"] = "torch_sdpa"
                be_mod._WAVE_STATE["fa_func"] = None
            return True
        except Exception as e:
            print(f"[probe] backend {name} unavailable: {e!r}", flush=True)
            return False

    def make_exch(K: int):
        """Return a KV context of exactly K chunks tiled from the current K/V."""
        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            ctx_k = roped_key if K == 1 else torch.cat([roped_key] * K, dim=1)
            ctx_v = value if K == 1 else torch.cat([value] * K, dim=1)
            return ctx_k, ctx_v
        return exch

    def timed_forward(K: int, iters: int, warmup: int) -> float:
        be.set_exchange_fn(make_exch(K))
        for _ in range(warmup):
            be.forward_chunk(noise, step0, start_frame=0)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            be.forward_chunk(noise, step0, start_frame=0)
        torch.cuda.synchronize()
        be.set_exchange_fn(None)
        return (time.perf_counter() - t) * 1000.0 / iters

    results: dict[str, dict[int, float]] = {}
    for bk in backends:
        if not set_backend(bk):
            continue
        results[bk] = {}
        for K in Ks:
            ms = timed_forward(K, args.iters, args.warmup)
            results[bk][K] = ms
            print(f"[probe] backend={bk:10s} K={K} (klen={K*npb*be.num_token_per_frame}) "
                  f"-> {ms:.1f} ms/forward", flush=True)

    # summary table
    print("\n=== sgl single-chunk forward latency (ms) vs KV context (chunks) ===", flush=True)
    hdr = "backend    " + "".join(f"K={k:<7d}" for k in Ks)
    print(hdr, flush=True)
    for bk, row in results.items():
        line = f"{bk:10s} " + "".join(f"{row[k]:<9.1f}" for k in Ks)
        print(line, flush=True)
    # per-chunk slope (K=1 -> max)
    for bk, row in results.items():
        if len(Ks) >= 2:
            slope = (row[Ks[-1]] - row[Ks[0]]) / (Ks[-1] - Ks[0])
            print(f"[probe] {bk}: base(K={Ks[0]})={row[Ks[0]]:.1f}ms, "
                  f"K={Ks[-1]}={row[Ks[-1]]:.1f}ms, +{slope:.1f}ms/chunk", flush=True)

    be.shutdown()
    print("[probe] DONE", flush=True)


if __name__ == "__main__":
    main()
