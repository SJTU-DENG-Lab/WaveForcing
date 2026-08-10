"""Dummy probe: does the FLOPs-based VAE-stage partition actually balance in
*wall-clock time*?  FLOPs is only an estimate; small units are launch-bound so
their real time is far more uniform than their FLOPs.  This measures per-unit
time under the real streaming (frame-by-frame, feat_cache) decode and compares:

  * FLOPs partition  (what we ship)          -> imbalance measured in TIME
  * time  partition  (balanced_partition on measured ms) -> best achievable

for n = 2,3,4 stages, both EAGER and torch.compile'd.  If the FLOPs partition's
time-imbalance is bad, we should time each layer at init and partition on time.

Run from repo root:  .venv/bin/python scripts/vae_balance_probe.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

import torch

from naive_rt.rolling_forcing.utils.wan_wrapper import WanVAEWrapper
from naive_rt.rolling_forcing.wan.modules.vae import (
    CausalConv3d, ResidualBlock, Resample, CACHE_T,
)
from naive_rt.wave import vae_pipe as vp

DEV = "cuda:0"
DTYPE = torch.bfloat16
H, W = 60, 104          # latent spatial (480x832 / 8)
NPB = 3                 # latent frames per chunk (num_frames_per_block)
WARMUP_CHUNKS = 5
TIMED_CHUNKS = 10


def build():
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    vae = WanVAEWrapper().to(DEV, DTYPE).eval()
    m = vae.model
    dec = m.decoder
    scale = [vae.mean.to(DEV, DTYPE), (1.0 / vae.std).to(DEV, DTYPE)]
    units = vp.flatten_decode_units(dec)
    _, flops = vp.layer_flops(vae, DEV)
    return vae, m, dec, scale, units, flops


def make_chunk_x(m, scale):
    """One chunk: random latent -> conv2 -> x (BCTHW), as stage0 sees it."""
    z = torch.randn(1, 16, NPB, H, W, device=DEV, dtype=DTYPE)
    z = z / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)
    return m.conv2(z)


def build_apply_fns(units, compiled: bool):
    """One (x,cache,idx)->x callable per unit.  Eager uses vp.apply_unit; compiled
    torch.compile's each module (dispatch by the ORIGINAL type, since an
    OptimizedModule fails isinstance)."""
    fns = []
    for _, L in units:
        if not compiled:
            fns.append(lambda x, cache, idx, L=L: vp.apply_unit(L, x, cache, idx))
            continue
        cL = torch.compile(L, dynamic=True)
        if isinstance(L, CausalConv3d):
            def f(x, cache, idx, cL=cL):
                i = idx[0]
                cx = x[:, :, -CACHE_T:, :, :].clone()
                if cx.shape[2] < 2 and cache[i] is not None:
                    cx = torch.cat(
                        [cache[i][:, :, -1, :, :].unsqueeze(2).to(cx.device), cx], dim=2)
                x = cL(x, cache[i])
                cache[i] = cx
                idx[0] += 1
                return x
        elif isinstance(L, (ResidualBlock, Resample)):
            def f(x, cache, idx, cL=cL):
                return cL(x, cache, idx)
        else:
            def f(x, cache, idx, cL=cL):
                return cL(x)
        fns.append(f)
    return fns


def time_units(m, scale, units, compiled: bool):
    """Per-unit GPU time (ms) accumulated over TIMED_CHUNKS chunks x NPB frames,
    under the real streaming feat_cache.  Returns per-unit total ms + conv2 ms."""
    U = len(units)
    fns = build_apply_fns(units, compiled)
    cache = [None] * 256
    acc = [0.0] * U
    conv2_ms = 0.0

    def run_frame(xf, timed):
        idx = [0]
        if not timed:
            for f in fns:
                xf = f(xf, cache, idx)
            return
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(U + 1)]
        evs[0].record()
        for i, f in enumerate(fns):
            xf = f(xf, cache, idx)
            evs[i + 1].record()
        run_frame._evs.append(evs)
    run_frame._evs = []

    # warmup (populate cache + warm/compile kernels), not timed
    for _ in range(WARMUP_CHUNKS):
        x = make_chunk_x(m, scale)
        for f in range(x.shape[2]):
            run_frame(x[:, :, f:f + 1], timed=False)
    torch.cuda.synchronize()

    # timed
    for _ in range(TIMED_CHUNKS):
        c2s = torch.cuda.Event(enable_timing=True)
        c2e = torch.cuda.Event(enable_timing=True)
        z = torch.randn(1, 16, NPB, H, W, device=DEV, dtype=DTYPE)
        z = z / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)
        c2s.record(); x = m.conv2(z); c2e.record()
        for f in range(x.shape[2]):
            run_frame(x[:, :, f:f + 1], timed=True)
        torch.cuda.synchronize()
        conv2_ms += c2s.elapsed_time(c2e)

    for evs in run_frame._evs:
        for i in range(U):
            acc[i] += evs[i].elapsed_time(evs[i + 1])
    return acc, conv2_ms


def imbalance(costs, cuts):
    n = len(cuts) - 1
    seg = [sum(costs[cuts[i]:cuts[i + 1]]) for i in range(n)]
    ideal = sum(costs) / n
    return seg, (max(seg) / ideal if ideal else 1.0)


def analyze(tag, t_ms, flops, names, nfr, conv2_ms):
    tot_t = sum(t_ms)
    tot_f = sum(flops)
    print(f"\n===== {tag} =====")
    print(f"{'unit':<12}{'GFLOPs':>9}{'flop%':>8}{'ms/frame':>10}{'time%':>8}")
    for nm, f, t in zip(names, flops, t_ms):
        print(f"{nm:<12}{f/1e9:9.2f}{100*f/tot_f:8.1f}{t/nfr:10.3f}{100*t/tot_t:8.1f}")
    print(f"{'conv2(s0)':<12}{'':>9}{'':>8}{conv2_ms/nfr:10.3f}")
    print(f"[{tag}] decoder total: {tot_t/nfr:.2f} ms/frame "
          f"({tot_t/nfr*NPB:.1f} ms/chunk) + conv2 {conv2_ms/nfr:.2f}")
    for n in (2, 3, 4):
        fcuts = vp.balanced_partition(flops, n)
        tcuts = vp.balanced_partition(t_ms, n)
        fseg_t, fimb = imbalance(t_ms, fcuts)
        tseg_t, timb = imbalance(t_ms, tcuts)
        gain = max(fseg_t) / max(tseg_t) if max(tseg_t) else 1.0
        print(f"n={n}: FLOPs-cut slowest={max(fseg_t)/nfr:.2f}ms/f (imb {fimb:.2f}x)"
              f" | time-cut {tcuts} slowest={max(tseg_t)/nfr:.2f}ms/f (imb {timb:.2f}x)"
              f" | time-cut {gain:.2f}x faster")


def main():
    vae, m, dec, scale, units, flops = build()
    names = [nm for nm, _ in units]
    print(f"[probe] {len(units)} decoder units, chunk=(1,16,{NPB},{H},{W}), "
          f"bf16, warmup={WARMUP_CHUNKS} timed={TIMED_CHUNKS} chunks")
    nfr = TIMED_CHUNKS * NPB

    te, c2e = time_units(m, scale, units, compiled=False)
    analyze("EAGER", te, flops, names, nfr, c2e)

    try:
        tc, c2c = time_units(m, scale, units, compiled=True)
        analyze("COMPILED", tc, flops, names, nfr, c2c)
        print(f"\n[probe] compile speedup on decoder: "
              f"{sum(te)/sum(tc):.2f}x  ({sum(te)/nfr:.1f} -> {sum(tc)/nfr:.1f} ms/frame)")
        # cross-check: does compile reorder the per-unit cost ranking?
        import statistics
        ratios = [ (tc[i]/te[i]) for i in range(len(te)) if te[i] > 0 ]
        print(f"[probe] per-unit compile speedup: min {min(ratios):.2f}x "
              f"max {max(ratios):.2f}x median {statistics.median(ratios):.2f}x "
              f"(non-uniform -> compiled time-partition differs from eager)")
    except Exception as e:
        import traceback
        print(f"\n[probe] COMPILED pass failed: {e!r}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
