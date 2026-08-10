"""Reusable n-stage VAE-decode pipeline helpers.

The Wan causal 3D VAE decoder is a strictly *sequential chain of decoupled
layers* (nn.Sequential is just a wrapper): conv1 -> middle -> upsamples -> head,
each a self-contained module whose feat_cache slot is layer-local.  So the decode
can be cut at ANY inter-layer boundary and run as an n-stage pipeline (activations
passed stage->stage), bit-exact vs single-GPU.

This module provides the model-side logic, independent of the transport
(mp.Queue / NCCL p2p lives in the caller, e.g. naive_rt/wave/serve.py):
  * flatten_decode_units : the ordered list of atomic decoder units to split on
  * apply_unit           : run one unit with the streaming feat_cache
  * layer_flops          : per-unit FLOPs (one dummy frame, hooks) -- deterministic
  * balanced_partition   : DP min-max contiguous split into n stages
  * report               : human-readable per-unit FLOPs + stage balance

FLOPs (not measured time) is the cost metric: deterministic, so every stage
computes identical cuts with no coordination needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from naive_rt.rolling_forcing.wan.modules.vae import (
    AttentionBlock,
    CausalConv3d,
    ResidualBlock,
    Resample,
    CACHE_T,
)

__all__ = [
    "flatten_decode_units",
    "conv_manual",
    "apply_unit",
    "layer_flops",
    "measure_unit_times",
    "balanced_partition",
    "report",
]


# --------------------------------------------------------------------------- units
def flatten_decode_units(decoder) -> list[tuple[str, nn.Module]]:
    """Ordered atomic decoder units to split on (conv2 is NOT here -- it is the
    post-quant entry, kept on stage 0 by the caller)."""
    units: list[tuple[str, nn.Module]] = [("conv1", decoder.conv1)]
    units += [(f"middle[{i}]", L) for i, L in enumerate(decoder.middle)]
    units += [(f"up[{i}]", L) for i, L in enumerate(decoder.upsamples)]
    units += [(f"head[{i}]", L) for i, L in enumerate(decoder.head)]
    return units


def conv_manual(conv: CausalConv3d, x, cache, idx):
    """Streaming CausalConv3d with per-slot feat_cache (moved from serve.py)."""
    i = idx[0]
    cx = x[:, :, -CACHE_T:, :, :].clone()
    if cx.shape[2] < 2 and cache[i] is not None:
        cx = torch.cat([cache[i][:, :, -1, :, :].unsqueeze(2).to(cx.device), cx], dim=2)
    x = conv(x, cache[i])
    cache[i] = cx
    idx[0] += 1
    return x


def apply_unit(layer: nn.Module, x, cache, idx):
    """Run one decoder unit under the streaming feat_cache, dispatching by type.
    ResidualBlock/Resample take (x, cache, idx); CausalConv3d goes through
    conv_manual; everything else (AttentionBlock/RMS_norm/SiLU) is (x)."""
    if isinstance(layer, CausalConv3d):
        return conv_manual(layer, x, cache, idx)
    if isinstance(layer, (ResidualBlock, Resample)):
        return layer(x, cache, idx)
    return layer(x)


# --------------------------------------------------------------------------- FLOPs
def layer_flops(
    vae,
    device,
    latent_frame_shape: tuple = (1, 16, 1, 60, 104),
) -> tuple[list[tuple[str, nn.Module]], list[float]]:
    """Per-unit FLOPs for one streaming latent frame (deterministic).

    Counts Conv1d/2d/3d + Linear (2*MACs) AND the AttentionBlock's SDPA matmul
    (4*N^2*C, N = spatial tokens) -- the latter is a functional call, not a
    module, so it must be added explicitly (else attention/Resample-Conv2d read
    as 0).  Returns (units, flops-per-unit)."""
    m = vae.model
    dec = m.decoder
    units = flatten_decode_units(dec)
    leaf2u: dict[int, int] = {}
    for ui, (_, u) in enumerate(units):
        for sub in u.modules():
            leaf2u[id(sub)] = ui
    flops = [0.0] * len(units)

    def conv_lin_hook(mod, inp, out):
        ui = leaf2u.get(id(mod))
        if ui is None:
            return
        o = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            cout = mod.out_channels
            cin = mod.in_channels // mod.groups
            k = 1
            for kk in mod.kernel_size:
                k *= kk
            osp = 1
            for s in o.shape[2:]:
                osp *= int(s)
            flops[ui] += 2.0 * cout * cin * k * osp
        elif isinstance(mod, nn.Linear):
            flops[ui] += 2.0 * mod.in_features * mod.out_features * (o.numel() / o.shape[-1])

    def attn_hook(mod, inp, out):
        ui = leaf2u.get(id(mod))
        if ui is None:
            return
        x = inp[0]
        b, c, t, h, w = x.shape
        n = h * w
        # scaled_dot_product_attention over (b*t) heads=1 seq=n dim=c: qk + av
        flops[ui] += 4.0 * (n ** 2) * c * b * t

    hooks = []
    for sub in dec.modules():
        if isinstance(sub, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            hooks.append(sub.register_forward_hook(conv_lin_hook))
        if isinstance(sub, AttentionBlock):
            hooks.append(sub.register_forward_hook(attn_hook))

    dtype = next(m.parameters()).dtype
    m.clear_cache()
    z = torch.randn(latent_frame_shape, device=device, dtype=dtype)
    scale = [vae.mean.to(device, dtype), (1.0 / vae.std).to(device, dtype)]
    zz = z / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)
    x = m.conv2(zz)
    with torch.no_grad():
        dec(x, feat_cache=[None] * 256, feat_idx=[0])
    for hk in hooks:
        hk.remove()
    m.clear_cache()
    return units, flops


# --------------------------------------------------------------------- partition
def measure_unit_times(
    vae,
    device,
    npb: int = 3,
    latent_hw: tuple = (60, 104),
    n_warm: int = 4,
    n_timed: int = 8,
) -> tuple[list, list[float]]:
    """Per-unit GPU time (ms/frame) under the REAL streaming feat_cache decode.

    FLOPs is a poor time-proxy for this VAE: the high-res upsample ResidualBlocks
    have ~equal FLOPs (channels halve as H,W double) but run 6x+ slower (memory
    bound at 480x832), so a FLOPs partition is badly time-imbalanced (~1.8x at
    n=3).  Timing each unit and partitioning on measured ms cuts the slowest stage
    ~1.6x.  Call on ONE rank and broadcast the cuts (timing noise would otherwise
    desync the stages).  Returns (units, ms-per-frame-per-unit)."""
    m = vae.model
    dec = m.decoder
    units = flatten_decode_units(dec)
    U = len(units)
    dtype = next(m.parameters()).dtype
    H, W = latent_hw
    scale = [vae.mean.to(device, dtype), (1.0 / vae.std).to(device, dtype)]

    def chunk_x():
        z = torch.randn(1, 16, npb, H, W, device=device, dtype=dtype)
        z = z / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)
        return m.conv2(z)

    cache = [None] * 256
    acc = [0.0] * U

    def run_frame(xf, evs=None):
        idx = [0]
        if evs is None:
            for _, L in units:
                xf = apply_unit(L, xf, cache, idx)
            return
        evs[0].record()
        for i, (_, L) in enumerate(units):
            xf = apply_unit(L, xf, cache, idx)
            evs[i + 1].record()

    m.clear_cache()
    for _ in range(n_warm):
        x = chunk_x()
        for f in range(x.shape[2]):
            run_frame(x[:, :, f:f + 1])
    torch.cuda.synchronize()

    ev_frames = []
    for _ in range(n_timed):
        x = chunk_x()
        for f in range(x.shape[2]):
            evs = [torch.cuda.Event(enable_timing=True) for _ in range(U + 1)]
            run_frame(x[:, :, f:f + 1], evs)
            ev_frames.append(evs)
    torch.cuda.synchronize()
    for evs in ev_frames:
        for i in range(U):
            acc[i] += evs[i].elapsed_time(evs[i + 1])
    nfr = n_timed * npb
    m.clear_cache()
    return units, [a / nfr for a in acc]


def balanced_partition(costs: list[float], n: int) -> list[int]:
    """Split the ordered `costs` into n contiguous segments minimising the max
    segment sum (classic linear-partition DP).  Returns n+1 boundaries
    [0, c1, ..., len(costs)] so segment i is costs[cuts[i]:cuts[i+1]]."""
    N = len(costs)
    if n <= 1:
        return [0, N]
    if n >= N:
        # one unit per stage where possible; pad trailing empty segments
        cuts = list(range(N + 1))
        while len(cuts) < n + 1:
            cuts.append(N)
        return cuts[: n + 1]
    pre = [0.0]
    for c in costs:
        pre.append(pre[-1] + c)
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(N + 1)]
    cut = [[0] * (n + 1) for _ in range(N + 1)]
    dp[0][0] = 0.0
    for i in range(1, N + 1):
        for k in range(1, n + 1):
            for j in range(k - 1, i):
                v = max(dp[j][k - 1], pre[i] - pre[j])
                if v < dp[i][k]:
                    dp[i][k] = v
                    cut[i][k] = j
    # recover boundaries
    bounds = [N]
    i, k = N, n
    while k > 0:
        j = cut[i][k]
        bounds.append(j)
        i, k = j, k - 1
    bounds.reverse()
    return bounds


def report(names: list[str], flops: list[float], cuts: list[int]) -> str:
    """Human-readable per-unit GFLOPs + per-stage totals + imbalance."""
    total = sum(flops) or 1.0
    n = len(cuts) - 1
    lines = ["[vae_pipe] per-unit GFLOPs:"]
    for nm, f in zip(names, flops):
        lines.append(f"    {nm:<10} {f / 1e9:8.2f}")
    seg = [sum(flops[cuts[i]:cuts[i + 1]]) for i in range(n)]
    ideal = total / n
    imb = (max(seg) / ideal) if ideal else 1.0
    lines.append(f"[vae_pipe] n={n} cuts={cuts} imbalance={imb:.2f}x")
    for i in range(n):
        seg_names = names[cuts[i]:cuts[i + 1]]
        span = f"{seg_names[0]}..{seg_names[-1]}" if seg_names else "(empty)"
        lines.append(f"    stage{i}: {span}  {seg[i] / 1e9:.1f} GFLOPs")
    return "\n".join(lines)
