"""Shared KV statistics for fused Sage preparation."""
import triton
import triton.language as tl

@triton.jit
def _select(group, values: tl.constexpr):
    result = tl.full((), 0, tl.int32)
    for i in tl.static_range(len(values)):
        result = tl.where(group == i, values[i], result)
    return result

@triton.jit
def _kv_stats(K, V, Sums, Maxima, N: tl.constexpr, H: tl.constexpr,
              D: tl.constexpr, TILES: tl.constexpr,
              K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
              V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr):
    tile, dc, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = bh//H, bh%H
    row = tile*256 + tl.arange(0, 256)
    dim = dc*16 + tl.arange(0, 16)
    kk = tl.load(K+batch*K0+row[:, None]*K1+head*K2+dim[None, :],
                 row[:, None] < N, other=0).to(tl.float32)
    vv = tl.load(V+batch*V0+row[:, None]*V1+head*V2+dim[None, :],
                 row[:, None] < N, other=0).to(tl.float32)
    dst = (bh*TILES+tile)*D + dim
    tl.store(Sums+dst, tl.sum(kk, 0))
    tl.store(Maxima+dst, tl.max(tl.abs(vv), 0))

@triton.jit
def _prefix_stats(K, V, Sums, Maxima, Mean, VScale,
                  B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                  TILES: tl.constexpr, REDUCE: tl.constexpr, ENDS: tl.constexpr,
                  K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                  V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                  VInv=None, MIN_V: tl.constexpr = 1.e-7):
    dc, head, gb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    group, batch = gb//B, gb%B
    length = _select(group, ENDS)
    dim = dc*16 + tl.arange(0, 16)
    tile = tl.arange(0, REDUCE)
    full = length//256
    ptr = ((batch*H+head)*TILES+tile[:, None])*D+dim[None, :]
    sums = tl.load(Sums+ptr, tile[:, None] < full, other=0)
    maxima = tl.load(Maxima+ptr, tile[:, None] < full, other=0)
    # Only the incomplete final tile is reread for each prefix. All complete
    # prefix tiles share statistics from a single read of the incoming KV.
    row = full*256 + tl.arange(0, 256)
    kk = tl.load(K+batch*K0+row[:, None]*K1+head*K2+dim[None, :],
                 row[:, None] < length, other=0).to(tl.float32)
    vv = tl.load(V+batch*V0+row[:, None]*V1+head*V2+dim[None, :],
                 row[:, None] < length, other=0).to(tl.float32)
    mean = (tl.sum(sums, 0)+tl.sum(kk, 0))/length
    vmax = tl.maximum(tl.max(maxima, 0), tl.max(tl.abs(vv), 0))
    dst = (gb*H+head)*D+dim
    tl.store(Mean+dst, mean)  # Round as torch.mean(K) in the input dtype.
    vmax = tl.maximum(vmax, MIN_V)
    if VInv is None:
        tl.store(VScale+dst, vmax/448.)
    else:
        tl.store(VScale+dst, tl.div_rn(vmax, 448.))
        inverse = tl.div_rn(448., tl.where(vmax > 0, vmax, 1.))
        tl.store(VInv+dst, tl.where(vmax > 0, inverse, 0.))
