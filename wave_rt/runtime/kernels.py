"""WP kernels: final-address QKV preparation, first norm and context assembly."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope(TC, TS, HC, HS, WC, WS, token, dim, valid,
          HEIGHT: tl.constexpr, WIDTH: tl.constexpr, START,
          DT: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr):
    t = token // (HEIGHT * WIDTH) + START
    y, x = token // WIDTH % HEIGHT, token % WIDTH
    ct = tl.load(TC + t*DT + dim, valid & (dim < DT), other=0)
    st = tl.load(TS + t*DT + dim, valid & (dim < DT), other=0)
    ch = tl.load(HC + y*DH + dim-DT, valid & (dim >= DT) & (dim < DT+DH), other=0)
    sh = tl.load(HS + y*DH + dim-DT, valid & (dim >= DT) & (dim < DT+DH), other=0)
    cw = tl.load(WC + x*DW + dim-DT-DH, valid & (dim >= DT+DH) & (dim < DT+DH+DW), other=0)
    sw = tl.load(WS + x*DW + dim-DT-DH, valid & (dim >= DT+DH) & (dim < DT+DH+DW), other=0)
    return ct+ch+cw, st+sh+sw


@triton.jit(do_not_specialize=['START'])
def _qkv(Q, K, V, WQ, WK, TC, TS, HC, HS, WC, WS, OQ, OK, OV, RAWK,
         S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
         Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
         K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
         V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
         HEIGHT: tl.constexpr, WIDTH: tl.constexpr, START,
         DT: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
         EQ: tl.constexpr, EK: tl.constexpr, RAW: tl.constexpr,
         KV_ONLY: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    batch, token = row // S, row % S
    p = tl.arange(0, BLOCK)
    valid = p < H*D
    head, dim = p // D, p % D
    k = tl.load(K + batch*K0 + token*K1 + head*K2 + dim, valid, other=0).to(tl.float32)
    rk = tl.rsqrt(tl.sum(k*k, 0)/(H*D) + EK)
    kn = (k*rk*tl.load(WK+p, valid, other=0).to(tl.float32)).to(OK.dtype.element_ty).to(tl.float32)
    # Adjacent dimensions form one complex pair. Split the already normalized
    # register vector instead of loading and normalizing each partner again.
    pair = tl.arange(0, BLOCK//2)
    cos, sin = _rope(TC, TS, HC, HS, WC, WS, token, pair % (D//2), pair < H*D//2,
                     HEIGHT, WIDTH, START, DT, DH, DW)
    ke, ko = tl.split(tl.reshape(kn, (BLOCK//2, 2)))
    rotated_k = tl.reshape(tl.join(tl.fma(ko, -sin, ke*cos), tl.fma(ke, sin, ko*cos)), (BLOCK,))
    offset = row*H*D+p
    tl.store(OK+offset, rotated_k, valid)
    tl.store(OV+offset, tl.load(V+batch*V0+token*V1+head*V2+dim, valid, other=0), valid)
    if RAW:
        tl.store(RAWK+offset, kn, valid)
    if not KV_ONLY:
        q = tl.load(Q+batch*Q0+token*Q1+head*Q2+dim, valid, other=0).to(tl.float32)
        rq = tl.rsqrt(tl.sum(q*q, 0)/(H*D)+EQ)
        qn = (q*rq*tl.load(WQ+p, valid, other=0).to(tl.float32)).to(OQ.dtype.element_ty).to(tl.float32)
        qe, qo = tl.split(tl.reshape(qn, (BLOCK//2, 2)))
        rotated_q = tl.reshape(tl.join(tl.fma(qo, -sin, qe*cos), tl.fma(qe, sin, qo*cos)), (BLOCK,))
        tl.store(OQ+offset, rotated_q, valid)


def prepare_qkv(q, k, v, q_norm, k_norm, axes, start, *, targets=None, raw=False, kv_only=False):
    b, s, h, d = k.shape
    height, width = axes.shape[1:]
    if (q.shape != k.shape or v.shape != k.shape or d % 2 or sum(axes.dims) != d
            or s % (height*width) or q_norm.weight.numel() != h*d or k_norm.weight.numel() != h*d):
        raise ValueError('Invalid WP QKV/norm/axis geometry')
    axes.check(s//(height*width), height, width, start)
    if any(x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError('WP QKV needs unit head-dimension stride')
    if targets is None:
        ok, ov = torch.empty_like(k, memory_format=torch.contiguous_format), torch.empty_like(v, memory_format=torch.contiguous_format)
    else:
        ok, ov = targets
        if any(x.shape != k.shape or not x.is_contiguous() or x.dtype != k.dtype or x.device != k.device for x in targets):
            raise ValueError('Invalid final-address WP KV targets')
    oq = ok if kv_only else torch.empty_like(q, memory_format=torch.contiguous_format)
    raw_k = torch.empty_like(ok) if raw else None
    def eps(norm):
        return norm.variance_epsilon if hasattr(norm, 'variance_epsilon') else norm.eps
    _qkv[(b*s,)](q, k, v, q_norm.weight, k_norm.weight, *axes.tables,
        oq, ok, ov, raw_k if raw else ok, s, h, d,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], height, width, max(0, start),
        *(n//2 for n in axes.dims), eps(q_norm), eps(k_norm), raw, kv_only,
        triton.next_power_of_2(h*d), num_warps=8 if h*d > 4096 else 4, enable_fp_fusion=False)
    return oq, raw_k, ov, ok


@triton.jit
def _norm(X, SHIFT, SCALE, OUT, FF, GATE, RESOUT,
          S: tl.constexpr, D: tl.constexpr, SPATIAL: tl.constexpr,
          X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, EPS: tl.constexpr,
          RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    p = tl.arange(0, BLOCK)
    x = tl.load(X+(row//S)*X0+(row%S)*X1+p*X2, p < D, other=0).to(tl.float32)
    if RESIDUAL:
        ff = tl.load(FF+row*D+p, p < D, other=0).to(tl.float32)
        gate = tl.load(GATE+(row//SPATIAL)*D+p, p < D, other=0).to(tl.float32)
        # Native MulAdd contracts the FP32 multiply/add, then stores BF16.
        # The next block normalizes that rounded value, not the FP32 residual.
        rounded = tl.fma(ff, gate, x).to(RESOUT.dtype.element_ty)
        tl.store(RESOUT+row*D+p, rounded, p < D)
        x = rounded.to(tl.float32)
    mean = tl.sum(x, 0)/D
    diff = tl.where(p < D, x-mean, 0.)
    variance = tl.sum(diff*diff, 0)/D
    normalized = (x-mean)*tl.rsqrt(variance+EPS)
    modulation = (row//SPATIAL)*D+p
    scale = tl.load(SCALE+modulation, p < D, other=0).to(tl.float32)
    shift = tl.load(SHIFT+modulation, p < D, other=0).to(tl.float32)
    tl.store(OUT+row*D+p, normalized*(1.+scale)+shift, p < D)


def norm_modulate(x, shift, scale, eps, *, addition=None, gate=None):
    b, s, d = x.shape
    if (shift.ndim != 4 or shift.shape != scale.shape or shift.shape != (b, shift.shape[1], 1, d)
            or s % shift.shape[1] or not shift.is_contiguous() or not scale.is_contiguous()):
        raise ValueError('Invalid WP norm/modulation geometry')
    fused = addition is not None
    if fused and (addition.shape != x.shape or addition.dtype != x.dtype
                  or addition.device != x.device or not addition.is_contiguous()
                  or gate is None or gate.shape != shift.shape or not gate.is_contiguous()
                  or gate.device != x.device or gate.dtype != torch.float32):
        raise ValueError('Invalid WP residual/norm geometry')
    out = torch.empty((b, s, d), device=x.device, dtype=x.dtype)
    residual = torch.empty_like(out) if fused else out
    _norm[(b*s,)](x, shift, scale, out, addition if fused else x,
                  gate if fused else shift, residual, s, d, s//shift.shape[1], *x.stride(), eps, fused,
                  triton.next_power_of_2(d), num_warps=8 if d > 4096 else 4, enable_fp_fusion=False)
    return (residual, out) if fused else out


@triton.jit(do_not_specialize=['START'])
def _assemble(KS, VS, OK, OV, TC, TS, HC, HS, WC, WS,
              T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
              HEIGHT: tl.constexpr, WIDTH: tl.constexpr, START,
              DT: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
              ANCHOR: tl.constexpr, BLOCK: tl.constexpr):
    chunk = tl.program_id(0)
    pos = tl.program_id(1)*BLOCK+tl.arange(0, BLOCK)
    valid = pos < T*H*D
    for i in tl.static_range(len(KS)):
        if chunk == i:
            v = tl.load(VS[i]+pos, valid, other=0)
            if ANCHOR and i == 0:
                pair = tl.program_id(1)*(BLOCK//2)+tl.arange(0, BLOCK//2)
                pair_valid = pair < T*H*D//2
                dim, token = pair % (D//2), pair // (H*D//2)
                cos, sin = _rope(TC, TS, HC, HS, WC, WS, token, dim, pair_valid,
                                HEIGHT, WIDTH, START, DT, DH, DW)
                # One lane owns both components: no repeated partner loads or
                # shared-memory layout conversion between loading and rotation.
                even = tl.load(KS[i]+2*pair, pair_valid, other=0).to(tl.float32)
                odd = tl.load(KS[i]+2*pair+1, pair_valid, other=0).to(tl.float32)
                tl.store(OK+chunk*T*H*D+2*pair, tl.fma(odd, -sin, even*cos), pair_valid)
                tl.store(OK+chunk*T*H*D+2*pair+1, tl.fma(even, sin, odd*cos), pair_valid)
            else:
                k = tl.load(KS[i]+pos, valid, other=0)
                tl.store(OK+chunk*T*H*D+pos, k, valid)
            tl.store(OV+chunk*T*H*D+pos, v, valid)


def assemble_context(keys, values, out_k, out_v, axes, start, *, anchor=True):
    if not keys:
        return
    t, h, d = keys[0].shape[-3:]
    if (len(keys) != len(values) or any(not x.is_contiguous() or x.numel() != t*h*d for x in (*keys, *values))
            or any(not x.is_contiguous() or x.numel() < len(keys)*t*h*d for x in (out_k, out_v))):
        raise ValueError('WP context sources must be contiguous KV chunks')
    axes.check(t//(axes.shape[1]*axes.shape[2]), *axes.shape[1:], start)
    _assemble[(len(keys), triton.cdiv(t*h*d, 512))](tuple(keys), tuple(values), out_k, out_v,
        *axes.tables, t, h, d, *axes.shape[1:], max(0, start), *(n//2 for n in axes.dims),
        anchor, 512, enable_fp_fusion=False)
