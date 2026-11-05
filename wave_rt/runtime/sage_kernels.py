"""Batched prefix statistics/quantization for SageAttention 2.2.0 on Hopper.

ABI references: thu-ml/SageAttention v2.2.0 quant_per_thread.py, fused.cu,
and qk_int_sv_f8_cuda_sm90.cu. Each prefix has independent K mean and V scale.
Q segment boundaries retain Sage's 64-row alignment; V is written directly
in the NHD backend's padded, transposed, sequence-permuted FP8 layout.
Single-prefix calls reuse tail statistics and launch K/V quantization together.
"""
from collections import OrderedDict
from itertools import accumulate

import torch
import triton
import triton.language as tl

from .sage_stats import _select, _kv_stats, _prefix_stats


@triton.jit
def _kv_stats_promote(K, V, Sums, Maxima, N: tl.constexpr, H: tl.constexpr,
                      D: tl.constexpr, TILES: tl.constexpr,
                      K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                      V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                      Page, START: tl.constexpr, LENGTH: tl.constexpr):
    # Same 256-row reduction tree as _kv_stats. Save BF16 bytes before their
    # conversion/reduction, so raw K and V need not both survive the reductions.
    tile, dc, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = bh//H, bh%H
    row = tile*256 + tl.arange(0, 256)
    dim = dc*16 + tl.arange(0, 16)
    local = row-START
    copy_mask = (local[:, None] >= 0) & (local[:, None] < LENGTH)
    copy_offset = (local[:, None]*H+head)*D+dim[None, :]
    dst = (bh*TILES+tile)*D+dim
    kk = tl.load(K+batch*K0+row[:, None]*K1+head*K2+dim[None, :],
                 row[:, None] < N, other=0)
    if tile*256 < START+LENGTH and (tile+1)*256 > START:
        tl.store(Page+copy_offset, kk, copy_mask)
    tl.store(Sums+dst, tl.sum(kk.to(tl.float32), 0))
    vv = tl.load(V+batch*V0+row[:, None]*V1+head*V2+dim[None, :],
                 row[:, None] < N, other=0)
    if tile*256 < START+LENGTH and (tile+1)*256 > START:
        tl.store(Page+LENGTH*H*D+copy_offset, vv, copy_mask)
    tl.store(Maxima+dst, tl.max(tl.abs(vv.to(tl.float32)), 0))


@triton.jit
def _prefix_quant_q(Q, O, Scale, B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                    STARTS: tl.constexpr, LENGTHS: tl.constexpr,
                    PADS: tl.constexpr, OFFSETS: tl.constexpr,
                    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                    warp_id=None, head_id=None, gb_id=None):
    warp = tl.program_id(0) if warp_id is None else warp_id
    head = tl.program_id(1) if head_id is None else head_id
    gb = tl.program_id(2) if gb_id is None else gb_id
    group, batch = gb//B, gb%B
    pad, offset = _select(group, PADS), _select(group, OFFSETS)
    if warp < pad//16:
        start, length = _select(group, STARTS), _select(group, LENGTHS)
        lane, dim = tl.arange(0, 8), tl.arange(0, D)
        row = warp*16+lane
        src = Q+batch*S0+(start+row[:, None])*S1+head*S2+dim[None, :]
        a = tl.load(src, row[:, None] < length, other=0).to(tl.float32)
        b = tl.load(src+8*S1, row[:, None]+8 < length, other=0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(a), 1), tl.max(tl.abs(b), 1))/127.+1.e-7
        a, b = a/scale[:, None], b/scale[:, None]
        a = (a+tl.where(a >= 0, .5, -.5)).to(tl.int8)
        b = (b+tl.where(b >= 0, .5, -.5)).to(tl.int8)
        dst = O+((offset*B+batch*pad+row[:, None])*H+head)*D+dim[None, :]
        tl.store(dst, a)
        tl.store(dst+8*H*D, b)
        tl.store(Scale+offset*B*H//2+(batch*H+head)*(pad//2)+warp*8+lane, scale)


@triton.jit
def _prefix_quant_k(K, Mean, O, Scale, B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                    ENDS: tl.constexpr, PADS: tl.constexpr, OFFSETS: tl.constexpr,
                    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                    block_id=None, head_id=None, gb_id=None):
    block = tl.program_id(0) if block_id is None else block_id
    head = tl.program_id(1) if head_id is None else head_id
    gb = tl.program_id(2) if gb_id is None else gb_id
    group, batch = gb//B, gb%B
    pad, offset = _select(group, PADS), _select(group, OFFSETS)
    if block < pad//32:
        length = _select(group, ENDS)
        row = block//4*128+tl.arange(0, 16)*8+block%4*2
        dim = tl.arange(0, D)
        mean = tl.load(Mean+(gb*H+head)*D+dim).to(tl.float32)
        src = K+batch*S0+row[:, None]*S1+head*S2+dim[None, :]
        a = tl.load(src, row[:, None] < length, other=0).to(tl.float32)
        b = tl.load(src+S1, row[:, None]+1 < length, other=0).to(tl.float32)
        # Preserve the rounding of Sage's materialized BF16/FP16 K-minus-mean.
        a = (a-mean[None, :]).to(K.dtype.element_ty).to(tl.float32)
        b = (b-mean[None, :]).to(K.dtype.element_ty).to(tl.float32)
        a = tl.where(row[:, None] < length, a, 0.)
        b = tl.where(row[:, None]+1 < length, b, 0.)
        scale = tl.maximum(tl.max(tl.max(tl.abs(a), 0), 0),
                           tl.max(tl.max(tl.abs(b), 0), 0))/127.+1.e-7
        a, b = a/scale, b/scale
        a = (a+tl.where(a >= 0, .5, -.5)).to(tl.int8)
        b = (b+tl.where(b >= 0, .5, -.5)).to(tl.int8)
        dst = O+((offset*B+batch*pad+row[:, None])*H+head)*D+dim[None, :]
        tl.store(dst, a)
        tl.store(dst+H*D, b)
        tl.store(Scale+offset*B*H//32+(batch*H+head)*(pad//32)+block, scale)


@triton.jit
def _prefix_quant_v(V, Inv, O, B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                    ENDS: tl.constexpr, PADS: tl.constexpr, OFFSETS: tl.constexpr,
                    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                    tile_id=None, dc_id=None, gbh_id=None, TILE_D: tl.constexpr = 16):
    tile = tl.program_id(0) if tile_id is None else tile_id
    dc = tl.program_id(1) if dc_id is None else dc_id
    gbh = tl.program_id(2) if gbh_id is None else gbh_id
    gb, head = gbh//H, gbh%H
    group, batch = gb//B, gb%B
    pad, offset = _select(group, PADS), _select(group, OFFSETS)
    if tile < pad//128:
        row, dim = tile*128+tl.arange(0, 128), dc*TILE_D+tl.arange(0, TILE_D)
        x = tl.load(V+batch*S0+row[:, None]*S1+head*S2+dim[None, :],
                    row[:, None] < _select(group, ENDS), other=0).to(tl.float32)
        inv = tl.load(Inv+(gb*H+head)*D+dim)
        x = tl.minimum(tl.maximum(x*inv[None, :], -448.), 448.)
        # Source row -> permuted destination. The inverse read permutation is
        # [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15], as in Sage's fused.cu.
        r = row%16
        permuted = row//16*16+r//8*2+(r//2%4)*4+r%2
        dst = O+offset*B*H*D+((batch*D+dim[None, :])*H+head)*pad+permuted[:, None]
        tl.store(dst, x.to(tl.float8e4nv))


@triton.jit
def _single_stats(Sums, Maxima, Mean, VScale, VInv,
                  N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                  TILES: tl.constexpr, REDUCE: tl.constexpr, dc_id=None, bh_id=None):
    dc = tl.program_id(0) if dc_id is None else dc_id
    bh = tl.program_id(1) if bh_id is None else bh_id
    dim = dc*16+tl.arange(0, 16)
    tile = tl.arange(0, REDUCE)
    ptr = (bh*TILES+tile[:, None])*D+dim[None, :]
    # Reuse masked tail statistics. GPU reduction layouts can change rounding.
    sums = tl.load(Sums+ptr, tile[:, None] < N//256, other=0)
    maxima = tl.load(Maxima+ptr, tile[:, None] < N//256, other=0)
    tail_sum = tl.full((16,), 0., tl.float32)
    tail_max = tl.full((16,), 0., tl.float32)
    if N % 256:
        tail_ptr = (bh*TILES+N//256)*D+dim
        tail_sum = tl.load(Sums+tail_ptr)
        tail_max = tl.load(Maxima+tail_ptr)
    mean = (tl.sum(sums, 0)+tail_sum)/N
    vmax = tl.maximum(tl.max(maxima, 0), tail_max)
    dst = bh*D+dim
    tl.store(Mean+dst, mean)
    tl.store(VScale+dst, tl.div_rn(vmax, 448.))
    inv = tl.div_rn(448., tl.where(vmax > 0, vmax, 1.))
    tl.store(VInv+dst, tl.where(vmax > 0, inv, 0.))


@triton.jit
def _single_q_stats(Q, Q8, QS, Sums, Maxima, Mean, VScale, VInv,
                    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                    START: tl.constexpr, LENGTH: tl.constexpr, QPAD: tl.constexpr,
                    N: tl.constexpr, TILES: tl.constexpr, REDUCE: tl.constexpr,
                    Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr):
    # Q quantization and final KV statistics have no mutual dependency.
    # The previous launch completed all partial statistics; each CTA owns
    # either Q outputs or one statistics tile, with no global synchronization.
    task = tl.program_id(0)
    q_tasks: tl.constexpr = QPAD//16*B*H
    if task < q_tasks:
        warp, bh = task % (QPAD//16), task // (QPAD//16)
        _prefix_quant_q(Q, Q8, QS, B, H, D, (START,), (LENGTH,), (QPAD,), (0,),
                        Q0, Q1, Q2, warp, bh % H, bh // H)
    else:
        task -= q_tasks
        _single_stats(Sums, Maxima, Mean, VScale, VInv, N, H, D, TILES, REDUCE,
                      task % (D//16), task // (D//16))


@triton.jit
def _single_quant_kv(K, V, Mean, Inv, K8, KS, V8,
                     B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                     N: tl.constexpr, PAD: tl.constexpr,
                     K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                     V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                     V_TILE_D: tl.constexpr = 16):
    # Independent K and V CTAs share one launch, not a reduction or a scale.
    # Reuse the exact multi-prefix arithmetic and Sage operand layouts.
    task = tl.program_id(0)
    k_tasks: tl.constexpr = PAD//32*B*H
    if task < k_tasks:
        block, bh = task % (PAD//32), task // (PAD//32)
        _prefix_quant_k(K, Mean, K8, KS, B, H, D, (N,), (PAD,), (0,),
                        K0, K1, K2, block, bh % H, bh // H)
    else:
        task -= k_tasks
        tile = task % (PAD//128)
        dc = task // (PAD//128) % (D//V_TILE_D)
        bh = task // ((PAD//128)*(D//V_TILE_D))
        _prefix_quant_v(V, Inv, V8, B, H, D, (N,), (PAD,), (0,),
                        V0, V1, V2, tile, dc, bh, V_TILE_D)


@triton.jit
def _correct_lse(Q, Mean, Raw, Out, B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 TOTAL: tl.constexpr, START: tl.constexpr, LENGTH: tl.constexpr,
                 Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
                 R0: tl.constexpr, R1: tl.constexpr, R2: tl.constexpr):
    row = tl.program_id(0)*32+tl.arange(0, 32)
    bh = tl.program_id(1)
    batch, head = bh//H, bh%H
    dim = tl.arange(0, D)
    q = tl.load(Q+batch*Q0+(row[:, None]+START)*Q1+head*Q2+dim[None, :],
                row[:, None] < LENGTH, other=0).to(tl.float32)
    mean = tl.load(Mean+bh*D+dim).to(tl.float32)
    correction = tl.sum(q*mean[None, :], 1).to(Q.dtype.element_ty).to(tl.float32)
    raw = tl.load(Raw+batch*R0+head*R1+row*R2, row < LENGTH, other=0)
    tl.store(Out+bh*TOTAL+START+row, raw/1.44269504+correction*(D**-.5), row < LENGTH)


class PrefixWorkspace:
    """A stream-local high-water scratch pool; contents are recomputed every call."""
    def __init__(self):
        self.buffers = {}
        self.plans = OrderedDict()
        self.kernel = None

    def _reserve(self, name, size, dtype, device):
        old = self.buffers.get(name)
        if old is None or old.numel() < size or old.dtype != dtype or old.device != device:
            old = self.buffers[name] = torch.empty(size, dtype=dtype, device=device)
            self.plans.clear()
        return old[:size]

    def plan(self, q, k, segments):
        b, _, h, d = q.shape
        key = (tuple(q.shape), tuple(k.shape), q.dtype, q.device, segments)
        if key in self.plans:
            self.plans.move_to_end(key)
            return self.plans[key]
        starts, lengths, ends = map(tuple, zip(*segments))
        qpads = tuple(triton.cdiv(n, 64)*64 for n in lengths)
        kpads = tuple(triton.cdiv(n, 128)*128 for n in ends)
        qo = tuple(accumulate(qpads, initial=0))
        ko = tuple(accumulate(kpads, initial=0))
        g, tiles = len(segments), triton.cdiv(max(ends), 256)
        def reserve(name, count, dtype):
            return self._reserve(name, count, dtype, q.device)
        q8 = reserve('q8', qo[-1]*b*h*d, torch.int8)
        qs = reserve('qs', qo[-1]*b*h//2, torch.float32)
        k8 = reserve('k8', ko[-1]*b*h*d, torch.int8)
        ks = reserve('ks', ko[-1]*b*h//32, torch.float32)
        v8 = reserve('v8', ko[-1]*b*h*d, torch.float8_e4m3fn)
        mean = reserve('mean', g*b*h*d, q.dtype).view(g, b, h, d)
        vs = reserve('vs', g*b*h*d, torch.float32).view(g, b, h, d)
        inv = reserve('inv', g*b*h*d, torch.float32).view(g, b, h, d)
        sums = reserve('sums', b*h*tiles*d, torch.float32)
        maxima = reserve('maxima', b*h*tiles*d, torch.float32)
        views = []
        for i, (_, length, end) in enumerate(segments):
            views.append((
                q8[qo[i]*b*h*d:qo[i+1]*b*h*d].view(b, qpads[i], h, d)[:, :length],
                k8[ko[i]*b*h*d:ko[i+1]*b*h*d].view(b, kpads[i], h, d)[:, :end],
                v8[ko[i]*b*h*d:ko[i+1]*b*h*d].view(b, d, h, kpads[i]),
                qs[qo[i]*b*h//2:qo[i+1]*b*h//2].view(b, h, qpads[i]//2),
                ks[ko[i]*b*h//32:ko[i+1]*b*h//32].view(b, h, kpads[i]//32), vs[i]))
        plan = dict(starts=starts, lengths=lengths, ends=ends, qpads=qpads, kpads=kpads,
                    qo=qo[:-1], ko=ko[:-1], groups=g, tiles=tiles, views=views,
                    whole_output=(len(segments) == 1 and starts[0] == 0 and lengths[0] == q.shape[1]),
                    q8=q8, qs=qs, k8=k8, ks=ks, v8=v8, mean=mean, vs=vs, inv=inv,
                    sums=sums, maxima=maxima)
        self.plans[key] = plan
        while len(self.plans) > 2:
            self.plans.popitem(last=False)
        return plan

    @staticmethod
    def _single_launcher(p, shape, strides, promotion_shape=None):
        """Bind scratch and geometry, never the current invocation's Q/K/V.

        The owner plan is invalidated when scratch grows. Strides are checked
        separately because equal shapes can arrive through different views.
        Keep the same kernels, launch order and arithmetic as the eager path.
        """
        b, _, h, d = shape
        tiles, n = p['tiles'], p['ends'][0]
        qpad, pad = p['qpads'][0], p['kpads'][0]
        qstride, kstride, vstride = strides
        # Wan 1.3B WP: halve V CTAs without changing scales or elementwise
        # arithmetic. Keep other geometries and the multi-prefix path at 16.
        v_tile_d = 32 if (b == 1 and h == 12 and d == 128
                         and p['mean'].dtype == torch.bfloat16) else 16
        stats = (_kv_stats if promotion_shape is None else _kv_stats_promote)[(tiles, d//16, b*h)]
        quant_q = _single_q_stats[(b*h*(qpad//16+d//16),)]
        quant_kv = _single_quant_kv[(b*h*(pad//32+(pad//128)*(d//v_tile_d)),)]
        stats_args = (p['sums'], p['maxima'], n, h, d, tiles, *kstride, *vstride)
        q_args = (p['q8'], p['qs'], p['sums'], p['maxima'],
                  p['mean'], p['vs'], p['inv'], b, h, d, p['starts'][0],
                  p['lengths'][0], qpad, n, tiles, triton.next_power_of_2(tiles), *qstride)
        kv_args = (p['mean'], p['inv'], p['k8'], p['ks'], p['v8'],
                   b, h, d, n, pad, *kstride, *vstride)

        def launch(q, k, v, kv_promotion=None):
            if promotion_shape is None:
                stats(k, v, *stats_args, num_warps=4)
            else:
                stats(k, v, *stats_args, kv_promotion[1], *promotion_shape, num_warps=4)
            quant_q(q, *q_args, num_warps=4, enable_fp_fusion=False)
            quant_kv(k, v, *kv_args, V_TILE_D=v_tile_d, num_warps=4, enable_fp_fusion=False)

        return launch

    def prepare(self, q, k, v, segments, kv_promotion=None):
        p = self.plan(q, k, segments)
        b, _, h, d = q.shape
        g, tiles = p['groups'], p['tiles']
        promotion_shape = None
        if kv_promotion is not None:
            start, page = kv_promotion
            if (g != 1 or b != 1 or page.ndim != 4 or page.shape[0] != 2
                    or tuple(page.shape[2:]) != (h, d) or not page.is_contiguous()
                    or page.dtype != k.dtype or page.device != k.device
                    or start < 0 or page.shape[1] <= 0 or start+page.shape[1] > p['ends'][0]
                    or page.untyped_storage().data_ptr() in
                       (k.untyped_storage().data_ptr(), v.untyped_storage().data_ptr())):
                raise ValueError('Invalid owned history page for Sage KV promotion')
            promotion_shape = (start, page.shape[1])
        if g == 1:
            strides = (q.stride()[:3], k.stride()[:3], v.stride()[:3])
            launch_key = strides, promotion_shape
            cached = p.get('single_launcher')
            if cached is None or cached[0] != launch_key:
                cached = p['single_launcher'] = (launch_key, self._single_launcher(p, q.shape, strides, promotion_shape))
            cached[1](q, k, v, kv_promotion)
            return p
        if g != 1:
            _prefix_quant_q[(max(p['qpads'])//16, h, g*b)](
                q, p['q8'], p['qs'], b, h, d, p['starts'], p['lengths'], p['qpads'], p['qo'],
                *q.stride()[:3], num_warps=4, enable_fp_fusion=False)
        _kv_stats[(tiles, d//16, b*h)](k, v, p['sums'], p['maxima'], max(p['ends']), h, d,
            tiles, *k.stride()[:3], *v.stride()[:3], num_warps=4)
        _prefix_stats[(d//16, h, g*b)](k, v, p['sums'], p['maxima'], p['mean'], p['vs'],
            b, h, d, tiles, triton.next_power_of_2(tiles), p['ends'],
            *k.stride()[:3], *v.stride()[:3], VInv=p['inv'], MIN_V=0.,
            # Eight warps keep the combined prefix sum/max reduction below the
            # register pressure of the four-warp variant, including FP16 tails.
            num_warps=8, enable_fp_fusion=False)
        _prefix_quant_k[(max(p['kpads'])//32, h, g*b)](k, p['mean'], p['k8'], p['ks'],
            b, h, d, p['ends'], p['kpads'], p['ko'], *k.stride()[:3],
            num_warps=4, enable_fp_fusion=False)
        _prefix_quant_v[(max(p['kpads'])//128, d//16, g*b*h)](v, p['inv'], p['v8'],
            b, h, d, p['ends'], p['kpads'], p['ko'], *v.stride()[:3],
            num_warps=4, enable_fp_fusion=False)
        return p

    def attention(self, q, k, v, segments, *, lse=False, kv_promotion=None):
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        logsum = torch.full((q.shape[0], q.shape[2], q.shape[1]), -torch.inf,
                            dtype=torch.float32, device=q.device) if lse else None
        if sum(length for _, length, _ in segments) != q.shape[1]:
            out.zero_()  # Future-only Ring partitions have zero mass.
        if segments:
            p = self.prepare(q, k, v, segments, kv_promotion)
            if self.kernel is None:
                from sageattention.sm90_compile import qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
                self.kernel = qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
            for i, (start, length, _) in enumerate(segments):
                qi, ki, vi, qs, ks, vs = p['views'][i]
                # WP writes the whole query output. Avoid two redundant aten
                # slice views on every layer; segmented CP still needs a view.
                target = out if p['whole_output'] else out[:, start:start+length]
                raw_lse = self.kernel(qi, ki, vi, target, qs, ks, vs,
                                      0, 0, 3, q.shape[-1]**-.5, int(lse))
                if lse:
                    _correct_lse[(triton.cdiv(length, 32), q.shape[0]*q.shape[2])](
                        q, p['mean'][i], raw_lse, logsum, q.shape[0], q.shape[2], q.shape[3],
                        q.shape[1], start, length, *q.stride()[:3], *raw_lse.stride(),
                        num_warps=4, enable_fp_fusion=False)
        return (out, logsum) if lse else out
