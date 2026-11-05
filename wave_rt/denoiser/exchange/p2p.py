"""P2PExchangeMixin: Wave runtime implementation."""
from __future__ import annotations
import time
import torch
import torch.distributed as dist
from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import debug_log


class P2PExchangeMixin:
    def _drain_p2p_sends(self) -> None:
        """Wait all deferred isends (+ any outstanding prefetch recv works) from this
        tick, then release buffers.  Called after forward_chunk returns and BEFORE any
        WORLD collective, so p2p_pg is quiescent (no cross-PG interleave)."""
        for reqs, _buf in self._p2p_send_pending:
            for rq in reqs:
                rq.wait()
        self._p2p_send_pending = []
        if self._use_prefetch:
            for works in self._pf_works:
                if works:
                    for w in works:
                        try:
                            w.wait()
                        except Exception:
                            pass
            self._pf_slots = [None, None]
            self._pf_works = [None, None]

    def _post_recv(self, slot, recv_srcs):
        """Post one layer's irecvs into ping-pong `slot` on the DEFAULT stream (torch
        runs them on p2p_pg's internal nccl stream regardless).  Returns (bufs, works)
        or (None, None) if no sources."""
        if not recv_srcs:
            return None, None
        # Request noise can be FP32 even when every published KV is BF16.
        # Use it only for device placement; send and recv must agree in dtype
        # and byte count before NCCL runs, not merely cast before attention.
        bufs = {}
        if self._edge_ts:
            # DIAGNOSTIC ONLY (WAVE_EDGE_TS): post each source's irecv as its OWN
            # batched group so it returns a distinct Work -> the exch closure can
            # wait + time each source separately.  The production path below coalesces
            # all srcs into ONE batch_isend_irecv, which returns a SINGLE Work for the
            # whole batch -> only one wait event -> per-edge arrival skew collapses to
            # 0.  P2P send/recv still matches per-peer, so correctness is preserved;
            # this only fans the recv into N groups (minor timing perturbation, OK for
            # a measurement run).  works[] order == recv_srcs order (per-src labeling).
            works = []
            for src in recv_srcs:
                b = self._buffers.get(('recv', slot, src), self.KV_SHAPE,
                                      self.be.full_noise_bcthw, dtype=self.dtype)
                bufs[src] = b
                works.extend(dist.batch_isend_irecv(
                    [dist.P2POp(dist.irecv, b, src, self.p2p_pg)]))
            self._pf_slots[slot] = bufs
            self._pf_works[slot] = works
            return bufs, works
        ops = []
        for src in recv_srcs:
            if self._machprof:
                _ta = time.perf_counter()
                b = self._buffers.get(('recv', slot, src), self.KV_SHAPE,
                                      self.be.full_noise_bcthw, dtype=self.dtype)
                self._mp_alloc_cpu += (time.perf_counter() - _ta) * 1000.0
            else:
                b = self._buffers.get(('recv', slot, src), self.KV_SHAPE,
                                      self.be.full_noise_bcthw, dtype=self.dtype)
            bufs[src] = b
            ops.append(dist.P2POp(dist.irecv, b, src, self.p2p_pg))
        if self._machprof:
            _ts = time.perf_counter()
            works = dist.batch_isend_irecv(ops)
            self._mp_submit_cpu += (time.perf_counter() - _ts) * 1000.0
        else:
            works = dist.batch_isend_irecv(ops)
        self._pf_slots[slot] = bufs
        self._pf_works[slot] = works
        return bufs, works

    def _bootstrap_recv(self, recv_srcs) -> None:
        """Post layer-0's irecvs before the forward (prefetch bootstrap)."""
        self._pf_slots = [None, None]
        self._pf_works = [None, None]
        if settings.DEBUG:
            debug_log(self.rank, f"t={self._cur_tick} bootstrap recv srcs={recv_srcs} submit")
        self._post_recv(0, recv_srcs)
        if settings.DEBUG:
            debug_log(self.rank, f"t={self._cur_tick} bootstrap recv submitted")

    def _send_mine(self, mine, dsts) -> None:
        """isend my layer KV to `dsts` on the DEFAULT stream (torch auto syncStream +
        recordStream keeps `mine` alive across the async send); stash for drain."""
        if not dsts:
            return
        if self._machprof:
            _ts = time.perf_counter()
            reqs = dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, mine, d, self.p2p_pg) for d in dsts])
            self._mp_submit_cpu += (time.perf_counter() - _ts) * 1000.0
        else:
            reqs = dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, mine, d, self.p2p_pg) for d in dsts])
        self._p2p_send_pending.append((reqs, mine))

    def _make_denoise_exch_prefetch(self, clean, act_ranks, tick_csf, tick_qf,
                                    store_active):
        """Depth-one prefetch: send current KV before posting the next recv.

        Bootstrap posts layer 0; the next layer's receive overlaps the current
        attention and remaining block. Only the current receive gates attention.
        """
        store_rank, r = self.store_rank, self.rank
        recv_srcs = [n for n in act_ranks if n > r] + (
            [store_rank] if store_active else [])
        # nearest-first: send to r-1 (the most urgent consumer, next on the
        # cascade) before r-2..0 so the tightest dependency resolves first.
        # WAVE_ORIG_SEND_ORDER=1 restores the pre-fix farthest-first (ascending).
        if settings.ORIGINAL_SEND_ORDER:
            send_dsts = [m for m in act_ranks if m < r]
        else:
            send_dsts = sorted([m for m in act_ranks if m < r], reverse=True)
        infl_ranks = [n for n in act_ranks if n >= r]
        Ln = self.Ln

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            L = layer_idx
            if settings.DEBUG:
                debug_log(r, f"t={self._cur_tick} L={L} overlap exchange enter recv={recv_srcs} send={send_dsts}")
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            # The current send must precede the future recv on the NCCL stream.
            if settings.DEBUG:
                debug_log(r, f"t={self._cur_tick} L={L} send submit")
            self._send_mine(mine, send_dsts)
            if settings.DEBUG:
                debug_log(r, f"t={self._cur_tick} L={L} send submitted")
            if recv_srcs and L + 1 < Ln:
                if settings.DEBUG:
                    debug_log(r, f"t={self._cur_tick} L={L} prefetch L={L+1} submit")
                self._post_recv((L + 1) % 2, recv_srcs)
                if settings.DEBUG:
                    debug_log(r, f"t={self._cur_tick} L={L} prefetch submitted")
            if _lp is not None:
                _e_pub = torch.cuda.Event(enable_timing=True); _e_pub.record()
            # WAVE_EDGE_TS (rank0): mark exch entry -- prior layer's compute is done
            # in stream order, so this is "attention would start here if no recv wait".
            _ets = self._edge_ts and r == 0
            _ev_enter = _ev_recv = None
            if _ets:
                _ev_enter = torch.cuda.Event(enable_timing=True)
                _ev_enter.record()
                _ev_recv = []
            # gate: wait layer L's recv work (enqueues cudaStreamWaitEvent on default
            # stream; non-blocking on CPU -> overlap preserved)
            works = self._pf_works[L % 2]
            if works:
                for _i, w in enumerate(works):
                    if settings.DEBUG:
                        debug_log(r, f"t={self._cur_tick} L={L} recv wait={_i}")
                    w.wait()
                    if settings.DEBUG:
                        debug_log(r, f"t={self._cur_tick} L={L} recv wait enqueued={_i}")
                    if _ets:
                        # event after each source's wait (recv_srcs order); serialized
                        # on the stream, so it captures cumulative "done after waiting
                        # sources 0..i" -- the last one == all recvs in for this layer.
                        _e = torch.cuda.Event(enable_timing=True)
                        _e.record()
                        _ev_recv.append((recv_srcs[_i], _e))
            if _lp is not None:
                _e_wt = torch.cuda.Event(enable_timing=True); _e_wt.record()
            bufs = self._pf_slots[L % 2] or {}
            aK, aV = self._anchor_working(
                clean[L], tick_csf, tick_qf,
                post_patch_height, post_patch_width, dim, rope_num_heads)
            liveK, liveV = [], []
            if store_active:
                liveK = [bufs[store_rank][0].unsqueeze(0)]
                liveV = [bufs[store_rank][1].unsqueeze(0)]
                if not settings.FULL_BROADCAST:
                    # clone: bufs is a ping-pong slot reused by layer L+2's recv.
                    self._exch_store_slot[L] = bufs[store_rank].clone()
            inflK, inflV = [], []
            for n in infl_ranks:
                src = mine if n == r else bufs[n]
                inflK.append(src[0].unsqueeze(0))
                inflV.append(src[1].unsqueeze(0))
            segK = aK + liveK + inflK
            segV = aV + liveV + inflV
            _cts = self._cat_ts and r == 0 and self._cur_tick >= self.world - 1
            _ev_c0 = None
            if _cts:
                _ev_c0 = torch.cuda.Event(enable_timing=True)
                _ev_c0.record()
            if settings.DISABLE_CAT:
                # slice-assign into a preallocated ping-pong buffer (no per-layer
                # cat allocation); byte-identical layout to torch.cat(dim=1).
                total = sum(s.shape[1] for s in segK)
                buf = self._nocat_buf[L % 2]
                if buf is None or buf[0].shape[1] != total:
                    Hh, Dd = segK[0].shape[2], segK[0].shape[3]
                    full_k = torch.empty((1, total, Hh, Dd),
                                         device=self.device, dtype=self.dtype)
                    full_v = torch.empty((1, total, Hh, Dd),
                                         device=self.device, dtype=self.dtype)
                    self._nocat_buf[L % 2] = (full_k, full_v)
                else:
                    full_k, full_v = buf
                off = 0
                for sk, sv in zip(segK, segV):
                    n = sk.shape[1]
                    full_k[:, off:off + n].copy_(sk)
                    full_v[:, off:off + n].copy_(sv)
                    off += n
                K, V = full_k, full_v
            else:
                K = torch.cat(segK, dim=1)
                V = torch.cat(segV, dim=1)
            if _lp is not None:
                _e_ct = torch.cuda.Event(enable_timing=True); _e_ct.record()
                _lp["layers"].append(dict(
                    layer=L, enter=_e_en, pub=_e_pub, wait=_e_wt, cat=_e_ct))
            if _cts:
                _ev_c1 = torch.cuda.Event(enable_timing=True)
                _ev_c1.record()
                self._cat_records.append(
                    dict(tick=self._cur_tick, layer=L, start=_ev_c0, end=_ev_c1))
            if _ets:
                # event after the K/V cat: attention is about to start on this layer.
                _ev_attn = torch.cuda.Event(enable_timing=True)
                _ev_attn.record()
                self._edge_records.append(dict(
                    tick=self._cur_tick, layer=L, srcs=list(recv_srcs),
                    enter=_ev_enter, recv=_ev_recv, attn=_ev_attn))
            if settings.DEBUG:
                debug_log(r, f"t={self._cur_tick} L={L} overlap exchange assembled")
            return K, V

        if settings.DEBUG:
            def trace_attention(phase, layer, **details):
                debug_log(r, f't={self._cur_tick} L={layer} backend {phase} {details}')
            exch.trace_attention = trace_attention
        return exch

    def _make_store_exch_prefetch(self, clean, store_kv, c, act_ranks):
        """Store variant for prefetch overlap: only sends its clean KV (default
        stream) to every active denoise rank; no recv.  Local context assembly."""
        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block, write_only=False):
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            self._send_mine(mine, sorted(act_ranks, reverse=True))  # EDF
            if _lp is not None:
                _e_pub = torch.cuda.Event(enable_timing=True); _e_pub.record()
            store_kv[layer_idx] = (roped_key, unroped_key, value)
            if write_only:
                return None, None
            aK, aV = self._anchor_working(
                clean[layer_idx], c * self.npb, self.npb,
                post_patch_height, post_patch_width, dim, rope_num_heads)
            K = torch.cat(aK + [roped_key], dim=1)
            V = torch.cat(aV + [value], dim=1)
            store_kv[layer_idx] = (roped_key, unroped_key, value)
            if _lp is not None:
                _e_ct = torch.cuda.Event(enable_timing=True); _e_ct.record()
                _lp["layers"].append(dict(
                    layer=layer_idx, enter=_e_en, pub=_e_pub, wait=_e_pub, cat=_e_ct))
            return K, V

        exch.requires_raw_key = settings.FULL_BROADCAST
        return exch

    def _make_denoise_exch_overlap(self, clean, act_ranks, tick_csf, tick_qf,
                                   store_active):
        """Causal denoise exchange via async P2P cascade on self.p2p_pg.
        Same assembled context (hence same numerics) as _make_denoise_exch with
        kv_context=causal, but: recv layer-L KV from faster/higher ranks (n>r) +
        store (wait these), isend own KV to slower/lower ranks (m<r, DEFERRED)."""
        store_rank = self.store_rank
        r = self.rank
        # sets derived purely from act_ranks/store_active -> identical on all ranks
        recv_srcs = [n for n in act_ranks if n > r] + ([store_rank] if store_active else [])
        send_dsts = [m for m in act_ranks if m < r]
        infl_ranks = [n for n in act_ranks if n >= r]   # assembly order (incl. self)
        pg = self.p2p_pg

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            # fresh contiguous send buffer each layer (must outlive the layer)
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            recv_bufs = {
                src: torch.empty(self.KV_SHAPE, device=self.device, dtype=self.dtype)
                for src in recv_srcs
            }
            # post recvs (needed to proceed) + sends (deferred)
            if recv_srcs:
                recv_reqs = dist.batch_isend_irecv(
                    [dist.P2POp(dist.irecv, recv_bufs[s], s, pg) for s in recv_srcs]
                )
            else:
                recv_reqs = []
            if send_dsts:
                send_reqs = dist.batch_isend_irecv(
                    [dist.P2POp(dist.isend, mine, d, pg) for d in send_dsts]
                )
                self._p2p_send_pending.append((send_reqs, mine))
            for rq in recv_reqs:
                rq.wait()  # wait ONLY recvs; sends stay in flight -> hidden
            # assemble EXACTLY as causal-sync: aK + liveK + inflK (n>=r)
            aK, aV = self._anchor_working(
                clean[layer_idx], tick_csf, tick_qf,
                post_patch_height, post_patch_width, dim, rope_num_heads,
            )
            liveK, liveV = [], []
            if store_active:
                liveK = [recv_bufs[store_rank][0].unsqueeze(0)]
                liveV = [recv_bufs[store_rank][1].unsqueeze(0)]
                if not settings.FULL_BROADCAST:
                    self._exch_store_slot[layer_idx] = recv_bufs[store_rank].clone()
            inflK, inflV = [], []
            for n in infl_ranks:
                src = mine if n == r else recv_bufs[n]
                inflK.append(src[0].unsqueeze(0))
                inflV.append(src[1].unsqueeze(0))
            K = torch.cat(aK + liveK + inflK, dim=1)
            V = torch.cat(aV + liveV + inflV, dim=1)
            return K, V

        return exch

    def _make_store_exch_overlap(self, clean, store_kv, c, act_ranks):
        """Store exchange via async P2P: isend own clean KV to every active denoise
        rank (DEFERRED), no recv.  Local context assembly is identical to sync."""
        send_dsts = sorted(act_ranks, reverse=True)  # EDF: immediate consumer first
        pg = self.p2p_pg

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block, write_only=False):
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            if send_dsts:
                send_reqs = dist.batch_isend_irecv(
                    [dist.P2POp(dist.isend, mine, d, pg) for d in send_dsts]
                )
                self._p2p_send_pending.append((send_reqs, mine))
            store_kv[layer_idx] = (roped_key, unroped_key, value)
            if write_only:
                return None, None
            aK, aV = self._anchor_working(
                clean[layer_idx], c * self.npb, self.npb,
                post_patch_height, post_patch_width, dim, rope_num_heads,
            )
            K = torch.cat(aK + [roped_key], dim=1)
            V = torch.cat(aV + [value], dim=1)
            store_kv[layer_idx] = (roped_key, unroped_key, value)
            return K, V

        exch.requires_raw_key = settings.FULL_BROADCAST
        return exch
