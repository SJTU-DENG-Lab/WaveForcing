"""Two-sided P2P exchange implementations."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import record_cuda_event


class P2PExchangeMixin:
    def _drain_p2p_sends(self) -> None:
        """Drain P2P work before the next WORLD collective."""
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
        """Post one layer of receives into a ping-pong slot."""
        if not recv_srcs:
            return None, None
        bufs = {}
        if self._edge_ts:
            # Separate work handles preserve per-source timing labels.
            works = []
            for src in recv_srcs:
                b = torch.empty(self._kv_shape, device=self.device, dtype=self.dtype)
                bufs[src] = b
                works.extend(
                    dist.batch_isend_irecv(
                        [dist.P2POp(dist.irecv, b, src, self.p2p_pg)]
                    )
                )
            self._pf_slots[slot] = bufs
            self._pf_works[slot] = works
            return bufs, works
        ops = []
        for src in recv_srcs:
            if self._machprof:
                _ta = time.perf_counter()
                b = torch.empty(self._kv_shape, device=self.device, dtype=self.dtype)
                self._mp_alloc_cpu += (time.perf_counter() - _ta) * 1000.0
            else:
                b = torch.empty(self._kv_shape, device=self.device, dtype=self.dtype)
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
        """Post the initial receive set before the forward pass."""
        self._pf_slots = [None, None]
        self._pf_works = [None, None]
        self._post_recv(0, recv_srcs)

    def _send_mine(self, mine, dsts) -> None:
        """Send one KV layer and retain its buffer until completion."""
        if not dsts:
            return
        if self._machprof:
            _ts = time.perf_counter()
            reqs = dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, mine, d, self.p2p_pg) for d in dsts]
            )
            self._mp_submit_cpu += (time.perf_counter() - _ts) * 1000.0
        else:
            reqs = dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, mine, d, self.p2p_pg) for d in dsts]
            )
        self._p2p_send_pending.append((reqs, mine))

    def _make_denoise_exch_prefetch(
        self, clean, act_ranks, tick_csf, tick_qf, store_active
    ):
        """Build the depth-1 prefetched causal exchange."""
        store_rank, r = self.store_rank, self.rank
        recv_srcs = [n for n in act_ranks if n > r] + (
            [store_rank] if store_active else []
        )
        # Nearest-first resolves the tightest downstream dependency first.
        if settings.ORIGINAL_SEND_ORDER:
            send_dsts = [m for m in act_ranks if m < r]
        else:
            send_dsts = sorted([m for m in act_ranks if m < r], reverse=True)
        infl_ranks = [n for n in act_ranks if n >= r]
        Ln = self.Ln

        def exch(
            *,
            layer_idx,
            roped_query,
            roped_key,
            unroped_key,
            value,
            current_start,
            tokens_per_frame,
            post_patch_height,
            post_patch_width,
            dim,
            rope_num_heads,
            num_frames_per_block,
        ):
            L = layer_idx
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = record_cuda_event()
            # Send before posting L+1; reversing this order deadlocks the cascade.
            self._send_mine(mine, send_dsts)
            if recv_srcs and L + 1 < Ln:
                self._post_recv((L + 1) % 2, recv_srcs)
            if _lp is not None:
                _e_pub = record_cuda_event()
            _ets = self._edge_ts and r == 0
            _ev_enter = _ev_recv = None
            if _ets:
                _ev_enter = record_cuda_event()
                _ev_recv = []
            works = self._pf_works[L % 2]
            if works:
                for _i, w in enumerate(works):
                    w.wait()
                    if _ets:
                        _e = record_cuda_event()
                        _ev_recv.append((recv_srcs[_i], _e))
            if _lp is not None:
                _e_wt = record_cuda_event()
            bufs = self._pf_slots[L % 2] or {}
            aK, aV = self._anchor_working(
                clean[L],
                tick_csf,
                tick_qf,
                post_patch_height,
                post_patch_width,
                dim,
                rope_num_heads,
            )
            liveK, liveV = [], []
            if store_active:
                liveK = [bufs[store_rank][0].unsqueeze(0)]
                liveV = [bufs[store_rank][1].unsqueeze(0)]
                if not settings.FULL_BROADCAST:
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
                _ev_c0 = record_cuda_event()
            if settings.DISABLE_CAT:
                total = sum(s.shape[1] for s in segK)
                buf = self._nocat_buf[L % 2]
                if buf is None or buf[0].shape[1] != total:
                    Hh, Dd = segK[0].shape[2], segK[0].shape[3]
                    full_k = torch.empty(
                        (1, total, Hh, Dd), device=self.device, dtype=self.dtype
                    )
                    full_v = torch.empty(
                        (1, total, Hh, Dd), device=self.device, dtype=self.dtype
                    )
                    self._nocat_buf[L % 2] = (full_k, full_v)
                else:
                    full_k, full_v = buf
                off = 0
                for sk, sv in zip(segK, segV):
                    n = sk.shape[1]
                    full_k[:, off : off + n].copy_(sk)
                    full_v[:, off : off + n].copy_(sv)
                    off += n
                K, V = full_k, full_v
            else:
                K = torch.cat(segK, dim=1)
                V = torch.cat(segV, dim=1)
            if _lp is not None:
                _e_ct = record_cuda_event()
                _lp["layers"].append(
                    dict(layer=L, enter=_e_en, pub=_e_pub, wait=_e_wt, cat=_e_ct)
                )
            if _cts:
                _ev_c1 = record_cuda_event()
                self._cat_records.append(
                    dict(tick=self._cur_tick, layer=L, start=_ev_c0, end=_ev_c1)
                )
            if _ets:
                _ev_attn = record_cuda_event()
                self._edge_records.append(
                    dict(
                        tick=self._cur_tick,
                        layer=L,
                        srcs=list(recv_srcs),
                        enter=_ev_enter,
                        recv=_ev_recv,
                        attn=_ev_attn,
                    )
                )
            return K, V

        return exch

    def _make_store_exch_prefetch(self, clean, store_kv, c, act_ranks):
        """Store variant for prefetch overlap: only sends its clean KV (default
        stream) to every active denoise rank; no recv.  Local context assembly."""

        def exch(
            *,
            layer_idx,
            roped_query,
            roped_key,
            unroped_key,
            value,
            current_start,
            tokens_per_frame,
            post_patch_height,
            post_patch_width,
            dim,
            rope_num_heads,
            num_frames_per_block,
        ):
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = record_cuda_event()
            self._send_mine(mine, sorted(act_ranks, reverse=True))  # EDF
            if _lp is not None:
                _e_pub = record_cuda_event()
            K, V = self._assemble_store_context(
                clean[layer_idx],
                c,
                roped_key,
                value,
                post_patch_height,
                post_patch_width,
                dim,
                rope_num_heads,
            )
            store_kv[layer_idx] = (roped_key, unroped_key, value)
            if _lp is not None:
                _e_ct = record_cuda_event()
                _lp["layers"].append(
                    dict(
                        layer=layer_idx, enter=_e_en, pub=_e_pub, wait=_e_pub, cat=_e_ct
                    )
                )
            return K, V

        return exch

    def _make_denoise_exch_overlap(
        self, clean, act_ranks, tick_csf, tick_qf, store_active
    ):
        """Causal denoise exchange via async P2P cascade on self.p2p_pg.
        Same assembled context (hence same numerics) as _make_denoise_exch with
        kv_context=causal, but: recv layer-L KV from faster/higher ranks (n>r) +
        store (wait these), isend own KV to slower/lower ranks (m<r, DEFERRED)."""
        store_rank = self.store_rank
        r = self.rank
        # sets derived purely from act_ranks/store_active -> identical on all ranks
        recv_srcs = [n for n in act_ranks if n > r] + (
            [store_rank] if store_active else []
        )
        send_dsts = [m for m in act_ranks if m < r]
        infl_ranks = [n for n in act_ranks if n >= r]  # assembly order (incl. self)
        pg = self.p2p_pg

        def exch(
            *,
            layer_idx,
            roped_query,
            roped_key,
            unroped_key,
            value,
            current_start,
            tokens_per_frame,
            post_patch_height,
            post_patch_width,
            dim,
            rope_num_heads,
            num_frames_per_block,
        ):
            # fresh contiguous send buffer each layer (must outlive the layer)
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            recv_bufs = {
                src: torch.empty(self._kv_shape, device=self.device, dtype=self.dtype)
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
                clean[layer_idx],
                tick_csf,
                tick_qf,
                post_patch_height,
                post_patch_width,
                dim,
                rope_num_heads,
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
        """Build the store-side asynchronous P2P exchange."""
        send_dsts = sorted(act_ranks, reverse=True)
        pg = self.p2p_pg

        def exch(
            *,
            layer_idx,
            roped_query,
            roped_key,
            unroped_key,
            value,
            current_start,
            tokens_per_frame,
            post_patch_height,
            post_patch_width,
            dim,
            rope_num_heads,
            num_frames_per_block,
        ):
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            if send_dsts:
                send_reqs = dist.batch_isend_irecv(
                    [dist.P2POp(dist.isend, mine, d, pg) for d in send_dsts]
                )
                self._p2p_send_pending.append((send_reqs, mine))
            K, V = self._assemble_store_context(
                clean[layer_idx],
                c,
                roped_key,
                value,
                post_patch_height,
                post_patch_width,
                dim,
                rope_num_heads,
            )
            store_kv[layer_idx] = (roped_key, unroped_key, value)
            return K, V

        return exch
