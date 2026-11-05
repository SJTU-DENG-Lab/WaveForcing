"""StaggeredExchangeMixin: Wave runtime implementation."""
from __future__ import annotations
import torch
from wave_rt.denoiser import settings


class StaggeredExchangeMixin:
    def _stagger_kick(self, recv_srcs, layer: int, gen: int, tick: int) -> None:
        """Prefetch the inbound acquire for `layer`: enqueue on the dedicated
        prefetch stream a device-side wait (>= tick) on every inbound src's
        generation flag for (gen, layer), then record a ready event into
        self._st_ev[layer].  The consumer's compute stream later wait_event()s
        this before reading slab[src, gen, layer] -- so the producer's transfer
        overlaps the consumer's compute of the preceding `stagger_lead` layers.
        No-op past the last layer or when this rank consumes nothing."""
        if not recv_srcs or layer >= self.Ln:
            return
        sh = self._st_prefetch.cuda_stream
        for src in recv_srcs:
            i = self._os_src_index[src]
            self._p2p.stream_wait_u32(
                sh, self._os_flags[i, gen, layer].data_ptr(), tick)
        ev = torch.cuda.Event()
        ev.record(self._st_prefetch)
        self._st_ev[layer] = ev

    def _stagger_bootstrap(self, recv_srcs, gen: int, tick: int) -> None:
        """Kick the first `stagger_lead` layers' inbound acquires before the
        forward (reset the per-tick event map first)."""
        self._st_ev = {}
        for L in range(min(self._stagger_lead, self.Ln)):
            self._stagger_kick(recv_srcs, L, gen, tick)

    def _make_denoise_exch_staggered(self, clean, act_ranks, tick_csf, tick_qf,
                                     store_active):
        """Denoise exchange: one-sided publication (identical to onesided) with a
        PREFETCHED inbound acquire.  My layer-L KV is published into every slower
        rank's slab the instant it is ready; the inbound layer-L acquire was
        already kicked `stagger_lead` layers earlier (on the prefetch stream), so
        this layer only wait_event()s a (usually already-complete) marker instead
        of stalling on the producer's same-layer flag.  Kicks the L+lead acquire
        so it overlaps this layer's compute.  Assembled context (aK + liveK +
        inflK, cat) and numerics are byte-identical to causal-sync / onesided."""
        store_rank, r = self.store_rank, self.rank
        tick = self._cur_tick
        gen = tick % self._os_gt
        recv_srcs = [n for n in act_ranks if n > r] + (
            [store_rank] if store_active else [])
        # nearest-first fan-out (see _make_denoise_exch_prefetch); WAVE_ORIG_SEND_ORDER
        # restores farthest-first.
        if settings.ORIGINAL_SEND_ORDER:
            send_dsts = [m for m in act_ranks if m < r]
        else:
            send_dsts = sorted([m for m in act_ranks if m < r], reverse=True)
        infl_ranks = [n for n in act_ranks if n >= r]
        lead = self._stagger_lead

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            L = layer_idx
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            # producer role: publish my layer-L KV into every slower rank's slab
            self._publish_mine(mine, send_dsts, L, gen, tick)
            if _lp is not None:
                _e_pub = torch.cuda.Event(enable_timing=True); _e_pub.record()
            # consumer role: gate on the PREFETCHED acquire for layer L, then kick
            # the acquire for L+lead so it overlaps this layer's compute.
            # residual_wait: CUDA-event pair AROUND ONLY the gate (no kick, no sync)
            # -> the pure exposed wait on the compute stream.  If the prefetch has
            # already landed (perfect hiding) the stream never stalls and w0->w1 ~ 0.
            _e_w0 = _e_w1 = None
            if recv_srcs:
                if _lp is not None:
                    _e_w0 = torch.cuda.Event(enable_timing=True); _e_w0.record()
                ev = self._st_ev.pop(L, None)
                if ev is not None:
                    torch.cuda.current_stream().wait_event(ev)
                else:
                    # safety fallback (should not happen: L is always kicked by the
                    # bootstrap for L<lead or by exch(L-lead) otherwise) -- gate
                    # directly on the compute stream so the read can't race.
                    sh = torch.cuda.current_stream().cuda_stream
                    for src in recv_srcs:
                        i = self._os_src_index[src]
                        self._p2p.stream_wait_u32(
                            sh, self._os_flags[i, gen, L].data_ptr(), tick)
                if _lp is not None:
                    # record BEFORE the L+lead kick so residual excludes kick enqueue.
                    _e_w1 = torch.cuda.Event(enable_timing=True); _e_w1.record()
                self._stagger_kick(recv_srcs, L + lead, gen, tick)
            if _lp is not None:
                _e_wt = torch.cuda.Event(enable_timing=True); _e_wt.record()
            remote = {src: self._os_kv[self._os_src_index[src], gen, L]
                      for src in recv_srcs}
            aK, aV = self._anchor_working(
                clean[L], tick_csf, tick_qf,
                post_patch_height, post_patch_width, dim, rope_num_heads)
            liveK, liveV = [], []
            if store_active:
                slot = remote[store_rank]
                liveK = [slot[0].unsqueeze(0)]
                liveV = [slot[1].unsqueeze(0)]
                if not settings.FULL_BROADCAST:
                    # clone out of the gen ring before its next-gen reuse (promoted
                    # into clean[] at tick end -- identical to the onesided path).
                    self._exch_store_slot[L] = slot.clone()
            inflK, inflV = [], []
            for n in infl_ranks:
                s = mine if n == r else remote[n]
                inflK.append(s[0].unsqueeze(0))
                inflV.append(s[1].unsqueeze(0))
            K = torch.cat(aK + liveK + inflK, dim=1)
            V = torch.cat(aV + liveV + inflV, dim=1)
            if _lp is not None:
                _e_ct = torch.cuda.Event(enable_timing=True); _e_ct.record()
                _lp["layers"].append(dict(
                    layer=L, enter=_e_en, pub=_e_pub, wait=_e_wt, cat=_e_ct,
                    w0=_e_w0, w1=_e_w1))
            return K, V

        return exch
