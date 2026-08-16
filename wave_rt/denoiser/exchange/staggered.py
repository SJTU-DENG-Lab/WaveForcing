"""One-sided exchange with prefetched generation acquires."""

from __future__ import annotations

import torch

from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import record_cuda_event


class StaggeredExchangeMixin:
    def _stagger_kick(self, recv_srcs, layer: int, gen: int, tick: int) -> None:
        """Prefetch one layer's inbound generation-flag acquires."""
        if not recv_srcs or layer >= self.Ln:
            return
        sh = self._st_prefetch.cuda_stream
        for src in recv_srcs:
            i = self._os_src_index[src]
            self._p2p.stream_wait_u32(
                sh, self._os_flags[i, gen, layer].data_ptr(), tick
            )
        ev = torch.cuda.Event()
        ev.record(self._st_prefetch)
        self._st_ev[layer] = ev

    def _stagger_bootstrap(self, recv_srcs, gen: int, tick: int) -> None:
        """Prefetch the initial staggered acquire window."""
        self._st_ev = {}
        for L in range(min(self._stagger_lead, self.Ln)):
            self._stagger_kick(recv_srcs, L, gen, tick)

    def _make_denoise_exch_staggered(
        self, clean, act_ranks, tick_csf, tick_qf, store_active
    ):
        """Build one-sided exchange with prefetched inbound acquires."""
        store_rank, r = self.store_rank, self.rank
        tick = self._cur_tick
        gen = tick % self._os_gt
        recv_srcs = [n for n in act_ranks if n > r] + (
            [store_rank] if store_active else []
        )
        if settings.ORIGINAL_SEND_ORDER:
            send_dsts = [m for m in act_ranks if m < r]
        else:
            send_dsts = sorted([m for m in act_ranks if m < r], reverse=True)
        infl_ranks = [n for n in act_ranks if n >= r]
        lead = self._stagger_lead

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
            self._publish_mine(mine, send_dsts, L, gen, tick)
            if _lp is not None:
                _e_pub = record_cuda_event()
            _e_w0 = _e_w1 = None
            if recv_srcs:
                if _lp is not None:
                    _e_w0 = record_cuda_event()
                ev = self._st_ev.pop(L, None)
                if ev is not None:
                    torch.cuda.current_stream().wait_event(ev)
                else:
                    # Preserve correctness if a prefetched event is unexpectedly absent.
                    sh = torch.cuda.current_stream().cuda_stream
                    for src in recv_srcs:
                        i = self._os_src_index[src]
                        self._p2p.stream_wait_u32(
                            sh, self._os_flags[i, gen, L].data_ptr(), tick
                        )
                if _lp is not None:
                    _e_w1 = record_cuda_event()
                self._stagger_kick(recv_srcs, L + lead, gen, tick)
            if _lp is not None:
                _e_wt = record_cuda_event()
            remote = {
                src: self._os_kv[self._os_src_index[src], gen, L] for src in recv_srcs
            }
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
                slot = remote[store_rank]
                liveK = [slot[0].unsqueeze(0)]
                liveV = [slot[1].unsqueeze(0)]
                if not settings.FULL_BROADCAST:
                    self._exch_store_slot[L] = slot.clone()
            inflK, inflV = [], []
            for n in infl_ranks:
                s = mine if n == r else remote[n]
                inflK.append(s[0].unsqueeze(0))
                inflV.append(s[1].unsqueeze(0))
            K = torch.cat(aK + liveK + inflK, dim=1)
            V = torch.cat(aV + liveV + inflV, dim=1)
            if _lp is not None:
                _e_ct = record_cuda_event()
                _lp["layers"].append(
                    dict(
                        layer=L,
                        enter=_e_en,
                        pub=_e_pub,
                        wait=_e_wt,
                        cat=_e_ct,
                        w0=_e_w0,
                        w1=_e_w1,
                    )
                )
            return K, V

        return exch
