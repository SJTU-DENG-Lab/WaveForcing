"""Fixed-slot paged KV exchange."""

from __future__ import annotations

import torch
import torch.distributed as dist

from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import debug_log, record_cuda_event


class PagedExchangeMixin:
    def _init_paged(self) -> None:
        """Create fixed-slot attention buffers for in-place one-sided exchange."""
        self._init_onesided()
        from wave_rt.distributed import ipc as p2p

        dev = self.device
        c = self.rank
        GT = self._os_gt
        Ln = self.Ln
        npb = self.npb
        T = self.tokens_per_chunk
        H = self._kv_shape[2]
        D = self._kv_shape[3]
        wmax_frames = self.max_attn_frames - self.rf_step * npb - npb
        wmax_chunks = max(0, wmax_frames // npb)
        self._pg_naw = 1 + wmax_chunks
        self._pg_wmax = self._pg_naw + 1 + self.rf_step
        Wmax = self._pg_wmax
        if self._os_fp8:
            raise RuntimeError("paged mode does not support WAVE_FP8KV_ONESIDED yet")
        elem = torch.empty(0, dtype=self.dtype).element_size()

        if c < self.rf_step:
            need = 2 * GT * Ln * Wmax * T * H * D * elem
            free_b, _tot = torch.cuda.mem_get_info()
            if need > free_b * 0.9:
                raise RuntimeError(
                    f"[paged r{c}] KV attn buffers {need / 2**30:.1f} GiB exceed 90% of "
                    f"free {free_b / 2**30:.1f} GiB (GT={GT} Ln={Ln} Wmax={Wmax})"
                )
            self._pg_k = torch.zeros(
                (GT, Ln, Wmax * T, H, D), device=dev, dtype=self.dtype
            )
            self._pg_v = torch.zeros(
                (GT, Ln, Wmax * T, H, D), device=dev, dtype=self.dtype
            )
            k_h, k_off = p2p.ipc_export(self._pg_k)
            v_h, v_off = p2p.ipc_export(self._pg_v)
            debug_log(
                c,
                f"paged: kv buffers {need / 2**30:.2f}GiB naw={self._pg_naw} Wmax={Wmax}",
            )
        else:
            self._pg_k = self._pg_v = None
            k_h = k_off = v_h = v_off = None

        my_info = dict(
            rank=c,
            k_h=k_h,
            k_off=k_off,
            v_h=v_h,
            v_off=v_off,
            Wmax=Wmax,
            Ln=Ln,
            GT=GT,
            THD=T * H * D,
            elem=elem,
        )
        gathered: list = [None] * self.world
        dist.all_gather_object(gathered, my_info)
        world_info = {g["rank"]: g for g in gathered}

        if c < self.rf_step:
            my_dsts = list(range(0, c))
        else:
            my_dsts = list(range(0, self.rf_step))
        self._pg_dst = {}
        for m in my_dsts:
            gi = world_info[m]
            if gi["k_h"] is None:
                continue
            kb = p2p.ipc_open_handle(gi["k_h"])
            vb = p2p.ipc_open_handle(gi["v_h"])
            od = self._os_dst[m]
            self._pg_dst[m] = dict(
                k_slab=kb + gi["k_off"],
                v_slab=vb + gi["v_off"],
                fl_slab=od["fl_slab"],
                i=od["i"],
                GT=gi["GT"],
                Ln=gi["Ln"],
                Wmax=gi["Wmax"],
                THD=gi["THD"],
                elem=gi["elem"],
            )

        if self._pg_dst:
            warm = torch.zeros((T, H, D), device=dev, dtype=self.dtype)
            sh = self._os_scopy.cuda_stream
            for m, d in self._pg_dst.items():
                p2p.memcpy_dtod_async(
                    d["k_slab"], warm.data_ptr(), warm.numel() * elem, sh
                )
                p2p.memcpy_dtod_async(
                    d["v_slab"], warm.data_ptr(), warm.numel() * elem, sh
                )
            self._os_scopy.synchronize()
        dist.barrier()

    def _paged_steady(self, store_active, act_ranks) -> bool:
        """Return whether fixed paged offsets are valid on every rank."""
        return (
            store_active
            and len(act_ranks) == self.rf_step
            and (self._cur_tick - self.store_rank) >= self._pg_naw
        )

    def _publish_paged(self, k, v, dst_slots, layer: int, gen: int, tick: int) -> None:
        """Publish K and V directly into fixed consumer slots."""
        if not self._pg_dst or not dst_slots:
            return
        scopy = self._os_scopy
        ev = torch.cuda.Event()
        ev.record()
        scopy.wait_event(ev)
        sh = scopy.cuda_stream
        kb = k.numel() * k.element_size()
        vb = v.numel() * v.element_size()
        kp = k.data_ptr()
        vp = v.data_ptr()
        for m, slot in dst_slots:
            d = self._pg_dst.get(m)
            if d is None:
                continue
            base = ((gen * d["Ln"] + layer) * d["Wmax"] + slot) * d["THD"] * d["elem"]
            self._p2p.memcpy_dtod_async(d["k_slab"] + base, kp, kb, sh)
            self._p2p.memcpy_dtod_async(d["v_slab"] + base, vp, vb, sh)
            fslot = (d["i"] * d["GT"] + gen) * d["Ln"] + layer
            self._p2p.stream_write_u32(sh, d["fl_slab"] + fslot * 4, tick)
        k.record_stream(scopy)
        v.record_stream(scopy)
        self._os_pending.append(k)
        self._os_pending.append(v)

    def _wait_paged_flag(self, src: int, layer: int, gen: int, tick: int) -> None:
        """Consumer: device-side acquire on src's generation flag for (gen, layer).
        The payload is already in _pg_k/_pg_v (DMA'd to its final slot), so there is
        nothing to return -- just gate the compute stream before attention reads it."""
        i = self._os_src_index[src]
        flag_addr = self._os_flags[i, gen, layer].data_ptr()
        sh = torch.cuda.current_stream().cuda_stream
        self._p2p.stream_wait_u32(sh, flag_addr, tick)

    def _make_denoise_exch_paged(
        self, clean, act_ranks, tick_csf, tick_qf, store_active
    ):
        """Build denoise exchange into fixed paged slots without concatenation."""
        store_rank, r = self.store_rank, self.rank
        tick = self._cur_tick
        gen = tick % self._os_gt
        T = self.tokens_per_chunk
        npb = self.npb
        # This geometry must remain identical to _anchor_working on every rank.
        clean_len = min(8, max(0, tick - store_rank))
        wmax_chunks = max(0, (self.max_attn_frames - len(act_ranks) * npb - npb) // npb)
        naw = 0 if clean_len == 0 else 1 + min(wmax_chunks, clean_len - 1)
        live = 1 if store_active else 0
        base_infl = naw + live
        infl_ranks = [n for n in act_ranks if n >= r]
        active_slots = base_infl + len(infl_ranks)
        if active_slots > self._pg_wmax:
            raise RuntimeError(
                f"[paged r{r}] active_slots={active_slots} > Wmax={self._pg_wmax} "
                f"(t={tick} naw={naw} live={live} n_infl={len(infl_ranks)})"
            )
        self_off = base_infl * T
        live_off = naw * T
        active_len = active_slots * T
        recv_srcs = [n for n in act_ranks if n > r] + (
            [store_rank] if store_active else []
        )
        send_slots = []
        for m in act_ranks:
            if m >= r:
                continue
            idx = sum(1 for n in act_ranks if m <= n < r)
            send_slots.append((m, base_infl + idx))

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
            k = roped_key[0].contiguous()
            v = value[0].contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = record_cuda_event()
            self._publish_paged(k, v, send_slots, L, gen, tick)
            if _lp is not None:
                _e_pub = record_cuda_event()
            Kb = self._pg_k[gen, L]
            Vb = self._pg_v[gen, L]
            aK, aV = self._anchor_working(
                clean[L],
                tick_csf,
                tick_qf,
                post_patch_height,
                post_patch_width,
                dim,
                rope_num_heads,
            )
            off = 0
            for ak, av in zip(aK, aV):
                Kb[off : off + T].copy_(ak[0])
                Vb[off : off + T].copy_(av[0])
                off += T
            if off != naw * T:
                raise RuntimeError(
                    f"[paged r{r}] anchor+working = {off // T} chunks != naw={naw} "
                    f"(t={tick} clean_len={clean_len})"
                )
            Kb[self_off : self_off + T].copy_(k)
            Vb[self_off : self_off + T].copy_(v)
            for src in recv_srcs:
                self._wait_paged_flag(src, L, gen, tick)
            if _lp is not None:
                _e_wt = record_cuda_event()
            if store_active and not settings.FULL_BROADCAST:
                self._exch_store_slot[L] = torch.stack(
                    [Kb[live_off : live_off + T], Vb[live_off : live_off + T]]
                ).clone()
            K = Kb[:active_len].unsqueeze(0)
            V = Vb[:active_len].unsqueeze(0)
            if _lp is not None:
                _e_ct = record_cuda_event()
                _lp["layers"].append(
                    dict(layer=L, enter=_e_en, pub=_e_pub, wait=_e_wt, cat=_e_ct)
                )
            return K, V

        return exch

    def _make_store_exch_paged(self, clean, store_kv, c, act_ranks):
        """Build the store-side paged exchange."""
        tick = self._cur_tick
        gen = tick % self._os_gt
        npb = self.npb
        clean_len = min(8, max(0, tick - self.store_rank))
        wmax_chunks = max(0, (self.max_attn_frames - len(act_ranks) * npb - npb) // npb)
        naw = 0 if clean_len == 0 else 1 + min(wmax_chunks, clean_len - 1)
        send_slots = [(m, naw) for m in act_ranks]

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
            k = roped_key[0].contiguous()
            v = value[0].contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = record_cuda_event()
            self._publish_paged(k, v, send_slots, layer_idx, gen, tick)
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
