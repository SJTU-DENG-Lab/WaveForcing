"""One-sided CUDA IPC KV exchange."""

from __future__ import annotations

import torch
import torch.distributed as dist

from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import debug_log, record_cuda_event


class OneSidedExchangeMixin:
    def _init_onesided(self) -> None:
        """Create IPC KV slabs and generation flags for one-sided exchange.

        Denoise rank ``r`` receives from higher ranks and publishes to lower ranks;
        the store only publishes. Migration provides credit for generation reuse.
        """
        from wave_rt.distributed import ipc as p2p

        self._p2p = p2p
        dev = self.device
        c = self.rank
        rf_step = self.rf_step
        store_rank = self.store_rank
        world = self.world
        GT = settings.GENERATION_SLOTS
        Ln = self.Ln
        self._os_gt = GT
        slot_numel = 1
        for s in self._kv_shape:
            slot_numel *= s
        self._os_fp8 = settings.FP8_KV_ONESIDED
        if self._os_fp8:
            _os_dtype = torch.float8_e4m3fn
            elem = 1
        else:
            _os_dtype = self.dtype
            elem = torch.empty(0, dtype=self.dtype).element_size()
        slot_bytes = slot_numel * elem

        if c < rf_step:
            in_srcs = list(range(c + 1, rf_step)) + [store_rank]
        else:
            in_srcs = []
        self._os_in_srcs = in_srcs
        self._os_src_index = {src: i for i, src in enumerate(in_srcs)}
        n_in = len(in_srcs)

        if n_in > 0:
            self._os_flags = torch.zeros((n_in, GT, Ln), device=dev, dtype=torch.int32)
            fl_h, fl_off = p2p.ipc_export(self._os_flags)
            if self._paged:
                self._os_kv = None
                kv_h = kv_off = None
                debug_log(c, f"onesided(paged): flags only, n_in={n_in} srcs={in_srcs}")
            else:
                need = n_in * GT * Ln * slot_bytes
                free_b, _tot = torch.cuda.mem_get_info()
                if need > free_b * 0.9:
                    raise RuntimeError(
                        f"[onesided r{c}] remote_kv slab {need / 2**30:.1f} GiB exceeds "
                        f"90% of free {free_b / 2**30:.1f} GiB (n_in={n_in} GT={GT} "
                        f"Ln={Ln} unit={slot_bytes / 2**20:.1f}MiB)"
                    )
                self._os_kv = torch.zeros(
                    (n_in, GT, Ln) + self._kv_shape, device=dev, dtype=_os_dtype
                )
                torch.cuda.synchronize()
                kv_h, kv_off = p2p.ipc_export(self._os_kv)
                debug_log(
                    c,
                    f"onesided: slab {need / 2**30:.2f}GiB n_in={n_in} srcs={in_srcs}",
                )
        else:
            self._os_kv = self._os_flags = None
            kv_h = kv_off = fl_h = fl_off = None

        my_info = dict(
            rank=c,
            src_index=self._os_src_index,
            kv_h=kv_h,
            kv_off=kv_off,
            fl_h=fl_h,
            fl_off=fl_off,
            GT=GT,
            Ln=Ln,
            slot_bytes=slot_bytes,
        )
        gathered: list = [None] * world
        dist.all_gather_object(gathered, my_info)
        world_info = {g["rank"]: g for g in gathered}

        if c < rf_step:
            my_dsts = list(range(0, c))
        else:
            my_dsts = list(range(0, rf_step))
        self._os_dst = {}
        for m in my_dsts:
            gi = world_info[m]
            my_i = gi["src_index"][c]
            fl_base = self._ipc_open(gi["fl_h"])
            if self._paged:
                self._os_dst[m] = dict(
                    kv_slab=0,
                    fl_slab=fl_base + gi["fl_off"],
                    i=my_i,
                    GT=gi["GT"],
                    Ln=gi["Ln"],
                    slot_bytes=gi["slot_bytes"],
                )
            else:
                kv_base = self._ipc_open(gi["kv_h"])
                self._os_dst[m] = dict(
                    kv_slab=kv_base + gi["kv_off"],
                    fl_slab=fl_base + gi["fl_off"],
                    i=my_i,
                    GT=gi["GT"],
                    Ln=gi["Ln"],
                    slot_bytes=gi["slot_bytes"],
                )

        self._os_scopy = torch.cuda.Stream(device=dev)
        self._os_pending = []

        # Fault in peer mappings before timed execution.
        if self._os_dst and not self._paged:
            warm_src = torch.zeros(self._kv_shape, device=dev, dtype=_os_dtype)
            sh = self._os_scopy.cuda_stream
            for m, d in self._os_dst.items():
                base_off = (d["i"] * d["GT"] + 0) * d["Ln"] + 0
                p2p.memcpy_dtod_async(
                    d["kv_slab"] + base_off * d["slot_bytes"],
                    warm_src.data_ptr(),
                    slot_bytes,
                    sh,
                )
                p2p.stream_write_u32(sh, d["fl_slab"] + base_off * 4, 0)
            self._os_scopy.synchronize()
        dist.barrier()

    def _publish_mine(self, mine, dsts, layer: int, gen: int, tick: int) -> None:
        """Publish one KV layer to remote slabs and release their flags."""
        if not self._os_dst or not dsts:
            return
        scopy = self._os_scopy
        ev = torch.cuda.Event()
        ev.record()
        scopy.wait_event(ev)
        sh = scopy.cuda_stream
        if self._os_fp8:
            mine_send = mine.to(torch.float8_e4m3fn).contiguous()
        else:
            mine_send = mine
        src_ptr = mine_send.data_ptr()
        nbytes = mine_send.numel() * mine_send.element_size()
        for m in dsts:
            d = self._os_dst.get(m)
            if d is None:
                continue
            slot = (d["i"] * d["GT"] + gen) * d["Ln"] + layer
            self._p2p.memcpy_dtod_async(
                d["kv_slab"] + slot * d["slot_bytes"], src_ptr, nbytes, sh
            )
            # The release flag must follow the payload copy on the same stream.
            self._p2p.stream_write_u32(sh, d["fl_slab"] + slot * 4, tick)
        mine_send.record_stream(scopy)
        self._os_pending.append(mine_send)

    def _wait_peer(self, src: int, layer: int, gen: int, tick: int):
        """Acquire a producer flag and return its remote KV slot."""
        i = self._os_src_index[src]
        flag_addr = self._os_flags[i, gen, layer].data_ptr()
        sh = torch.cuda.current_stream().cuda_stream
        self._p2p.stream_wait_u32(sh, flag_addr, tick)
        slot = self._os_kv[i, gen, layer]
        if self._os_fp8:
            return slot.to(self.dtype)
        return slot

    def _drain_publish(self) -> None:
        """Order WORLD migration after this tick's copy-stream publications."""
        if self._os_scopy is not None:
            ev = torch.cuda.Event()
            ev.record(self._os_scopy)
            torch.cuda.current_stream().wait_event(ev)
        self._os_pending = []

    def _make_denoise_exch_onesided(
        self, clean, act_ranks, tick_csf, tick_qf, store_active
    ):
        """Build the denoise-side one-sided exchange."""
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
            self._publish_mine(mine, send_dsts, layer_idx, gen, tick)
            if _lp is not None:
                _e_pub = record_cuda_event()
            remote = {src: self._wait_peer(src, L, gen, tick) for src in recv_srcs}
            if _lp is not None:
                _e_wt = record_cuda_event()
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
                    dict(layer=L, enter=_e_en, pub=_e_pub, wait=_e_wt, cat=_e_ct)
                )
            return K, V

        return exch

    def _make_store_exch_onesided(self, clean, store_kv, c, act_ranks):
        """Build the store-side one-sided exchange."""
        tick = self._cur_tick
        gen = tick % self._os_gt
        send_dsts = sorted(act_ranks, reverse=True)

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
            self._publish_mine(mine, send_dsts, layer_idx, gen, tick)
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
