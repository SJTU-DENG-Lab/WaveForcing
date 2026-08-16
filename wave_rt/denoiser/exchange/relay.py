"""Neighbor-accumulating relay exchange."""

from __future__ import annotations

import torch
import torch.distributed as dist

from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import debug_log


class RelayExchangeMixin:
    def _init_relay(self) -> None:
        """Create IPC state for the store-to-rank-0 accumulating relay chain."""
        from wave_rt.distributed import ipc as p2p

        self._p2p = p2p
        dev = self.device
        c = self.rank
        rf_step = self.rf_step
        world = self.world
        GT = settings.GENERATION_SLOTS
        Ln = self.Ln
        self._os_gt = GT
        self._rl_world = world
        slot_numel = 1
        for s in self._kv_shape:
            slot_numel *= s
        self._os_fp8 = settings.FP8_KV_ONESIDED
        if self._os_fp8:
            elem = 1
        else:
            elem = torch.empty(0, dtype=self.dtype).element_size()
        slot_bytes = slot_numel * elem
        self._rl_slot_bytes = slot_bytes

        if c < rf_step:
            need = GT * Ln * world * slot_bytes
            free_b, _tot = torch.cuda.mem_get_info()
            if need > free_b * 0.9:
                raise RuntimeError(
                    f"[relay r{c}] recv slab {need / 2**30:.1f} GiB exceeds 90% of "
                    f"free {free_b / 2**30:.1f} GiB (GT={GT} Ln={Ln} world={world} "
                    f"unit={slot_bytes / 2**20:.1f}MiB)"
                )
            self._rl_kv = torch.zeros(
                (GT, Ln, world) + self._kv_shape, device=dev, dtype=self.dtype
            )
            self._rl_flags = torch.zeros((GT, Ln), device=dev, dtype=torch.int32)
            torch.cuda.synchronize()
            kv_h, kv_off = p2p.ipc_export(self._rl_kv)
            fl_h, fl_off = p2p.ipc_export(self._rl_flags)
            debug_log(c, f"relay: recv slab {need / 2**30:.2f}GiB world={world}")
        else:
            self._rl_kv = self._rl_flags = None
            kv_h = kv_off = fl_h = fl_off = None

        my_info = dict(rank=c, kv_h=kv_h, kv_off=kv_off, fl_h=fl_h, fl_off=fl_off)
        gathered: list = [None] * world
        dist.all_gather_object(gathered, my_info)
        world_info = {g["rank"]: g for g in gathered}

        if c == 0:
            down = None
        elif c < rf_step:
            down = c - 1
        else:
            down = rf_step - 1
        self._rl_down = None
        if down is not None:
            gi = world_info[down]
            kv_base = p2p.ipc_open_handle(gi["kv_h"])
            fl_base = p2p.ipc_open_handle(gi["fl_h"])
            self._rl_down = dict(
                rank=down,
                kv_slab=kv_base + gi["kv_off"],
                fl_slab=fl_base + gi["fl_off"],
            )
            debug_log(c, f"relay: downstream={down}")

        self._os_scopy = torch.cuda.Stream(device=dev)
        self._os_pending = []

        # Fault in the downstream mapping before timed execution.
        if self._rl_down is not None:
            warm = torch.zeros(self._kv_shape, device=dev, dtype=self.dtype)
            sh = self._os_scopy.cuda_stream
            d = self._rl_down
            slot = c
            p2p.memcpy_dtod_async(
                d["kv_slab"] + slot * slot_bytes, warm.data_ptr(), slot_bytes, sh
            )
            p2p.stream_write_u32(sh, d["fl_slab"] + (0 * Ln + 0) * 4, 0)
            self._os_scopy.synchronize()
        dist.barrier()

    def _relay_forward(self, mine, L, gen, tick, r, top, has_up, has_down) -> None:
        """Merge local KV into an upstream bundle and relay it downstream."""
        if not has_down:
            return
        scopy = self._os_scopy
        ev = torch.cuda.Event()
        ev.record()
        scopy.wait_event(ev)
        sh = scopy.cuda_stream
        world = self._rl_world
        Ln = self.Ln
        sb = self._rl_slot_bytes
        base = self._rl_kv.data_ptr()
        slot_off = ((gen * Ln + L) * world + r) * sb
        self._p2p.memcpy_dtod_async(base + slot_off, mine.data_ptr(), sb, sh)
        if has_up:
            self._p2p.stream_wait_u32(sh, self._rl_flags[gen, L].data_ptr(), tick)
        d = self._rl_down
        nbytes = (top - r + 1) * sb
        self._p2p.memcpy_dtod_async(
            d["kv_slab"] + slot_off, base + slot_off, nbytes, sh
        )
        self._p2p.stream_write_u32(sh, d["fl_slab"] + (gen * Ln + L) * 4, tick)
        mine.record_stream(scopy)
        self._os_pending.append(mine)

    def _relay_store_publish(self, mine, L, gen, tick) -> None:
        """Publish store KV into the head of the relay chain."""
        scopy = self._os_scopy
        ev = torch.cuda.Event()
        ev.record()
        scopy.wait_event(ev)
        sh = scopy.cuda_stream
        world = self._rl_world
        Ln = self.Ln
        sb = self._rl_slot_bytes
        d = self._rl_down
        slot_off = ((gen * Ln + L) * world + self.store_rank) * sb
        self._p2p.memcpy_dtod_async(d["kv_slab"] + slot_off, mine.data_ptr(), sb, sh)
        self._p2p.stream_write_u32(sh, d["fl_slab"] + (gen * Ln + L) * 4, tick)
        mine.record_stream(scopy)
        self._os_pending.append(mine)

    def _make_denoise_exch_relay(
        self, clean, act_ranks, tick_csf, tick_qf, store_active
    ):
        """Build the denoise-side accumulating relay exchange."""
        store_rank, r = self.store_rank, self.rank
        tick = self._cur_tick
        gen = tick % self._os_gt
        lo_a, hi_a = min(act_ranks), max(act_ranks)
        top = store_rank if store_active else hi_a
        if r < store_rank - 1:
            has_up = (r + 1) in act_ranks
        else:
            has_up = store_active
        has_down = r > lo_a
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
            self._relay_forward(mine, L, gen, tick, r, top, has_up, has_down)
            if has_up:
                sh = torch.cuda.current_stream().cuda_stream
                self._p2p.stream_wait_u32(sh, self._rl_flags[gen, L].data_ptr(), tick)
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
                slot = self._rl_kv[gen, L, store_rank]
                liveK = [slot[0].unsqueeze(0)]
                liveV = [slot[1].unsqueeze(0)]
                if not settings.FULL_BROADCAST:
                    self._exch_store_slot[L] = slot.clone()
            inflK, inflV = [], []
            for n in infl_ranks:
                s = mine if n == r else self._rl_kv[gen, L, n]
                inflK.append(s[0].unsqueeze(0))
                inflV.append(s[1].unsqueeze(0))
            K = torch.cat(aK + liveK + inflK, dim=1)
            V = torch.cat(aV + liveV + inflV, dim=1)
            return K, V

        return exch

    def _make_store_exch_relay(self, clean, store_kv, c, act_ranks):
        """Build the store-side relay exchange."""
        tick = self._cur_tick
        gen = tick % self._os_gt
        has_down = len(act_ranks) > 0

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
            if has_down and self._rl_down is not None:
                self._relay_store_publish(mine, L, gen, tick)
            K, V = self._assemble_store_context(
                clean[L],
                c,
                roped_key,
                value,
                post_patch_height,
                post_patch_width,
                dim,
                rope_num_heads,
            )
            store_kv[L] = (roped_key, unroped_key, value)
            return K, V

        return exch
