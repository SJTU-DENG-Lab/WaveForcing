"""RelayExchangeMixin: Wave runtime implementation."""
from __future__ import annotations
import torch
import torch.distributed as dist
from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import debug_log


class RelayExchangeMixin:
    def _init_relay(self) -> None:
        """Neighbor-accumulating relay setup.

        Static chain (world = rf_step+1, store_rank = rf_step):
            store -> rf_step-1 -> rf_step-2 -> ... -> 1 -> 0
        Each rank writes ONLY its immediate downstream (fan-out 1); each denoise
        rank is written ONLY by its immediate upstream (single writer -> a single
        gen flag per (gen, L)):
            denoise r (1<=r<=rf_step-1): downstream = r-1
            denoise 0                  : no downstream (chain tail)
            store  (rf_step)           : downstream = rf_step-1
        The recv slab is indexed by ABSOLUTE rank id so an accumulated bundle
        [r..top] (the KV of all ranks >= r that r must forward) is one CONTIGUOUS
        slice -> a single cuMemcpyDtoDAsync per hop.  Store is a pure producer and
        owns no recv slab.
            rl_kv[gen, L, rank_id]  shape KV_SHAPE=(2,T,H,D)  bf16
            rl_flags[gen, L]        int32 (release/acquire, set by my upstream)
        gen = tick % G_T (double buffer; the tick-end migration all_gather is the
        implicit credit that makes G_T=2 race-free, same as the onesided path)."""
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
        self._rl_world = world
        slot_numel = 1
        for s in self.KV_SHAPE:
            slot_numel *= s
        self._os_fp8 = settings.FP8_KV_ONESIDED
        if self._os_fp8:
            _os_dtype = torch.float8_e4m3fn
            elem = 1  # fp8 = 1 byte
        else:
            _os_dtype = self.dtype
            elem = torch.empty(0, dtype=self.dtype).element_size()
        slot_bytes = slot_numel * elem
        self._rl_slot_bytes = slot_bytes

        # --- consumer role: every denoise rank owns a recv slab; store owns none ---
        if c < rf_step:
            need = GT * Ln * world * slot_bytes
            free_b, _tot = torch.cuda.mem_get_info()
            if need > free_b * 0.9:
                raise RuntimeError(
                    f"[relay r{c}] recv slab {need / 2**30:.1f} GiB exceeds 90% of "
                    f"free {free_b / 2**30:.1f} GiB (GT={GT} Ln={Ln} world={world} "
                    f"unit={slot_bytes / 2**20:.1f}MiB)")
            self._rl_kv = torch.zeros(
                (GT, Ln, world) + tuple(self.KV_SHAPE), device=dev, dtype=self.dtype)
            self._rl_flags = torch.zeros((GT, Ln), device=dev, dtype=torch.int32)
            torch.cuda.synchronize()
            kv_h, kv_off = p2p.ipc_export(self._rl_kv)
            fl_h, fl_off = p2p.ipc_export(self._rl_flags)
            debug_log(c, f"relay: recv slab {need / 2**30:.2f}GiB world={world}")
        else:
            self._rl_kv = self._rl_flags = None
            kv_h = kv_off = fl_h = fl_off = None

        # --- share export info (one-time, default PG) ---
        my_info = dict(rank=c, kv_h=kv_h, kv_off=kv_off, fl_h=fl_h, fl_off=fl_off)
        gathered: list = [None] * world
        dist.all_gather_object(gathered, my_info)
        world_info = {g["rank"]: g for g in gathered}

        # --- producer role: open my single downstream slab ---
        if c == 0:
            down = None                     # chain tail (lowest denoise)
        elif c < rf_step:
            down = c - 1                    # denoise r -> r-1
        else:
            down = rf_step - 1              # store -> highest denoise rank
        self._rl_down = None
        if down is not None:
            gi = world_info[down]
            kv_base = self._ipc_open(gi["kv_h"])   # peer-mapped (DtoD)
            fl_base = self._ipc_open(gi["fl_h"])
            self._rl_down = dict(
                rank=down,
                kv_slab=kv_base + gi["kv_off"],
                fl_slab=fl_base + gi["fl_off"])
            debug_log(c, f"relay: downstream={down}")

        self._os_scopy = torch.cuda.Stream(device=dev)
        self._os_pending = []

        # --- fault-in the downstream mapping (materialize lazy IPC pages) so the
        # first timed copy is not a cold VMM fault.  Touch (gen0, L0, my rank slot)
        # + flag=0 (inert: no consumer ever waits >= 0). ---
        if self._rl_down is not None:
            warm = torch.zeros(self.KV_SHAPE, device=dev, dtype=self.dtype)
            sh = self._os_scopy.cuda_stream
            d = self._rl_down
            slot = (0 * Ln + 0) * world + c          # my rank's slot in the bundle
            p2p.memcpy_dtod_async(
                d["kv_slab"] + slot * slot_bytes, warm.data_ptr(), slot_bytes, sh)
            p2p.stream_write_u32(sh, d["fl_slab"] + (0 * Ln + 0) * 4, 0)
            self._os_scopy.synchronize()
        dist.barrier()

    def _relay_forward(self, mine, L, gen, tick, r, top, has_up, has_down) -> None:
        """Copy-stream (denoise rank): stage own KV into rl_kv[gen,L,r], wait the
        upstream bundle (so slots [r+1..top] are valid), then forward the merged
        contiguous slice [r..top] into the downstream neighbor's slab with ONE
        memcpy and release its gen flag.  No matching recv (one-sided)."""
        if not has_down:
            return                       # chain tail: nothing to stage/forward
        scopy = self._os_scopy
        ev = torch.cuda.Event()
        ev.record()                      # default stream: `mine` materialized
        scopy.wait_event(ev)
        sh = scopy.cuda_stream
        world = self._rl_world
        Ln = self.Ln
        sb = self._rl_slot_bytes
        base = self._rl_kv.data_ptr()
        # stage own KV into slot r (the head of the bundle I forward)
        slot_off = ((gen * Ln + L) * world + r) * sb
        self._p2p.memcpy_dtod_async(base + slot_off, mine.data_ptr(), sb, sh)
        # wait upstream so the relayed slots [r+1..top] are present before forwarding
        if has_up:
            self._p2p.stream_wait_u32(
                sh, self._rl_flags[gen, L].data_ptr(), tick)
        # forward the merged bundle [r..top] -> downstream, then release its flag
        d = self._rl_down
        nbytes = (top - r + 1) * sb
        self._p2p.memcpy_dtod_async(
            d["kv_slab"] + slot_off, base + slot_off, nbytes, sh)
        self._p2p.stream_write_u32(sh, d["fl_slab"] + (gen * Ln + L) * 4, tick)
        mine.record_stream(scopy)
        self._os_pending.append(mine)

    def _relay_store_publish(self, mine, L, gen, tick) -> None:
        """Copy-stream (store rank): publish own clean KV (1 chunk) into the highest
        denoise rank's slab at slot [store_rank] and release its flag.  Store is the
        chain head -> no upstream wait, no local staging."""
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
        self._p2p.memcpy_dtod_async(
            d["kv_slab"] + slot_off, mine.data_ptr(), sb, sh)
        self._p2p.stream_write_u32(sh, d["fl_slab"] + (gen * Ln + L) * 4, tick)
        mine.record_stream(scopy)
        self._os_pending.append(mine)

    def _make_denoise_exch_relay(self, clean, act_ranks, tick_csf, tick_qf,
                                 store_active):
        """Denoise exchange via neighbor-accumulating relay.  Assembles EXACTLY the
        causal-sync context (aK + liveK + inflK, n>=r) -> byte-identical numerics;
        only the transport differs (single-neighbor merged relay hops)."""
        store_rank, r = self.store_rank, self.rank
        tick = self._cur_tick
        gen = tick % self._os_gt
        lo_a, hi_a = min(act_ranks), max(act_ranks)
        # global chain head this tick (== store when store-active, else the highest
        # active denoise rank).  Every rank forwards the contiguous slice [r..top].
        top = store_rank if store_active else hi_a
        # single upstream writer into my slab: r+1 (denoise) or store (when r is the
        # highest denoise rank).  Present only if that rank is active this tick.
        if r < store_rank - 1:
            has_up = (r + 1) in act_ranks
        else:  # r == rf_step-1
            has_up = store_active
        has_down = r > lo_a             # my downstream (r-1) is an active denoise rank
        infl_ranks = [n for n in act_ranks if n >= r]

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            L = layer_idx
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            # copy-stream: stage + relay the merged bundle to my one neighbor
            self._relay_forward(mine, L, gen, tick, r, top, has_up, has_down)
            # compute-stream: device-side acquire on the upstream bundle, then read
            if has_up:
                sh = torch.cuda.current_stream().cuda_stream
                self._p2p.stream_wait_u32(
                    sh, self._rl_flags[gen, L].data_ptr(), tick)
            aK, aV = self._anchor_working(
                clean[L], tick_csf, tick_qf,
                post_patch_height, post_patch_width, dim, rope_num_heads)
            liveK, liveV = [], []
            if store_active:
                slot = self._rl_kv[gen, L, store_rank]
                liveK = [slot[0].unsqueeze(0)]
                liveV = [slot[1].unsqueeze(0)]
                if not settings.FULL_BROADCAST:
                    # clone out of the gen ring before its next-gen reuse (promoted
                    # into clean[] at tick end -- identical to the onesided path).
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
        """Store variant for relay: publish own clean KV into the highest denoise
        rank's slab (chain head, no recv); local context assembly identical to sync."""
        tick = self._cur_tick
        gen = tick % self._os_gt
        has_down = len(act_ranks) > 0    # last tick has no denoise ranks -> nobody

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            L = layer_idx
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            if has_down and self._rl_down is not None:
                self._relay_store_publish(mine, L, gen, tick)
            aK, aV = self._anchor_working(
                clean[L], c * self.npb, self.npb,
                post_patch_height, post_patch_width, dim, rope_num_heads)
            K = torch.cat(aK + [roped_key], dim=1)
            V = torch.cat(aV + [value], dim=1)
            store_kv[L] = (roped_key, unroped_key, value)
            return K, V

        return exch
