"""OneSidedExchangeMixin: Wave runtime implementation."""
from __future__ import annotations
import torch
import torch.distributed as dist
from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import debug_log


class OneSidedExchangeMixin:
    def _init_onesided(self) -> None:
        """Allocate the consumer remote-KV slab + generation flags, IPC-export them
        to every producer, and open every consumer slab this rank produces INTO.

        Topology (5-rank causal cascade): denoise rank r attends to in-flight ranks
        n>=r, so it CONSUMES layer-L KV from every active higher denoise rank n>r
        plus (when store-active) the store rank, and PRODUCES its own KV to every
        active lower denoise rank m<r.  The store rank is a pure producer -> writes
        every denoise rank's slab; it consumes nothing.  Hence:
          consumer r (r<rf_step): inbound srcs = [r+1 .. rf_step-1] + [store_rank]
          producer r (r<rf_step): outbound dsts = [0 .. r-1]
          producer store         : outbound dsts = [0 .. rf_step-1]
        Slab layout (contiguous, one big tensor so a single IPC handle covers all):
          remote_kv[i, gen, L]  shape KV_SHAPE=(2,T,H,D)  bf16   (i = src row index)
          gen_flag[i, gen, L]   uint32                            (release/acquire)
        gen = tick % G_T (double buffer; the tick-end migration all_gather is the
        implicit credit that makes G_T=2 race-free -- see design GateC_final1)."""
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
        for s in self.KV_SHAPE:
            slot_numel *= s
        self._os_fp8 = (settings.FP8_KV_ONESIDED or
                        getattr(self.cfg, "kv_transport_dtype", "bf16") == "fp8")
        if self._os_fp8:
            _os_dtype = torch.float8_e4m3fn
            elem = 1  # fp8 = 1 byte
        else:
            _os_dtype = self.dtype
            elem = torch.empty(0, dtype=self.dtype).element_size()
        slot_bytes = slot_numel * elem

        # --- consumer role: which srcs write INTO me ---
        if c < rf_step:
            in_srcs = list(range(c + 1, rf_step)) + [store_rank]
        else:
            in_srcs = []
        self._os_in_srcs = in_srcs
        self._os_src_index = {src: i for i, src in enumerate(in_srcs)}
        n_in = len(in_srcs)

        if n_in > 0:
            self._os_flags = torch.zeros(
                (n_in, GT, Ln), device=dev, dtype=torch.int32)
            fl_h, fl_off = p2p.ipc_export(self._os_flags)
            if self._paged:
                # paged transports KV through its own contiguous attention buffers
                # (see _init_paged); the onesided remote-KV slab is not allocated.
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
                        f"Ln={Ln} unit={slot_bytes / 2**20:.1f}MiB)")
                self._os_kv = torch.zeros(
                    (n_in, GT, Ln) + tuple(self.KV_SHAPE), device=dev, dtype=_os_dtype)
                torch.cuda.synchronize()
                kv_h, kv_off = p2p.ipc_export(self._os_kv)
                debug_log(c, f"onesided: slab {need / 2**30:.2f}GiB n_in={n_in} srcs={in_srcs}")
        else:
            self._os_kv = self._os_flags = None
            kv_h = kv_off = fl_h = fl_off = None

        # --- share export info across the world (one-time, default PG) ---
        my_info = dict(
            rank=c, src_index=self._os_src_index,
            kv_h=kv_h, kv_off=kv_off, fl_h=fl_h, fl_off=fl_off,
            GT=GT, Ln=Ln, slot_bytes=slot_bytes)
        gathered: list = [None] * world
        dist.all_gather_object(gathered, my_info)
        world_info = {g["rank"]: g for g in gathered}

        # --- producer role: open every consumer slab I write into ---
        if c < rf_step:
            my_dsts = list(range(0, c))
        else:  # store rank -> all denoise ranks
            my_dsts = list(range(0, rf_step))
        self._os_dst = {}

        for m in my_dsts:
            gi = world_info[m]
            my_i = gi["src_index"][c]            # my row in consumer m's slab
            fl_base = self._ipc_open(gi["fl_h"])
            if self._paged:
                # paged: only the flag slab is shared here; KV goes to _pg_dst
                self._os_dst[m] = dict(
                    kv_slab=0, fl_slab=fl_base + gi["fl_off"],
                    i=my_i, GT=gi["GT"], Ln=gi["Ln"], slot_bytes=gi["slot_bytes"])
            else:
                kv_base = self._ipc_open(gi["kv_h"])  # self ctx, peer-mapped (DtoD)
                self._os_dst[m] = dict(
                    kv_slab=kv_base + gi["kv_off"], fl_slab=fl_base + gi["fl_off"],
                    i=my_i, GT=gi["GT"], Ln=gi["Ln"], slot_bytes=gi["slot_bytes"])

        self._os_scopy = torch.cuda.Stream(device=dev)
        self._os_pending = []

        # --- fault-in the peer mappings once (materialize lazy IPC pages) so the
        # first timed copy is not a cold VMM fault.  One full-size copy + flag per
        # edge into (row, gen0, L0); flag value 0 == the init value (no consumer
        # ever waits >= 0), so this is inert w.r.t. the generation protocol. ---
        if self._os_dst and not self._paged:
            warm_src = torch.zeros(self.KV_SHAPE, device=dev, dtype=_os_dtype)
            sh = self._os_scopy.cuda_stream
            for m, d in self._os_dst.items():
                base_off = ((d["i"] * d["GT"] + 0) * d["Ln"] + 0)
                p2p.memcpy_dtod_async(
                    d["kv_slab"] + base_off * d["slot_bytes"],
                    warm_src.data_ptr(), slot_bytes, sh)
                p2p.stream_write_u32(sh, d["fl_slab"] + base_off * 4, 0)
            self._os_scopy.synchronize()
        dist.barrier()

    def _publish_mine(self, mine, dsts, layer: int, gen: int, tick: int) -> None:
        """Producer side: one-sided publish of `mine` (layer-L KV, (2,T,H,D)) into
        each active consumer's remote_kv[my_row][gen][L] slot on the copy stream,
        then release its generation flag (write value=tick, ordered after the copy).
        No matching recv.  `mine` is kept alive on the copy stream via record_stream
        so the caching allocator can't reuse it before the async copy finishes."""
        if not self._os_dst or not dsts:
            return
        scopy = self._os_scopy
        if self._os_fp8:
            mine_send = mine.to(torch.float8_e4m3fn).contiguous()
        else:
            mine_send = mine
        ev = torch.cuda.Event()
        ev.record()                 # include the optional FP8 cast in readiness
        scopy.wait_event(ev)
        sh = scopy.cuda_stream
        src_ptr = mine_send.data_ptr()
        nbytes = mine_send.numel() * mine_send.element_size()
        for m in dsts:
            d = self._os_dst.get(m)
            if d is None:
                continue
            slot = (d["i"] * d["GT"] + gen) * d["Ln"] + layer
            self._p2p.memcpy_dtod_async(
                d["kv_slab"] + slot * d["slot_bytes"], src_ptr, nbytes, sh)
            # release AFTER the payload copy on the same stream (Gate-B st.release.sys)
            self._p2p.stream_write_u32(sh, d["fl_slab"] + slot * 4, tick)
        mine_send.record_stream(scopy)
        self._os_pending.append(mine_send)

    def _wait_peer(self, src: int, layer: int, gen: int, tick: int):
        """Consumer side: device-side acquire on src's generation flag for
        (gen, layer), then return the slab slot (2,T,H,D).  Enqueues a
        cuStreamWaitValue32 on the current compute stream -> CPU returns
        immediately; the GPU stalls only if the producer has not published yet
        (at steady state the producer leads -> the flag is already set -> ~0)."""
        i = self._os_src_index[src]
        flag_addr = self._os_flags[i, gen, layer].data_ptr()
        sh = torch.cuda.current_stream().cuda_stream
        self._p2p.stream_wait_u32(sh, flag_addr, tick)
        slot = self._os_kv[i, gen, layer]
        if self._os_fp8:
            return slot.to(self.dtype)
        return slot

    def _drain_publish(self) -> None:
        """Make all this tick's peer writes globally visible before the WORLD
        migration all_gather, WITHOUT a CPU block: record an event on the copy
        stream and have the compute stream wait it (the migration runs on the
        compute stream, so it is ordered after every copy).  record_stream in
        _publish_mine already guards the `mine` buffers, so dropping the refs is
        safe."""
        if self._os_scopy is not None:
            ev = torch.cuda.Event()
            ev.record(self._os_scopy)
            torch.cuda.current_stream().wait_event(ev)
        self._os_pending = []

    def _make_denoise_exch_onesided(self, clean, act_ranks, tick_csf, tick_qf,
                                    store_active):
        """Denoise exchange via one-sided remote-slab publication.  Same assembled
        context (aK + liveK + inflK, cat) and numerics as causal-sync / prefetch;
        only the transport differs: publish my KV into slower ranks' slabs, and
        gate each inbound src on its device-side generation flag (no irecv)."""
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

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            L = layer_idx
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            self._publish_mine(mine, send_dsts, layer_idx, gen, tick)
            if _lp is not None:
                _e_pub = torch.cuda.Event(enable_timing=True); _e_pub.record()
            remote = {src: self._wait_peer(src, L, gen, tick) for src in recv_srcs}
            if _lp is not None:
                _e_wt = torch.cuda.Event(enable_timing=True); _e_wt.record()
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
                    # into clean[] at tick end -- identical to the prefetch path).
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
                    layer=L, enter=_e_en, pub=_e_pub, wait=_e_wt, cat=_e_ct))
            return K, V

        return exch

    def _make_store_exch_onesided(self, clean, store_kv, c, act_ranks):
        """Store variant for one-sided: publish clean KV into every active denoise
        rank's slab (no recv); local context assembly identical to sync."""
        tick = self._cur_tick
        gen = tick % self._os_gt
        send_dsts = sorted(act_ranks, reverse=True)  # EDF: immediate consumer first

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block, write_only=False):
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            self._publish_mine(mine, send_dsts, layer_idx, gen, tick)
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
                # store is a pure producer: no _wait_peer -> wait bucket is empty
                # (wait == pub -> wait_ms=0).
                _lp["layers"].append(dict(
                    layer=layer_idx, enter=_e_en, pub=_e_pub, wait=_e_pub, cat=_e_ct))
            return K, V

        exch.requires_raw_key = settings.FULL_BROADCAST
        return exch
