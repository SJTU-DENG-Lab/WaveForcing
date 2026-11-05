"""PagedExchangeMixin: Wave runtime implementation."""
from __future__ import annotations
import torch
import torch.distributed as dist
from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import debug_log
from wave_rt.denoiser.profile import current as runtime_profile_current, span as runtime_span


class PagedExchangeMixin:
    def _init_paged(self) -> None:
        """One slot per layer, with explicit post-attention reuse credits.
        Producers write each layer's fixed token slots directly. Ready flags remain
        per-(tick generation,layer); the payload drops the tick-generation axis.
        A complete forward fits without waiting for any same-tick release. Next
        tick's compute follows migration, ordered after all attention and copies.

        Slot order per consumer r (identical to the onesided cat order, so the
        attention input is byte-identical): [anchor, working..., live(store), self,
        in-flight(r+1..)].  At steady state anchor+working == _pg_naw is constant, so
        every slot offset is a compile-time-constant function of (producer, consumer)
        and both sides agree on offsets. Overwrite waits for the consumer's credit."""
        self._init_onesided()   # flags, _os_src_index, _os_dst(fl_slab,i), scopy, _os_kv
        from wave_rt.denoiser.exchange.slots import PagedSlots
        self._pg_slots = PagedSlots(self.Ln)
        self._pg_slots.address(self.num_ticks - 1, 0, self.Ln - 1)
        from wave_rt.distributed import ipc as p2p
        dev = self.device
        c = self.rank
        GT = self._os_gt
        Ln = self.Ln
        npb = self.npb
        T = self.tokens_per_chunk
        H = self.KV_SHAPE[2]
        D = self.KV_SHAPE[3]
        # steady-state window geometry (matches _anchor_working with qf=rf_step*npb)
        wmax_frames = self.max_attn_frames - self.rf_step * npb - npb
        wmax_chunks = max(0, wmax_frames // npb)
        self._pg_naw = 1 + wmax_chunks                       # anchor + working
        self._pg_wmax = self._pg_naw + 1 + self.rf_step      # + live + max in-flight
        Wmax = self._pg_wmax
        if self._os_fp8:
            raise RuntimeError("paged mode does not support WAVE_FP8KV_ONESIDED yet")
        elem = torch.empty(0, dtype=self.dtype).element_size()

        # --- consumer role: denoise ranks allocate their attention buffers ---
        if c < self.rf_step:
            need = 2 * self._pg_slots.depth * Wmax * T * H * D * elem
            free_b, _tot = torch.cuda.mem_get_info()
            # Include clean history that is allocated later, plus scratch space.
            # This is a lower-bound budget, not a promise about kernel workspace.
            history = 2 * Ln * min(8, self.num_blocks) * T * H * D * elem
            scratch = max(2**30, 8 * self.rf_step * T * H * D * elem)
            if need + history + scratch > free_b * 0.95:
                raise RuntimeError(
                    f"[paged r{c}] request KV budget: slabs={need / 2**30:.1f}, "
                    f"history={history / 2**30:.1f}, scratch reserve={scratch / 2**30:.1f} GiB "
                    f"exceeds 95% of free {free_b / 2**30:.1f} GiB")
            # Every active prefix is locally written or acquired from producers.
            # Flags remain zero-initialized; unused payload bytes are never read.
            self._pg_payload = torch.empty((self._pg_slots.depth, 2, Wmax * T, H, D), device=dev, dtype=self.dtype)
            self._pg_k, self._pg_v = self._pg_payload.unbind(1)
            self._pg_consumed = torch.zeros(self._pg_slots.depth, device=dev, dtype=torch.int32)
            credit_h, credit_off = p2p.ipc_export(self._pg_consumed)
            k_h, k_off = p2p.ipc_export(self._pg_k)
            v_h, v_off = k_h, k_off + Wmax*T*H*D*elem
            debug_log(c, f"paged: kv buffers {need / 2**30:.2f}GiB naw={self._pg_naw} Wmax={Wmax}")
        else:
            self._pg_k = self._pg_v = None
            self._pg_consumed = None
            credit_h = credit_off = None
            k_h = k_off = v_h = v_off = None

        my_info = dict(rank=c, k_h=k_h, k_off=k_off, v_h=v_h, v_off=v_off,
                       credit_h=credit_h, credit_off=credit_off,
                       Wmax=Wmax, Ln=Ln, GT=GT, THD=T * H * D, elem=elem)
        gathered: list = [None] * self.world
        dist.all_gather_object(gathered, my_info)
        world_info = {g["rank"]: g for g in gathered}

        # --- producer role: open every consumer buffer I write into ---
        if c < self.rf_step:
            my_dsts = list(range(0, c))
        else:  # store rank -> all denoise ranks
            my_dsts = list(range(0, self.rf_step))
        self._pg_dst = {}
        self._pg_publish_events = [torch.cuda.Event() for _ in range(Ln)]
        for m in my_dsts:
            gi = world_info[m]
            if gi["k_h"] is None:
                continue
            # Import into the producer's current context for DtoD copies.
            # K/V share one allocation: open and close its IPC mapping once.
            kb = self._ipc_open(gi["k_h"])
            vb = kb  # K and V share the same exported allocation.
            credit = self._ipc_open(gi['credit_h'])
            od = self._os_dst[m]   # reuse onesided flag slab + my row index
            self._pg_dst[m] = dict(
                k_slab=kb + gi["k_off"], v_slab=vb + gi["v_off"],
                credits=credit + gi['credit_off'],
                fl_slab=od["fl_slab"], i=od["i"], GT=gi["GT"], Ln=gi["Ln"],
                Wmax=gi["Wmax"], THD=gi["THD"], elem=gi["elem"])

        # --- fault-in the peer mappings once (materialize lazy IPC pages) ---
        if self._pg_dst:
            warm = torch.zeros((T, H, D), device=dev, dtype=self.dtype)
            self._os_scopy.wait_stream(torch.cuda.current_stream())
            sh = self._os_scopy.cuda_stream
            for m, d in self._pg_dst.items():
                for ptr in (d['k_slab'], d['v_slab']):
                    p2p.memcpy_dtod_async(ptr, warm.data_ptr(), warm.numel()*elem, sh)
            self._os_scopy.synchronize()
        dist.barrier()

    def _paged_steady(self, store_active, act_ranks) -> bool:
        """True iff this tick's window is at steady state on EVERY rank, so the fixed
        paged slot layout is valid.  Pure function of global schedule (t, act_ranks,
        store_active) + constants -> producer and consumer decide identically without
        a handshake.  clean_len at tick start == clamp(t-store_rank, 0, 8); require it
        >= _pg_naw so _anchor_working yields exactly _pg_naw anchor+working pieces."""
        return (store_active and len(act_ranks) == self.rf_step
                and (self._cur_tick - self.store_rank) >= self._pg_naw)

    def _publish_paged(self, k, v, dst_slots, layer: int, gen: int, tick: int) -> None:
        """Producer: DMA k and v (each (T,H,D), contiguous) into the fixed slot of each
        consumer's paged K/V buffer, then release the generation flag (after the copy,
        same stream).  dst_slots = [(consumer_rank, slot_index), ...]."""
        if not self._pg_dst or not dst_slots:
            return
        scopy = self._os_scopy
        ev = self._pg_publish_events[layer]
        profile = runtime_profile_current()
        if profile is not None:
            profile.mark('kv_ready', layer=layer, destinations=[m for m, _ in dst_slots])
        ev.record()
        scopy.wait_event(ev)         # copy waits for layer-L KV to materialize
        sh = scopy.cuda_stream
        kb = k.numel() * k.element_size()
        kp = k.data_ptr()
        vp = v.data_ptr()
        if (k.untyped_storage().data_ptr() != v.untyped_storage().data_ptr() or vp-kp < kb
                or k.shape != v.shape or not k.is_contiguous() or not v.is_contiguous()):
            raise ValueError('Paged publication requires final-address K/V in one owned allocation')
        # Preserve the original rank-0-first publication order.
        for (m, slot) in sorted(dst_slots):
            d = self._pg_dst.get(m)
            if d is None:
                continue
            ring_slot, _, required = self._pg_slots.address(tick, m, layer)
            if required:
                with runtime_span('kv_credit_wait', stream=scopy, layer=layer, destination=m):
                    self._p2p.stream_wait_u32(sh, d['credits'] + ring_slot * 4, required)
            base = (2 * ring_slot * d["Wmax"] + slot) * d["THD"] * d["elem"]
            with runtime_span('kv_transfer', stream=scopy, layer=layer, destination=m, payload_bytes=2*kb):
                self._p2p.memcpy_dtod_async(d['k_slab'] + base, kp, kb, sh)
                self._p2p.memcpy_dtod_async(d['v_slab'] + base, vp, kb, sh)
                fslot = (d["i"] * d["GT"] + gen) * d["Ln"] + layer
                self._p2p.stream_write_u32(sh, d["fl_slab"] + fslot * 4, tick)
        k.record_stream(scopy)
        v.record_stream(scopy)

    def _make_denoise_exch_paged(self, clean, act_ranks, tick_csf, tick_qf,
                                 store_active):
        """Denoise exchange with in-place paged KV assembly (no cat), for ALL ticks.
        Slot order is byte-identical to the onesided cat order
        [anchor, working, live(store), self, in-flight(n>r ascending)].  The window
        geometry (naw = anchor+working count) is a pure function of the global
        schedule -- the clean list has identical length on every denoise rank -- so
        producer and consumer agree on every slot offset without a handshake."""
        store_rank, r = self.store_rank, self.rank
        tick = self._cur_tick
        gen = tick % self._os_gt
        T = self.tokens_per_chunk
        npb = self.npb
        # window geometry; MUST match _anchor_working (same clean[] on every rank)
        clean_len = min(8, max(0, tick - store_rank))
        wmax_chunks = max(0, (self.max_attn_frames - len(act_ranks) * npb - npb) // npb)
        naw = 0 if clean_len == 0 else 1 + min(wmax_chunks, clean_len - 1)
        live = 1 if store_active else 0
        base_infl = naw + live
        infl_ranks = [n for n in act_ranks if n >= r]      # ascending; self=r is first
        active_slots = base_infl + len(infl_ranks)
        if active_slots > self._pg_wmax:
            raise RuntimeError(
                f"[paged r{r}] active_slots={active_slots} > Wmax={self._pg_wmax} "
                f"(t={tick} naw={naw} live={live} n_infl={len(infl_ranks)})")
        self_off = base_infl * T          # self (in-flight r) is index 0 of infl_ranks
        live_off = naw * T                # store live slot (valid iff store_active)
        active_len = active_slots * T
        recv_srcs = [n for n in act_ranks if n > r] + (
            [store_rank] if store_active else [])
        # producer: for each slower active consumer m, my slot = base_infl + index of r
        # in m's in-flight list = number of active ranks in [m, r).
        send_slots = []
        for m in act_ranks:
            if m >= r:
                continue
            idx = sum(1 for n in act_ranks if m <= n < r)
            send_slots.append((m, base_infl + idx))

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            L = layer_idx
            k = roped_key[0].contiguous()
            v = value[0].contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            self._publish_paged(k, v, send_slots, L, gen, tick)
            if _lp is not None:
                _e_pub = torch.cuda.Event(enable_timing=True); _e_pub.record()
            ring_slot, _, _ = self._pg_slots.address(tick, r, L)
            Kb = self._pg_k[ring_slot]
            Vb = self._pg_v[ring_slot]
            # local placement: anchor + working into [0, naw*T)
            count = self._assemble_paged_context(
                clean[L], tick_csf, tick_qf,
                post_patch_height, post_patch_width, dim, rope_num_heads, Kb, Vb)
            if count != naw:
                raise RuntimeError(
                    f"[paged r{r}] anchor+working = {count} chunks != naw={naw} "
                    f"(t={tick} clean_len={clean_len})")
            # self (in-flight r) into its fixed slot; live (store) slot is DMA-filled
            if k.data_ptr() != Kb[self_off:self_off + T].data_ptr():
                Kb[self_off:self_off + T].copy_(k)
                Vb[self_off:self_off + T].copy_(v)
            if _lp is not None:
                _e_assembled = torch.cuda.Event(enable_timing=True); _e_assembled.record()
            # gate on every remote src (store live + in-flight n>r); data already placed
            self._wait_paged_flags(recv_srcs, L, gen, tick)
            if _lp is not None:
                _e_wt = torch.cuda.Event(enable_timing=True); _e_wt.record()
            # promote store live slot into clean[] at tick end (same as onesided)
            if store_active and not settings.FULL_BROADCAST and tick > store_rank:
                if exch.defer_live_copy:
                    # The following Sage statistics launch owns the copy. Local
                    # assembly above has finished reading the recycled page.
                    page = self._working_target(L, tick-store_rank)
                    exch.kv_promotion = (live_off, page)
                else:
                    page = self._working_page(L, tick-store_rank,
                        Kb[live_off:live_off + T], Vb[live_off:live_off + T])
                self._exch_store_slot[L] = page
            K = Kb[:active_len].unsqueeze(0)
            V = Vb[:active_len].unsqueeze(0)
            if _lp is not None:
                _e_ct = torch.cuda.Event(enable_timing=True); _e_ct.record()
                _lp["layers"].append(dict(
                    layer=L, enter=_e_en, pub=_e_pub, wait=_e_wt, cat=_e_ct,
                    assembled=_e_assembled))
            return K, V

        exch.defer_live_copy = False
        exch.kv_promotion = None
        exch.release = lambda layer: self._release_paged(tick, layer)
        targets = [(self._pg_k[L, self_off:self_off+T].unsqueeze(0),
                    self._pg_v[L, self_off:self_off+T].unsqueeze(0)) for L in range(self.Ln)]
        exch.prepare_targets = targets.__getitem__
        return exch

    def _make_store_exch_paged(self, clean, store_kv, c, act_ranks):
        """Store variant for paged: publish live KV into each active denoise
        consumer's live slot; assemble local context in reusable compute scratch.
        Publication sources are separate, persistent per-layer history pages."""
        tick = self._cur_tick
        gen = tick % self._os_gt
        npb = self.npb
        # consumers' window geometry (same formula the denoise ranks use)
        clean_len = min(8, max(0, tick - self.store_rank))
        wmax_chunks = max(0, (self.max_attn_frames - len(act_ranks) * npb - npb) // npb)
        naw = 0 if clean_len == 0 else 1 + min(wmax_chunks, clean_len - 1)
        send_slots = [(m, naw) for m in act_ranks]   # store -> live slot in each consumer

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block,
                 write_only=False):
            k = roped_key[0].contiguous()
            v = value[0].contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            self._publish_paged(k, v, send_slots, layer_idx, gen, tick)
            if c > 0 and not settings.FULL_BROADCAST:
                page = self._working_page(layer_idx, c, k, v)
                store_kv[layer_idx] = (page[0].unsqueeze(0), None, page[1].unsqueeze(0))
            else:
                store_kv[layer_idx] = (roped_key, unroped_key, value)
            if _lp is not None:
                _e_pub = torch.cuda.Event(enable_timing=True); _e_pub.record()
            if write_only:
                # The final clean layer has no attention consumer locally.
                # Publication and store history are still required by peers.
                if _lp is not None:
                    _lp["layers"].append(dict(
                        layer=layer_idx, enter=_e_en, pub=_e_pub, wait=_e_pub, cat=_e_pub))
                return roped_key, value
            needed = min(len(clean[layer_idx]), 1 + max(0, (self.max_attn_frames-2*self.npb)//self.npb)) + 1
            shape = (2, needed*k.shape[0], *k.shape[1:])
            workspace = getattr(self, '_pg_store_workspace', None)
            if workspace is None or workspace.shape[1] < shape[1] or workspace.shape[2:] != shape[2:]:
                # Allocate the full bounded window once (including the current
                # chunk); this workspace is used only on the compute stream.
                capacity = (2 + min(7, max(0, (self.max_attn_frames-2*self.npb)//self.npb)))*k.shape[0]
                workspace = self._pg_store_workspace = k.new_empty((2, capacity, *k.shape[1:]))
            count = self._assemble_paged_context(
                clean[layer_idx], c * self.npb, self.npb,
                post_patch_height, post_patch_width, dim, rope_num_heads,
                workspace[0], workspace[1], extra=(roped_key, value))
            K, V = (part[:count*k.shape[0]].unsqueeze(0) for part in workspace)
            if _lp is not None:
                _e_ct = torch.cuda.Event(enable_timing=True); _e_ct.record()
                _lp["layers"].append(dict(
                    layer=layer_idx, enter=_e_en, pub=_e_pub, wait=_e_pub, cat=_e_ct))
            return K, V

        def prepare_targets(layer):
            # Chunk zero has no older anchor. Keep its projection source separate
            # from the seven working chunks and the next write destination.
            page = self._working_target(layer, c)
            return page[0].unsqueeze(0), page[1].unsqueeze(0)
        exch.prepare_targets = prepare_targets
        exch.requires_raw_key = settings.FULL_BROADCAST
        return exch

    def _wait_paged_flags(self, sources, layer, gen, tick):
        """Acquire each producer's ready flag on the compute stream."""
        if not sources:
            return
        sh = torch.cuda.current_stream().cuda_stream
        with runtime_span('kv_exposed_wait', layer=layer, sources=list(sources)):
            for source in sources:
                addr = self._os_flags[self._os_src_index[source], gen, layer].data_ptr()
                self._p2p.stream_wait_u32(sh, addr, tick)

    def _release_paged(self, tick, layer):
        # Enqueued AFTER attention on the compute stream, not after acquisition.
        # The same layer may reuse this slot in the next tick, after migration.
        slot, sequence, _ = self._pg_slots.address(tick, self.rank, layer)
        self._p2p.stream_write_u32(torch.cuda.current_stream().cuda_stream,
                                  self._pg_consumed[slot].data_ptr(), sequence)
