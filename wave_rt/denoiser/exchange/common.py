"""CommonExchangeMixin: Wave runtime implementation."""
from __future__ import annotations
import os
import time
import torch
import torch.distributed as dist
from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import debug_log


class CommonExchangeMixin:
    def _anchor_rope(self, num_frames, pph, ppw, start_frame, dim, num_heads, device):
        """Cached cos/sin table for the anchor re-rope (see __init__ note)."""
        key = (num_frames, pph, ppw, start_frame, dim, num_heads)
        hit = self._rope_cache.get(key)
        if hit is None:
            axes = getattr(self, '_anchor_axes', None)
            if axes is not None:
                axes.check(num_frames, pph, ppw, start_frame)
                hit = axes.grid(num_frames, start_frame)
                self._rope_cache[key] = hit
                if len(self._rope_cache) > 32:
                    self._rope_cache.popitem(last=False)
                return hit
            head_dim = dim // num_heads
            rope_dim_list = [
                head_dim - 4 * (head_dim // 6),
                2 * (head_dim // 6),
                2 * (head_dim // 6),
            ]
            cos, sin = self._get_rotary_pos_embed(
                (num_frames, pph, ppw), dim, num_heads, rope_dim_list,
                dtype=torch.float64 if self._rope_f64 else torch.float32,
                rope_theta=10000, start_frame=start_frame, device=device,
            )
            hit = (cos.float(), sin.float())
            self._rope_cache[key] = hit
            if len(self._rope_cache) > 32:
                self._rope_cache.popitem(last=False)
        self._rope_cache.move_to_end(key)
        return hit

    def _anchor_working(self, clean_L, csf, qf, pph, ppw, dim, rope_num_heads):
        """anchor(re-RoPEd) + bounded working set from the replicated clean list."""
        if len(clean_L) == 0:
            return [], []
        npb = self.npb
        wmax_frames = self.max_attn_frames - qf - npb
        wmax_chunks = max(0, wmax_frames // npb)
        anchor = clean_L[0]
        rest = clean_L[1:]
        working = rest[-wmax_chunks:] if wmax_chunks > 0 else []
        wflen = len(working) * npb
        rope_start = csf - wflen - npb
        cos, sin = self._anchor_rope(
            npb, pph, ppw, rope_start, dim, rope_num_heads, anchor["ku"].device
        )
        ak = self._apply_rotary_emb(
            anchor["ku"], cos, sin, is_neox_style=False
        ).type_as(anchor["v"])
        return (
            [ak] + [w["kr"] for w in working],
            [anchor["v"]] + [w["v"] for w in working],
        )

    def _all_gather(self, mine: torch.Tensor) -> list[torch.Tensor]:
        if settings.FP8_KV:
            return self._all_gather_fp8(mine)
        if self._machprof:
            _ta = time.perf_counter()
            out = list(self._buffers.get('all_gather', (self.world, *self.KV_SHAPE), mine).unbind(0))
            self._mp_alloc_cpu += (time.perf_counter() - _ta) * 1000.0
            _ts = time.perf_counter()
            dist.all_gather(out, mine.contiguous())
            self._mp_submit_cpu += (time.perf_counter() - _ts) * 1000.0
            return out
        out = list(self._buffers.get('all_gather', (self.world, *self.KV_SHAPE), mine).unbind(0))
        if self._ag_ts and self.rank == 0 and self._cur_tick >= self.world - 1:
            _e0 = torch.cuda.Event(enable_timing=True)
            _e1 = torch.cuda.Event(enable_timing=True)
            _e0.record()
            dist.all_gather(out, mine.contiguous())
            _e1.record()
            self._ag_records.append((self._cur_tick, _e0, _e1))
            return out
        dist.all_gather(out, mine.contiguous())
        return out

    def _all_gather_fp8(self, mine: torch.Tensor) -> list[torch.Tensor]:
        """all_gather the KV as fp8 e4m3 (half the bytes) + per-tensor scale, then
        dequantize back to bf16.  Two collectives (fp8 bytes + fp32 scale) -> the count
        is still uniform across ranks (all call sites go through _all_gather)."""
        scale = (mine.abs().amax() / settings.FP8_E4M3_MAX).clamp(min=1e-8).to(torch.float32)
        mine_u8 = (mine / scale).to(settings.FP8_DTYPE).view(torch.uint8).contiguous()
        out_u8 = [torch.empty(self.KV_SHAPE, device=self.device, dtype=torch.uint8)
                  for _ in range(self.world)]
        scales = [torch.empty(1, device=self.device, dtype=torch.float32)
                  for _ in range(self.world)]
        timed = self._ag_ts and self.rank == 0 and self._cur_tick >= self.world - 1
        if timed:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        dist.all_gather(out_u8, mine_u8)
        dist.all_gather(scales, scale.reshape(1))
        if timed:
            end.record()
            self._ag_records.append((self._cur_tick, begin, end))
        # The gathered scale is a length-one FP32 tensor. Multiplying it by
        # BF16 promotes the result to FP32, so cast only AFTER dequantization.
        return [(out_u8[n].view(settings.FP8_DTYPE).float() * scales[n]).to(self.dtype)
                for n in range(self.world)]

    def _dummy_allgathers(self) -> None:
        if settings.DISABLE_ALL_GATHER:
            return  # diagnostic: no per-layer collectives at all
        dummy = torch.zeros(self.KV_SHAPE, device=self.device, dtype=self.dtype)
        for i in range(self.Ln):
            if i == 0:
                debug_log(self.rank, "  dummy L0: all_gather start")
            self._all_gather(dummy)
            if i == 0:
                debug_log(self.rank, "  dummy L0: all_gather done")

    def _warmup(self, n_iters: int = 3) -> None:
        """Warm up DiT + VAE kernels/cuBLAS/NCCL at init so the timed wavefront
        runs at steady state.  Every rank runs the same number of forwards (each
        issuing the per-layer all_gathers) so the collective stream stays aligned;
        the store rank additionally warms local VAE decode only in serial-VAE
        mode. Streaming VAE weights belong to separate worker processes."""
        be = self.be
        dev = self.device
        C = be.latent_channels
        H, W = be.ctx.height, be.ctx.width
        dummy = torch.zeros([1, C, self.npb, H, W], device=dev, dtype=self.dtype)

        def warm_exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                      **kw):
            mine = torch.stack([roped_key[0], value[0]])
            self._all_gather(mine)  # exercise the collective + NCCL channels
            return roped_key, value  # self-only context is enough to warm kernels

        debug_log(self.rank, f"warmup: {n_iters} DiT forwards")
        for _ in range(n_iters):
            be.set_exchange_fn(warm_exch)
            try:
                be.forward_chunk(dummy, self.dsl_list[0], start_frame=0)
            finally:
                be.set_exchange_fn(None)
        if (self.rank == self.store_rank and self.cfg.vae_stages == 0
                and not self.cfg.dit_only):
            debug_log(self.rank, "warmup: VAE decode")
            be.decode(dummy.clone())
        torch.cuda.synchronize()
        debug_log(self.rank, "warmup done")

        # --- one-off profile: isolate all_gather (comm) cost from forward compute ---
        if not settings.TICK_PROFILE:
            return

        def noag_exch(*, roped_query, roped_key, value, **kw):
            return roped_key, value  # self-only, NO collective

        def timed(exch_fn, n=3):
            be.barrier()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                be.set_exchange_fn(exch_fn)
                be.forward_chunk(dummy, self.dsl_list[0], start_frame=0)
                be.set_exchange_fn(None)
            torch.cuda.synchronize()
            return (time.perf_counter() - t) * 1000.0 / n

        # NOTE: noag makes all ranks issue the SAME (zero) collectives, so it stays
        # aligned.  warm_exch does the 30 per-layer all_gathers like the real loop.
        t_ag = timed(warm_exch)
        t_noag = timed(noag_exch)
        if self.rank == 0:
            print(
                f"[wave_rt/profile] forward with all_gather={t_ag:.0f}ms, "
                f"without={t_noag:.0f}ms, comm≈{t_ag - t_noag:.0f}ms/forward "
                f"({100*(t_ag-t_noag)/t_ag:.0f}% of tick)",
                flush=True,
            )
        be.barrier()

    def _nccl_selftest(self) -> None:
        """Exercise every collective WaveRT uses (all_gather / broadcast / isend /
        irecv) once, so a hang here isolates a comm-usage bug from a wavefront-loop
        bug."""
        if not settings.DEBUG:
            return
        r, world = self.rank, self.world
        debug_log(r, "selftest: all_gather ...")
        self._all_gather(torch.zeros(self.KV_SHAPE, device=self.device, dtype=self.dtype))
        debug_log(r, "selftest: broadcast(src=store) ...")
        buf = torch.full((4,), float(r), device=self.device)
        dist.broadcast(buf, src=self.store_rank)
        debug_log(r, f"selftest: broadcast got {buf[0].item():.0f} (expect {self.store_rank})")
        debug_log(r, "selftest: ring batch_isend_irecv ...")
        send_buf = torch.full((4,), float(r), device=self.device)
        recv_buf = torch.empty((4,), device=self.device)
        ops = [
            dist.P2POp(dist.irecv, recv_buf, (r - 1) % world),
            dist.P2POp(dist.isend, send_buf, (r + 1) % world),
        ]
        for rq in dist.batch_isend_irecv(ops):
            rq.wait()
        debug_log(r, f"selftest OK: recv from {(r-1)%world} = {recv_buf[0].item():.0f}")

    def _make_denoise_exch(self, clean, act_ranks, tick_csf, tick_qf, store_active):
        store_rank = self.store_rank
        # chunk-causal ("overlap") semantics: rank r processes chunk c=t-r, so
        # in-flight ranks n>=r hold chunks at position <= r's (its causal past +
        # self); ranks n<r are FUTURE chunks -> dropped.  "joint" keeps all.
        # NOTE: the all_gather below is UNCHANGED either way (every rank still
        # gathers all ranks' KV, for NCCL alignment); we only change what we cat.
        if self.cfg.kv_context == "causal":
            infl_ranks = [n for n in act_ranks if n >= self.rank]
        else:
            infl_ranks = act_ranks

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            if layer_idx == 0:
                debug_log(self.rank, "  exch(denoise) L0: all_gather start")
            if settings.DISABLE_ALL_GATHER:
                return roped_key, value  # diagnostic: skip all_gather
            mine = torch.stack([roped_key[0], value[0]])
            gathered = self._all_gather(mine)
            if layer_idx == 0:
                debug_log(self.rank, "  exch(denoise) L0: all_gather done")
            inflK = [gathered[n][0].unsqueeze(0) for n in infl_ranks]
            inflV = [gathered[n][1].unsqueeze(0) for n in infl_ranks]
            aK, aV = self._anchor_working(
                clean[layer_idx], tick_csf, tick_qf,
                post_patch_height, post_patch_width, dim, rope_num_heads,
            )
            liveK, liveV = [], []
            if store_active:
                # the store/finalizing chunk (ct) is the oldest position -> causal
                # past for every denoise rank; always included.
                liveK = [gathered[store_rank][0].unsqueeze(0)]
                liveV = [gathered[store_rank][1].unsqueeze(0)]
                # promote this exchanged slot into a working clean page at tick end
                # (== what the tick-end broadcast used to carry).  clone: run() reads
                # it after the forward returns, past this all_gather's buffer scope.
                if not settings.FULL_BROADCAST:
                    self._exch_store_slot[layer_idx] = gathered[store_rank].clone()
                if settings.BROADCAST_CHECK and self.rank == 0:
                    # consumer-side receipt of the store slot (cross-rank check).
                    self._bc_rank0_recv[layer_idx] = gathered[store_rank].clone()
            K = torch.cat(aK + liveK + inflK, dim=1)
            V = torch.cat(aV + liveV + inflV, dim=1)
            return K, V

        return exch

    def _make_store_exch(self, clean, store_kv, c):
        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block, write_only=False):
            if settings.DISABLE_ALL_GATHER:
                store_kv[layer_idx] = (roped_key, unroped_key, value)
                return roped_key, value  # diagnostic: skip all_gather
            # CRITICAL: the store rank MUST issue the per-layer all_gather just
            # like every denoise rank (and the inactive-rank dummy path).  It
            # contributes its freshly-finalized clean KV as ``gathered[store_rank]``
            # -- which active denoise ranks read as the "live" block -- and it
            # keeps the world's per-tick collective count aligned (Ln all_gathers
            # on every rank).  Omitting it desyncs NCCL and deadlocks the whole
            # wavefront the moment the store rank goes active (t >= store_rank).
            # The store's own attention context does not need the in-flight
            # denoise KV, so the gathered result itself is discarded here.
            mine = torch.stack([roped_key[0], value[0]])
            self._all_gather(mine)
            if settings.BROADCAST_CHECK:
                # store's own all_gather contribution == gathered[store_rank] on
                # every consumer (NCCL all_gather copies byte-identically).
                self._bc_store_mine[layer_idx] = mine.clone()
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

    def _bcast_check(self, t, cid, kr, ku, vv) -> None:
        """WAVE_BCAST_CHECK: compare the broadcast blob components against what the
        per-layer exchange actually moved.  kr/vv should be byte-identical to the
        store's all_gather contribution (store-local proxy for every consumer's
        gathered[store_rank], since NCCL all_gather copies byte-for-byte); ku has
        NO exchange counterpart.  Runs on the store rank."""
        Ln = self.Ln
        mine = self._bc_store_mine  # layer -> stack([roped_key[0], value[0]])
        if len(mine) != Ln:
            print(f"[bcast_check t={t} cid={cid}] SKIP: store stash "
                  f"{len(mine)}/{Ln} layers", flush=True)
            return
        kr_ok = all(torch.equal(kr[L], mine[L][0]) for L in range(Ln))
        vv_ok = all(torch.equal(vv[L], mine[L][1]) for L in range(Ln))
        # is ku present anywhere in the exchange payload? (expected: never)
        ku_in_ag = any(
            torch.equal(ku[L], mine[L][0]) or torch.equal(ku[L], mine[L][1])
            for L in range(Ln)
        )
        kr_eq_ku = all(torch.equal(kr[L], ku[L]) for L in range(Ln))
        print(
            f"[bcast_check t={t} cid={cid}] kr==exch(roped_key): {kr_ok} | "
            f"vv==exch(value): {vv_ok} | ku(unroped) in exch payload: {ku_in_ag} | "
            f"kr==ku: {kr_eq_ku}",
            flush=True,
        )
        # persist store-side layer-0 tensors for a cross-rank torch.equal vs rank0.
        os.makedirs(settings.BROADCAST_CHECK_DIR, exist_ok=True)
        torch.save(
            {"kr0": kr[0].cpu(), "ku0": ku[0].cpu(), "vv0": vv[0].cpu(),
             "mine0": mine[0].cpu()},
            os.path.join(settings.BROADCAST_CHECK_DIR, f"store_t{t}.pt"),
        )

    def _working_target(self, layer, chunk, key=None):
        """Seven retained chunks plus one destination not aliasing that history."""
        if not hasattr(self, '_working_pages'):
            self._working_pages = {}
        # Denoise promotes only after context assembly, so it can reuse the
        # oldest of seven pages. Store writes before assembly and needs a spare.
        depth = 8 if getattr(self, 'rank', self.store_rank) == self.store_rank else 7
        index = (layer, -1 if chunk == 0 else (chunk - 1) % depth)
        page = self._working_pages.get(index)
        shape = ((2, self.tokens_per_chunk, *self.KV_SHAPE[-2:]) if key is None else (2, *key.shape))
        if page is None or tuple(page.shape) != shape:
            page = (torch.empty(shape, device=self.device, dtype=self.dtype)
                    if key is None else key.new_empty(shape))
            self._working_pages[index] = page
        return page

    def _working_page(self, layer, chunk, key, value):
        """Own K/V only; skip the copy when projection already wrote this page."""
        page = self._working_target(layer, chunk, key)
        if page[0].data_ptr() != key.data_ptr():
            if key.is_cuda:
                from wave_rt.runtime.kernels import assemble_context
                assemble_context((key,), (value,), page[0], page[1], self._anchor_axes, 0, anchor=False)
            else:
                page[0].copy_(key)
                page[1].copy_(value)
        return page

    def _assemble_paged_context(self, clean_L, csf, qf, h, w, dim, heads, out_k, out_v, extra=None):
        """One CUDA launch for anchor rotation and all local context placement."""
        if out_k.is_cuda:
            count = max(0, (self.max_attn_frames-qf-self.npb)//self.npb)
            working = clean_L[1:][-count:] if count else []
            keys = ([clean_L[0]['ku']] + [p['kr'] for p in working]) if clean_L else []
            values = ([clean_L[0]['v']] + [p['v'] for p in working]) if clean_L else []
            start = csf-len(working)*self.npb-self.npb
        else:
            # Numerical oracle only; the CUDA path never materializes a rotated
            # anchor or launches one copy for every history chunk.
            keys, values = self._anchor_working(clean_L, csf, qf, h, w, dim, heads)
            start = 0
        if extra is not None:
            keys.append(extra[0]); values.append(extra[1])
        if out_k.is_cuda:
            from wave_rt.runtime.kernels import assemble_context
            assemble_context(keys, values, out_k, out_v, self._anchor_axes, start, anchor=bool(clean_L))
        else:
            for i, (key, value) in enumerate(zip(keys, values)):
                t = key.shape[-3]
                out_k[i*t:(i+1)*t].copy_(key.reshape(-1, *key.shape[-2:]))
                out_v[i*t:(i+1)*t].copy_(value.reshape(-1, *value.shape[-2:]))
        return len(keys)
