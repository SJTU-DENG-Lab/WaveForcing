"""Context assembly and synchronized KV exchange."""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import debug_log, record_cuda_event


class CommonExchangeMixin:
    def _init_geometry(self) -> None:
        backend = self.be
        self.max_attn_frames = backend.rf_stage.sliding_window_num_frames
        max_frames = os.environ.get("WAVE_MAX_ATTN_FRAMES", "").strip()
        if max_frames:
            self.max_attn_frames = int(max_frames)

        self.tokens_per_chunk = self.npb * backend.num_token_per_frame
        self._kv_shape = (
            2,
            self.tokens_per_chunk,
            backend.num_attention_heads,
            backend.attention_head_dim,
        )

    def _init_rope(self) -> None:
        from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
            _apply_rotary_emb,
        )
        from sglang.multimodal_gen.runtime.layers.rotary_embedding.factory import (
            get_rotary_pos_embed,
        )
        from sglang.multimodal_gen.runtime.platforms import current_platform

        self._apply_rotary_emb = _apply_rotary_emb
        self._get_rotary_pos_embed = get_rotary_pos_embed
        self._rope_f64 = current_platform.is_float64_supported()
        self._rope_cache: dict = {}

    def _anchor_rope(self, num_frames, pph, ppw, start_frame, dim, num_heads, device):
        """Cached cos/sin table for the anchor re-rope (see __init__ note)."""
        key = (num_frames, pph, ppw, start_frame, dim, num_heads)
        hit = self._rope_cache.get(key)
        if hit is None:
            head_dim = dim // num_heads
            rope_dim_list = [
                head_dim - 4 * (head_dim // 6),
                2 * (head_dim // 6),
                2 * (head_dim // 6),
            ]
            cos, sin = self._get_rotary_pos_embed(
                (num_frames, pph, ppw),
                dim,
                num_heads,
                rope_dim_list,
                dtype=torch.float64 if self._rope_f64 else torch.float32,
                rope_theta=10000,
                start_frame=start_frame,
                device=device,
            )
            hit = (cos.float(), sin.float())
            self._rope_cache[key] = hit
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

    def _assemble_store_context(
        self, clean_layer, chunk, roped_key, value, pph, ppw, dim, rope_num_heads
    ):
        anchor_k, anchor_v = self._anchor_working(
            clean_layer,
            chunk * self.npb,
            self.npb,
            pph,
            ppw,
            dim,
            rope_num_heads,
        )
        return (
            torch.cat(anchor_k + [roped_key], dim=1),
            torch.cat(anchor_v + [value], dim=1),
        )

    @staticmethod
    def _append_clean_page(clean_layer: list, page: dict) -> None:
        clean_layer.append(page)
        if len(clean_layer) > 8:
            clean_layer[:] = clean_layer[:1] + clean_layer[-7:]

    def _all_gather(self, mine: torch.Tensor) -> list[torch.Tensor]:
        if settings.FP8_KV:
            return self._all_gather_fp8(mine)
        if self._machprof:
            _ta = time.perf_counter()
            out = [
                torch.empty(self._kv_shape, device=self.device, dtype=self.dtype)
                for _ in range(self.world)
            ]
            self._mp_alloc_cpu += (time.perf_counter() - _ta) * 1000.0
            _ts = time.perf_counter()
            dist.all_gather(out, mine.contiguous())
            self._mp_submit_cpu += (time.perf_counter() - _ts) * 1000.0
            return out
        out = [
            torch.empty(self._kv_shape, device=self.device, dtype=self.dtype)
            for _ in range(self.world)
        ]
        if self._ag_ts and self.rank == 0 and self._cur_tick >= self.world - 1:
            _e0 = record_cuda_event()
            dist.all_gather(out, mine.contiguous())
            _e1 = record_cuda_event()
            self._ag_records.append((self._cur_tick, _e0, _e1))
            return out
        dist.all_gather(out, mine.contiguous())
        return out

    def _all_gather_fp8(self, mine: torch.Tensor) -> list[torch.Tensor]:
        """Gather FP8 KV and per-tensor scales, then restore the runtime dtype."""
        scale = (
            (mine.abs().amax() / settings.FP8_E4M3_MAX)
            .clamp(min=1e-8)
            .to(torch.float32)
        )
        mine_u8 = (mine / scale).to(settings.FP8_DTYPE).view(torch.uint8).contiguous()
        out_u8 = [
            torch.empty(self._kv_shape, device=self.device, dtype=torch.uint8)
            for _ in range(self.world)
        ]
        dist.all_gather(out_u8, mine_u8)
        scales = [
            torch.empty(1, device=self.device, dtype=torch.float32)
            for _ in range(self.world)
        ]
        dist.all_gather(scales, scale.reshape(1))
        return [
            out_u8[n].view(settings.FP8_DTYPE).to(self.dtype) * scales[n]
            for n in range(self.world)
        ]

    def _dummy_allgathers(self) -> None:
        if settings.DISABLE_ALL_GATHER:
            return
        dummy = torch.zeros(self._kv_shape, device=self.device, dtype=self.dtype)
        for i in range(self.Ln):
            if i == 0:
                debug_log(self.rank, "  dummy L0: all_gather start")
            self._all_gather(dummy)
            if i == 0:
                debug_log(self.rank, "  dummy L0: all_gather done")

    def _warmup(self, n_iters: int = 3) -> None:
        """Warm DiT, NCCL, and the store-rank VAE before timed execution."""
        be = self.be
        dev = self.device
        C = be.latent_channels
        H, W = be.ctx.height, be.ctx.width
        dummy = torch.zeros([1, C, self.npb, H, W], device=dev, dtype=self.dtype)

        def warm_exch(*, layer_idx, roped_query, roped_key, unroped_key, value, **kw):
            mine = torch.stack([roped_key[0], value[0]])
            self._all_gather(mine)
            return roped_key, value

        debug_log(self.rank, f"warmup: {n_iters} DiT forwards")
        for _ in range(n_iters):
            be.set_exchange_fn(warm_exch)
            be.forward_chunk(dummy, self.dsl_list[0], start_frame=0)
            be.set_exchange_fn(None)
        if self.rank == self.store_rank:
            debug_log(self.rank, "warmup: VAE decode")
            be.decode(dummy.clone())
        torch.cuda.synchronize()
        debug_log(self.rank, "warmup done")

        if not settings.TICK_PROFILE:
            return

        def noag_exch(*, roped_query, roped_key, value, **kw):
            return roped_key, value

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

        t_ag = timed(warm_exch)
        t_noag = timed(noag_exch)
        if self.rank == 0:
            print(
                f"[wave_rt/profile] forward with all_gather={t_ag:.0f}ms, "
                f"without={t_noag:.0f}ms, comm≈{t_ag - t_noag:.0f}ms/forward "
                f"({100 * (t_ag - t_noag) / t_ag:.0f}% of tick)",
                flush=True,
            )
        be.barrier()

    def _nccl_selftest(self) -> None:
        """Exercise each communication primitive when debug mode is enabled."""
        if not settings.DEBUG:
            return
        r, world = self.rank, self.world
        debug_log(r, "selftest: all_gather ...")
        self._all_gather(
            torch.zeros(self._kv_shape, device=self.device, dtype=self.dtype)
        )
        debug_log(r, "selftest: broadcast(src=store) ...")
        buf = torch.full((4,), float(r), device=self.device)
        dist.broadcast(buf, src=self.store_rank)
        debug_log(
            r, f"selftest: broadcast got {buf[0].item():.0f} (expect {self.store_rank})"
        )
        debug_log(r, "selftest: ring batch_isend_irecv ...")
        send_buf = torch.full((4,), float(r), device=self.device)
        recv_buf = torch.empty((4,), device=self.device)
        ops = [
            dist.P2POp(dist.irecv, recv_buf, (r - 1) % world),
            dist.P2POp(dist.isend, send_buf, (r + 1) % world),
        ]
        for rq in dist.batch_isend_irecv(ops):
            rq.wait()
        debug_log(
            r, f"selftest OK: recv from {(r - 1) % world} = {recv_buf[0].item():.0f}"
        )

    def _make_denoise_exch(self, clean, act_ranks, tick_csf, tick_qf, store_active):
        store_rank = self.store_rank
        # The collective stays world-aligned; causal mode filters only assembly.
        if self.cfg.kv_context == "causal":
            infl_ranks = [n for n in act_ranks if n >= self.rank]
        else:
            infl_ranks = act_ranks

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
            if layer_idx == 0:
                debug_log(self.rank, "  exch(denoise) L0: all_gather start")
            if settings.DISABLE_ALL_GATHER:
                return roped_key, value
            mine = torch.stack([roped_key[0], value[0]])
            gathered = self._all_gather(mine)
            if layer_idx == 0:
                debug_log(self.rank, "  exch(denoise) L0: all_gather done")
            inflK = [gathered[n][0].unsqueeze(0) for n in infl_ranks]
            inflV = [gathered[n][1].unsqueeze(0) for n in infl_ranks]
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
                liveK = [gathered[store_rank][0].unsqueeze(0)]
                liveV = [gathered[store_rank][1].unsqueeze(0)]
                if not settings.FULL_BROADCAST:
                    self._exch_store_slot[layer_idx] = gathered[store_rank].clone()
                if settings.BROADCAST_CHECK and self.rank == 0:
                    self._bc_rank0_recv[layer_idx] = gathered[store_rank].clone()
            K = torch.cat(aK + liveK + inflK, dim=1)
            V = torch.cat(aV + liveV + inflV, dim=1)
            return K, V

        return exch

    def _make_store_exch(self, clean, store_kv, c):
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
            if settings.DISABLE_ALL_GATHER:
                store_kv[layer_idx] = (roped_key, unroped_key, value)
                return roped_key, value
            # Store must participate once per layer to preserve collective ordering.
            mine = torch.stack([roped_key[0], value[0]])
            self._all_gather(mine)
            if settings.BROADCAST_CHECK:
                self._bc_store_mine[layer_idx] = mine.clone()
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

    def _bcast_check(self, t, cid, kr, ku, vv) -> None:
        """Compare store-side exchange and broadcast payloads."""
        Ln = self.Ln
        mine = self._bc_store_mine
        if len(mine) != Ln:
            print(
                f"[bcast_check t={t} cid={cid}] SKIP: store stash "
                f"{len(mine)}/{Ln} layers",
                flush=True,
            )
            return
        kr_ok = all(torch.equal(kr[L], mine[L][0]) for L in range(Ln))
        vv_ok = all(torch.equal(vv[L], mine[L][1]) for L in range(Ln))
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
        os.makedirs(settings.BROADCAST_CHECK_DIR, exist_ok=True)
        torch.save(
            {
                "kr0": kr[0].cpu(),
                "ku0": ku[0].cpu(),
                "vv0": vv[0].cpu(),
                "mine0": mine[0].cpu(),
            },
            os.path.join(settings.BROADCAST_CHECK_DIR, f"store_t{t}.pt"),
        )
