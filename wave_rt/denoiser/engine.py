"""WaveDenoiser lifecycle and systolic scheduling."""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist

from wave_rt import bench
from wave_rt.config import WaveConfig
from wave_rt.denoiser import settings
from wave_rt.denoiser.diagnostics import DiagnosticsMixin
from wave_rt.denoiser.exchange import (
    CommonExchangeMixin,
    OneSidedExchangeMixin,
    P2PExchangeMixin,
    PagedExchangeMixin,
    RelayExchangeMixin,
    StaggeredExchangeMixin,
)
from wave_rt.denoiser.utils import debug_log, record_cuda_event
from wave_rt.runtime.backend import WaveBackend


class WaveDenoiser(
    CommonExchangeMixin,
    P2PExchangeMixin,
    OneSidedExchangeMixin,
    PagedExchangeMixin,
    StaggeredExchangeMixin,
    RelayExchangeMixin,
    DiagnosticsMixin,
):
    def __init__(
        self, backend: WaveBackend, cfg: WaveConfig, q=None, meta_q=None
    ) -> None:
        self.be = backend
        self.cfg = cfg
        self.q = q
        self.meta_q = meta_q
        self.rank = backend.rank
        self.world = cfg.wp_size
        self.device = backend.device
        self.dtype = backend.ctx.target_dtype
        self.npb = cfg.num_frames_per_block
        self.Ln = backend.num_layers
        self.rf_step = cfg.rf_step
        self.store_rank = cfg.store_rank
        self.num_blocks = cfg.num_blocks
        self.num_ticks = self.num_blocks + self.world - 1
        self.dsl = backend.dsl
        # Keep timesteps on CPU to avoid per-tick device synchronization.
        self.dsl_list = [float(x) for x in self.dsl.tolist()]
        self.context_noise = float(backend.server_args.pipeline_config.context_noise)

        self._init_geometry()
        self._init_rope()
        self._init_overlap_state()
        self._init_profiler_state()
        self._init_transport_state()

    def _init_overlap_state(self) -> None:
        self.p2p_pg = None
        if self.cfg.exchange_mode == "overlap":
            self.p2p_pg = dist.new_group(list(range(self.world)))
            # Every rank must initialize the communicator before subset P2P begins.
            dist.barrier(group=self.p2p_pg)

        self._p2p_send_pending: list = []
        self._use_prefetch = self.cfg.exchange_mode == "overlap" and settings.PREFETCH
        self._pf_slots: list = [None, None]
        self._pf_works: list = [None, None]
        self._bc_store_mine: dict = {}
        self._bc_rank0_recv: dict = {}
        self._exch_store_slot: dict = {}
        self._nocat_buf: list = [None, None]

    def _init_profiler_state(self) -> None:
        self._edge_ts = settings.EDGE_TIMING
        self._edge_records: list = []
        self._edge_ref = None
        self._cur_tick = -1
        self._cat_ts = settings.CAT_TIMING
        self._cat_records: list = []
        self._layer_prof = settings.LAYER_PROFILE
        self._lp_records: list = []
        self._lp_cur = None
        self._ag_ts = settings.ALL_GATHER_TIMING
        self._ag_records: list = []
        self._machprof = settings.MACHINE_PROFILE
        self._mp_records: list = []
        self._mp_submit_cpu = 0.0
        self._mp_alloc_cpu = 0.0

    def _init_transport_state(self) -> None:
        mode = self.cfg.exchange_mode
        self._onesided = mode == "onesided"
        self._relay = mode == "relay"
        self._staggered = mode == "staggered"
        self._paged = mode == "paged"

        self._os_kv = None
        self._os_flags = None
        self._os_scopy = None
        self._os_pending: list = []
        self._os_dst: dict = {}
        self._os_src_index: dict = {}

        self._rl_kv = None
        self._rl_flags = None
        self._rl_down = None
        self._rl_world = self.world
        self._rl_slot_bytes = 0

        self._stagger_lead = max(1, int(getattr(self.cfg, "stagger_lead", 1)))
        self._st_prefetch = None
        self._st_ev: dict = {}

        self._pg_k = None
        self._pg_v = None
        self._pg_dst: dict = {}
        self._pg_naw = 0
        self._pg_wmax = 0
        # Every cuIpcOpenMemHandle call must be paired with an explicit close while
        # the importing CUDA context is still current.  A resident server creates a
        # fresh WaveDenoiser per request, so relying on process teardown leaks one
        # full set of peer mappings per request.
        self._ipc_handles: list[int] = []
        self._closed = False

        self._physical_skew_ms = float(os.environ.get("WAVE_PHYSICAL_SKEW_MS", "0"))
        self._skew_cycles = 0
        if self._physical_skew_ms > 0:
            khz = torch.cuda.get_device_properties(self.device).clock_rate
            self._skew_cycles = int(self._physical_skew_ms * 1e-3 * khz * 1e3)

        if self._onesided:
            self._init_onesided()
        elif self._relay:
            self._init_relay()
        elif self._staggered:
            self._init_onesided()
            self._st_prefetch = torch.cuda.Stream(device=self.device)
        elif self._paged:
            self._init_paged()

    def _ipc_open(self, handle_bytes: bytes) -> int:
        """Open and retain the raw base pointer needed by cuIpcCloseMemHandle."""
        ptr = self._p2p.ipc_open_handle(handle_bytes)
        self._ipc_handles.append(ptr)
        return ptr

    def close(self, *, coordinated: bool = True) -> None:
        """Release per-request CUDA IPC mappings and their exported slabs.

        Successful requests close cooperatively: all ranks first drain their CUDA
        work, then close imported handles, then cross a second barrier before any
        rank releases allocations exported to its peers.  On an exception path the
        launcher passes ``coordinated=False`` because another rank may already have
        left the process group; cleanup remains best-effort without collectives.
        """
        if self._closed:
            return
        self._closed = True

        errors: list[tuple[str, Exception]] = []

        def attempt(label, fn) -> None:
            try:
                fn()
            except Exception as exc:
                errors.append((label, exc))

        if self._os_scopy is not None:
            attempt("copy-stream synchronize", self._os_scopy.synchronize)
        attempt("device synchronize", lambda: torch.cuda.synchronize(self.device))
        if coordinated:
            attempt("pre-close barrier", self.be.barrier)

        handles = self._ipc_handles
        self._ipc_handles = []
        closer = getattr(self, "_p2p", None)
        if closer is not None:
            for ptr in reversed(handles):
                attempt(
                    f"ipc_close_handle({ptr:#x})",
                    lambda ptr=ptr: closer.ipc_close_handle(ptr),
                )

        if coordinated:
            attempt("post-close barrier", self.be.barrier)

        # Imported mappings are gone on every healthy rank, so exporters may now
        # release their local allocations.  Clearing these references also makes
        # the method idempotent and lets the caching allocator reuse the slabs.
        self._os_dst.clear()
        self._pg_dst.clear()
        self._rl_down = None
        self._os_pending.clear()
        self._os_scopy = None
        self._st_prefetch = None
        self._st_ev.clear()
        self._os_kv = self._os_flags = None
        self._pg_k = self._pg_v = None
        self._rl_kv = self._rl_flags = None

        if errors:
            details = "; ".join(f"{label}: {exc!r}" for label, exc in errors)
            raise RuntimeError(f"WaveDenoiser cleanup failed: {details}") from errors[0][1]

    def _act_denoise(self, n: int, t: int) -> bool:
        c = t - n
        return (0 <= n < self.rf_step) and (0 <= c < self.num_blocks)

    def _act_store(self, t: int) -> bool:
        return 0 <= (t - self.store_rank) < self.num_blocks

    def _det_renoise(self, x0_bcthw, next_t, chunk, stage) -> torch.Tensor:
        """Deterministic re-noising to the next DMD step (seed = 1000*chunk+stage,
        matching naive_rt so every rank reproduces the same trajectory)."""
        scheduler = self.be.ctx.scheduler
        b, c, t_, h, w = x0_bcthw.shape
        x0_btchw = x0_bcthw.permute(0, 2, 1, 3, 4)
        flat = x0_btchw.flatten(0, 1)
        g = torch.Generator(device=self.device).manual_seed(1000 * chunk + stage)
        nz = torch.randn(flat.shape, generator=g, device=self.device, dtype=flat.dtype)
        renoised = scheduler.add_noise(
            flat,
            nz,
            float(next_t) * torch.ones([b * t_], device=self.device, dtype=torch.long),
        ).unflatten(0, (b, t_))
        return renoised.permute(0, 2, 1, 3, 4).contiguous()

    def _select_denoise_exchange(
        self, clean, act_ranks, tick_csf, tick_qf, store_active
    ):
        args = (clean, act_ranks, tick_csf, tick_qf, store_active)
        if self._onesided:
            return self._make_denoise_exch_onesided(*args)
        if self._paged:
            return self._make_denoise_exch_paged(*args)
        if self._staggered:
            recv_srcs = [n for n in act_ranks if n > self.rank]
            if store_active:
                recv_srcs.append(self.store_rank)
            self._stagger_bootstrap(
                recv_srcs, self._cur_tick % self._os_gt, self._cur_tick
            )
            return self._make_denoise_exch_staggered(*args)
        if self._relay:
            return self._make_denoise_exch_relay(*args)
        if self.cfg.exchange_mode == "overlap" and self._use_prefetch:
            recv_srcs = [n for n in act_ranks if n > self.rank]
            if store_active:
                recv_srcs.append(self.store_rank)
            self._bootstrap_recv(recv_srcs)
            return self._make_denoise_exch_prefetch(*args)
        if self.cfg.exchange_mode == "overlap":
            return self._make_denoise_exch_overlap(*args)
        return self._make_denoise_exch(*args)

    def _select_store_exchange(self, clean, store_kv, chunk, act_ranks):
        args = (clean, store_kv, chunk, act_ranks)
        if self._paged:
            return self._make_store_exch_paged(*args)
        if self._onesided or self._staggered:
            return self._make_store_exch_onesided(*args)
        if self._relay:
            return self._make_store_exch_relay(*args)
        if self.cfg.exchange_mode == "overlap" and self._use_prefetch:
            self._pf_slots = [None, None]
            self._pf_works = [None, None]
            return self._make_store_exch_prefetch(*args)
        if self.cfg.exchange_mode == "overlap":
            return self._make_store_exch_overlap(*args)
        return self._make_store_exch(clean, store_kv, chunk)

    def _drain_tick_exchange(self) -> None:
        if self.cfg.exchange_mode == "overlap":
            self._drain_p2p_sends()
        elif self._onesided or self._relay or self._staggered or self._paged:
            self._drain_publish()

    def _systolic(self, full_noise, *, warmup_pass, num_ticks):
        """Run the wavefront once, optionally as an untimed warmup pass."""
        be = self.be
        dev = self.device
        npb, Ln = self.npb, self.Ln
        rf_step, store_rank = self.rf_step, self.store_rank
        _, C, _, H, W = full_noise.shape
        clean = [[] for _ in range(Ln)]
        my_latent = None
        my_x0 = None
        final_latents: dict[int, torch.Tensor] = {}
        tick_ms: list[float] = []
        timeline: list = []  # WAVE_TIMELINE: per-tick phase timestamps
        for t in range(num_ticks):
            tb = time.perf_counter()
            self._cur_tick = t
            is_dn = self._act_denoise(self.rank, t)
            is_st = (self.rank == store_rank) and self._act_store(t)
            c = (t - self.rank) if is_dn else ((t - store_rank) if is_st else -1)

            act_ranks = [n for n in range(rf_step) if self._act_denoise(n, t)]
            oldest_c = (t - max(act_ranks)) if act_ranks else 0
            tick_csf = oldest_c * npb
            tick_qf = len(act_ranks) * npb
            store_active = self._act_store(t)
            debug_log(
                self.rank,
                f"t={t} is_dn={is_dn} is_st={is_st} c={c} "
                f"act_ranks={act_ranks} store_active={store_active}",
            )

            if is_dn and self.rank == 0:
                debug_log(self.rank, f"t={t} slicing noise chunk c={c}")
                my_latent = full_noise[:, :, c * npb : (c + 1) * npb].contiguous()
                debug_log(
                    self.rank, f"t={t} noise chunk ready {tuple(my_latent.shape)}"
                )

            store_kv: list = [None] * Ln
            self._p2p_send_pending = []
            self._exch_store_slot = {}
            _mp = self._machprof and not warmup_pass
            _mp_rec = None
            if _mp:
                self._mp_submit_cpu = 0.0
                self._mp_alloc_cpu = 0.0
                _mp_drain_cpu = 0.0
                _e_start = record_cuda_event()
            _lp_active = self._layer_prof and not warmup_pass and (is_dn or is_st)
            if _lp_active:
                self._lp_cur = {
                    "tick": t,
                    "rank": self.rank,
                    "is_dn": is_dn,
                    "is_st": is_st,
                    "layers": [],
                }
            else:
                self._lp_cur = None
            if settings.BROADCAST_CHECK:
                self._bc_store_mine = {}
                self._bc_rank0_recv = {}

            _prof = (
                settings.TICK_PROFILE
                and self.rank == 0
                and t in settings.PROFILE_TICKS
                and not warmup_pass
            )
            _tlprof = settings.TIMELINE_PROFILE and (is_dn or is_st) and not warmup_pass
            if _prof or _tlprof:
                from wave_rt.runtime.backend import prof_reset

                prof_reset(True)

            if is_dn:
                step = self.dsl_list[self.rank]
                debug_log(self.rank, f"t={t} building denoise exch (step={step:.0f})")
                fn = self._select_denoise_exchange(
                    clean, act_ranks, tick_csf, tick_qf, store_active
                )
                be.set_exchange_fn(fn)
                debug_log(self.rank, f"t={t} denoise forward start (step={step:.0f})")
                if self._skew_cycles > 0 and self._onesided and self.rank != store_rank:
                    torch.cuda._sleep(self._skew_cycles)
                x0 = be.forward_chunk(my_latent, step, start_frame=c * npb)
                be.set_exchange_fn(None)
                if _mp:
                    _td = time.perf_counter()
                    self._drain_tick_exchange()
                    _mp_drain_cpu += (time.perf_counter() - _td) * 1000.0
                else:
                    self._drain_tick_exchange()
                my_x0 = x0.permute(0, 2, 1, 3, 4).contiguous()
                debug_log(self.rank, f"t={t} denoise forward done")
            elif is_st:
                fn = self._select_store_exchange(clean, store_kv, c, act_ranks)
                be.set_exchange_fn(fn)
                debug_log(self.rank, f"t={t} store forward start")
                be.forward_chunk(my_latent, self.context_noise, start_frame=c * npb)
                be.set_exchange_fn(None)
                if _mp:
                    _td = time.perf_counter()
                    self._drain_tick_exchange()
                    _mp_drain_cpu += (time.perf_counter() - _td) * 1000.0
                else:
                    self._drain_tick_exchange()
                final_latents[c] = my_latent.clone()
                if self.q is not None and not warmup_pass:
                    self.q.put(
                        (
                            c,
                            my_latent.permute(0, 2, 1, 3, 4)
                            .contiguous()
                            .float()
                            .cpu()
                            .numpy(),
                        )
                    )
                debug_log(self.rank, f"t={t} store forward done")
            else:
                async_exchange = (
                    self.cfg.exchange_mode == "overlap"
                    or self._onesided
                    or self._relay
                    or self._staggered
                    or self._paged
                )
                if not async_exchange:
                    debug_log(self.rank, f"t={t} dummy allgathers x{Ln}")
                    self._dummy_allgathers()
                debug_log(self.rank, f"t={t} dummy allgathers done")

            if settings.TIMELINE:
                torch.cuda.synchronize()
                _tl_fwd = time.perf_counter()

            if _mp:
                _e_fwd = record_cuda_event()

            _prof2 = _prof
            if _prof2:
                torch.cuda.synchronize()
                _t_fwd = time.perf_counter()

            # Working pages reuse exchanged KV. The anchor still broadcasts its
            # unroped key; WAVE_FULL_BCAST restores the original full broadcast.
            cid = t - store_rank
            if store_active and (settings.FULL_BROADCAST or cid == 0):
                debug_log(self.rank, f"t={t} broadcast clean KV start (cid={cid})")
                Hh = self.be.num_attention_heads
                Dd = self.be.attention_head_dim
                Tt = self.tokens_per_chunk
                if settings.FULL_BROADCAST:
                    if is_st:
                        kr = torch.stack([store_kv[L][0][0] for L in range(Ln)])
                        ku = torch.stack([store_kv[L][1][0] for L in range(Ln)])
                        vv = torch.stack([store_kv[L][2][0] for L in range(Ln)])
                        blob = torch.stack([kr, ku, vv]).contiguous()
                        if settings.BROADCAST_CHECK:
                            self._bcast_check(t, cid, kr, ku, vv)
                    else:
                        blob = torch.empty(
                            (3, Ln, Tt, Hh, Dd), device=dev, dtype=self.dtype
                        )
                    dist.broadcast(blob, src=store_rank)
                    for L in range(Ln):
                        self._append_clean_page(
                            clean[L],
                            dict(
                                kr=blob[0, L].unsqueeze(0),
                                ku=blob[1, L].unsqueeze(0),
                                v=blob[2, L].unsqueeze(0),
                                cid=cid,
                            ),
                        )
                else:
                    if is_st:
                        ku = torch.stack([store_kv[L][1][0] for L in range(Ln)])
                        vv = torch.stack([store_kv[L][2][0] for L in range(Ln)])
                        blob = torch.stack([ku, vv]).contiguous()  # (2,Ln,T,H,D)
                    else:
                        blob = torch.empty(
                            (2, Ln, Tt, Hh, Dd), device=dev, dtype=self.dtype
                        )
                    dist.broadcast(blob, src=store_rank)
                    for L in range(Ln):
                        self._append_clean_page(
                            clean[L],
                            dict(
                                ku=blob[0, L].unsqueeze(0),
                                v=blob[1, L].unsqueeze(0),
                                cid=cid,
                            ),
                        )
                debug_log(self.rank, f"t={t} broadcast clean KV done (cid={cid})")
                if settings.BROADCAST_CHECK and self.rank == 0 and self._bc_rank0_recv:
                    os.makedirs(settings.BROADCAST_CHECK_DIR, exist_ok=True)
                    torch.save(
                        {"recv0": self._bc_rank0_recv[0].cpu()},
                        os.path.join(settings.BROADCAST_CHECK_DIR, f"rank0_t{t}.pt"),
                    )
            elif store_active:
                debug_log(
                    self.rank,
                    f"t={t} promote clean KV (cid={cid}) is_st={is_st} is_dn={is_dn}",
                )
                if is_st:
                    for L in range(Ln):
                        self._append_clean_page(
                            clean[L],
                            dict(
                                kr=store_kv[L][0][0].unsqueeze(0),
                                v=store_kv[L][2][0].unsqueeze(0),
                                cid=cid,
                            ),
                        )
                elif is_dn and len(self._exch_store_slot) == Ln:
                    for L in range(Ln):
                        slot = self._exch_store_slot[L]
                        self._append_clean_page(
                            clean[L],
                            dict(
                                kr=slot[0].unsqueeze(0),
                                v=slot[1].unsqueeze(0),
                                cid=cid,
                            ),
                        )
                debug_log(self.rank, f"t={t} promote clean KV done (cid={cid})")

            if _prof:
                torch.cuda.synchronize()
                _t_bc = time.perf_counter()

            if settings.TIMELINE:
                torch.cuda.synchronize()
                _tl_bc = time.perf_counter()

            # A world collective keeps latent migration ordered on every rank.
            will_recv = (
                (1 <= self.rank <= rf_step - 1) and self._act_denoise(self.rank, t + 1)
            ) or (self.rank == store_rank and self._act_store(t + 1))
            will_send = is_dn
            debug_log(
                self.rank, f"t={t} migrate will_send={will_send} will_recv={will_recv}"
            )
            mig_shape = [1, C, npb, H, W]
            if will_send:
                if self.rank < rf_step - 1:
                    send_payload = self._det_renoise(
                        my_x0, self.dsl_list[self.rank + 1], c, self.rank
                    )
                else:
                    send_payload = my_x0
            else:
                send_payload = torch.zeros(mig_shape, device=dev, dtype=self.dtype)
            if _mp:
                _e_mig0 = record_cuda_event()
                _tm = time.perf_counter()
            gathered_mig = [
                torch.empty(mig_shape, device=dev, dtype=self.dtype)
                for _ in range(self.world)
            ]
            dist.all_gather(gathered_mig, send_payload.contiguous())
            if _mp:
                _mp_mig_submit_cpu = (time.perf_counter() - _tm) * 1000.0
            if will_recv:
                my_latent = gathered_mig[self.rank - 1].clone()
            debug_log(self.rank, f"t={t} migrate done")
            if _mp:
                _e_mig1 = record_cuda_event()
                self._mp_records.append(
                    dict(
                        tick=t,
                        rank=self.rank,
                        is_dn=is_dn,
                        is_st=is_st,
                        e_start=_e_start,
                        e_fwd=_e_fwd,
                        e_mig0=_e_mig0,
                        e_mig1=_e_mig1,
                        submit_cpu=self._mp_submit_cpu,
                        alloc_cpu=self._mp_alloc_cpu,
                        drain_cpu=_mp_drain_cpu,
                        mig_submit_cpu=_mp_mig_submit_cpu,
                    )
                )

            if _lp_active:
                self._lp_records.append(self._lp_cur)
                self._lp_cur = None

            if _prof:
                torch.cuda.synchronize()
                _t_mig = time.perf_counter()
                from wave_rt.runtime.backend import prof_read, prof_reset

                _attn_ms, _exch_ms, _klen = prof_read()
                prof_reset(False)
                print(
                    f"[wave_rt/tickprof t={t}] fwd={(_t_fwd - tb) * 1000:.0f}ms "
                    f"(exch={_exch_ms:.0f}ms attn={_attn_ms:.0f}ms klen={_klen}) "
                    f"bcast={(_t_bc - _t_fwd) * 1000:.0f}ms "
                    f"migrate={(_t_mig - _t_bc) * 1000:.0f}ms",
                    flush=True,
                )

            # Production avoids a per-tick host sync so cross-tick lead survives.
            if settings.TICK_PROFILE or settings.TIMELINE or settings.FORCE_TICK_SYNC:
                torch.cuda.synchronize()
            tick_ms.append((time.perf_counter() - tb) * 1000)
            if settings.TIMELINE:
                _tl_mig = time.perf_counter()
                role = "denoise" if is_dn else ("store" if is_st else "idle")
                _a_ms = _e_ms = _kl = 0
                if _tlprof:
                    from wave_rt.runtime.backend import prof_read, prof_reset

                    _a_ms, _e_ms, _kl = prof_read()
                    prof_reset(False)
                timeline.append(
                    dict(
                        tick=t,
                        rank=self.rank,
                        role=role,
                        chunk=c,
                        t0=tb,
                        fwd=_tl_fwd,
                        bc=_tl_bc,
                        mig=_tl_mig,
                        attn_ms=_a_ms,
                        exch_ms=_e_ms,
                        klen=_kl,
                    )
                )

        if not warmup_pass and self._layer_prof and self._lp_records:
            self._dump_layer_prof()

        return final_latents, tick_ms, timeline

    def run(
        self, out_dir: str | None = None, warmup: bool = True, save: bool = True
    ) -> None:
        be = self.be
        store_rank = self.store_rank
        full_noise = be.full_noise_bcthw

        debug_log(self.rank, f"reached run(); role_ranks store={store_rank}")
        self._nccl_selftest()
        be.barrier()
        if warmup:
            self._warmup()
            # Build overlap channels incrementally using the real fill schedule.
            if self.cfg.exchange_mode == "overlap":
                _wt0 = time.perf_counter()
                _wsave = (self._machprof, self._edge_ts, self._ag_ts, self._cat_ts)
                self._machprof = self._edge_ts = self._ag_ts = self._cat_ts = False
                _wn = min(self.num_ticks, self.world + 2)
                self._systolic(full_noise, warmup_pass=True, num_ticks=_wn)
                (self._machprof, self._edge_ts, self._ag_ts, self._cat_ts) = _wsave
                self._drain_p2p_sends()
                torch.cuda.synchronize()
                be.barrier()
                if self.rank == 0:
                    print(
                        f"[wave_rt/warmup] overlap systolic fill "
                        f"{(time.perf_counter() - _wt0) * 1000:.0f}ms "
                        f"({_wn} ticks)",
                        flush=True,
                    )
        be.barrier()
        debug_log(self.rank, "passed start barrier; entering systolic loop")
        if self._edge_ts and self.rank == 0:
            self._edge_ref = record_cuda_event()
        t_start = time.perf_counter()
        final_latents, tick_ms, timeline = self._systolic(
            full_noise, warmup_pass=False, num_ticks=self.num_ticks
        )
        be.barrier()
        diffusion_ms = (time.perf_counter() - t_start) * 1000

        if self._edge_ts and self.rank == 0:
            self._dump_edge_ts()
        if self._ag_ts and self.rank == 0:
            self._dump_ag_ts()
        if self._machprof:
            self._dump_machprof()

        if self._cat_ts and self.rank == 0 and self._cat_records:
            torch.cuda.synchronize()
            from collections import defaultdict

            per_layer = defaultdict(list)
            for rec in self._cat_records:
                per_layer[rec["layer"]].append(rec["start"].elapsed_time(rec["end"]))
            all_ms = [m for v in per_layer.values() for m in v]
            print(
                "[wave_rt/cat_ts] per-layer mean K/V-cat (ms), steady ticks, "
                f"mode={'no_cat' if settings.DISABLE_CAT else 'cat'}:",
                flush=True,
            )
            for L in sorted(per_layer):
                v = per_layer[L]
                print(
                    f"  layer {L:2d}: {sum(v) / len(v):.4f}ms  (n={len(v)})", flush=True
                )
            print(
                f"[wave_rt/cat_ts] overall mean {sum(all_ms) / len(all_ms):.4f}ms/layer "
                f"across {len(all_ms)} samples; "
                f"~{sum(all_ms) / len(all_ms) * self.Ln:.2f}ms/tick",
                flush=True,
            )

        if settings.TIMELINE:
            os.makedirs(settings.TIMELINE_DIR, exist_ok=True)
            with open(
                os.path.join(settings.TIMELINE_DIR, f"diff_r{self.rank}.json"), "w"
            ) as _f:
                json.dump(
                    dict(
                        rank=self.rank,
                        mode=self.cfg.exchange_mode,
                        kv=self.cfg.kv_context,
                        t_start=t_start,
                        events=timeline,
                    ),
                    _f,
                )

        if self.rank == 0:
            if settings.TICK_PROFILE or settings.TIMELINE:
                mean_tick = sum(tick_ms) / len(tick_ms)
            else:
                mean_tick = diffusion_ms / self.num_ticks
            print(
                f"[wavefront] {self.world} ranks, {self.num_blocks} chunks, "
                f"{self.num_ticks} ticks, diffusion {diffusion_ms:.0f} ms, "
                f"mean tick {mean_tick:.1f} ms",
                flush=True,
            )

        if self.rank == store_rank and final_latents:
            if self.cfg.vae_stages > 0 and self.q is not None:
                self.q.put(None)
                idx = sorted(final_latents.keys())
                latents = torch.cat(
                    [final_latents[i] for i in idx], dim=2
                )  # (1,C,nf,H,W)
                if save:
                    od = out_dir or bench.run_dir(
                        self.cfg.out_root, self.cfg.task, self.cfg.run_tag
                    )
                    os.makedirs(od, exist_ok=True)
                    torch.save(latents.detach().cpu(), os.path.join(od, "latents.pt"))
                    print(
                        f"[wavefront/store] streamed {len(idx)} chunks to VAE, "
                        f"latents -> {od}/latents.pt",
                        flush=True,
                    )
                else:
                    print(
                        f"[wavefront/store] warmup: streamed {len(idx)} chunks (no save)",
                        flush=True,
                    )
                if self.meta_q is not None:
                    self.meta_q.put(
                        ("diff", diffusion_ms, t_start, tick_ms, latents.shape[2])
                    )
            else:
                from wave_rt.pipelines.vae import finalize_on_store

                finalize_on_store(be, self.cfg, final_latents, diffusion_ms, tick_ms)

        be.barrier()
