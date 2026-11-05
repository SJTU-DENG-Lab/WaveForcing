"""WaveDenoiser: Wave runtime implementation."""
from __future__ import annotations
import os
import time
from collections import OrderedDict
import torch
import torch.distributed as dist
from wave_rt import bench
from wave_rt.config import WaveConfig
from wave_rt.runtime.backend import WaveBackend
from wave_rt.denoiser import settings
from wave_rt.denoiser.utils import debug_log
from wave_rt.denoiser.profile import current as runtime_profile_current, span as runtime_span
from wave_rt.denoiser.diagnostics import DiagnosticsMixin
from wave_rt.denoiser.exchange import (CommonExchangeMixin, P2PExchangeMixin, OneSidedExchangeMixin, PagedExchangeMixin, StaggeredExchangeMixin, RelayExchangeMixin)


class WaveDenoiser(CommonExchangeMixin, P2PExchangeMixin, OneSidedExchangeMixin, PagedExchangeMixin, StaggeredExchangeMixin, RelayExchangeMixin, DiagnosticsMixin):
    def __init__(self, backend: WaveBackend, cfg: WaveConfig, q=None, meta_q=None) -> None:
        self._ipc_handles = []
        self._closed = False
        self.be = backend
        self.cfg = cfg
        self.q = q            # mp.Queue to the streaming VAE stages (None -> serial VAE)
        self.meta_q = meta_q  # mp.Queue for timing aggregation in the launcher
        self.rank = backend.rank
        self.world = cfg.wp_size
        self.device = backend.device
        self.dtype = backend.ctx.target_dtype
        if backend.full_noise_bcthw.shape[0] != 1:
            raise ValueError('Wave transport currently requires batch size 1')
        if cfg.exchange_mode in ('onesided', 'staggered', 'paged') and settings.GENERATION_SLOTS < 2:
            raise ValueError('WAVE_OS_GT must be at least 2: migration grants tick-buffer credit')
        if settings.FP8_KV_ONESIDED and cfg.exchange_mode != 'onesided':
            raise ValueError('WAVE_FP8KV_ONESIDED requires onesided exchange')
        from wave_rt.distributed.buffers import Buffers
        self._buffers = Buffers(capacity=64)

        self.npb = cfg.num_frames_per_block
        self.Ln = backend.num_layers
        self.rf_step = cfg.rf_step
        self.store_rank = cfg.store_rank
        self.num_blocks = cfg.num_blocks
        self.num_ticks = self.num_blocks + self.world - 1
        self.dsl = backend.dsl  # (rf_step,) warped denoising steps
        self.dsl_list = [float(x) for x in self.dsl.tolist()]  # CPU copy; avoid
        # per-tick .item() CUDA syncs on the hot path (they can stall a rank right
        # before it issues its forward all_gathers and deadlock the wavefront).
        self.context_noise = float(
            backend.server_args.pipeline_config.context_noise
        )

        self._init_geometry()
        self._init_rope()
        self._init_overlap_state()
        self._init_profiler_state()
        self._init_transport_state()

    def _init_geometry(self) -> None:
        backend, cfg = self.be, self.cfg
        self.max_attn_frames = backend.rf_stage.sliding_window_num_frames  # 21
        # WAVE_MAX_ATTN_FRAMES: override the attention sliding window (frames).  Caps the
        # clean "working" history in _anchor_working (sink/anchor is always kept).  Set
        # to e.g. 6 (=2 chunks) to drop the clean prefix -> context = sink + live +
        # in-flight only.  Used to emulate a smaller rolling window (sink + N running).
        _maf = os.environ.get("WAVE_MAX_ATTN_FRAMES", "")
        if _maf.strip():
            self.max_attn_frames = int(_maf)
        tpf = backend.num_token_per_frame
        H = backend.num_attention_heads
        D = backend.attention_head_dim
        self.tokens_per_chunk = self.npb * tpf
        self.KV_SHAPE = (2, self.tokens_per_chunk, H, D)      # stacked [k,v]
        self.CLEAN_SHAPE = (1, self.tokens_per_chunk, H, D)   # a single k / v

    def _init_rope(self) -> None:
        backend, cfg = self.be, self.cfg
        D = self.KV_SHAPE[-1]
        # RoPE helpers for the anchor re-rope.  We cache the cos/sin table per
        # (num_frames, pph, ppw, start_frame): within a tick all 30 layers re-rope
        # the anchor with IDENTICAL params, so building the float64 freq table once
        # instead of 30x is the single biggest per-tick win (RoPE rebuild, not the
        # attention kernel, was the bottleneck -- sdpa/fa/sage all tied).
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
        self._rope_cache = OrderedDict()
        if cfg.uses_optimized_wan:
            from wave_rt.runtime.compute import AxisRoPE
            ph, pw = backend.transformer.patch_size[1:]
            self._anchor_axes = AxisRoPE((self.num_ticks + 1)*self.npb,
                backend.ctx.height//ph, backend.ctx.width//pw, D, self.device,
                compute_device=self.device,
                dtype=torch.float64 if self._rope_f64 else torch.float32)

    def _init_overlap_state(self) -> None:
        backend, cfg = self.be, self.cfg
        # Overlap mode: a DEDICATED process group for the async KV cascade, so its
        # asymmetric P2P never interleaves with the WORLD collectives (broadcast +
        # migration) -- see the NCCL-alignment note in run()/migration.  Created
        # once on ALL ranks (dist.new_group is itself a collective).  None in sync.
        self.p2p_pg = None
        if cfg.exchange_mode == "overlap":
            self.p2p_pg = dist.new_group(list(range(self.world)))
            # EAGER-init the pg communicator with an all-member collective: NCCL
            # inits a group's comm lazily on first use, and that init is collective
            # over ALL members -- but our per-tick cascade P2P only involves a
            # subset, so idle ranks would never trigger their init side and the
            # first P2P would hang on the ncclUniqueId rendezvous.  A one-time
            # barrier on the pg forces the handshake while all ranks are here.
            dist.barrier(group=self.p2p_pg)
        # per-tick list of (send_reqs, backing_tensor) kept alive until drained.
        self._p2p_send_pending: list = []
        # PREFETCH overlap (depth 1): NO side CUDA streams -- torch runs p2p_pg P2P
        # on the PG's INTERNAL nccl stream regardless of the python stream, so overlap
        # comes for free (internal stream || default compute stream); we just defer
        # each layer's recv gate (work.wait, non-blocking on CPU) to the point of use.
        self._use_prefetch = (cfg.exchange_mode == "overlap") and settings.PREFETCH
        self._pf_slots: list = [None, None]   # ping-pong: {src: recv_buffer}
        self._pf_works: list = [None, None]   # ping-pong: list[Work] per slot
        # WAVE_BCAST_CHECK stashes (reset per tick in run()).
        self._bc_store_mine: dict = {}   # store rank: layer -> all_gather contrib
        self._bc_rank0_recv: dict = {}   # rank 0: layer -> gathered[store_rank]
        # Page-lifecycle promotion (Gate C): the per-layer exchange already delivers
        # the store rank's (roped_key, value) to every ACTIVE denoise rank -- exactly
        # what a working (non-anchor) clean page needs.  Active denoise ranks stash
        # that slot here per layer during the store-active forward, and run() promotes
        # it into clean[] at tick end instead of re-broadcasting it.  Only the anchor
        # (chunk 0) still needs a broadcast, because it reads the UNROPED key (ku),
        # which the exchange never carries.  Reset per tick.
        self._exch_store_slot: dict = {}  # active denoise rank: layer -> [kr, v]

    def _init_profiler_state(self) -> None:
        backend, cfg = self.be, self.cfg
        # WAVE_EDGE_TS diagnostic state (rank 0 only): list of per-(tick,layer) dicts
        # holding CUDA events, a global reference event, and the current tick number
        # (the exch closure has no tick arg -> run() stamps it each tick).
        self._edge_ts = settings.EDGE_TIMING
        self._edge_records: list = []
        self._edge_ref = None
        self._cur_tick = -1
        # WAVE_NO_CAT: ping-pong preallocated (full_k, full_v) reused across layers so
        # the exch slice-copies segments in instead of allocating a fresh cat output.
        # Keyed by L%2; content for layer L is fully consumed by L's attention before
        # L+2 reuses the slot (default-stream ordering), so depth 2 is safe.
        self._nocat_buf: list = [None, None]
        # WAVE_CAT_TS: rank0 per-(tick,layer) (start,end) CUDA event pairs around cat.
        self._cat_ts = settings.CAT_TIMING
        self._cat_records: list = []
        # WAVE_LAYER_PROF: per-layer attribution state (all active ranks).  _lp_cur is
        # the current tick's record (begin event + per-layer event dicts + post-attn
        # event sink filled by the backend); appended to _lp_records at tick end.
        self._layer_prof = settings.LAYER_PROFILE
        self._lp_records: list = []
        self._lp_cur = None
        # WAVE_AG_TS: rank0 per-tick list of (start,end) CUDA event pairs bracketing
        # each dist.all_gather call (sync-path exposed comm).  Layer index unknown to
        # _all_gather -> per-tick totals; per-layer = per-tick / Ln.
        self._ag_ts = settings.ALL_GATHER_TIMING
        self._ag_records: list = []
        # WAVE_MACHPROF: per-tick region CUDA events (all ranks) + CPU accumulators.
        self._machprof = settings.MACHINE_PROFILE
        self._mp_records: list = []   # per-tick dicts of events + cpu accumulators
        self._mp_submit_cpu = 0.0     # per-tick: batch_isend_irecv / all_gather enqueue (ms)
        self._mp_alloc_cpu = 0.0      # per-tick: recv/all_gather output torch.empty (ms)

    def _init_transport_state(self) -> None:
        backend, cfg = self.be, self.cfg
        # ------- one-sided (Gate C-final-1) state -------
        self._onesided = cfg.exchange_mode == "onesided"
        self._os_kv = None            # consumer slab: (n_in, GT, DL, 2, T, H, D)
        self._os_flags = None         # consumer gen flags: (n_in, GT, DL) uint32
        self._os_scopy = None         # dedicated copy stream (per rank)
        self._os_pending: list = []   # keep published `mine` tensors alive per tick
        self._os_dst: dict = {}       # dst_rank -> opened peer slab ptrs + my slot index
        self._os_src_index: dict = {} # inbound src_rank -> row index in my slab
        # ------- neighbor-accumulating relay (exchange_mode=relay) state -------
        self._relay = cfg.exchange_mode == "relay"
        self._rl_kv = None            # consumer recv slab: (GT, Ln, world, 2, T, H, D)
        self._rl_flags = None         # consumer gen flags: (GT, Ln) int32 (one upstream)
        self._rl_down = None          # my single downstream: opened slab ptrs (or None)
        self._rl_world = self.world
        self._rl_slot_bytes = 0
        # ------- staggered (onesided transport + prefetched inbound acquire) state -------
        self._staggered = cfg.exchange_mode == "staggered"
        self._stagger_lead = max(1, int(getattr(cfg, "stagger_lead", 1)))
        self._st_prefetch = None       # dedicated CUDA stream for inbound flag-acquires
        self._st_ev: dict = {}         # layer -> prefetch-ready CUDA event (per tick)
        # ------- paged (onesided transport + in-place attention KV buffer) state -------
        # Producers DMA KV straight into fixed slots of the consumer's contiguous
        # per-layer slots, so attention reads a contiguous prefix in place.
        # Post-attention credits protect reuse during fill, steady and drain.
        # Reuses the onesided ready-flag/IPC/copy-stream machinery.
        self._paged = cfg.exchange_mode == "paged"
        self._pg_k = None              # consumer K buffer (Ln, Wmax*T, H, D)
        self._pg_v = None              # consumer V attn buffer
        self._pg_dst: dict = {}        # dst_rank -> opened peer K/V buffer ptrs + my row
        self._pg_naw = 0               # steady-state anchor+working chunk count
        self._pg_wmax = 0              # max slots per layer (anchor+working+live+infl)
        # ------- physical skew: give store_rank a device-side head start -------
        # WAVE_PHYSICAL_SKEW_MS>0 makes every NON-store rank burn `skew` ms on its
        # compute stream at the start of each denoise forward (before layer 0), so the
        # store rank's KV production runs ahead and the downstream cascade is filled
        # before the denoise ranks reach their exchange points.  onesided-only; 0=off
        # (no code-path change).  GPU-side (torch.cuda._sleep) so the delay lands on
        # the compute stream, not the host, and overlaps nothing it shouldn't.
        self._physical_skew_ms = float(os.environ.get("WAVE_PHYSICAL_SKEW_MS", "0"))
        self._skew_cycles = 0
        if self._physical_skew_ms > 0:
            # clock_rate is kHz -> Hz; cycles = ms * 1e-3 * Hz
            khz = torch.cuda.get_device_properties(self.device).clock_rate
            self._skew_cycles = int(self._physical_skew_ms * 1e-3 * khz * 1e3)
        if self._onesided:
            self._init_onesided()
        elif self._relay:
            self._init_relay()
        elif self._staggered:
            # staggered reuses the ENTIRE one-sided slab / IPC / copy-stream machinery
            # (allocation, IPC export/open, generation flags, fault-in warmup) -- only
            # the CONSUMER acquire is prefetched onto a side stream (see
            # _make_denoise_exch_staggered).  Producers publish exactly as onesided.
            self._init_onesided()
            self._st_prefetch = torch.cuda.Stream(device=self.device)
        elif self._paged:
            self._init_paged()

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
        cached = getattr(self.be, 'rf_schedule', None)
        if cached is not None:
            renoised = cached.add_noise_at(flat, nz, next_t)
        else:
            renoised = scheduler.add_noise(flat, nz,
                float(next_t) * torch.ones([b * t_], device=self.device, dtype=torch.long))
        renoised = renoised.unflatten(0, (b, t_))
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
        """The systolic wavefront loop.  Run once as a dummy fill (warmup_pass=
        True: skips VAE-queue feed + diagnostic records) to warm the overlap P2P
        cascade, then again as the timed pass.  Body is IDENTICAL either way --
        no algorithm change."""
        be = self.be
        dev = self.device
        npb, Ln = self.npb, self.Ln
        rf_step, store_rank = self.rf_step, self.store_rank
        dsl = self.dsl
        _, C, _, H, W = full_noise.shape
        clean = [[] for _ in range(Ln)]
        my_latent = None
        my_x0 = None
        final_latents: dict[int, torch.Tensor] = {}
        self._final_output = torch.empty_like(full_noise) if self.rank == store_rank else None
        migration = torch.empty((self.world, 1, C, npb, H, W), device=dev, dtype=self.dtype)
        migration_slots = list(migration.unbind(0))
        migration_zero = torch.zeros((1, C, npb, H, W), device=dev, dtype=self.dtype)
        tick_events = []
        tick_ms: list[float] = []
        timeline: list = []   # WAVE_TIMELINE: per-tick phase timestamps
        for t in range(num_ticks):
            tb = time.perf_counter()
            if full_noise.is_cuda:
                tick_begin, tick_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                tick_begin.record()
            self._cur_tick = t
            is_dn = self._act_denoise(self.rank, t)
            is_st = (self.rank == store_rank) and self._act_store(t)
            c = (t - self.rank) if is_dn else ((t - store_rank) if is_st else -1)
            profile = runtime_profile_current()
            if profile is not None:
                profile.at(t, c)
                tick_span = profile.begin('round', chunk=c, gpu=True)
                tick_span.update(tick=t, role='denoise' if is_dn else 'store' if is_st else 'idle',
                                 phase='fill' if t<rf_step else 'drain' if t>=self.num_blocks else 'steady')

            act_ranks = [n for n in range(rf_step) if self._act_denoise(n, t)]
            oldest_c = (t - max(act_ranks)) if act_ranks else 0
            tick_csf = oldest_c * npb
            tick_qf = len(act_ranks) * npb
            store_active = self._act_store(t)
            skip_final_clean = (self.cfg.uses_optimized_wan
                                and not warmup_pass and t == self.num_ticks - 1)
            if skip_final_clean:
                store_active = False
            debug_log(self.rank, f"t={t} is_dn={is_dn} is_st={is_st} c={c} "
                            f"act_ranks={act_ranks} store_active={store_active}")

            if is_dn and self.rank == 0:
                debug_log(self.rank, f"t={t} slicing noise chunk c={c}")
                my_latent = full_noise[:, :, c * npb : (c + 1) * npb].contiguous()
                debug_log(self.rank, f"t={t} noise chunk ready {tuple(my_latent.shape)}")

            store_kv: list = [None] * Ln
            overlap = self.cfg.exchange_mode == "overlap"
            onesided = self._onesided
            relay = self._relay
            staggered = self._staggered
            paged = self._paged
            self._p2p_send_pending = []   # reset per tick (drained after forward)
            self._exch_store_slot = {}    # reset per tick (promoted at tick end)
            # WAVE_MACHPROF: reset per-tick CPU accumulators, record tick-start event
            # (default stream, async).  drain_cpu times the _drain_p2p_sends block.
            _mp = self._machprof and not warmup_pass
            _mp_rec = None
            if _mp:
                self._mp_submit_cpu = 0.0
                self._mp_alloc_cpu = 0.0
                _mp_drain_cpu = 0.0
                _e_start = torch.cuda.Event(enable_timing=True)
                _e_start.record()
            # WAVE_LAYER_PROF: create per-tick layer sink (active denoise/store ticks
            # only, non-warmup).  The exch closures fill _lp_cur["layers"] with CUDA
            # events; saved to _lp_records at tick end.
            _lp_active = self._layer_prof and not warmup_pass and (is_dn or is_st)
            if _lp_active:
                self._lp_cur = {"tick": t, "rank": self.rank,
                                "is_dn": is_dn, "is_st": is_st, "layers": []}
            else:
                self._lp_cur = None
            if settings.BROADCAST_CHECK:
                self._bc_store_mine = {}
                self._bc_rank0_recv = {}

            # Profile an ACTIVE denoise forward (rank 0 denoises at t == its chunk
            # index c == t, so it is active for t in [0, num_blocks)).  The old gate
            # (8<=t<=10) profiled rank0 while it was INACTIVE -> it timed the dummy
            # all_gathers, not a forward.  WAVE_PROF_TICKS overrides the ticks.
            _prof = (settings.TICK_PROFILE and self.rank == 0 and t in settings.PROFILE_TICKS
                     and not warmup_pass)
            # WAVE_TL_PROF: split attn/exch on active ranks (defeats overlap -> only
            # for slack analysis, not for overlap timelines).
            _tlprof = settings.TIMELINE_PROFILE and (is_dn or is_st) and not warmup_pass
            if _prof or _tlprof:
                from wave_rt.runtime.backend import prof_reset
                prof_reset(True)

            if is_dn:
                step = self.dsl_list[self.rank]
                debug_log(self.rank, f"t={t} building denoise exch (step={step:.0f})")
                fn = self._select_denoise_exchange(clean, act_ranks, tick_csf, tick_qf, store_active)
                be.set_exchange_fn(fn)
                debug_log(self.rank, f"t={t} denoise forward start (step={step:.0f})")
                # Physical skew: non-store ranks burn GPU cycles at tick start
                # to give store_rank (rank4) a head start for KV copy.  onesided-only.
                if self._skew_cycles > 0 and onesided and self.rank != store_rank:
                    torch.cuda._sleep(self._skew_cycles)
                try:
                    with runtime_span('denoise_forward', detail=False):
                        x0 = be.forward_chunk(my_latent, step, start_frame=c * npb)
                finally:
                    be.set_exchange_fn(None)
                if _mp:
                    _td = time.perf_counter()
                    self._drain_tick_exchange()
                    _mp_drain_cpu += (time.perf_counter() - _td) * 1000.0
                else:
                    self._drain_tick_exchange()
                my_x0 = x0.permute(0, 2, 1, 3, 4).contiguous()  # (1,C,npb,H,W)
                debug_log(self.rank, f"t={t} denoise forward done")
            elif is_st:
                fn = self._select_store_exchange(clean, store_kv, c, act_ranks)
                be.set_exchange_fn(fn)
                debug_log(self.rank, f"t={t} store forward start")
                try:
                    if not skip_final_clean:
                        with runtime_span('store_forward', detail=False):
                            be.forward_store(my_latent, self.context_noise, start_frame=c * npb)
                finally:
                    be.set_exchange_fn(None)
                if _mp:
                    _td = time.perf_counter()
                    self._drain_tick_exchange()
                    _mp_drain_cpu += (time.perf_counter() - _td) * 1000.0
                else:
                    self._drain_tick_exchange()
                target = self._final_output[:, :, c*npb:(c+1)*npb]
                target.copy_(my_latent)
                final_latents[c] = target
                # Stream this finalized chunk to the VAE pipeline AS IT DRAINS off the
                # diagonal (overlap VAE with the ongoing diffusion).  BTCHW numpy to
                # match vae_pipe/vae_stage stage0's permute(0,2,1,3,4) convention.
                if self.q is not None and not warmup_pass:
                    writer = getattr(self, '_latent_writer', None)
                    if writer is not None:
                        writer.submit(c, my_latent)
                    else:
                        self.q.put((c, my_latent.permute(0, 2, 1, 3, 4)
                                    .contiguous().float().cpu().numpy()))
                debug_log(self.rank, f"t={t} store forward done")
            else:
                # overlap/onesided/staggered/paged: inactive ranks do NO per-layer comm
                # (they idle on the P2P/IPC path); sync: they must issue matching
                # dummy all_gathers for WORLD alignment.
                if self.cfg.exchange_mode == "sync" and not skip_final_clean:
                    debug_log(self.rank, f"t={t} dummy allgathers x{Ln}")
                    self._dummy_allgathers()
                debug_log(self.rank, f"t={t} dummy allgathers done")

            if settings.TIMELINE:
                torch.cuda.synchronize()
                _tl_fwd = time.perf_counter()

            if _mp:
                _e_fwd = torch.cuda.Event(enable_timing=True)
                _e_fwd.record()

            _prof2 = _prof
            if _prof2:
                torch.cuda.synchronize()
                _t_fwd = time.perf_counter()

            # Clean-KV promotion (Gate C).  Original design: at every store-active
            # tick the store rank broadcasts the finalized chunk's (kr, ku, v) to
            # all ranks.  But the per-layer exchange ALREADY delivered (kr, v) to
            # every active denoise rank (verified byte-exact, WAVE_BCAST_CHECK), and
            # the scheduling invariant guarantees any rank idle at a store tick is
            # FINISHED (never needs this chunk).  So a WORKING chunk (cid>=1) needs
            # NO broadcast: each rank promotes what it already has into clean[].
            # The ANCHOR (cid==0) still broadcasts -- but only {ku, v}: it is read
            # via _anchor_working's re-RoPE of the UNROPED key, which the exchange
            # never carries (the roped kr the exchange delivered is dead for the
            # anchor).  WAVE_FULL_BCAST=1 restores the original 3-tensor broadcast.
            cid = t - store_rank
            if store_active and (settings.FULL_BROADCAST or cid == 0):
                debug_log(self.rank, f"t={t} broadcast clean KV start (cid={cid})")
                Hh = self.be.num_attention_heads
                Dd = self.be.attention_head_dim
                Tt = self.tokens_per_chunk
                if settings.FULL_BROADCAST:
                    # A/B baseline: original 3-tensor broadcast for every chunk.
                    if is_st:
                        kr = torch.stack([store_kv[L][0][0] for L in range(Ln)])
                        ku = torch.stack([store_kv[L][1][0] for L in range(Ln)])
                        vv = torch.stack([store_kv[L][2][0] for L in range(Ln)])
                        blob = torch.stack([kr, ku, vv]).contiguous()
                        if settings.BROADCAST_CHECK:
                            self._bcast_check(t, cid, kr, ku, vv)
                    else:
                        blob = torch.empty(
                            (3, Ln, Tt, Hh, Dd), device=dev, dtype=self.dtype)
                    dist.broadcast(blob, src=store_rank)
                    for L in range(Ln):
                        clean[L].append(dict(
                            kr=blob[0, L].unsqueeze(0), ku=blob[1, L].unsqueeze(0),
                            v=blob[2, L].unsqueeze(0), cid=cid))
                        if len(clean[L]) > 8:
                            clean[L] = clean[L][:1] + clean[L][-7:]
                else:
                    # ANCHOR only: broadcast {ku, v} (2 tensors; kr is dead here).
                    if is_st:
                        ku = torch.stack([store_kv[L][1][0] for L in range(Ln)])
                        vv = torch.stack([store_kv[L][2][0] for L in range(Ln)])
                        blob = torch.stack([ku, vv]).contiguous()  # (2,Ln,T,H,D)
                    else:
                        blob = torch.empty(
                            (2, Ln, Tt, Hh, Dd), device=dev, dtype=self.dtype)
                    dist.broadcast(blob, src=store_rank)
                    for L in range(Ln):
                        clean[L].append(dict(
                            ku=blob[0, L].unsqueeze(0),
                            v=blob[1, L].unsqueeze(0), cid=cid))
                debug_log(self.rank, f"t={t} broadcast clean KV done (cid={cid})")
                if paged and is_st and cid == 0:
                    # Broadcast now owns anchor {raw K,V}; outgoing copies were
                    # drained before this collective. The rotated publication
                    # source has no future reader and must not retain Ln pages.
                    for L in range(Ln):
                        getattr(self, '_working_pages', {}).pop((L, -1), None)
                if settings.BROADCAST_CHECK and self.rank == 0 and self._bc_rank0_recv:
                    os.makedirs(settings.BROADCAST_CHECK_DIR, exist_ok=True)
                    torch.save(
                        {"recv0": self._bc_rank0_recv[0].cpu()},
                        os.path.join(settings.BROADCAST_CHECK_DIR, f"rank0_t{t}.pt"))
            elif store_active:
                # WORKING chunk (cid>=1), no broadcast: promote the (kr, v) each rank
                # already holds into a clean working page.  Store rank: local
                # store_kv.  Active denoise rank: the exchange-cached slot.  Idle
                # ranks are finished -> skip (they never read this chunk again).
                debug_log(self.rank, f"t={t} promote clean KV (cid={cid}) is_st={is_st} is_dn={is_dn}")
                if is_st:
                    for L in range(Ln):
                        clean[L].append(dict(
                            kr=store_kv[L][0][0].unsqueeze(0),
                            v=store_kv[L][2][0].unsqueeze(0), cid=cid))
                        if len(clean[L]) > 8:
                            clean[L] = clean[L][:1] + clean[L][-7:]
                elif is_dn and len(self._exch_store_slot) == Ln:
                    for L in range(Ln):
                        slot = self._exch_store_slot[L]  # (2, T, H, D) = [kr, v]
                        clean[L].append(dict(
                            kr=slot[0].unsqueeze(0),
                            v=slot[1].unsqueeze(0), cid=cid))
                        if len(clean[L]) > 8:
                            clean[L] = clean[L][:1] + clean[L][-7:]
                debug_log(self.rank, f"t={t} promote clean KV done (cid={cid})")


            if _prof:
                torch.cuda.synchronize()
                _t_bc = time.perf_counter()

            if settings.TIMELINE:
                torch.cuda.synchronize()
                _tl_bc = time.perf_counter()

            # migration: shift chunks down the diagonal (chain 0->..->store).
            # Implemented with all_gather (NOT per-rank isend/irecv): mixing p2p
            # that only *some* ranks issue into the default PG desyncs NCCL's
            # per-rank op-stream ordering vs the every-rank layer all_gathers and
            # deadlocks.  A collective every tick keeps the op stream aligned.
            will_recv = (
                (1 <= self.rank <= rf_step - 1)
                and self._act_denoise(self.rank, t + 1)
            ) or (self.rank == store_rank and self._act_store(t + 1))
            will_send = is_dn
            debug_log(self.rank, f"t={t} migrate will_send={will_send} will_recv={will_recv}")
            mig_shape = [1, C, npb, H, W]
            if will_send:
                if self.rank < rf_step - 1:
                    send_payload = self._det_renoise(
                        my_x0, self.dsl_list[self.rank + 1], c, self.rank
                    )
                else:  # last denoise rank -> raw finalized latent to the store rank
                    send_payload = my_x0
            else:
                send_payload = migration_zero
            # migration stays an all_gather even in overlap: like the broadcast, a
            # P2P version measured slower than the optimized collective, and it
            # already carries only small latents.
            if _mp:
                _e_mig0 = torch.cuda.Event(enable_timing=True)
                _e_mig0.record()
                _tm = time.perf_counter()
            gathered_mig = migration_slots
            with runtime_span('round_boundary_migration', detail=False,
                              local_payload_bytes=send_payload.numel()*send_payload.element_size()):
                dist.all_gather(gathered_mig, send_payload.contiguous())
            if _mp:
                _mp_mig_submit_cpu = (time.perf_counter() - _tm) * 1000.0
            if will_recv:
                # The next tick consumes this slot before migration overwrites it.
                # The collective remains the credit barrier for IPC generations.
                my_latent = gathered_mig[self.rank - 1]
            debug_log(self.rank, f"t={t} migrate done")
            if _mp:
                _e_mig1 = torch.cuda.Event(enable_timing=True)
                _e_mig1.record()
                self._mp_records.append(dict(
                    tick=t, rank=self.rank, is_dn=is_dn, is_st=is_st,
                    e_start=_e_start, e_fwd=_e_fwd, e_mig0=_e_mig0, e_mig1=_e_mig1,
                    submit_cpu=self._mp_submit_cpu, alloc_cpu=self._mp_alloc_cpu,
                    drain_cpu=_mp_drain_cpu, mig_submit_cpu=_mp_mig_submit_cpu))

            # WAVE_LAYER_PROF: save this tick's per-layer sink.
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

            # Per-tick CPU<->GPU sync is only needed to make tick_ms a true
            # per-tick wall time (timing).  On the production path we skip it so a
            # rank that finishes early can launch the next tick's kernels while its
            # GPU still drains the migration all_gather -> cross-tick lead survives
            # (the migration all_gather remains the WORLD op-stream barrier; stream
            # ordering keeps correctness).  diffusion_ms (be.barrier below) stays
            # the honest wall clock; mean tick falls back to diffusion/num_ticks.
            if settings.TICK_PROFILE or settings.TIMELINE or settings.FORCE_TICK_SYNC:
                torch.cuda.synchronize()
            tick_ms.append((time.perf_counter() - tb) * 1000)
            if full_noise.is_cuda:
                tick_end.record()
                tick_events.append((tick_begin, tick_end))
            if profile is not None:
                profile.end(tick_span)
            if settings.TIMELINE:
                _tl_mig = time.perf_counter()
                role = "denoise" if is_dn else ("store" if is_st else "idle")
                _a_ms = _e_ms = _kl = 0
                if _tlprof:
                    from wave_rt.runtime.backend import prof_read, prof_reset
                    _a_ms, _e_ms, _kl = prof_read()
                    prof_reset(False)
                timeline.append(dict(
                    tick=t, rank=self.rank, role=role, chunk=c,
                    t0=tb, fwd=_tl_fwd, bc=_tl_bc, mig=_tl_mig,
                    attn_ms=_a_ms, exch_ms=_e_ms, klen=_kl,
                ))

        if not warmup_pass and self._layer_prof and self._lp_records:
            self._dump_layer_prof()

        self._tick_events = tick_events
        return final_latents, tick_ms, timeline

    def run(self, out_dir: str | None = None, warmup: bool = True,
            save: bool = True) -> dict:
        be = self.be
        dev = self.device
        npb, Ln = self.npb, self.Ln
        rf_step, store_rank = self.rf_step, self.store_rank
        dsl = self.dsl
        latents = None
        full_noise = be.full_noise_bcthw  # (1, C, nf, H, W)
        _, C, _, H, W = full_noise.shape

        debug_log(self.rank, f"reached run(); role_ranks store={store_rank}")
        self._nccl_selftest()
        be.barrier()
        # In serving the model is warmed once at startup by a full dummy generation
        # (warmup=False here for every request); one-shot warms before its timed run.
        if warmup:
            self._warmup()
            # Dummy pass through the REAL overlap systolic path: the async P2P
            # cascade builds its 10 directed NCCL channels INCREMENTALLY over the
            # fill (t0->t4), exactly as the timed run does -- so the cold,
            # CPU-blocking channel setup is paid HERE, not on the timed loop.
            # (A cold all-at-once P2P warmup deadlocks: bundled bootstrap recvs
            # form a circular channel-build wait; the incremental fill does not.)
            # Overlap only -- sync's all_gather is already warmed by _warmup.
            if self.cfg.exchange_mode == "overlap":
                _wt0 = time.perf_counter()
                _wsave = (self._machprof, self._edge_ts, self._ag_ts,
                          self._cat_ts)
                self._machprof = self._edge_ts = self._ag_ts = \
                    self._cat_ts = False
                _wn = min(self.num_ticks, self.world + 2)
                self._systolic(full_noise, warmup_pass=True, num_ticks=_wn)
                (self._machprof, self._edge_ts, self._ag_ts,
                 self._cat_ts) = _wsave
                self._drain_p2p_sends()
                torch.cuda.synchronize()
                be.barrier()
                if self.rank == 0:
                    print(f"[wave_rt/warmup] overlap systolic fill "
                          f"{(time.perf_counter() - _wt0) * 1000:.0f}ms "
                          f"({_wn} ticks)", flush=True)
        if self.cfg.exchange_mode == 'paged' and hasattr(self, '_pg_consumed'):
            # Permit a fresh run on the same engine without stale ready/credit
            # generations satisfying an acquire from this request prematurely.
            self._os_scopy.synchronize()
            be.barrier()
            if self._pg_consumed is not None:
                self._pg_consumed.zero_()
                self._os_flags.zero_()
            torch.cuda.synchronize(self.device)
        be.barrier()
        debug_log(self.rank, "passed start barrier; entering systolic loop")
        if self._edge_ts and self.rank == 0:
            self._edge_ref = torch.cuda.Event(enable_timing=True)
            self._edge_ref.record()
        t_start = time.perf_counter()
        self._latent_writer = None
        if self.rank == store_rank and self.q is not None and self.cfg.uses_optimized_wan:
            from wave_rt.pipelines.latent_output import LatentQueueWriter
            self._latent_writer = LatentQueueWriter(self.q, self.device,
                                                    trace=getattr(self, 'output_trace', None))
        try:
            final_latents, tick_ms, timeline = self._systolic(
                full_noise, warmup_pass=False, num_ticks=self.num_ticks)
        except BaseException:
            # Do not wait for CUDA/D2H on a failed rank. The launcher terminates
            # this worker and its daemon writer after observing the failure.
            self._latent_writer = None
            raise
        else:
            if self._latent_writer is not None:
                self._latent_writer.close()
                self._latent_writer = None
        be.barrier()
        diffusion_ms = (time.perf_counter() - t_start) * 1000
        if self._tick_events:
            self._tick_events[-1][1].synchronize()
            tick_ms = [a.elapsed_time(b) for a, b in self._tick_events]

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
            print("[wave_rt/cat_ts] per-layer mean K/V-cat (ms), steady ticks, "
                  f"mode={'no_cat' if settings.DISABLE_CAT else 'cat'}:", flush=True)
            for L in sorted(per_layer):
                v = per_layer[L]
                print(f"  layer {L:2d}: {sum(v)/len(v):.4f}ms  (n={len(v)})", flush=True)
            print(f"[wave_rt/cat_ts] overall mean {sum(all_ms)/len(all_ms):.4f}ms/layer "
                  f"across {len(all_ms)} samples; "
                  f"~{sum(all_ms)/len(all_ms)*self.Ln:.2f}ms/tick", flush=True)

        if settings.TIMELINE:
            import json as _json
            os.makedirs(settings.TIMELINE_DIR, exist_ok=True)
            with open(os.path.join(settings.TIMELINE_DIR, f"diff_r{self.rank}.json"), "w") as _f:
                _json.dump(dict(rank=self.rank, mode=self.cfg.exchange_mode,
                                kv=self.cfg.kv_context, t_start=t_start,
                                events=timeline), _f)

        if self.rank == 0:
            # Without the per-tick sync, tick_ms is only CPU-launch time; the honest
            # per-tick wall clock is diffusion_ms / num_ticks.  Use tick_ms only when
            # it was actually synced (timing runs).
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

        # finalize on the store rank
        if self.rank == store_rank and final_latents:
            if self.cfg.dit_only:
                latents = self._final_output
            elif self.cfg.vae_stages > 0 and self.q is not None:
                # streaming n-stage VAE: chunks were already pushed to q as they
                # finalized; signal end, persist latents (PSNR gate) + diffusion
                # timing, and let the VAE stages save the video (metrics in launcher).
                self.q.put(None)
                idx = sorted(final_latents.keys())
                latents = self._final_output
                if save:
                    od = out_dir or bench.run_dir(self.cfg.out_root, self.cfg.task, self.cfg.run_tag)
                    os.makedirs(od, exist_ok=True)
                    torch.save(latents.detach().cpu(), os.path.join(od, "latents.pt"))
                    print(f"[wavefront/store] streamed {len(idx)} chunks to VAE, "
                          f"latents -> {od}/latents.pt", flush=True)
                else:
                    print(f"[wavefront/store] warmup: streamed {len(idx)} chunks (no save)", flush=True)
                if self.meta_q is not None:
                    self.meta_q.put(("diff", diffusion_ms, t_start, tick_ms,
                                     latents.shape[2]))
            else:
                # serial fallback (vae_stages == 0)
                from wave_rt.pipelines.vae import finalize_on_store
                latents = self._final_output
                finalize_on_store(be, self.cfg, latents, diffusion_ms, tick_ms, save=save)

        be.barrier()
        return dict(diffusion_ms=diffusion_ms, start_time=t_start,
                    tick_ms=tick_ms, latents=latents)

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

        self._pg_payload = self._pg_consumed = None

        if errors:
            details = "; ".join(f"{label}: {exc!r}" for label, exc in errors)
            raise RuntimeError(f"WaveDenoiser cleanup failed: {details}") from errors[0][1]
