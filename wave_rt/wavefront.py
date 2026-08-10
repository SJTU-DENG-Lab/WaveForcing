"""WaveRT systolic wavefront (ported from naive_rt/wave/history/s2_wave_dist.py).

wp_size ranks form a diagonal:
  * ranks 0..rf_step-1 are denoising stages (rank r runs DMD step dsl[r]);
  * rank rf_step (== store_rank) re-encodes each finalized chunk at
    context_noise to produce its "clean" KV, and keeps the final latents.

At every tick each active rank runs ONE single-chunk DiT forward.  Inside that
forward, WaveRT's monkeypatched self-attention calls a per-layer exchange
closure that all_gathers the in-flight KV of the whole wavefront and assembles
the joint attention context:  [anchor + working(clean history) + live(chunk
being stored this tick) + in-flight(all active denoise chunks)].  This is the
per-layer, same-tick KV exchange that recovers the naive joint-window quality.

Inactive ranks issue matching dummy all_gathers so the collective count stays
aligned across the world.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

from wave_rt import bench
from wave_rt.backend import WaveBackend
from wave_rt.config import WaveConfig
_DEBUG = os.environ.get("WAVE_DEBUG", "") not in ("", "0", "false")
_DBG_DIR = os.environ.get("WAVE_DEBUG_DIR", "/tmp/wrt_dbg")
_TICKPROF = os.environ.get("WAVE_TICKPROF", "") not in ("", "0", "false")
# Per-rank per-tick phase-timing for the Gantt timeline (scripts/wp_timeline.py).
# Adds a cuda.synchronize before each phase timestamp (perturbs perf -- diagnostic
# runs only), writes /tmp/wrt_timeline/diff_r{rank}.json.
_TIMELINE = os.environ.get("WAVE_TIMELINE", "") not in ("", "0", "false")
_TL_DIR = os.environ.get("WAVE_TIMELINE_DIR", "/tmp/wrt_timeline")
# WAVE_TL_PROF: additionally split each rank's forward into attn_ms/exch_ms via the
# per-layer prof hook.  OFF by default because that per-layer cuda.sync SERIALIZES
# and DEFEATS the prefetch overlap -- so a plain WAVE_TIMELINE run keeps the real
# (overlap-preserving) forward wall time; enable TL_PROF only for slack analysis.
_TL_PROF = os.environ.get("WAVE_TL_PROF", "") not in ("", "0", "false")
# Stream-based PREFETCH overlap: prefetch each layer's KV recv (depth 1) so the
# transfer runs on p2p_pg's internal nccl stream concurrently with the prior layer's
# compute; gate via work.wait() (non-blocking).  DEFAULT ON for exchange_mode=overlap
# (validated: bitwise-identical to causal-sync, deadlock-free, fastest config: 14B
# steady tick 711->671ms).  WAVE_PREFETCH=0 falls back to the simple per-layer-wait
# overlap.  NOTE: overlap is PARTIAL (NCCL P2P recv kernels share SMs with compute),
# so it hides ~half the comm, not all.
_PREFETCH = os.environ.get("WAVE_PREFETCH", "1") not in ("", "0", "false")
# diagnostic: skip the per-layer all_gathers entirely (self-only context on every
# rank -> collectives stay aligned).  Breaks correctness; isolates the per-layer
# collective sync/straggler cost from the forward compute.
_NOAG = os.environ.get("WAVE_NOAG", "") not in ("", "0", "false")
# FP8-KV: transfer the exchanged KV as fp8 e4m3 (per-tensor scale) instead of bf16,
# halving the exchange bytes (14B: ~79->~40ms of comm on the critical path).  Applies
# to the SYNC all_gather path (every call site branches inside _all_gather, so the
# per-tick collective COUNT stays aligned across ranks).  Quant/dequant adds a few
# elementwise kernels; net win when comm dominates.  Overlap/P2P path not yet wired.
_FP8KV = os.environ.get("WAVE_FP8_KV", "") not in ("", "0", "false")
_FP8KV_OS = os.environ.get("WAVE_FP8KV_ONESIDED", "") not in ("", "0", "false")
_FP8KV_E4M3_MAX = 448.0
_FP8KV_DT = torch.float8_e4m3fn
# WAVE_BCAST_CHECK: Gate-C diagnostic.  Verify whether the tick-end clean-KV
# broadcast is byte-redundant with what the per-layer exchange already moved.
# Store rank stashes its all_gather contribution (mine = [roped_key, value]) per
# layer; consumer rank 0 stashes the store slot it receives.  At broadcast time
# we torch.equal the blob components (kr,ku,vv) against those.  Also dumps one
# layer to /tmp for a cross-rank torch.equal.  OFF by default (adds clones).
_BCAST_CHECK = os.environ.get("WAVE_BCAST_CHECK", "") not in ("", "0", "false")
_BCAST_CHECK_DIR = os.environ.get("WAVE_BCAST_CHECK_DIR", "/tmp/wrt_bcast_check")
# Gate C page-lifecycle promotion: skip the tick-end broadcast for WORKING chunks
# (cid>=1) -- their (roped_key, value) is already delivered to every active denoise
# rank by the per-layer exchange, so run() promotes that cached slot into clean[]
# instead of re-broadcasting.  The anchor (cid==0) still broadcasts, but only its
# UNROPED key + value ({ku, v}), since the exchange carries the roped key.  DEFAULT
# ON; WAVE_FULL_BCAST=1 reverts to the original 3-tensor broadcast for every chunk
# (A/B baseline).  See docs + the WAVE_BCAST_CHECK byte-exact verification.
_FULL_BCAST = os.environ.get("WAVE_FULL_BCAST", "") not in ("", "0", "false")
# A/B gates for Task-A benefit verification (default OFF == current "fix" behavior):
#   WAVE_ORIG_SEND_ORDER=1 : prefetch cascade sends farthest-first (ascending), the
#     pre-fix order, instead of nearest-first (reverse).
#   WAVE_FORCE_TICK_SYNC=1 : restore the unconditional per-tick torch.cuda.synchronize
#     (the pre-fix baseline that serialized ticks; the fix skips it).
_ORIG_SEND_ORDER = os.environ.get("WAVE_ORIG_SEND_ORDER", "") not in ("", "0", "false")
_FORCE_TICK_SYNC = os.environ.get("WAVE_FORCE_TICK_SYNC", "") not in ("", "0", "false")
# WAVE_EDGE_TS: per-edge/per-layer timestamp diagnostic for the PREFETCH overlap path.
# On rank 0 only (the heaviest consumer: recv from ranks 1,2,3 + store), record CUDA
# events around each layer's exch closure -- exch entry ("prior compute done"), one
# event after each source's recv work.wait() (in recv_srcs order), and one after the
# K/V cat (attention about to start).  To time sources SEPARATELY, _post_recv fans the
# per-layer recv into one Work PER source under this flag (production coalesces them
# into a single Work -> per-edge skew would collapse to 0).  Events are async (no
# per-layer sync -> overlap preserved); a single torch.cuda.synchronize at end of
# run() reads all elapsed times.
# Dumps a per-layer table + raw JSON to WAVE_EDGE_TS_DIR.  See docs/exp.md.
_EDGE_TS = os.environ.get("WAVE_EDGE_TS", "") not in ("", "0", "false")
_EDGE_TS_DIR = os.environ.get("WAVE_EDGE_TS_DIR", "/tmp/wrt_edge_ts")
# WAVE_CAT_TS: rank-0 diagnostic that brackets the per-layer K/V torch.cat in the
# PREFETCH overlap exch with CUDA events (async, read once at end of run) to isolate
# how much of the exposed per-layer time is the KV assembly (cat) vs P2P.  Steady
# ticks only (>= world-1).  Prints a per-layer table + mean.  OFF by default.
_CAT_TS = os.environ.get("WAVE_CAT_TS", "") not in ("", "0", "false")
# WAVE_NO_CAT: replace the two torch.cat calls in the PREFETCH overlap exch with a
# preallocated ping-pong buffer + per-segment slice copy_ (byte-identical assembly,
# no fresh allocation per layer).  A/B lever against the cat baseline.  OFF by default.
_NO_CAT = os.environ.get("WAVE_NO_CAT", "") not in ("", "0", "false")
# WAVE_AG_TS: rank-0 diagnostic that brackets each per-layer dist.all_gather (the
# SYNC-path collective) with CUDA events (async, read once at end of run) to measure
# the per-layer all_gather GPU time -- i.e. the comm that is FULLY exposed on the
# critical path in sync mode.  Steady ticks only (>= world-1).  OFF by default.
_AG_TS = os.environ.get("WAVE_AG_TS", "") not in ("", "0", "false")
# WAVE_MACHPROF: Gate C-final-0 machinery profiler.  Runs on ALL ranks, default OFF.
# Attributes the overlap-vs-sync per-tick residual bottom-up WITHOUT perturbing the
# hot path: (a) CUDA events bracket each per-tick region on the DEFAULT stream --
# forward / drain / promote+bcast / migration -- read once via a single end-of-run
# synchronize (async -> overlap preserved, no per-tick sync); (b) CPU perf_counter
# accumulators (per tick) time the P2P submit (batch_isend_irecv enqueue), the recv
# buffer allocation (torch.empty), the sync all_gather submit+alloc, and the drain
# wait (_drain_p2p_sends, a real CPU block).  The event-region durations sum to the
# GPU-timeline tick (== diffusion/num_ticks); the CPU accumulators isolate the
# python-side machinery cost inside the forward region.  Dumps a per-rank per-tick
# table + steady-state means to WAVE_MACHPROF_DIR.  ALGORITHM UNCHANGED.
_MACHPROF = os.environ.get("WAVE_MACHPROF", "") not in ("", "0", "false")
_MACHPROF_DIR = os.environ.get("WAVE_MACHPROF_DIR", "/tmp/wrt_machprof")
# WAVE_OS_GT: onesided tick-generation double-buffer depth (default 2, the safe
# value: tick t+2 reuses gen=t%2, and the tick-end migration barrier guarantees
# the consumer finished reading gen=t%2 before then).  1 would race; >2 costs
# memory with no benefit at the current cross-tick lead.
_OS_GT = int(os.environ.get("WAVE_OS_GT", "2"))
# WAVE_LAYER_PROF: per-layer attribution (all active ranks, non-warmup ticks).  For
# the onesided / prefetch-overlap denoise+store exch paths, brackets every layer's
# publish / wait / cat with CUDA events (async; elapsed_time read once at end -> no
# per-layer synchronize, overlap preserved), and the backend records one post-attn
# event per layer so sage (attention) is separable from the pre-attn compute.  Per
# layer: compute = prev-attn-end->exch-enter, publish, wait, cat, sage = cat->attn-end.
# Dumps per-rank JSON rows {rank,tick,layer,compute_ms,publish_ms,wait_ms,cat_ms,
# sage_ms,total_ms} to WAVE_LAYER_PROF_DIR + a steady-state per-layer mean.  OFF by
# default.  Instrumentation only -- algorithm unchanged.
_LAYER_PROF = os.environ.get("WAVE_LAYER_PROF", "") not in ("", "0", "false")
_LAYER_PROF_DIR = os.environ.get("WAVE_LAYER_PROF_DIR", "/tmp/wrt_layer_os")
# Ticks at which rank 0's active denoise forward is profiled (WAVE_TICKPROF=1).
# Default {6,7}: past the fill ramp so the attention window is (near) saturated.
_PROF_TICKS = {
    int(x) for x in os.environ.get("WAVE_PROF_TICKS", "6,7").split(",") if x.strip()
}


def _dbg(rank: int, msg: str) -> None:
    if not _DEBUG:
        return
    # Per-rank file: avoids interleaving/line-tearing from the shared stdout that
    # NCCL floods, and survives SIGABRT (append + close each call).
    os.makedirs(_DBG_DIR, exist_ok=True)
    with open(os.path.join(_DBG_DIR, f"r{rank}.log"), "a") as f:
        f.write(msg + "\n")


class WaveDenoiser:
    def __init__(self, backend: WaveBackend, cfg: WaveConfig, q=None, meta_q=None) -> None:
        self.be = backend
        self.cfg = cfg
        self.q = q            # mp.Queue to the streaming VAE stages (None -> serial VAE)
        self.meta_q = meta_q  # mp.Queue for timing aggregation in the launcher
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
        self.dsl = backend.dsl  # (rf_step,) warped denoising steps
        self.dsl_list = [float(x) for x in self.dsl.tolist()]  # CPU copy; avoid
        # per-tick .item() CUDA syncs on the hot path (they can stall a rank right
        # before it issues its forward all_gathers and deadlock the wavefront).
        self.context_noise = float(
            backend.server_args.pipeline_config.context_noise
        )

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
        self._rope_cache: dict = {}

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
        self._use_prefetch = (cfg.exchange_mode == "overlap") and _PREFETCH
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
        # WAVE_EDGE_TS diagnostic state (rank 0 only): list of per-(tick,layer) dicts
        # holding CUDA events, a global reference event, and the current tick number
        # (the exch closure has no tick arg -> run() stamps it each tick).
        self._edge_ts = _EDGE_TS
        self._edge_records: list = []
        self._edge_ref = None
        self._cur_tick = -1
        # WAVE_NO_CAT: ping-pong preallocated (full_k, full_v) reused across layers so
        # the exch slice-copies segments in instead of allocating a fresh cat output.
        # Keyed by L%2; content for layer L is fully consumed by L's attention before
        # L+2 reuses the slot (default-stream ordering), so depth 2 is safe.
        self._nocat_buf: list = [None, None]
        # WAVE_CAT_TS: rank0 per-(tick,layer) (start,end) CUDA event pairs around cat.
        self._cat_ts = _CAT_TS
        self._cat_records: list = []
        # WAVE_LAYER_PROF: per-layer attribution state (all active ranks).  _lp_cur is
        # the current tick's record (begin event + per-layer event dicts + post-attn
        # event sink filled by the backend); appended to _lp_records at tick end.
        self._layer_prof = _LAYER_PROF
        self._lp_records: list = []
        self._lp_cur = None
        # WAVE_AG_TS: rank0 per-tick list of (start,end) CUDA event pairs bracketing
        # each dist.all_gather call (sync-path exposed comm).  Layer index unknown to
        # _all_gather -> per-tick totals; per-layer = per-tick / Ln.
        self._ag_ts = _AG_TS
        self._ag_records: list = []
        # WAVE_MACHPROF: per-tick region CUDA events (all ranks) + CPU accumulators.
        self._machprof = _MACHPROF
        self._mp_records: list = []   # per-tick dicts of events + cpu accumulators
        self._mp_submit_cpu = 0.0     # per-tick: batch_isend_irecv / all_gather enqueue (ms)
        self._mp_alloc_cpu = 0.0      # per-tick: recv/all_gather output torch.empty (ms)
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
        # per-(gen,layer) attention buffer, so attention reads a contiguous prefix in
        # place -- no per-layer torch.cat.  Steady-state ticks only (all denoise ranks
        # active + clean window full); fill/drain ticks fall back to onesided (cat).
        # Reuses the ENTIRE onesided flag/IPC/copy-stream machinery via _init_onesided.
        self._paged = cfg.exchange_mode == "paged"
        self._pg_k = None              # consumer K attn buffer (GT, Ln, Wmax*T, H, D)
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
    # ------------------------------------------------------------- scheduling
    def _act_denoise(self, n: int, t: int) -> bool:
        c = t - n
        return (0 <= n < self.rf_step) and (0 <= c < self.num_blocks)

    def _act_store(self, t: int) -> bool:
        return 0 <= (t - self.store_rank) < self.num_blocks

    # --------------------------------------------------------- KV assembly
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
                (num_frames, pph, ppw), dim, num_heads, rope_dim_list,
                dtype=torch.float64 if self._rope_f64 else torch.float32,
                rope_theta=10000, start_frame=start_frame, device=device,
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

    def _all_gather(self, mine: torch.Tensor) -> list[torch.Tensor]:
        if _FP8KV:
            return self._all_gather_fp8(mine)
        if self._machprof:
            _ta = time.perf_counter()
            out = [
                torch.empty(self.KV_SHAPE, device=self.device, dtype=self.dtype)
                for _ in range(self.world)
            ]
            self._mp_alloc_cpu += (time.perf_counter() - _ta) * 1000.0
            _ts = time.perf_counter()
            dist.all_gather(out, mine.contiguous())
            self._mp_submit_cpu += (time.perf_counter() - _ts) * 1000.0
            return out
        out = [
            torch.empty(self.KV_SHAPE, device=self.device, dtype=self.dtype)
            for _ in range(self.world)
        ]
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
        scale = (mine.abs().amax() / _FP8KV_E4M3_MAX).clamp(min=1e-8).to(torch.float32)
        mine_u8 = (mine / scale).to(_FP8KV_DT).view(torch.uint8).contiguous()
        out_u8 = [torch.empty(self.KV_SHAPE, device=self.device, dtype=torch.uint8)
                  for _ in range(self.world)]
        dist.all_gather(out_u8, mine_u8)
        scales = [torch.empty(1, device=self.device, dtype=torch.float32)
                  for _ in range(self.world)]
        dist.all_gather(scales, scale.reshape(1))
        return [out_u8[n].view(_FP8KV_DT).to(self.dtype) * scales[n]
                for n in range(self.world)]

    def _dummy_allgathers(self) -> None:
        if _NOAG:
            return  # diagnostic: no per-layer collectives at all
        dummy = torch.zeros(self.KV_SHAPE, device=self.device, dtype=self.dtype)
        for i in range(self.Ln):
            if i == 0:
                _dbg(self.rank, "  dummy L0: all_gather start")
            self._all_gather(dummy)
            if i == 0:
                _dbg(self.rank, "  dummy L0: all_gather done")

    def _warmup(self, n_iters: int = 3) -> None:
        """Warm up DiT + VAE kernels/cuBLAS/NCCL at init so the timed wavefront
        runs at steady state.  Every rank runs the same number of forwards (each
        issuing the per-layer all_gathers) so the collective stream stays aligned;
        the store rank additionally warms the VAE decode (a local op)."""
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

        _dbg(self.rank, f"warmup: {n_iters} DiT forwards")
        for _ in range(n_iters):
            be.set_exchange_fn(warm_exch)
            be.forward_chunk(dummy, self.dsl_list[0], start_frame=0)
            be.set_exchange_fn(None)
        if self.rank == self.store_rank:
            _dbg(self.rank, "warmup: VAE decode")
            be.decode(dummy.clone())
        torch.cuda.synchronize()
        _dbg(self.rank, "warmup done")

        # --- one-off profile: isolate all_gather (comm) cost from forward compute ---
        if not _TICKPROF:
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
        if not _DEBUG:
            return
        r, world = self.rank, self.world
        _dbg(r, "selftest: all_gather ...")
        self._all_gather(torch.zeros(self.KV_SHAPE, device=self.device, dtype=self.dtype))
        _dbg(r, "selftest: broadcast(src=store) ...")
        buf = torch.full((4,), float(r), device=self.device)
        dist.broadcast(buf, src=self.store_rank)
        _dbg(r, f"selftest: broadcast got {buf[0].item():.0f} (expect {self.store_rank})")
        _dbg(r, "selftest: ring batch_isend_irecv ...")
        send_buf = torch.full((4,), float(r), device=self.device)
        recv_buf = torch.empty((4,), device=self.device)
        ops = [
            dist.P2POp(dist.irecv, recv_buf, (r - 1) % world),
            dist.P2POp(dist.isend, send_buf, (r + 1) % world),
        ]
        for rq in dist.batch_isend_irecv(ops):
            rq.wait()
        _dbg(r, f"selftest OK: recv from {(r-1)%world} = {recv_buf[0].item():.0f}")

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
                _dbg(self.rank, "  exch(denoise) L0: all_gather start")
            if _NOAG:
                return roped_key, value  # diagnostic: skip all_gather
            mine = torch.stack([roped_key[0], value[0]])
            gathered = self._all_gather(mine)
            if layer_idx == 0:
                _dbg(self.rank, "  exch(denoise) L0: all_gather done")
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
                if not _FULL_BCAST:
                    self._exch_store_slot[layer_idx] = gathered[store_rank].clone()
                if _BCAST_CHECK and self.rank == 0:
                    # consumer-side receipt of the store slot (cross-rank check).
                    self._bc_rank0_recv[layer_idx] = gathered[store_rank].clone()
            K = torch.cat(aK + liveK + inflK, dim=1)
            V = torch.cat(aV + liveV + inflV, dim=1)
            return K, V

        return exch

    def _make_store_exch(self, clean, store_kv, c):
        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            if _NOAG:
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
            if _BCAST_CHECK:
                # store's own all_gather contribution == gathered[store_rank] on
                # every consumer (NCCL all_gather copies byte-identically).
                self._bc_store_mine[layer_idx] = mine.clone()
            aK, aV = self._anchor_working(
                clean[layer_idx], c * self.npb, self.npb,
                post_patch_height, post_patch_width, dim, rope_num_heads,
            )
            K = torch.cat(aK + [roped_key], dim=1)
            V = torch.cat(aV + [value], dim=1)
            store_kv[layer_idx] = (roped_key, unroped_key, value)
            return K, V

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
        os.makedirs(_BCAST_CHECK_DIR, exist_ok=True)
        torch.save(
            {"kr0": kr[0].cpu(), "ku0": ku[0].cpu(), "vv0": vv[0].cpu(),
             "mine0": mine[0].cpu()},
            os.path.join(_BCAST_CHECK_DIR, f"store_t{t}.pt"),
        )

    # ------------------------------------------------- async overlap (P2P cascade)
    def _drain_p2p_sends(self) -> None:
        """Wait all deferred isends (+ any outstanding prefetch recv works) from this
        tick, then release buffers.  Called after forward_chunk returns and BEFORE any
        WORLD collective, so p2p_pg is quiescent (no cross-PG interleave)."""
        for reqs, _buf in self._p2p_send_pending:
            for rq in reqs:
                rq.wait()
        self._p2p_send_pending = []
        if self._use_prefetch:
            for works in self._pf_works:
                if works:
                    for w in works:
                        try:
                            w.wait()
                        except Exception:
                            pass
            self._pf_slots = [None, None]
            self._pf_works = [None, None]

    def _post_recv(self, slot, recv_srcs):
        """Post one layer's irecvs into ping-pong `slot` on the DEFAULT stream (torch
        runs them on p2p_pg's internal nccl stream regardless).  Returns (bufs, works)
        or (None, None) if no sources."""
        if not recv_srcs:
            return None, None
        bufs = {}
        if self._edge_ts:
            # DIAGNOSTIC ONLY (WAVE_EDGE_TS): post each source's irecv as its OWN
            # batched group so it returns a distinct Work -> the exch closure can
            # wait + time each source separately.  The production path below coalesces
            # all srcs into ONE batch_isend_irecv, which returns a SINGLE Work for the
            # whole batch -> only one wait event -> per-edge arrival skew collapses to
            # 0.  P2P send/recv still matches per-peer, so correctness is preserved;
            # this only fans the recv into N groups (minor timing perturbation, OK for
            # a measurement run).  works[] order == recv_srcs order (per-src labeling).
            works = []
            for src in recv_srcs:
                b = torch.empty(self.KV_SHAPE, device=self.device, dtype=self.dtype)
                bufs[src] = b
                works.extend(dist.batch_isend_irecv(
                    [dist.P2POp(dist.irecv, b, src, self.p2p_pg)]))
            self._pf_slots[slot] = bufs
            self._pf_works[slot] = works
            return bufs, works
        ops = []
        for src in recv_srcs:
            if self._machprof:
                _ta = time.perf_counter()
                b = torch.empty(self.KV_SHAPE, device=self.device, dtype=self.dtype)
                self._mp_alloc_cpu += (time.perf_counter() - _ta) * 1000.0
            else:
                b = torch.empty(self.KV_SHAPE, device=self.device, dtype=self.dtype)
            bufs[src] = b
            ops.append(dist.P2POp(dist.irecv, b, src, self.p2p_pg))
        if self._machprof:
            _ts = time.perf_counter()
            works = dist.batch_isend_irecv(ops)
            self._mp_submit_cpu += (time.perf_counter() - _ts) * 1000.0
        else:
            works = dist.batch_isend_irecv(ops)
        self._pf_slots[slot] = bufs
        self._pf_works[slot] = works
        return bufs, works

    def _bootstrap_recv(self, recv_srcs) -> None:
        """Post layer-0's irecvs before the forward (prefetch bootstrap)."""
        self._pf_slots = [None, None]
        self._pf_works = [None, None]
        self._post_recv(0, recv_srcs)

    def _send_mine(self, mine, dsts) -> None:
        """isend my layer KV to `dsts` on the DEFAULT stream (torch auto syncStream +
        recordStream keeps `mine` alive across the async send); stash for drain."""
        if not dsts:
            return
        if self._machprof:
            _ts = time.perf_counter()
            reqs = dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, mine, d, self.p2p_pg) for d in dsts])
            self._mp_submit_cpu += (time.perf_counter() - _ts) * 1000.0
        else:
            reqs = dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, mine, d, self.p2p_pg) for d in dsts])
        self._p2p_send_pending.append((reqs, mine))

    def _make_denoise_exch_prefetch(self, clean, act_ranks, tick_csf, tick_qf,
                                    store_active):
        """Real compute/comm overlap (prefetch depth 1): recv for layer 0 is posted
        by _bootstrap_recv; each layer prefetches L+1's recv, sends its own KV, then
        gates layer L via work.wait() (non-blocking on CPU -> transfer overlaps the
        prior layer's compute).  Same assembled context (bitwise) as causal-sync."""
        store_rank, r = self.store_rank, self.rank
        recv_srcs = [n for n in act_ranks if n > r] + (
            [store_rank] if store_active else [])
        # nearest-first: send to r-1 (the most urgent consumer, next on the
        # cascade) before r-2..0 so the tightest dependency resolves first.
        # WAVE_ORIG_SEND_ORDER=1 restores the pre-fix farthest-first (ascending).
        if _ORIG_SEND_ORDER:
            send_dsts = [m for m in act_ranks if m < r]
        else:
            send_dsts = sorted([m for m in act_ranks if m < r], reverse=True)
        infl_ranks = [n for n in act_ranks if n >= r]
        Ln = self.Ln

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            L = layer_idx
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            # ORDER MATTERS: send layer-L FIRST, then prefetch layer L+1's recv.
            # (posting recv_(L+1) before send_L would block p2p_pg's internal stream
            # on the not-yet-available recv ahead of the send -> cascade can't resolve
            # -> deadlock at the first full-pipeline tick.)
            self._send_mine(mine, send_dsts)
            if recv_srcs and L + 1 < Ln:
                self._post_recv((L + 1) % 2, recv_srcs)
            if _lp is not None:
                _e_pub = torch.cuda.Event(enable_timing=True); _e_pub.record()
            # WAVE_EDGE_TS (rank0): mark exch entry -- prior layer's compute is done
            # in stream order, so this is "attention would start here if no recv wait".
            _ets = self._edge_ts and r == 0
            _ev_enter = _ev_recv = None
            if _ets:
                _ev_enter = torch.cuda.Event(enable_timing=True)
                _ev_enter.record()
                _ev_recv = []
            # gate: wait layer L's recv work (enqueues cudaStreamWaitEvent on default
            # stream; non-blocking on CPU -> overlap preserved)
            works = self._pf_works[L % 2]
            if works:
                for _i, w in enumerate(works):
                    w.wait()
                    if _ets:
                        # event after each source's wait (recv_srcs order); serialized
                        # on the stream, so it captures cumulative "done after waiting
                        # sources 0..i" -- the last one == all recvs in for this layer.
                        _e = torch.cuda.Event(enable_timing=True)
                        _e.record()
                        _ev_recv.append((recv_srcs[_i], _e))
            if _lp is not None:
                _e_wt = torch.cuda.Event(enable_timing=True); _e_wt.record()
            bufs = self._pf_slots[L % 2] or {}
            aK, aV = self._anchor_working(
                clean[L], tick_csf, tick_qf,
                post_patch_height, post_patch_width, dim, rope_num_heads)
            liveK, liveV = [], []
            if store_active:
                liveK = [bufs[store_rank][0].unsqueeze(0)]
                liveV = [bufs[store_rank][1].unsqueeze(0)]
                if not _FULL_BCAST:
                    # clone: bufs is a ping-pong slot reused by layer L+2's recv.
                    self._exch_store_slot[L] = bufs[store_rank].clone()
            inflK, inflV = [], []
            for n in infl_ranks:
                src = mine if n == r else bufs[n]
                inflK.append(src[0].unsqueeze(0))
                inflV.append(src[1].unsqueeze(0))
            segK = aK + liveK + inflK
            segV = aV + liveV + inflV
            _cts = self._cat_ts and r == 0 and self._cur_tick >= self.world - 1
            _ev_c0 = None
            if _cts:
                _ev_c0 = torch.cuda.Event(enable_timing=True)
                _ev_c0.record()
            if _NO_CAT:
                # slice-assign into a preallocated ping-pong buffer (no per-layer
                # cat allocation); byte-identical layout to torch.cat(dim=1).
                total = sum(s.shape[1] for s in segK)
                buf = self._nocat_buf[L % 2]
                if buf is None or buf[0].shape[1] != total:
                    Hh, Dd = segK[0].shape[2], segK[0].shape[3]
                    full_k = torch.empty((1, total, Hh, Dd),
                                         device=self.device, dtype=self.dtype)
                    full_v = torch.empty((1, total, Hh, Dd),
                                         device=self.device, dtype=self.dtype)
                    self._nocat_buf[L % 2] = (full_k, full_v)
                else:
                    full_k, full_v = buf
                off = 0
                for sk, sv in zip(segK, segV):
                    n = sk.shape[1]
                    full_k[:, off:off + n].copy_(sk)
                    full_v[:, off:off + n].copy_(sv)
                    off += n
                K, V = full_k, full_v
            else:
                K = torch.cat(segK, dim=1)
                V = torch.cat(segV, dim=1)
            if _lp is not None:
                _e_ct = torch.cuda.Event(enable_timing=True); _e_ct.record()
                _lp["layers"].append(dict(
                    layer=L, enter=_e_en, pub=_e_pub, wait=_e_wt, cat=_e_ct))
            if _cts:
                _ev_c1 = torch.cuda.Event(enable_timing=True)
                _ev_c1.record()
                self._cat_records.append(
                    dict(tick=self._cur_tick, layer=L, start=_ev_c0, end=_ev_c1))
            if _ets:
                # event after the K/V cat: attention is about to start on this layer.
                _ev_attn = torch.cuda.Event(enable_timing=True)
                _ev_attn.record()
                self._edge_records.append(dict(
                    tick=self._cur_tick, layer=L, srcs=list(recv_srcs),
                    enter=_ev_enter, recv=_ev_recv, attn=_ev_attn))
            return K, V

        return exch

    def _make_store_exch_prefetch(self, clean, store_kv, c, act_ranks):
        """Store variant for prefetch overlap: only sends its clean KV (default
        stream) to every active denoise rank; no recv.  Local context assembly."""
        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            self._send_mine(mine, sorted(act_ranks, reverse=True))  # EDF
            if _lp is not None:
                _e_pub = torch.cuda.Event(enable_timing=True); _e_pub.record()
            aK, aV = self._anchor_working(
                clean[layer_idx], c * self.npb, self.npb,
                post_patch_height, post_patch_width, dim, rope_num_heads)
            K = torch.cat(aK + [roped_key], dim=1)
            V = torch.cat(aV + [value], dim=1)
            store_kv[layer_idx] = (roped_key, unroped_key, value)
            if _lp is not None:
                _e_ct = torch.cuda.Event(enable_timing=True); _e_ct.record()
                _lp["layers"].append(dict(
                    layer=layer_idx, enter=_e_en, pub=_e_pub, wait=_e_pub, cat=_e_ct))
            return K, V

        return exch

    def _make_denoise_exch_overlap(self, clean, act_ranks, tick_csf, tick_qf,
                                   store_active):
        """Causal denoise exchange via async P2P cascade on self.p2p_pg.
        Same assembled context (hence same numerics) as _make_denoise_exch with
        kv_context=causal, but: recv layer-L KV from faster/higher ranks (n>r) +
        store (wait these), isend own KV to slower/lower ranks (m<r, DEFERRED)."""
        store_rank = self.store_rank
        r = self.rank
        # sets derived purely from act_ranks/store_active -> identical on all ranks
        recv_srcs = [n for n in act_ranks if n > r] + ([store_rank] if store_active else [])
        send_dsts = [m for m in act_ranks if m < r]
        infl_ranks = [n for n in act_ranks if n >= r]   # assembly order (incl. self)
        pg = self.p2p_pg

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            # fresh contiguous send buffer each layer (must outlive the layer)
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            recv_bufs = {
                src: torch.empty(self.KV_SHAPE, device=self.device, dtype=self.dtype)
                for src in recv_srcs
            }
            # post recvs (needed to proceed) + sends (deferred)
            if recv_srcs:
                recv_reqs = dist.batch_isend_irecv(
                    [dist.P2POp(dist.irecv, recv_bufs[s], s, pg) for s in recv_srcs]
                )
            else:
                recv_reqs = []
            if send_dsts:
                send_reqs = dist.batch_isend_irecv(
                    [dist.P2POp(dist.isend, mine, d, pg) for d in send_dsts]
                )
                self._p2p_send_pending.append((send_reqs, mine))
            for rq in recv_reqs:
                rq.wait()  # wait ONLY recvs; sends stay in flight -> hidden
            # assemble EXACTLY as causal-sync: aK + liveK + inflK (n>=r)
            aK, aV = self._anchor_working(
                clean[layer_idx], tick_csf, tick_qf,
                post_patch_height, post_patch_width, dim, rope_num_heads,
            )
            liveK, liveV = [], []
            if store_active:
                liveK = [recv_bufs[store_rank][0].unsqueeze(0)]
                liveV = [recv_bufs[store_rank][1].unsqueeze(0)]
                if not _FULL_BCAST:
                    self._exch_store_slot[layer_idx] = recv_bufs[store_rank].clone()
            inflK, inflV = [], []
            for n in infl_ranks:
                src = mine if n == r else recv_bufs[n]
                inflK.append(src[0].unsqueeze(0))
                inflV.append(src[1].unsqueeze(0))
            K = torch.cat(aK + liveK + inflK, dim=1)
            V = torch.cat(aV + liveV + inflV, dim=1)
            return K, V

        return exch

    def _make_store_exch_overlap(self, clean, store_kv, c, act_ranks):
        """Store exchange via async P2P: isend own clean KV to every active denoise
        rank (DEFERRED), no recv.  Local context assembly is identical to sync."""
        send_dsts = sorted(act_ranks, reverse=True)  # EDF: immediate consumer first
        pg = self.p2p_pg

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            if send_dsts:
                send_reqs = dist.batch_isend_irecv(
                    [dist.P2POp(dist.isend, mine, d, pg) for d in send_dsts]
                )
                self._p2p_send_pending.append((send_reqs, mine))
            aK, aV = self._anchor_working(
                clean[layer_idx], c * self.npb, self.npb,
                post_patch_height, post_patch_width, dim, rope_num_heads,
            )
            K = torch.cat(aK + [roped_key], dim=1)
            V = torch.cat(aV + [value], dim=1)
            store_kv[layer_idx] = (roped_key, unroped_key, value)
            return K, V

        return exch

    # ------------------------------------------------- one-sided (Gate C-final-1)
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
        from wave_rt import p2p_mem as p2p
        self._p2p = p2p
        dev = self.device
        c = self.rank
        rf_step = self.rf_step
        store_rank = self.store_rank
        world = self.world
        GT = _OS_GT
        Ln = self.Ln
        self._os_gt = GT
        slot_numel = 1
        for s in self.KV_SHAPE:
            slot_numel *= s
        self._os_fp8 = _FP8KV_OS
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
                _dbg(c, f"onesided(paged): flags only, n_in={n_in} srcs={in_srcs}")
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
                _dbg(c, f"onesided: slab {need / 2**30:.2f}GiB n_in={n_in} srcs={in_srcs}")
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
            fl_base = p2p.ipc_open_handle(gi["fl_h"])
            if self._paged:
                # paged: only the flag slab is shared here; KV goes to _pg_dst
                self._os_dst[m] = dict(
                    kv_slab=0, fl_slab=fl_base + gi["fl_off"],
                    i=my_i, GT=gi["GT"], Ln=gi["Ln"], slot_bytes=gi["slot_bytes"])
            else:
                kv_base = p2p.ipc_open_handle(gi["kv_h"])  # self ctx, peer-mapped (DtoD)
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
        ev = torch.cuda.Event()
        ev.record()                 # on the current (default/compute) stream
        scopy.wait_event(ev)        # copy waits for layer-L KV to materialize
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
        if _ORIG_SEND_ORDER:
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
                if not _FULL_BCAST:
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
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            self._publish_mine(mine, send_dsts, layer_idx, gen, tick)
            if _lp is not None:
                _e_pub = torch.cuda.Event(enable_timing=True); _e_pub.record()
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

        return exch

    # -------------------------------------------------- paged (in-place KV buffer)
    def _init_paged(self) -> None:
        """Paged = onesided transport + a per-(gen,layer) contiguous attention KV
        buffer that producers DMA straight into (fixed slots), so the consumer reads
        a contiguous prefix in place -- no per-layer torch.cat.  Reuses the onesided
        flag slab / src_index / copy stream / _os_kv fallback (via _init_onesided);
        adds separate contiguous K and V buffers (attention takes k,v separately) and
        IPC-exchanges them so every producer can write its slot.

        Slot order per consumer r (identical to the onesided cat order, so the
        attention input is byte-identical): [anchor, working..., live(store), self,
        in-flight(r+1..)].  At steady state anchor+working == _pg_naw is constant, so
        every slot offset is a compile-time-constant function of (producer, consumer)
        and both sides agree without a handshake."""
        self._init_onesided()   # flags, _os_src_index, _os_dst(fl_slab,i), scopy, _os_kv
        from wave_rt import p2p_mem as p2p
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
            need = 2 * GT * Ln * Wmax * T * H * D * elem
            free_b, _tot = torch.cuda.mem_get_info()
            if need > free_b * 0.9:
                raise RuntimeError(
                    f"[paged r{c}] KV attn buffers {need / 2**30:.1f} GiB exceed 90% of "
                    f"free {free_b / 2**30:.1f} GiB (GT={GT} Ln={Ln} Wmax={Wmax})")
            self._pg_k = torch.zeros((GT, Ln, Wmax * T, H, D), device=dev, dtype=self.dtype)
            self._pg_v = torch.zeros((GT, Ln, Wmax * T, H, D), device=dev, dtype=self.dtype)
            k_h, k_off = p2p.ipc_export(self._pg_k)
            v_h, v_off = p2p.ipc_export(self._pg_v)
            _dbg(c, f"paged: kv buffers {need / 2**30:.2f}GiB naw={self._pg_naw} Wmax={Wmax}")
        else:
            self._pg_k = self._pg_v = None
            k_h = k_off = v_h = v_off = None

        my_info = dict(rank=c, k_h=k_h, k_off=k_off, v_h=v_h, v_off=v_off,
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
        for m in my_dsts:
            gi = world_info[m]
            if gi["k_h"] is None:
                continue
            kb = p2p.ipc_open_handle(gi["k_h"])
            vb = p2p.ipc_open_handle(gi["v_h"])
            od = self._os_dst[m]   # reuse onesided flag slab + my row index
            self._pg_dst[m] = dict(
                k_slab=kb + gi["k_off"], v_slab=vb + gi["v_off"],
                fl_slab=od["fl_slab"], i=od["i"], GT=gi["GT"], Ln=gi["Ln"],
                Wmax=gi["Wmax"], THD=gi["THD"], elem=gi["elem"])

        # --- fault-in the peer mappings once (materialize lazy IPC pages) ---
        if self._pg_dst:
            warm = torch.zeros((T, H, D), device=dev, dtype=self.dtype)
            sh = self._os_scopy.cuda_stream
            for m, d in self._pg_dst.items():
                p2p.memcpy_dtod_async(d["k_slab"], warm.data_ptr(),
                                      warm.numel() * elem, sh)
                p2p.memcpy_dtod_async(d["v_slab"], warm.data_ptr(),
                                      warm.numel() * elem, sh)
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
        ev = torch.cuda.Event()
        ev.record()
        scopy.wait_event(ev)         # copy waits for layer-L KV to materialize
        sh = scopy.cuda_stream
        kb = k.numel() * k.element_size()
        vb = v.numel() * v.element_size()
        kp = k.data_ptr()
        vp = v.data_ptr()
        for (m, slot) in dst_slots:
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
            Kb = self._pg_k[gen, L]
            Vb = self._pg_v[gen, L]
            # local placement: anchor + working into [0, naw*T)
            aK, aV = self._anchor_working(
                clean[L], tick_csf, tick_qf,
                post_patch_height, post_patch_width, dim, rope_num_heads)
            off = 0
            for ak, av in zip(aK, aV):
                Kb[off:off + T].copy_(ak[0]); Vb[off:off + T].copy_(av[0])
                off += T
            if off != naw * T:
                raise RuntimeError(
                    f"[paged r{r}] anchor+working = {off // T} chunks != naw={naw} "
                    f"(t={tick} clean_len={clean_len})")
            # self (in-flight r) into its fixed slot; live (store) slot is DMA-filled
            Kb[self_off:self_off + T].copy_(k)
            Vb[self_off:self_off + T].copy_(v)
            # gate on every remote src (store live + in-flight n>r); data already placed
            for src in recv_srcs:
                self._wait_paged_flag(src, L, gen, tick)
            if _lp is not None:
                _e_wt = torch.cuda.Event(enable_timing=True); _e_wt.record()
            # promote store live slot into clean[] at tick end (same as onesided)
            if store_active and not _FULL_BCAST:
                self._exch_store_slot[L] = torch.stack(
                    [Kb[live_off:live_off + T], Vb[live_off:live_off + T]]).clone()
            K = Kb[:active_len].unsqueeze(0)
            V = Vb[:active_len].unsqueeze(0)
            if _lp is not None:
                _e_ct = torch.cuda.Event(enable_timing=True); _e_ct.record()
                _lp["layers"].append(dict(
                    layer=L, enter=_e_en, pub=_e_pub, wait=_e_wt, cat=_e_ct))
            return K, V

        return exch

    def _make_store_exch_paged(self, clean, store_kv, c, act_ranks):
        """Store variant for paged: publish live KV into each active denoise
        consumer's live slot (slot index = consumers' naw); assemble its own (small,
        local) context by cat (store is not the bottleneck rank).  Runs on all ticks."""
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
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            k = roped_key[0].contiguous()
            v = value[0].contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            self._publish_paged(k, v, send_slots, layer_idx, gen, tick)
            if _lp is not None:
                _e_pub = torch.cuda.Event(enable_timing=True); _e_pub.record()
            aK, aV = self._anchor_working(
                clean[layer_idx], c * self.npb, self.npb,
                post_patch_height, post_patch_width, dim, rope_num_heads)
            K = torch.cat(aK + [roped_key], dim=1)
            V = torch.cat(aV + [value], dim=1)
            store_kv[layer_idx] = (roped_key, unroped_key, value)
            if _lp is not None:
                _e_ct = torch.cuda.Event(enable_timing=True); _e_ct.record()
                _lp["layers"].append(dict(
                    layer=layer_idx, enter=_e_en, pub=_e_pub, wait=_e_pub, cat=_e_ct))
            return K, V

        return exch


    # ------------------------------------------- staggered (prefetched onesided)
    def _stagger_kick(self, recv_srcs, layer: int, gen: int, tick: int) -> None:
        """Prefetch the inbound acquire for `layer`: enqueue on the dedicated
        prefetch stream a device-side wait (>= tick) on every inbound src's
        generation flag for (gen, layer), then record a ready event into
        self._st_ev[layer].  The consumer's compute stream later wait_event()s
        this before reading slab[src, gen, layer] -- so the producer's transfer
        overlaps the consumer's compute of the preceding `stagger_lead` layers.
        No-op past the last layer or when this rank consumes nothing."""
        if not recv_srcs or layer >= self.Ln:
            return
        sh = self._st_prefetch.cuda_stream
        for src in recv_srcs:
            i = self._os_src_index[src]
            self._p2p.stream_wait_u32(
                sh, self._os_flags[i, gen, layer].data_ptr(), tick)
        ev = torch.cuda.Event()
        ev.record(self._st_prefetch)
        self._st_ev[layer] = ev

    def _stagger_bootstrap(self, recv_srcs, gen: int, tick: int) -> None:
        """Kick the first `stagger_lead` layers' inbound acquires before the
        forward (reset the per-tick event map first)."""
        self._st_ev = {}
        for L in range(min(self._stagger_lead, self.Ln)):
            self._stagger_kick(recv_srcs, L, gen, tick)

    def _make_denoise_exch_staggered(self, clean, act_ranks, tick_csf, tick_qf,
                                     store_active):
        """Denoise exchange: one-sided publication (identical to onesided) with a
        PREFETCHED inbound acquire.  My layer-L KV is published into every slower
        rank's slab the instant it is ready; the inbound layer-L acquire was
        already kicked `stagger_lead` layers earlier (on the prefetch stream), so
        this layer only wait_event()s a (usually already-complete) marker instead
        of stalling on the producer's same-layer flag.  Kicks the L+lead acquire
        so it overlaps this layer's compute.  Assembled context (aK + liveK +
        inflK, cat) and numerics are byte-identical to causal-sync / onesided."""
        store_rank, r = self.store_rank, self.rank
        tick = self._cur_tick
        gen = tick % self._os_gt
        recv_srcs = [n for n in act_ranks if n > r] + (
            [store_rank] if store_active else [])
        # nearest-first fan-out (see _make_denoise_exch_prefetch); WAVE_ORIG_SEND_ORDER
        # restores farthest-first.
        if _ORIG_SEND_ORDER:
            send_dsts = [m for m in act_ranks if m < r]
        else:
            send_dsts = sorted([m for m in act_ranks if m < r], reverse=True)
        infl_ranks = [n for n in act_ranks if n >= r]
        lead = self._stagger_lead

        def exch(*, layer_idx, roped_query, roped_key, unroped_key, value,
                 current_start, tokens_per_frame, post_patch_height,
                 post_patch_width, dim, rope_num_heads, num_frames_per_block):
            L = layer_idx
            mine = torch.stack([roped_key[0], value[0]]).contiguous()
            _lp = self._lp_cur
            if _lp is not None:
                _e_en = torch.cuda.Event(enable_timing=True); _e_en.record()
            # producer role: publish my layer-L KV into every slower rank's slab
            self._publish_mine(mine, send_dsts, L, gen, tick)
            if _lp is not None:
                _e_pub = torch.cuda.Event(enable_timing=True); _e_pub.record()
            # consumer role: gate on the PREFETCHED acquire for layer L, then kick
            # the acquire for L+lead so it overlaps this layer's compute.
            # residual_wait: CUDA-event pair AROUND ONLY the gate (no kick, no sync)
            # -> the pure exposed wait on the compute stream.  If the prefetch has
            # already landed (perfect hiding) the stream never stalls and w0->w1 ~ 0.
            _e_w0 = _e_w1 = None
            if recv_srcs:
                if _lp is not None:
                    _e_w0 = torch.cuda.Event(enable_timing=True); _e_w0.record()
                ev = self._st_ev.pop(L, None)
                if ev is not None:
                    torch.cuda.current_stream().wait_event(ev)
                else:
                    # safety fallback (should not happen: L is always kicked by the
                    # bootstrap for L<lead or by exch(L-lead) otherwise) -- gate
                    # directly on the compute stream so the read can't race.
                    sh = torch.cuda.current_stream().cuda_stream
                    for src in recv_srcs:
                        i = self._os_src_index[src]
                        self._p2p.stream_wait_u32(
                            sh, self._os_flags[i, gen, L].data_ptr(), tick)
                if _lp is not None:
                    # record BEFORE the L+lead kick so residual excludes kick enqueue.
                    _e_w1 = torch.cuda.Event(enable_timing=True); _e_w1.record()
                self._stagger_kick(recv_srcs, L + lead, gen, tick)
            if _lp is not None:
                _e_wt = torch.cuda.Event(enable_timing=True); _e_wt.record()
            remote = {src: self._os_kv[self._os_src_index[src], gen, L]
                      for src in recv_srcs}
            aK, aV = self._anchor_working(
                clean[L], tick_csf, tick_qf,
                post_patch_height, post_patch_width, dim, rope_num_heads)
            liveK, liveV = [], []
            if store_active:
                slot = remote[store_rank]
                liveK = [slot[0].unsqueeze(0)]
                liveV = [slot[1].unsqueeze(0)]
                if not _FULL_BCAST:
                    # clone out of the gen ring before its next-gen reuse (promoted
                    # into clean[] at tick end -- identical to the onesided path).
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
                    layer=L, enter=_e_en, pub=_e_pub, wait=_e_wt, cat=_e_ct,
                    w0=_e_w0, w1=_e_w1))
            return K, V

        return exch

    # ------------------------------------------- neighbor-accumulating relay
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
        from wave_rt import p2p_mem as p2p
        self._p2p = p2p
        dev = self.device
        c = self.rank
        rf_step = self.rf_step
        store_rank = self.store_rank
        world = self.world
        GT = _OS_GT
        Ln = self.Ln
        self._os_gt = GT
        self._rl_world = world
        slot_numel = 1
        for s in self.KV_SHAPE:
            slot_numel *= s
        self._os_fp8 = _FP8KV_OS
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
            _dbg(c, f"relay: recv slab {need / 2**30:.2f}GiB world={world}")
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
            kv_base = p2p.ipc_open_handle(gi["kv_h"])   # peer-mapped (DtoD)
            fl_base = p2p.ipc_open_handle(gi["fl_h"])
            self._rl_down = dict(
                rank=down,
                kv_slab=kv_base + gi["kv_off"],
                fl_slab=fl_base + gi["fl_off"])
            _dbg(c, f"relay: downstream={down}")

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
        mine_send.record_stream(scopy)
        self._os_pending.append(mine_send)

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
        mine_send.record_stream(scopy)
        self._os_pending.append(mine_send)

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
                if not _FULL_BCAST:
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

    # -------------------------------------------------------------- re-noise
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
            flat, nz,
            float(next_t) * torch.ones([b * t_], device=self.device, dtype=torch.long),
        ).unflatten(0, (b, t_))
        return renoised.permute(0, 2, 1, 3, 4).contiguous()

    # ------------------------------------------------------- edge-ts diagnostic
    def _dump_ag_ts(self) -> None:
        """WAVE_AG_TS (rank0): synchronize once, read every all_gather CUDA event pair
        as elapsed ms, aggregate per steady tick and print per-layer/per-tick totals.
        Measurement only -- the collectives ran exactly as in production."""
        torch.cuda.synchronize()
        from collections import defaultdict
        per_tick = defaultdict(list)
        for tick, e0, e1 in self._ag_records:
            per_tick[tick].append(e0.elapsed_time(e1))  # ms
        if not per_tick:
            print("[ag_ts] no all_gather records (sync path only)", flush=True)
            return
        ticks = sorted(per_tick)
        all_ms = [ms for t in ticks for ms in per_tick[t]]
        n_calls = len(all_ms)
        per_layer = sum(all_ms) / n_calls
        # per-tick total = mean over steady ticks of that tick's summed all_gather ms
        tick_totals = [sum(per_tick[t]) for t in ticks]
        per_tick_mean = sum(tick_totals) / len(tick_totals)
        print(f"\n[ag_ts] rank0 sync all_gather CUDA-event timing over steady ticks "
              f"{ticks}", flush=True)
        for t in ticks:
            v = per_tick[t]
            print(f"[ag_ts]   t={t}: {len(v)} calls, "
                  f"sum={sum(v):.1f}ms, mean/call={sum(v)/len(v):.4f}ms", flush=True)
        print(f"[ag_ts] SUMMARY: {n_calls} calls over {len(ticks)} steady ticks; "
              f"per-layer all_gather mean={per_layer:.4f} ms/call; "
              f"per-tick total mean={per_tick_mean:.1f} ms/tick "
              f"(Ln={self.Ln})", flush=True)

    # ------------------------------------------------------- machprof diagnostic
    def _dump_machprof(self) -> None:
        """WAVE_MACHPROF (all ranks): one synchronize, then read the per-tick region
        CUDA events as ms and combine with the CPU accumulators.  Regions:
          fwd   = e_start->e_fwd   (forward compute + exposed work.wait + P2P drain)
          promo = e_fwd->e_mig0    (clean-KV promote / anchor broadcast)
          mig   = e_mig0->e_mig1   (WORLD migration all_gather)
          total = e_start->e_mig1  (== diffusion/num_ticks at steady state)
        CPU accumulators (perf_counter, inside fwd): submit (isend/irecv or all_gather
        enqueue), alloc (recv/all_gather output torch.empty), drain (_drain_p2p_sends
        block).  Prints per-tick rows for this rank + a steady-state mean."""
        import json as _json
        torch.cuda.synchronize()
        rows = []
        for r in self._mp_records:
            fwd = r["e_start"].elapsed_time(r["e_fwd"])
            promo = r["e_fwd"].elapsed_time(r["e_mig0"])
            mig = r["e_mig0"].elapsed_time(r["e_mig1"])
            total = r["e_start"].elapsed_time(r["e_mig1"])
            rows.append(dict(
                tick=r["tick"], rank=r["rank"], is_dn=r["is_dn"], is_st=r["is_st"],
                fwd=fwd, promo=promo, mig=mig, total=total,
                submit=r["submit_cpu"], alloc=r["alloc_cpu"],
                drain=r["drain_cpu"], mig_submit=r["mig_submit_cpu"]))
        os.makedirs(_MACHPROF_DIR, exist_ok=True)
        with open(os.path.join(_MACHPROF_DIR, f"mp_r{self.rank}.json"), "w") as f:
            _json.dump(dict(rank=self.rank, world=self.world,
                            store_rank=self.store_rank, num_layers=self.Ln,
                            num_ticks=self.num_ticks, mode=self.cfg.exchange_mode,
                            kv=self.cfg.kv_context, rows=rows), f)
        # steady = active ticks past the fill ramp, before the drain tail.
        lo, hi = self.world - 1, self.num_blocks - 1
        steady = [x for x in rows if lo <= x["tick"] <= hi and (x["is_dn"] or x["is_st"])]
        tag = f"r{self.rank}"
        print(f"\n[machprof {tag}] mode={self.cfg.exchange_mode} "
              f"kv={self.cfg.kv_context} steady ticks [{lo},{hi}] "
              f"({len(steady)} active rows); regions=GPU-event ms, "
              f"submit/alloc/drain/mig_sub=CPU ms", flush=True)
        hdr = (f"{'tick':>4} {'role':>4} | {'total':>7} {'fwd':>7} {'promo':>6} "
               f"{'mig':>6} | {'submit':>6} {'alloc':>6} {'drain':>6} {'mig_sub':>7}")
        print(f"[machprof {tag}] " + hdr, flush=True)
        for x in steady:
            role = "dn" if x["is_dn"] else ("st" if x["is_st"] else "--")
            print(f"[machprof {tag}] {x['tick']:>4} {role:>4} | {x['total']:>7.1f} "
                  f"{x['fwd']:>7.1f} {x['promo']:>6.1f} {x['mig']:>6.1f} | "
                  f"{x['submit']:>6.2f} {x['alloc']:>6.2f} {x['drain']:>6.2f} "
                  f"{x['mig_submit']:>7.2f}", flush=True)
        if steady:
            n = len(steady)
            def _m(k): return sum(x[k] for x in steady) / n
            print(f"[machprof {tag}] MEAN over {n} steady ticks: "
                  f"total={_m('total'):.1f} fwd={_m('fwd'):.1f} "
                  f"promo={_m('promo'):.1f} mig={_m('mig'):.1f} || "
                  f"submit={_m('submit'):.2f} alloc={_m('alloc'):.2f} "
                  f"drain={_m('drain'):.2f} mig_submit={_m('mig_submit'):.2f}",
                  flush=True)

    # ------------------------------------------------------- layer-prof diagnostic
    def _dump_layer_prof(self) -> None:
        """WAVE_LAYER_PROF: resolve the per-layer CUDA events captured by the exch
        closures into pub/wait/cat/total ms, write raw JSON per rank, and (rank0) a
        steady-state per-layer mean.  Buckets:
          pub  = enter->pub  (publish latency)
          wait = pub->wait   (wait_peer latency; store has none -> 0)
          cat  = wait->cat   (context assembly latency)
          total= enter->cat  (per-layer total)"""
        import json, os
        out_dir = _LAYER_PROF_DIR
        os.makedirs(out_dir, exist_ok=True)
        # 1) raw JSON per rank
        torch.cuda.synchronize()
        raw = []
        for rec in self._lp_records:
            layers_out = []
            for ly in rec["layers"]:
                ly["enter"].synchronize()
                d = dict(
                    layer=ly["layer"],
                    pub_ms=ly["enter"].elapsed_time(ly["pub"]),
                    wait_ms=ly["pub"].elapsed_time(ly["wait"]),
                    cat_ms=ly["wait"].elapsed_time(ly["cat"]),
                    total_ms=ly["enter"].elapsed_time(ly["cat"]),
                )
                # staggered: pure residual gate latency (w0->w1, excludes L+lead kick)
                if ly.get("w0") is not None and ly.get("w1") is not None:
                    d["residual_wait_ms"] = ly["w0"].elapsed_time(ly["w1"])
                layers_out.append(d)
            raw.append(dict(tick=rec["tick"], rank=rec["rank"],
                            is_dn=rec["is_dn"], is_st=rec["is_st"],
                            layers=layers_out))
        path = os.path.join(out_dir, f"layer_prof_r{self.rank}.json")
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)
        # 2) steady-state mean per layer (ticks >= world, exclude fill ramp).
        # All ranks emit their own summary so critical-rank migration is visible.
        from collections import defaultdict
        layer_sums = defaultdict(
            lambda: {"pub_ms": 0, "wait_ms": 0, "cat_ms": 0, "total_ms": 0, "n": 0,
                     "residual_wait_ms": 0.0, "res_n": 0})
        ss_recs = [r for r in raw if r["tick"] >= self.world]
        for r in ss_recs:
            for ly in r["layers"]:
                k = layer_sums[ly["layer"]]
                k["pub_ms"] += ly["pub_ms"]
                k["wait_ms"] += ly["wait_ms"]
                k["cat_ms"] += ly["cat_ms"]
                k["total_ms"] += ly["total_ms"]
                k["n"] += 1
                if "residual_wait_ms" in ly:
                    k["residual_wait_ms"] += ly["residual_wait_ms"]
                    k["res_n"] += 1
        summary = []
        sum_total = 0.0
        for L in sorted(layer_sums):
            k = layer_sums[L]
            n = max(k["n"], 1)
            mean_total = k["total_ms"] / n
            sum_total += mean_total
            row = dict(layer=L,
                       pub_ms=round(k["pub_ms"] / n, 4),
                       wait_ms=round(k["wait_ms"] / n, 4),
                       cat_ms=round(k["cat_ms"] / n, 4),
                       total_ms=round(mean_total, 4),
                       n=k["n"])
            if k["res_n"] > 0:
                row["residual_wait_ms"] = round(k["residual_wait_ms"] / k["res_n"], 4)
            summary.append(row)
        # ss ticks / mean per-tick sum_total so cross-rank comparison is one number
        n_ss_ticks = len(set(r["tick"] for r in ss_recs))
        # staggered residual: mean over layers>=1 (layer0 warmup excluded) + layer0
        res_rows = [s for s in summary if "residual_wait_ms" in s]
        res_body = [s["residual_wait_ms"] for s in res_rows if s["layer"] >= 1]
        res_l0 = next((s["residual_wait_ms"] for s in res_rows if s["layer"] == 0), None)
        mean_residual = round(sum(res_body) / len(res_body), 4) if res_body else None
        out_obj = dict(rank=self.rank,
                       n_ticks=len(raw),
                       n_ss_ticks=n_ss_ticks,
                       sum_total_ms=round(sum_total, 4),
                       mean_residual_wait_ms=mean_residual,   # layers 1..N-1
                       residual_wait_ms_layer0=res_l0,
                       layers=summary)
        sum_path = os.path.join(out_dir, f"layer_summary_r{self.rank}.json")
        with open(sum_path, "w") as f:
            json.dump(out_obj, f, indent=2)
        _res_str = (f", mean_residual(L>=1)={mean_residual:.4f}ms (L0={res_l0})"
                    if mean_residual is not None else "")
        print(f"[WAVE_LAYER_PROF] rank{self.rank}: {len(raw)} ticks, "
              f"{len(ss_recs)} steady-state recs, sum_total={sum_total:.2f}ms"
              f"{_res_str}, dumped to {out_dir}/", flush=True)

    # ------------------------------------------------------- edge-ts diagnostic
    def _dump_edge_ts(self) -> None:
        """WAVE_EDGE_TS (rank0): synchronize once, read every recorded CUDA event as
        ms-since-reference, write raw JSON + print a per-layer table.  No analysis --
        just the numbers (per the experiment spec)."""
        import json as _json
        torch.cuda.synchronize()
        ref = self._edge_ref
        recs = []
        for rec in self._edge_records:
            enter = ref.elapsed_time(rec["enter"])
            attn = ref.elapsed_time(rec["attn"])
            recv = [(int(src), ref.elapsed_time(ev)) for src, ev in rec["recv"]]
            recv_ms = [m for _, m in recv]
            earliest = min(recv_ms) if recv_ms else enter
            latest = max(recv_ms) if recv_ms else enter
            recs.append(dict(
                tick=rec["tick"], layer=rec["layer"], srcs=rec["srcs"],
                enter=enter, attn=attn, recv=recv,
                earliest=earliest, latest=latest,
                skew=latest - earliest, wait=attn - earliest,
                exposed=attn - enter))
        os.makedirs(_EDGE_TS_DIR, exist_ok=True)
        path = os.path.join(_EDGE_TS_DIR, "edge_r0.json")
        with open(path, "w") as f:
            _json.dump(dict(
                rank=0, world=self.world, store_rank=self.store_rank,
                num_layers=self.Ln, num_ticks=self.num_ticks,
                mode=self.cfg.exchange_mode, kv=self.cfg.kv_context,
                records=recs), f)

        # steady-state = past the pipeline-fill ramp (tick >= world-1); rank0 is
        # active only for ticks in [0, num_blocks).
        steady = [x for x in recs if x["tick"] >= self.world - 1]
        if not steady:
            steady = recs
        steady_ticks = sorted({x["tick"] for x in steady})
        Ln = self.Ln

        def _agg(rows, key):
            vals = [r[key] for r in rows]
            return sum(vals) / len(vals) if vals else 0.0

        print(f"\n[edge_ts] rank0 prefetch-overlap per-edge timestamps "
              f"(mode={self.cfg.exchange_mode} kv={self.cfg.kv_context})", flush=True)
        print(f"[edge_ts] {len(recs)} records; steady ticks {steady_ticks} "
              f"({len(steady)} records); srcs order = recv_srcs "
              f"(ranks>0 then store={self.store_rank}); raw -> {path}", flush=True)
        print(f"[edge_ts] ms averaged over steady ticks per layer; "
              f"skew=latest-earliest recv, wait=attn_start-earliest, "
              f"exposed=attn_start-enter", flush=True)
        hdr = (f"{'layer':>5} | {'n':>3} | {'mean_skew':>9} | {'mean_wait':>9} | "
               f"{'mean_exp':>8} | {'max_skew':>8} | {'max_exp':>8}")
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)
        per_layer = []
        for L in range(Ln):
            rows = [x for x in steady if x["layer"] == L]
            if not rows:
                continue
            ms_skew = _agg(rows, "skew")
            ms_wait = _agg(rows, "wait")
            ms_exp = _agg(rows, "exposed")
            mx_skew = max(r["skew"] for r in rows)
            mx_exp = max(r["exposed"] for r in rows)
            per_layer.append(dict(layer=L, n=len(rows), skew=ms_skew,
                                  wait=ms_wait, exp=ms_exp,
                                  max_skew=mx_skew, max_exp=mx_exp))
            print(f"{L:>5} | {len(rows):>3} | {ms_skew:>9.3f} | {ms_wait:>9.3f} | "
                  f"{ms_exp:>8.3f} | {mx_skew:>8.3f} | {mx_exp:>8.3f}", flush=True)

        # per-source incremental wait (delta from prior wait in recv_srcs order; the
        # first delta is measured from exch entry).  Attributes stall to each source.
        src_delta: dict = {}
        src_cnt: dict = {}
        for x in steady:
            prev = x["enter"]
            for src, m in x["recv"]:
                src_delta[src] = src_delta.get(src, 0.0) + (m - prev)
                src_cnt[src] = src_cnt.get(src, 0) + 1
                prev = m
        if src_delta:
            print("[edge_ts] mean incremental wait per source (ms, recv order; "
                  "delta beyond the previous source's wait):", flush=True)
            for src in sorted(src_delta):
                print(f"[edge_ts]   src {src}: "
                      f"{src_delta[src] / src_cnt[src]:.3f} ms  "
                      f"(n={src_cnt[src]})", flush=True)

        if per_layer:
            mean_skew = sum(p["skew"] for p in per_layer) / len(per_layer)
            mean_wait = sum(p["wait"] for p in per_layer) / len(per_layer)
            mean_exp = sum(p["exp"] for p in per_layer) / len(per_layer)
            top_skew = sorted(per_layer, key=lambda p: p["skew"], reverse=True)[:5]
            half = Ln // 2
            fh_rows = [p for p in per_layer if p["layer"] < half]
            sh_rows = [p for p in per_layer if p["layer"] >= half]
            fh = sum(p["skew"] for p in fh_rows) / max(1, len(fh_rows))
            sh = sum(p["skew"] for p in sh_rows) / max(1, len(sh_rows))
            print(f"[edge_ts] SUMMARY: mean_skew/layer={mean_skew:.3f}ms  "
                  f"mean_wait/layer={mean_wait:.3f}ms  "
                  f"mean_exposed/layer={mean_exp:.3f}ms", flush=True)
            print(f"[edge_ts] top-5 skew layers: "
                  f"{[(p['layer'], round(p['skew'], 3)) for p in top_skew]}",
                  flush=True)
            print(f"[edge_ts] skew first-half(L<{half})={fh:.3f}ms  "
                  f"second-half={sh:.3f}ms  (slack-accumulation trend)", flush=True)

    # -------------------------------------------------------------------- run
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
        tick_ms: list[float] = []
        timeline: list = []   # WAVE_TIMELINE: per-tick phase timestamps
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
            _dbg(self.rank, f"t={t} is_dn={is_dn} is_st={is_st} c={c} "
                            f"act_ranks={act_ranks} store_active={store_active}")

            if is_dn and self.rank == 0:
                _dbg(self.rank, f"t={t} slicing noise chunk c={c}")
                my_latent = full_noise[:, :, c * npb : (c + 1) * npb].contiguous()
                _dbg(self.rank, f"t={t} noise chunk ready {tuple(my_latent.shape)}")

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
            if _BCAST_CHECK:
                self._bc_store_mine = {}
                self._bc_rank0_recv = {}

            # Profile an ACTIVE denoise forward (rank 0 denoises at t == its chunk
            # index c == t, so it is active for t in [0, num_blocks)).  The old gate
            # (8<=t<=10) profiled rank0 while it was INACTIVE -> it timed the dummy
            # all_gathers, not a forward.  WAVE_PROF_TICKS overrides the ticks.
            _prof = (_TICKPROF and self.rank == 0 and t in _PROF_TICKS
                     and not warmup_pass)
            # WAVE_TL_PROF: split attn/exch on active ranks (defeats overlap -> only
            # for slack analysis, not for overlap timelines).
            _tlprof = _TL_PROF and (is_dn or is_st) and not warmup_pass
            if _prof or _tlprof:
                from wave_rt.backend import prof_reset
                prof_reset(True)

            if is_dn:
                step = self.dsl_list[self.rank]
                _dbg(self.rank, f"t={t} building denoise exch (step={step:.0f})")
                if onesided:
                    fn = self._make_denoise_exch_onesided(
                        clean, act_ranks, tick_csf, tick_qf, store_active)
                elif paged:
                    fn = self._make_denoise_exch_paged(
                        clean, act_ranks, tick_csf, tick_qf, store_active)
                elif staggered:
                    gen = t % self._os_gt
                    recv_srcs = [n for n in act_ranks if n > self.rank] + (
                        [store_rank] if store_active else [])
                    self._stagger_bootstrap(recv_srcs, gen, t)
                    fn = self._make_denoise_exch_staggered(
                        clean, act_ranks, tick_csf, tick_qf, store_active)
                elif relay:
                    fn = self._make_denoise_exch_relay(
                        clean, act_ranks, tick_csf, tick_qf, store_active)
                elif overlap and self._use_prefetch:
                    recv_srcs = [n for n in act_ranks if n > self.rank] + (
                        [store_rank] if store_active else [])
                    self._bootstrap_recv(recv_srcs)
                    fn = self._make_denoise_exch_prefetch(
                        clean, act_ranks, tick_csf, tick_qf, store_active)
                elif overlap:
                    fn = self._make_denoise_exch_overlap(
                        clean, act_ranks, tick_csf, tick_qf, store_active)
                else:
                    fn = self._make_denoise_exch(
                        clean, act_ranks, tick_csf, tick_qf, store_active)
                be.set_exchange_fn(fn)
                _dbg(self.rank, f"t={t} denoise forward start (step={step:.0f})")
                # Physical skew: non-store ranks burn GPU cycles at tick start
                # to give store_rank (rank4) a head start for KV copy.  onesided-only.
                if self._skew_cycles > 0 and onesided and self.rank != store_rank:
                    torch.cuda._sleep(self._skew_cycles)
                x0 = be.forward_chunk(my_latent, step, start_frame=c * npb)
                be.set_exchange_fn(None)
                if overlap:
                    if _mp:
                        _td = time.perf_counter()
                        self._drain_p2p_sends()
                        _mp_drain_cpu += (time.perf_counter() - _td) * 1000.0
                    else:
                        self._drain_p2p_sends()
                elif onesided or staggered or paged:
                    if _mp:
                        _td = time.perf_counter()
                        self._drain_publish()
                        _mp_drain_cpu += (time.perf_counter() - _td) * 1000.0
                    else:
                        self._drain_publish()
                my_x0 = x0.permute(0, 2, 1, 3, 4).contiguous()  # (1,C,npb,H,W)
                _dbg(self.rank, f"t={t} denoise forward done")
            elif is_st:
                if paged:
                    fn = self._make_store_exch_paged(clean, store_kv, c, act_ranks)
                elif onesided or staggered:
                    # store is a pure producer -> staggered reuses the one-sided
                    # store publish verbatim (no inbound acquire to prefetch).
                    fn = self._make_store_exch_onesided(clean, store_kv, c, act_ranks)
                elif overlap and self._use_prefetch:
                    self._pf_slots = [None, None]   # store only sends (no recv)
                    self._pf_works = [None, None]
                    fn = self._make_store_exch_prefetch(clean, store_kv, c, act_ranks)
                elif overlap:
                    fn = self._make_store_exch_overlap(clean, store_kv, c, act_ranks)
                else:
                    fn = self._make_store_exch(clean, store_kv, c)
                be.set_exchange_fn(fn)
                _dbg(self.rank, f"t={t} store forward start")
                be.forward_chunk(my_latent, self.context_noise, start_frame=c * npb)
                be.set_exchange_fn(None)
                if overlap:
                    if _mp:
                        _td = time.perf_counter()
                        self._drain_p2p_sends()
                        _mp_drain_cpu += (time.perf_counter() - _td) * 1000.0
                    else:
                        self._drain_p2p_sends()
                elif onesided or staggered or paged:
                    if _mp:
                        _td = time.perf_counter()
                        self._drain_publish()
                        _mp_drain_cpu += (time.perf_counter() - _td) * 1000.0
                    else:
                        self._drain_publish()
                final_latents[c] = my_latent.clone()
                # Stream this finalized chunk to the VAE pipeline AS IT DRAINS off the
                # diagonal (overlap VAE with the ongoing diffusion).  BTCHW numpy to
                # match vae_pipe/vae_stage stage0's permute(0,2,1,3,4) convention.
                if self.q is not None and not warmup_pass:
                    self.q.put((c, my_latent.permute(0, 2, 1, 3, 4)
                                .contiguous().float().cpu().numpy()))
                _dbg(self.rank, f"t={t} store forward done")
            else:
                # overlap/onesided/staggered/paged: inactive ranks do NO per-layer comm
                # (they idle on the P2P/IPC path); sync: they must issue matching
                # dummy all_gathers for WORLD alignment.
                if not overlap and not onesided and not staggered and not paged:
                    _dbg(self.rank, f"t={t} dummy allgathers x{Ln}")
                    self._dummy_allgathers()
                _dbg(self.rank, f"t={t} dummy allgathers done")

            if _TIMELINE:
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
            if store_active and (_FULL_BCAST or cid == 0):
                _dbg(self.rank, f"t={t} broadcast clean KV start (cid={cid})")
                Hh = self.be.num_attention_heads
                Dd = self.be.attention_head_dim
                Tt = self.tokens_per_chunk
                if _FULL_BCAST:
                    # A/B baseline: original 3-tensor broadcast for every chunk.
                    if is_st:
                        kr = torch.stack([store_kv[L][0][0] for L in range(Ln)])
                        ku = torch.stack([store_kv[L][1][0] for L in range(Ln)])
                        vv = torch.stack([store_kv[L][2][0] for L in range(Ln)])
                        blob = torch.stack([kr, ku, vv]).contiguous()
                        if _BCAST_CHECK:
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
                _dbg(self.rank, f"t={t} broadcast clean KV done (cid={cid})")
                if _BCAST_CHECK and self.rank == 0 and self._bc_rank0_recv:
                    os.makedirs(_BCAST_CHECK_DIR, exist_ok=True)
                    torch.save(
                        {"recv0": self._bc_rank0_recv[0].cpu()},
                        os.path.join(_BCAST_CHECK_DIR, f"rank0_t{t}.pt"))
            elif store_active:
                # WORKING chunk (cid>=1), no broadcast: promote the (kr, v) each rank
                # already holds into a clean working page.  Store rank: local
                # store_kv.  Active denoise rank: the exchange-cached slot.  Idle
                # ranks are finished -> skip (they never read this chunk again).
                _dbg(self.rank, f"t={t} promote clean KV (cid={cid}) is_st={is_st} is_dn={is_dn}")
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
                _dbg(self.rank, f"t={t} promote clean KV done (cid={cid})")


            if _prof:
                torch.cuda.synchronize()
                _t_bc = time.perf_counter()

            if _TIMELINE:
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
            _dbg(self.rank, f"t={t} migrate will_send={will_send} will_recv={will_recv}")
            mig_shape = [1, C, npb, H, W]
            if will_send:
                if self.rank < rf_step - 1:
                    send_payload = self._det_renoise(
                        my_x0, self.dsl_list[self.rank + 1], c, self.rank
                    )
                else:  # last denoise rank -> raw finalized latent to the store rank
                    send_payload = my_x0
            else:
                send_payload = torch.zeros(mig_shape, device=dev, dtype=self.dtype)
            # migration stays an all_gather even in overlap: like the broadcast, a
            # P2P version measured slower than the optimized collective, and it
            # already carries only small latents.
            if _mp:
                _e_mig0 = torch.cuda.Event(enable_timing=True)
                _e_mig0.record()
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
            _dbg(self.rank, f"t={t} migrate done")
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
                from wave_rt.backend import prof_read, prof_reset
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
            if _TICKPROF or _TIMELINE or _FORCE_TICK_SYNC:
                torch.cuda.synchronize()
            tick_ms.append((time.perf_counter() - tb) * 1000)
            if _TIMELINE:
                _tl_mig = time.perf_counter()
                role = "denoise" if is_dn else ("store" if is_st else "idle")
                _a_ms = _e_ms = _kl = 0
                if _tlprof:
                    from wave_rt.backend import prof_read, prof_reset
                    _a_ms, _e_ms, _kl = prof_read()
                    prof_reset(False)
                timeline.append(dict(
                    tick=t, rank=self.rank, role=role, chunk=c,
                    t0=tb, fwd=_tl_fwd, bc=_tl_bc, mig=_tl_mig,
                    attn_ms=_a_ms, exch_ms=_e_ms, klen=_kl,
                ))

        if not warmup_pass and self._layer_prof and self._lp_records:
            self._dump_layer_prof()

        return final_latents, tick_ms, timeline

    def run(self, out_dir: str | None = None, warmup: bool = True,
            save: bool = True) -> None:
        be = self.be
        dev = self.device
        npb, Ln = self.npb, self.Ln
        rf_step, store_rank = self.rf_step, self.store_rank
        dsl = self.dsl
        full_noise = be.full_noise_bcthw  # (1, C, nf, H, W)
        _, C, _, H, W = full_noise.shape

        _dbg(self.rank, f"reached run(); role_ranks store={store_rank}")
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
        be.barrier()
        _dbg(self.rank, "passed start barrier; entering systolic loop")
        if self._edge_ts and self.rank == 0:
            self._edge_ref = torch.cuda.Event(enable_timing=True)
            self._edge_ref.record()
        t_start = time.perf_counter()
        final_latents, tick_ms, timeline = self._systolic(
            full_noise, warmup_pass=False, num_ticks=self.num_ticks)
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
            print("[wave_rt/cat_ts] per-layer mean K/V-cat (ms), steady ticks, "
                  f"mode={'no_cat' if _NO_CAT else 'cat'}:", flush=True)
            for L in sorted(per_layer):
                v = per_layer[L]
                print(f"  layer {L:2d}: {sum(v)/len(v):.4f}ms  (n={len(v)})", flush=True)
            print(f"[wave_rt/cat_ts] overall mean {sum(all_ms)/len(all_ms):.4f}ms/layer "
                  f"across {len(all_ms)} samples; "
                  f"~{sum(all_ms)/len(all_ms)*self.Ln:.2f}ms/tick", flush=True)

        if _TIMELINE:
            import json as _json
            os.makedirs(_TL_DIR, exist_ok=True)
            with open(os.path.join(_TL_DIR, f"diff_r{self.rank}.json"), "w") as _f:
                _json.dump(dict(rank=self.rank, mode=self.cfg.exchange_mode,
                                kv=self.cfg.kv_context, t_start=t_start,
                                events=timeline), _f)

        if self.rank == 0:
            # Without the per-tick sync, tick_ms is only CPU-launch time; the honest
            # per-tick wall clock is diffusion_ms / num_ticks.  Use tick_ms only when
            # it was actually synced (timing runs).
            if _TICKPROF or _TIMELINE:
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
            if self.cfg.vae_stages > 0 and self.q is not None:
                # streaming n-stage VAE: chunks were already pushed to q as they
                # finalized; signal end, persist latents (PSNR gate) + diffusion
                # timing, and let the VAE stages save the video (metrics in launcher).
                self.q.put(None)
                idx = sorted(final_latents.keys())
                latents = torch.cat([final_latents[i] for i in idx], dim=2)  # (1,C,nf,H,W)
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
                from wave_rt.vae import finalize_on_store
                finalize_on_store(be, self.cfg, final_latents, diffusion_ms, tick_ms)

        be.barrier()
