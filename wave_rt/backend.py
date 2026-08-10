"""WaveRT backend -- the ONLY module that reaches into sglang internals.

Responsibilities (per rank):
  0. At import time, monkeypatch ``CausalWanSelfAttention.forward`` so that, when
     WaveRT installs a per-tick "exchange" closure, the self-attention bypasses
     the single-GPU KV cache and instead attends over a KV context assembled by
     WaveRT (anchor + working + live-clean + in-flight chunks).  This is the same
     hook we prototyped on the abandoned wave-parallel branch, now expressed as a
     runtime patch so sglang source stays pristine.
  1. Own the torch.distributed world (size == wp_size).  sglang model-parallel
     degrees are all 1 (init via data_parallel_size=wp_size so every rank gets a
     valid singleton TP/SP group -> the model does zero collective comm), leaving
     the default WORLD process group free for WaveRT's wavefront collectives.
  2. build_pipeline() and expose transformer / vae / scheduler / RF stage plus a
     single-chunk DiT forward helper and a VAE decode helper.

Everything wave-specific lives here or in wavefront.py; sglang is unmodified.
"""

from __future__ import annotations

import os
from typing import Callable

import torch
import torch.distributed as dist

from wave_rt.config import WaveConfig

# --- module-level state read by the monkeypatched attention -----------------
# ``exchange_fn`` is set by wavefront.py right before an active rank runs a
# single-chunk DiT forward, and cleared afterwards.  When None, the patched
# forward falls through to the original sglang implementation (so a wp_size==1
# smoke run is bit-identical to single-GPU RF).
_WAVE_STATE: dict = {"exchange_fn": None, "attn": "torch_sdpa", "fa_func": None,
                     "layer_prof_sink": None}


def set_exchange_fn(fn: Callable | None) -> None:
    _WAVE_STATE["exchange_fn"] = fn


def set_layer_prof_sink(sink) -> None:
    """WAVE_LAYER_PROF: install a list into which the patched attention appends one
    CUDA event (enable_timing) right AFTER each layer's attention -> lets wavefront
    separate sage time from pre-attn compute.  None disables (no per-layer overhead
    on the hot path when off)."""
    _WAVE_STATE["layer_prof_sink"] = sink


def prof_reset(on: bool) -> None:
    _WAVE_STATE["prof"] = on
    _WAVE_STATE["attn_ms"] = 0.0
    _WAVE_STATE["exch_ms"] = 0.0


def prof_read() -> tuple:
    return (
        _WAVE_STATE.get("attn_ms", 0.0),
        _WAVE_STATE.get("exch_ms", 0.0),
        _WAVE_STATE.get("klen", 0),
    )


def enable_fa_custom() -> None:
    """Route the wavefront attention through our own pip-compiled flash_attn
    (flash_attn 2.8.4), bypassing sglang's LocalAttention -> jit_kernel path
    (LocalAttention doesn't even list FA2, and FA routes to the cu13 jit kernel
    that's broken on this driver).  Only affects the exchange path."""
    from flash_attn import flash_attn_func

    _WAVE_STATE["attn"] = "fa_custom"
    _WAVE_STATE["fa_func"] = flash_attn_func


def enable_sage() -> None:
    """Route the wavefront attention through SageAttention (int8-quantized, ~2x
    faster than flash on large contexts, near-lossless here: vs SDPA maxdiff
    ~1.7e-3).  Only affects the exchange path."""
    from sageattention import sageattn

    _WAVE_STATE["attn"] = "sa"
    _WAVE_STATE["fa_func"] = sageattn


def _install_attention_monkeypatch() -> None:
    """Patch CausalWanSelfAttention.forward to honour ``_WAVE_STATE`` closure."""
    from sglang.multimodal_gen.runtime.models.dits import causal_wanvideo as cw

    if getattr(cw.CausalWanSelfAttention, "_wave_patched", False):
        return

    orig_forward = cw.CausalWanSelfAttention.forward
    # NOTE: _apply_rotary_emb was bound into the causal_wanvideo namespace at its
    # import time, so read it from there (not the origin module).
    apply_rope = cw._apply_rotary_emb

    def patched_forward(
        self,
        q,
        k,
        v,
        freqs_cis,
        block_mask,
        kv_cache=None,
        current_start: int = 0,
        cache_start=None,
        updating_cache: bool = False,
        tokens_per_frame=None,
        post_patch_height=None,
        post_patch_width=None,
    ):
        exch = _WAVE_STATE["exchange_fn"]
        if exch is None:
            return orig_forward(
                self, q, k, v, freqs_cis, block_mask, kv_cache,
                current_start, cache_start, updating_cache,
                tokens_per_frame, post_patch_height, post_patch_width,
            )
        cos, sin = freqs_cis
        roped_query = apply_rope(q, cos, sin, is_neox_style=False).type_as(v)
        roped_key = apply_rope(k, cos, sin, is_neox_style=False).type_as(v)
        # The closure assembles the full (K, V) context for this layer across the
        # wavefront and returns them ready for attention.  ``k`` is the un-roped
        # key (needed to re-RoPE the anchor at a shifted start frame).
        if _WAVE_STATE.get("prof"):
            import time as _t
            torch.cuda.synchronize(); _e = _t.perf_counter()
        ctx_k, ctx_v = exch(
            layer_idx=self._wave_layer_idx,
            roped_query=roped_query,
            roped_key=roped_key,
            unroped_key=k,
            value=v,
            current_start=current_start,
            tokens_per_frame=tokens_per_frame,
            post_patch_height=post_patch_height,
            post_patch_width=post_patch_width,
            dim=self.dim,
            rope_num_heads=self.rope_num_heads,
            num_frames_per_block=self.num_frames_per_block,
        )
        if _WAVE_STATE.get("prof"):
            import time as _t
            torch.cuda.synchronize()
            _WAVE_STATE["exch_ms"] = _WAVE_STATE.get("exch_ms", 0.0) + (_t.perf_counter() - _e) * 1000
            _a = _t.perf_counter()
        if _WAVE_STATE["attn"] == "fa_custom":
            out = _WAVE_STATE["fa_func"](roped_query, ctx_k, ctx_v, causal=False)
        elif _WAVE_STATE["attn"] == "sa":
            out = _WAVE_STATE["fa_func"](
                roped_query, ctx_k, ctx_v, tensor_layout="NHD", is_causal=False
            )
        else:
            out = self.attn(roped_query, ctx_k, ctx_v)
        if _WAVE_STATE.get("prof"):
            import time as _t
            torch.cuda.synchronize()
            _WAVE_STATE["attn_ms"] = _WAVE_STATE.get("attn_ms", 0.0) + (_t.perf_counter() - _a) * 1000
            _WAVE_STATE["klen"] = ctx_k.shape[1]
        # WAVE_LAYER_PROF: mark this layer's attention end (async event; no sync ->
        # negligible hot-path cost).  Ordered on the compute stream after `out`.
        _lp_sink = _WAVE_STATE.get("layer_prof_sink")
        if _lp_sink is not None:
            _e_pa = torch.cuda.Event(enable_timing=True)
            _e_pa.record()
            _lp_sink.append(_e_pa)
        return out

    patched_forward._wave_orig = orig_forward
    cw.CausalWanSelfAttention.forward = patched_forward
    cw.CausalWanSelfAttention._wave_patched = True


class WaveBackend:
    """Per-rank sglang wrapper for WaveRT."""

    def __init__(self, cfg: WaveConfig, rank: int) -> None:
        self.cfg = cfg
        self.rank = rank
        self.world = cfg.wp_size
        self.device = torch.device(f"cuda:{rank}")

        # populated by init()
        self.server_args = None
        self.pipeline = None
        self.transformer = None
        self.vae = None
        self.scheduler = None
        self.rf_stage = None
        self.decode_stage = None
        self.batch = None
        self.ctx = None  # CausalDMDForwardContext
        self.num_token_per_frame = None
        self.num_layers = None
        self.latent_channels = None

    # ------------------------------------------------------------------ init
    def init(self) -> None:
        torch.cuda.set_device(self.rank)
        torch.set_grad_enabled(False)
        # Per-rank intra-op thread cap (WAVE_THREADS=N; launcher defaults it to an
        # even core split -- critical: uncapped 6x128 OMP threads crush the forward,
        # 1050ms -> 180ms).  "0"/"" leaves torch's default (no cap).
        import os as _os
        _wt = _os.environ.get("WAVE_THREADS", "")
        if _wt not in ("", "0"):
            torch.set_num_threads(int(_wt))

        # env for sglang's distributed bring-up
        os.environ["WORLD_SIZE"] = str(self.world)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.rank)

        # patch BEFORE any model construction
        _install_attention_monkeypatch()
        # WaveRT-level attention choice for the wavefront path
        if self.cfg.attention_backend == "fa_custom":
            enable_fa_custom()
        elif self.cfg.attention_backend == "sa":
            enable_sage()

        from sglang.multimodal_gen.runtime.server_args import (
            ServerArgs,
            set_global_server_args,
        )
        from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline

        from wave_rt.dist_compat import (
            disable_rf_config_patch,
            init_wave_dist,
        )

        # sglang always gets a driver-safe backend for any non-wavefront path;
        # the wavefront attention is controlled by cfg.attention_backend above.
        sgl_attn = (
            "torch_sdpa"
            if self.cfg.attention_backend in ("fa_custom", "sa")
            else self.cfg.attention_backend
        )
        server_args = ServerArgs.from_dict(
            {
                "model_path": self.cfg.model_path,
                "pipeline_class_name": self.cfg.pipeline_class_name,
                "num_gpus": 1,
                "tp_size": 1,
                "ulysses_degree": 1,
                "ring_degree": 1,
                "cfg_parallel_degree": 1,
                "attention_backend": sgl_attn,
                "dit_cpu_offload": False,
                # serving keeps T5 resident (avoid reloading it from CPU per request);
                # one-shot can offload it (single encode).
                "text_encoder_cpu_offload": not self.cfg.serve,
                "enable_torch_compile": self.cfg.compile,
                # WaveRT wants each rank to hold a FULL replica of every model.
                # FSDP inference-sharding would (a) shard weights across the world
                # and (b) build an init_device_mesh whose product must equal the
                # world size -> crashes with num_gpus=1 on a >1-rank world.  Force
                # the plain (non-FSDP) loader path instead.
                "use_fsdp_inference": False,
            }
        )
        # hsdp_shard_dim is forced to num_gpus in __post_init__; clear it so the
        # T5/bridge loader's "hsdp_shard_dim is not None" FSDP trigger is skipped.
        server_args.hsdp_shard_dim = None
        # override RF checkpoint (path lives on the pipeline_config)
        server_args.pipeline_config.rolling_forcing_checkpoint_path = self.cfg.gen_ckpt
        server_args.pipeline_config.rolling_forcing_use_ema = True
        # System-level scaling test: build a LARGER random-init DiT and skip the RF
        # overlay (no checkpoint exists at that scale).  Must patch the loader BEFORE
        # build_pipeline; point the RF path at a missing file so the overlay is
        # skipped cleanly (initialize_pipeline falls back to base weights).
        if self.cfg.dit_scale not in (None, "1.3b"):
            from wave_rt.dummy_model import install as install_dummy_dit
            install_dummy_dit(self.cfg.dit_scale, verbose=(self.rank == 0))
            server_args.pipeline_config.rolling_forcing_checkpoint_path = (
                "/nonexistent/wave_dummy_skip_rf.pt"
            )
            if self.rank == 0:
                print(f"[backend] DUMMY DiT scale={self.cfg.dit_scale} "
                      f"(random init, RF overlay skipped)", flush=True)
        # CRITICAL: force the DMD schedule to match the loaded checkpoint.  sglang's
        # RollingForcingWanT2V480PConfig ships a WRONG 5-step [1000,800,600,400,200]
        # default, but longvideo.pt is a 4-step DMD [1000,750,500,250] (see
        # Causal-Forcing/long_video/configs/rolling_forcing_dmd.yaml).  Running the
        # wrong step count/values on the 4-step checkpoint is off-distribution.
        # sgl warps these identically to the real pipeline (timesteps[1000-steps]),
        # so overriding here reproduces naive's exact schedule.
        server_args.pipeline_config.dmd_denoising_steps = list(self.cfg.denoising_step_list)
        server_args.pipeline_config.warp_denoising_step = True
        set_global_server_args(server_args)
        self.server_args = server_args

        # WaveRT owns distribution the SAME way naive does: ONE bare NCCL process
        # group (which doubles as the wavefront comm) + a degree-1 identity shim for
        # sglang's parallel_state.  We deliberately do NOT call
        # maybe_init_distributed_environment_and_model_parallel -- that builds ~6
        # GroupCoordinators (gloo groups always + pynccl + srt custom-allreduce for
        # world>1) whose cross-process shared resources contend on one node.  The shim
        # (dist_compat._install_sgl_shim) points _WORLD/_TP/_SP/_CFG/_VAE_DECODE at a
        # size-1 coordinator and leaves _DP/_PP None, so the model does zero collective
        # comm and the default WORLD PG is free for the wavefront's raw collectives.
        # Must run BEFORE build_pipeline: linear __init__ reads get_tp_group() (_TP).
        #
        # WAVE_DIST_MODE=sgl reverts to sglang's GroupCoordinator path (A/B baseline
        # for the feasibility report); default "shim" is the naive-style bare NCCL.
        dist_mode = os.environ.get("WAVE_DIST_MODE", "shim")
        if dist_mode == "sgl":
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                maybe_init_distributed_environment_and_model_parallel,
            )
            maybe_init_distributed_environment_and_model_parallel(
                tp_size=1, sp_size=1, cfg_degree=1, ulysses_degree=1,
                ring_degree=1, dp_size=self.world, dist_timeout=180,
            )
            if self.rank == 0:
                print("[backend] dist mode = sgl (GroupCoordinator)", flush=True)
        else:
            init_wave_dist(
                world_size=self.world,
                rank=self.rank,
                local_rank=self.rank,
                timeout_s=180,
            )
            self._assert_single_card_degrees()
            if self.rank == 0:
                print("[backend] dist mode = shim (bare NCCL)", flush=True)

        # No-op the pipeline's per-rank transformer-config patch/restore (the
        # launcher pre-patched the shared file once); avoids the wp_size-way
        # read-modify-write race that crashed a rank and hung the rest.
        disable_rf_config_patch()

        self.pipeline = build_pipeline(server_args)
        self.transformer = self.pipeline.get_module("transformer")
        self.vae = self.pipeline.get_module("vae")
        self.scheduler = self.pipeline.get_module("scheduler")

        # tag each self-attention module with its layer index for the closure
        for i, block in enumerate(self.transformer.blocks):
            block.attn1._wave_layer_idx = i
        self.num_layers = len(self.transformer.blocks)

        # Optional post-load FP8 (W8A8) quantization of the DiT block linears.
        # Done AFTER the bf16 build + RF overlay (sglang's load-time --quantization
        # path is incompatible with the RF load_state_dict; see wave_rt/fp8_linear).
        if self.cfg.quantization == "fp8":
            from wave_rt.fp8_linear import quantize_transformer_fp8
            nq = quantize_transformer_fp8(self.transformer, verbose=(self.rank == 0))
            if self.rank == 0:
                print(f"[backend] FP8 W8A8: {nq} linears quantized", flush=True)
        elif self.cfg.quantization not in (None, "", "bf16"):
            raise ValueError(
                f"unknown --quantization {self.cfg.quantization!r} (wave_rt supports "
                f"'fp8' or None)"
            )

        self.latent_channels = self.transformer.config.in_channels
        self.num_attention_heads = self.transformer.config.num_attention_heads
        self.attention_head_dim = self.transformer.config.attention_head_dim

        # reuse the pipeline's own stage instances
        self.rf_stage = self.pipeline.stages[3]
        self.decode_stage = self.pipeline.stages[4]

        self._prepare_conditioning()

        # the wavefront needs one denoise rank per DMD step
        n_steps = int(self.dsl.numel())
        if self.cfg.wp_size > 1 and n_steps != self.cfg.rf_step:
            raise ValueError(
                f"rf_step ({self.cfg.rf_step}) must equal the number of DMD "
                f"denoising steps ({n_steps}); set --rf-step {n_steps} "
                f"--wp-size {n_steps + 1}."
            )

        if self.rank == 0:
            print(
                f"[backend] ready: {self.num_layers} layers, "
                f"C={self.latent_channels}, tokens/frame={self.num_token_per_frame}, "
                f"dsl={self.dsl.tolist()}",
                flush=True,
            )
        if os.environ.get("WAVE_DEBUG", "") not in ("", "0", "false"):
            print(f"[wrt-dbg r{self.rank}] backend.init DONE", flush=True)

    def _assert_single_card_degrees(self) -> None:
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_world_size,
            get_tp_world_size,
            model_parallel_is_initialized,
        )

        sp, tp = get_sp_world_size(), get_tp_world_size()
        assert sp == 1 and tp == 1, (
            f"WaveRT needs model degrees == 1, got sp={sp} tp={tp} on rank "
            f"{self.rank}. Model would do collective comm and deadlock the wavefront."
        )
        # The degree-1 shim MUST keep _DP/_PP None so model_parallel_is_initialized()
        # stays False -> VAE spatial-parallel decode disabled + size-1 fallbacks active.
        assert not model_parallel_is_initialized(), (
            f"WaveRT shim broke: model_parallel_is_initialized() is True on rank "
            f"{self.rank} (>1 degree groups built). VAE parallel decode would expect "
            f"real collectives and deadlock."
        )

    def _prepare_conditioning(self) -> None:
        """Init-time conditioning with the config defaults (also warms up + validates
        the DMD schedule).  Per-request serving re-runs prepare_request() directly."""
        self.prepare_request(
            self.cfg.prompt, self.cfg.seed, self.cfg.num_frames,
            self.cfg.height, self.cfg.width,
        )

    def prepare_request(self, prompt: str, seed: int, num_frames: int,
                        height: int | None = None, width: int | None = None) -> None:
        """Per-request conditioning: run the pipeline prefix (InputValidation ->
        Text encode -> LatentPrep) for this prompt/seed/shape, build the RF forward
        context, and (re)allocate the kv/crossattn caches.  Re-running resets those
        caches so serving requests don't leak state across videos.  Model weights are
        NOT reloaded (that's backend.init, once)."""
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

        h = height if height is not None else self.cfg.height
        w = width if width is not None else self.cfg.width
        sp_cls = type(self.pipeline).sampling_params_cls
        sampling_params = sp_cls(
            prompt=prompt, num_frames=num_frames, height=h, width=w, seed=seed,
        )
        batch = Req(sampling_params=sampling_params)

        # stages[0]=InputValidation, [1]=TextEncoding, [2]=LatentPreparation
        for stage in self.pipeline.stages[:3]:
            batch = stage(batch, self.server_args)

        self.batch = batch
        self.ctx = self.rf_stage._prepare_causal_dmd_forward_context(
            batch, self.server_args
        )
        self.num_token_per_frame = self.rf_stage.num_token_per_frame

        # (re)allocate caches (kv self-attn cache indexed by the transformer forward;
        # crossattn cache IS used).  Fresh alloc per request => no cross-request leak.
        self.rf_stage._initialize_causal_caches(
            batch_size=self.ctx.batch_size,
            max_text_len=self.rf_stage._get_max_text_len(self.server_args),
            dtype=self.ctx.target_dtype,
            device=self.device,
        )

        # The wave exchange path never reads/writes the sgl self-attn KV cache
        # (patched_forward attends over the wavefront-assembled context instead),
        # but _initialize_causal_caches still allocates it -- ~27 GiB dead weight
        # per rank on 14B.  Shrink each block's k/v to a 1-token placeholder so
        # the pass-through plumbing (shape reads, reset_indices) keeps working.
        if os.environ.get("WAVE_KEEP_SGL_KVCACHE", "0") != "1":
            freed = 0
            for cb in self.rf_stage.causal_kv_cache:
                freed += (cb.k.numel() + cb.v.numel()) * cb.k.element_size()
                cb.k = cb.k[:, :1].clone()
                cb.v = cb.v[:, :1].clone()
            torch.cuda.empty_cache()
            if self.rank == 0:
                print(
                    f"[backend] shrunk unused sgl self-attn KV cache, "
                    f"freed {freed / 2**30:.1f} GiB/rank",
                    flush=True,
                )

    # ---------------------------------------------------------------- exposed
    def set_exchange_fn(self, fn) -> None:
        """Install (or clear with None) the per-tick wavefront KV-exchange closure
        that the monkeypatched attention reads."""
        set_exchange_fn(fn)

    @property
    def dsl(self) -> torch.Tensor:
        """Denoising step list (warped DMD timesteps), e.g. [1000,750,500,250]."""
        return self.ctx.timesteps

    @property
    def full_noise_bcthw(self) -> torch.Tensor:
        """Full-window initial noise (B, C, nf, H, W) -- same tensor single-GPU
        RF slices its chunks from, for bit-level noise parity."""
        return self.ctx.latents

    def forward_chunk(
        self, noisy_chunk_bcthw: torch.Tensor, step_value: float, start_frame: int
    ) -> torch.Tensor:
        """One DiT forward for a single ``num_frames_per_block`` chunk at a single
        denoising ``step_value``.  Returns x0 in (B, T, C, H, W).  The active
        wavefront exchange closure must already be installed via set_exchange_fn.
        """
        npb = self.cfg.num_frames_per_block
        b = noisy_chunk_bcthw.shape[0]
        latent_model_input = noisy_chunk_bcthw.to(self.ctx.target_dtype)
        noise_latents_btchw = noisy_chunk_bcthw.permute(0, 2, 1, 3, 4)
        timestep_bf = torch.full(
            [b, npb], float(step_value), device=self.device, dtype=torch.float32
        )
        x0_btchw, _ = self.rf_stage._predict_x0_per_frame_btchw(
            self.batch,
            self.server_args,
            latent_model_input=latent_model_input,
            noise_latents_btchw=noise_latents_btchw,
            timestep_bf=timestep_bf,
            scheduler=self.ctx.scheduler,
            prompt_embeds=self.ctx.prompt_embeds,
            kv_cache=self.rf_stage.causal_kv_cache,
            crossattn_cache=self.rf_stage.crossattn_cache,
            current_start_tokens=start_frame * self.num_token_per_frame,
            start_frame=start_frame,
            image_kwargs=self.ctx.image_kwargs,
            pos_cond_kwargs=self.ctx.pos_cond_kwargs,
            attn_raw_latent_shape=(npb, self.ctx.height, self.ctx.width),
            current_timestep=0,
            target_dtype=self.ctx.target_dtype,
            autocast_enabled=self.ctx.autocast_enabled,
            device=self.device,
        )
        return x0_btchw

    @torch.no_grad()
    def decode(self, latents_bcthw: torch.Tensor) -> torch.Tensor:
        """VAE decode via the pipeline's DecodingStage (fp32 vae per config)."""
        return self.decode_stage.decode(
            latents_bcthw, self.server_args, vae_dtype=torch.bfloat16
        )

    # -------------------------------------------------------------- lifecycle
    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    def shutdown(self) -> None:
        if dist.is_initialized():
            # No barrier here: on an exception path a barrier would hang waiting
            # for ranks that already died.  Just tear the PG down.
            try:
                dist.destroy_process_group()
            except Exception:
                pass
