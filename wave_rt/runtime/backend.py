"""WaveRT integration with the SGLang diffusion pipeline.

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

Local compute is in runtime/model.py; scheduling and communication live in
denoiser/. SGLang source is unmodified.
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


def enable_flash_attention(backend="fa3") -> None:
    """Use the same explicit FA2/FA3 adapter for WP and SP attention."""
    from wave_rt.runtime.attention import Attention

    attention = Attention(backend)
    _WAVE_STATE["attn"] = backend
    _WAVE_STATE["fa_func"] = attention


def enable_fa3() -> None:
    enable_flash_attention("fa3")


def enable_sage() -> None:
    """Use shared fused Sage preparation on the WP exchange attention path."""
    from sageattention import sageattn
    from wave_rt.runtime.sage import SagePrefixAttention

    _WAVE_STATE["attn"] = "sa"
    _WAVE_STATE["fa_func"] = SagePrefixAttention(sageattn)


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
        write_only: bool = False,
        prepared_rope=None,
    ):
        exch = _WAVE_STATE["exchange_fn"]
        if exch is None:
            if write_only:
                raise RuntimeError("KV-only store requires an active WP exchange")
            return orig_forward(
                self, q, k, v, freqs_cis, block_mask, kv_cache,
                current_start, cache_start, updating_cache,
                tokens_per_frame, post_patch_height, post_patch_width,
            )
        cos, sin = freqs_cis
        if prepared_rope is None:
            roped_query = (None if write_only else
                           apply_rope(q, cos, sin, is_neox_style=False).type_as(v))
            roped_key = apply_rope(k, cos, sin, is_neox_style=False).type_as(v)
        else:
            roped_query, roped_key = prepared_rope
        if hasattr(exch, 'defer_live_copy'):
            can_promote = getattr(_WAVE_STATE['fa_func'], 'can_promote', None)
            exch.defer_live_copy = (not write_only and _WAVE_STATE['attn'] == 'sa'
                                   and can_promote is not None
                                   and can_promote(roped_query, roped_key, v))
            exch.kv_promotion = None
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
            **({"write_only": True} if write_only else {}),
        )
        trace = getattr(exch, 'trace_attention', None)
        from wave_rt.denoiser.profile import current as runtime_profile_current
        runtime_trace = runtime_profile_current()
        if runtime_trace is not None:
            previous_trace = trace
            def trace(phase, layer, **details):
                runtime_trace.attention(phase, layer, **details)
                if previous_trace is not None:
                    previous_trace(phase, layer, **details)
        if trace is not None:
            trace('exchange_returned', self._wave_layer_idx,
                  profiling=bool(_WAVE_STATE.get('prof')), write_only=write_only)
        if _WAVE_STATE.get("prof"):
            import time as _t
            torch.cuda.synchronize()
            _WAVE_STATE["exch_ms"] = _WAVE_STATE.get("exch_ms", 0.0) + (_t.perf_counter() - _e) * 1000
            _a = _t.perf_counter()
        if write_only:
            return None  # KV has been published; no query attention or output
        if trace is not None:
            # Metadata only: never read tensor contents or synchronize CUDA here.
            geometry = {name: dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                        dtype=str(tensor.dtype), device=str(tensor.device),
                        contiguous=tensor.is_contiguous(), requires_grad=tensor.requires_grad)
                        for name, tensor in (('q', roped_query), ('k', ctx_k), ('v', ctx_v))}
            cuda = (dict(current_device=torch.cuda.current_device(),
                         stream=torch.cuda.current_stream(roped_query.device).cuda_stream,
                         autocast=torch.is_autocast_enabled('cuda')) if roped_query.is_cuda else {})
            trace('attention_enter', self._wave_layer_idx, backend=_WAVE_STATE['attn'],
                  tensors=geometry, cuda=cuda)
        if _WAVE_STATE["attn"] in ("fa2", "fa3"):
            if ctx_k.dtype != roped_query.dtype or ctx_v.dtype != roped_query.dtype:
                raise ValueError(f"{_WAVE_STATE['attn']} Q/K/V dtype mismatch after WP exchange: "
                                 f"Q={roped_query.dtype}, K={ctx_k.dtype}, V={ctx_v.dtype}")
            out = _WAVE_STATE["fa_func"](roped_query, ctx_k, ctx_v)
        elif _WAVE_STATE["attn"] == "sa":
            promotion = getattr(exch, 'kv_promotion', None)
            out = _WAVE_STATE["fa_func"](
                roped_query, ctx_k, ctx_v, tensor_layout="NHD", is_causal=False,
                **({'kv_promotion': promotion} if promotion is not None else {})
            )
            if promotion is not None:
                exch.kv_promotion = None
        else:
            out = self.attn(roped_query, ctx_k, ctx_v)
        if trace is not None:
            trace('attention_returned', self._wave_layer_idx)
        release = getattr(exch, 'release', None)
        if release is not None:
            release(self._wave_layer_idx)
        if trace is not None:
            trace('release_returned', self._wave_layer_idx)
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
        if not self.cfg.dummy_weights and self.cfg.dit_scale in (None, '1.3b'):
            from pathlib import Path
            if not self.cfg.gen_ckpt or not Path(self.cfg.gen_ckpt).is_file():
                raise ValueError('Real-weight inference requires an existing generator checkpoint; no base-weight fallback')
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
        if self.cfg.attention_backend in ("fa2", "fa3"):
            enable_flash_attention(self.cfg.attention_backend)
        elif self.cfg.attention_backend == "sa":
            enable_sage()

        from sglang.multimodal_gen.runtime.server_args import (
            ServerArgs,
            set_global_server_args,
        )
        from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline

        from wave_rt.distributed.compat import (
            disable_rf_config_patch,
            init_wave_dist,
        )

        # Construct sglang with its SDPA backend; FA2/FA3 DiT modules are replaced
        # below before execution, without invoking sglang's kernel dispatcher.
        sgl_attn = (
            "torch_sdpa"
            if self.cfg.attention_backend in ("fa2", "fa3", "sa")
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
                # and (b) create collectives outside WaveRT's schedule. This
                # disables DiT FSDP; T5's separate offload mesh is suppressed by
                # local_text_encoder_loading below.
                "use_fsdp_inference": False,
            }
        )
        # Clear HSDP layout inferred by ServerArgs; model replicas are rank-local.
        server_args.hsdp_shard_dim = None
        # Direct text-stage execution below owns local whole-module residency.
        # Do not let auto layerwise policy install a second set of text hooks.
        server_args.layerwise_offload_components = [
            name for name in (server_args.layerwise_offload_components or [])
            if not name.startswith('text_encoder')
        ]
        server_args.text_encoder_cpu_offload = not self.cfg.serve
        # Current sgldiff DecodingStage resolves precision from pipeline_config
        # again and ignores its vae_dtype argument. Set the loading AND execution
        # policy, so replicated decode matches the BF16 streaming VAE workers.
        server_args.pipeline_config.vae_precision = "bf16"
        # override RF checkpoint (path lives on the pipeline_config)
        server_args.pipeline_config.rolling_forcing_checkpoint_path = self.cfg.gen_ckpt
        server_args.pipeline_config.rolling_forcing_use_ema = True
        # System-level scaling test: build a LARGER random-init DiT and skip the RF
        # overlay (no checkpoint exists at that scale).  Must patch the loader BEFORE
        # build_pipeline; point the RF path at a missing file so the overlay is
        # skipped cleanly (initialize_pipeline falls back to base weights).
        if self.cfg.dummy_weights or self.cfg.dit_scale not in (None, "1.3b"):
            from wave_rt.runtime.dummy_model import install as install_dummy_dit
            install_dummy_dit(self.cfg.dit_scale or "1.3b", verbose=(self.rank == 0),
                              seed=self.cfg.weight_seed, dims=self.cfg.dummy_dims)
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

        self.shared_conditioning = self.world > 1 and self.cfg.uses_optimized_wan
        decode_rank = self.cfg.store_rank
        load_vae = (not self.cfg.dit_only and self.rank == decode_rank
                    and self.cfg.vae_stages == 0)
        load_text = not self.shared_conditioning or self.rank == 0
        from wave_rt.runtime.text_encoder import local_text_encoder_loading
        if self.rank == 0:
            print('[backend] text encoder: rank-local loading/offload, no FSDP mesh', flush=True)
        with local_text_encoder_loading(server_args):
            if not load_vae or not load_text:
                from wave_rt.runtime.loading import build_role_pipeline
                self.pipeline = build_role_pipeline(server_args, load_vae=load_vae, load_text=load_text)
            else:
                self.pipeline = build_pipeline(server_args)
        self.transformer = self.pipeline.get_module("transformer")
        self.vae = self.pipeline.get_module("vae")
        self.scheduler = self.pipeline.get_module("scheduler")

        # tag each self-attention module with its layer index for the closure
        for i, block in enumerate(self.transformer.blocks):
            block.attn1._wave_layer_idx = i
            if self.cfg.attention_backend in ("fa2", "fa3"):
                from wave_rt.runtime.attention import FlashAttentionModule
                block.attn1.attn = FlashAttentionModule(self.cfg.attention_backend)
                block.attn2.attn = FlashAttentionModule(self.cfg.attention_backend)
        self.num_layers = len(self.transformer.blocks)

        self.wp_adapter = None
        if self.cfg.uses_optimized_wan:
            from wave_rt.runtime.model import WaveParallelWan
            cross_backend = 'fa2' if self.cfg.attention_backend == 'fa2' else 'fa3'
            self.wp_adapter = WaveParallelWan(self.transformer, backend=cross_backend).eval()
            if self.rank == 0:
                print('[backend] shared causal WP local optimizations: packed QKV, '
                      f'conditioning/text KV caches, {cross_backend.upper()} cross-attention, clean KV-only/head pruning',
                      flush=True)
            if self.cfg.compile:
                self.wp_adapter.compile(fullgraph=False, dynamic=None)

        # Optional post-load FP8 (W8A8) quantization of the DiT block linears.
        # Done AFTER the bf16 build + RF overlay (sglang's load-time --quantization
        # path is incompatible with the RF load_state_dict; see wave_rt/fp8_linear).
        if self.cfg.quantization == "fp8":
            from wave_rt.runtime.fp8 import quantize_transformer_fp8
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
        if self.wp_adapter is not None:
            self.rf_stage.transformer = self.wp_adapter
        self.decode_stage = self.pipeline.stages[4] if load_vae else None

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
        batch = self.pipeline.stages[0](batch, self.server_args)
        from wave_rt.runtime.text_encoder import encode_local_text
        if self.shared_conditioning:
            from wave_rt.distributed.conditioning import broadcast_tree
            from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import TextEncodingStage
            payload = None
            if self.rank == 0:
                batch = encode_local_text(self.pipeline.stages[1], batch, self.server_args, self.device)
                payload = {name: getattr(batch, name, None) for name in TextEncodingStage.deduplicated_output_fields}
            payload = broadcast_tree(payload, self.device)
            for name, value in payload.items():
                setattr(batch, name, value)
        else:
            batch = encode_local_text(self.pipeline.stages[1], batch, self.server_args, self.device)
        stage = self.pipeline.stages[2]
        if self.cfg.uses_optimized_wan and self.rank != 0:
            spec = stage.get_latent_preparation_spec(batch, self.server_args, batch.batch_size,
                stage.get_forward_latent_num_frames(batch, self.server_args), self.device)
            # Only rank 0 injects initial noise into WP. Other ranks need geometry
            # for metadata and receive their actual inputs through migration.
            batch.latents = torch.zeros((*spec.shape[:2], 1, *spec.shape[3:]),
                device=spec.device, dtype=spec.dtype).expand(spec.shape)
        batch = stage(batch, self.server_args)

        self.batch = batch
        self.ctx = self.rf_stage._prepare_causal_dmd_forward_context(
            batch, self.server_args
        )
        self.num_token_per_frame = self.rf_stage.num_token_per_frame
        self.rf_schedule = None
        if self.cfg.uses_optimized_wan:
            from wave_rt.runtime.schedule import RFSchedule
            from sglang.multimodal_gen.runtime.platforms import current_platform
            self.rf_schedule = RFSchedule(self.ctx.scheduler, self.dsl, self.device,
                torch.float64 if current_platform.is_float64_supported() else torch.float32,
                fused_pointwise=(self.cfg.uses_optimized_wan and self.cfg.attention_backend == 'sa'
                                 and self.num_attention_heads == 12 and self.attention_head_dim == 128))
        if self.wp_adapter is not None:
            # Include the actual context-noise timestep, not just assumed zero.
            times = torch.cat((self.dsl, self.dsl.new_tensor([
                self.server_args.pipeline_config.context_noise])))
            self.wp_adapter.reset()
            self.wp_adapter.conditioning.configure(times)
            ph, pw = self.transformer.patch_size[1:]
            self.wp_adapter.prepare_geometry(num_frames, self.ctx.height // ph,
                                             self.ctx.width // pw, self.device)

        # Wave/CP own history and text KV. Only small native plumbing objects
        # remain; never allocate the unused full-window native caches.
        self.rf_stage.causal_kv_cache = self.rf_stage._allocate_causal_kv_cache(
            batch_size=self.ctx.batch_size, kv_cache_size=1,
            num_attention_heads=self.num_attention_heads,
            attention_head_dim=self.attention_head_dim,
            dtype=self.ctx.target_dtype, device=self.device)
        self.rf_stage.crossattn_cache = [None] * self.num_layers

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
        if self.wp_adapter is not None:
            timestep_bf = self.wp_adapter.timestep_tensor(b, npb, step_value, self.device)
        else:
            timestep_bf = torch.full([b, npb], float(step_value), device=self.device, dtype=torch.float32)
        x0_btchw, _ = self.predict_x0(
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

    def predict_x0(self, batch, server_args, **kwargs):
        if self.rf_schedule is not None:
            from wave_rt.runtime.schedule import predict_x0
            return predict_x0(self.rf_stage, self.rf_schedule, batch, server_args, **kwargs)
        return self.rf_stage._predict_x0_per_frame_btchw(batch, server_args, **kwargs)

    @torch.no_grad()
    def forward_store(self, clean_bcthw: torch.Tensor, step_value: float,
                      start_frame: int) -> None:
        if self.wp_adapter is None:
            self.forward_chunk(clean_bcthw, step_value, start_frame)
            return
        ctx, stage = self.ctx, self.rf_stage
        frames = clean_bcthw.shape[2]
        if hasattr(self.wp_adapter, 'timestep_tensor'):
            times = self.wp_adapter.timestep_tensor(clean_bcthw.shape[0], frames, step_value, self.device)
        else:
            times = torch.full((clean_bcthw.shape[0], frames), float(step_value),
                               device=self.device, dtype=torch.float32).long()
        metadata = stage._build_causal_attn_metadata(
            self.batch, self.server_args, current_timestep=0,
            raw_latent_shape=(frames, ctx.height, ctx.width), device=self.device)
        # Enter the same autocast and RF forward context, but bypass flow->x0:
        # the adapter returns None after publishing the last layer's K/V.
        stage._forward_causal_transformer(
            self.batch, latent_model_input=clean_bcthw.to(ctx.target_dtype),
            prompt_embeds=ctx.prompt_embeds, timestep=times,
            kv_cache=stage.causal_kv_cache, crossattn_cache=stage.crossattn_cache,
            current_start_tokens=start_frame * self.num_token_per_frame,
            start_frame=start_frame, image_kwargs=ctx.image_kwargs,
            pos_cond_kwargs=ctx.pos_cond_kwargs, current_timestep=0,
            attn_metadata=metadata, target_dtype=ctx.target_dtype,
            autocast_enabled=ctx.autocast_enabled, updating_cache=True)

    @torch.no_grad()
    def decode(self, latents_bcthw: torch.Tensor) -> torch.Tensor:
        """VAE decode using the BF16 policy set before component loading."""
        if self.decode_stage is None:
            raise RuntimeError("This rank does not own a VAE decoder")
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
