"""Shared local Wan optimizations for causal WP; transport owns KV placement.

Use local block/conditioning/head implementations without sequence or tensor
sharding. The denoiser owns communication and history pages.
Self-attention still enters WaveBackend's native exchange hook, including its
selected self-attention kernel and profiling. Cross-attention uses explicit FA2/FA3.
"""
import torch

from .model_base import LocalWan
from .projection import pack_wan_qkv
from .compute import AxisRoPE, FixedConditioning, first_norm


class WaveExchange:
    def __init__(self, owner, forwards):
        self.owner, self.forwards = owner, forwards

    def reset(self):
        # WaveDenoiser owns and resets clean history and transport generations.
        pass

    def __call__(self, q, k, v, index, frames, height, width, start, clean,
                 rotate, write_only=False, prepared=None):
        extra = {}
        if prepared is not None:
            q, k, v, rk = prepared
            extra['prepared_rope'] = (q, rk)
        freqs = ((None, None) if prepared is not None else
                 self.owner.rope(frames, height, width, start, v.device))
        return self.forwards[index](
            q, k, v, freqs, None, current_start=start * height * width,
            cache_start=start * height * width, updating_cache=clean,
            tokens_per_frame=height * width, post_patch_height=height,
            post_patch_width=width, write_only=write_only, **extra)


class WaveParallelWan(LocalWan):
    def __init__(self, model, backend='fa3'):
        # Capture the installed WaveBackend hook before the adapter installs its
        # instance dispatch. Keep these bound methods, not the overwritten ones.
        forwards = [block.attn1.forward for block in model.blocks]
        super().__init__(model, backend=backend)
        self.joint = WaveExchange(self, forwards)
        pack_wan_qkv(model)  # checkpoint overlay precedes packing; FP8 follows it
        self.fixed = FixedConditioning()
        self._times = {}
        self._step_hint = None
        self._axes = None
        self._next_normalized = None

    def reset(self):
        super().reset()
        self.fixed.reset()
        self._times.clear()
        self._step_hint = None
        self._next_normalized = None

    def prepare_geometry(self, frames, height, width, device):
        from sglang.multimodal_gen.runtime.platforms import current_platform
        self._axes = AxisRoPE(frames + 3, height, width,
            self.model.hidden_size // self.model.num_attention_heads, device,
            dtype=torch.float64 if current_platform.is_float64_supported() else torch.float32)

    def timestep_tensor(self, batch, frames, step, device):
        from ctypes import c_float
        # Match the RF stage's float32 tensor -> long conversion, including
        # context-noise values just below an integer in Python float64.
        step = int(c_float(step).value)
        self._step_hint = step
        key = batch, frames, step, torch.device(device)
        if key not in self._times:
            self._times[key] = torch.full((batch, frames), int(step), device=device, dtype=torch.long)
        return self._times[key]

    def _conditioning(self, timestep, context):
        entry = self.fixed.get(self.model, self.conditioning, timestep, context, self._step_hint)
        self._modulation, self._head, self._zero = entry[3:]
        return entry[:3]

    def _head_modulation(self, temb, timestep):
        return self._head

    def _input_norm(self, index, x):
        ready, self._next_normalized = self._next_normalized, None
        if ready is not None:
            layer, residual, normalized = ready
            if layer != index or residual is not x:
                raise RuntimeError('WP fused normalization consumed out of order')
            return normalized
        shift, scale = self._modulation[index][:2]
        return first_norm(self.model.blocks[index], x, shift, scale)

    def _finish_block(self, index, addition, gate, x):
        block = self.model.blocks[index]
        if index + 1 == len(self.model.blocks):
            return block.mlp_residual(addition, gate, x).to(x.dtype)
        following = self.model.blocks[index + 1]
        shift, scale = self._modulation[index + 1][:2]
        if x.is_cuda:
            from .kernels import norm_modulate
            residual, normalized = norm_modulate(x, shift, scale, following.norm1.eps,
                                                 addition=addition, gate=gate)
        else:
            # Keep the native CPU oracle while exercising identical ownership
            # and clean-tail control flow in model-level regression tests.
            residual = block.mlp_residual(addition, gate, x).to(x.dtype)
            normalized = first_norm(following, residual, shift, scale)
        self._next_normalized = index + 1, residual, normalized
        return residual

    def _clean_projection(self, index, x, proj):
        block = self.model.blocks[index]
        normalized = self._input_norm(index, x)
        # PackedProjection's KV-only GEMM shares the original packed weights.
        k, v = block.to_k(normalized)[0], block.to_v(normalized)[0]
        k = block.norm_k(k)
        shape = block.local_num_heads, block.dim_head
        return k.unflatten(2, shape), v.unflatten(2, shape)

    def _run_block(self, index, x, context, proj, **kwargs):
        block = self.model.blocks[index]
        _, _, gate, cshift, cscale, cgate = self._modulation[index]
        dtype = x.dtype
        normalized = self._input_norm(index, x)
        owner = block._wave_qkv
        q, k, v = owner.linear(normalized)[0].split(owner.widths, dim=-1)
        q, k = block.norm_q(q), block.norm_k(k)
        shape = block.local_num_heads, block.dim_head
        attended = block.attn1(q.unflatten(2, shape), k.unflatten(2, shape),
                              v.unflatten(2, shape)).flatten(2)
        attended = block.to_out(attended)[0]
        normalized, x = block.self_attn_residual_norm(x, attended, gate, self._zero, self._zero)
        normalized, x = normalized.to(dtype), x.to(dtype)
        attended = block.attn2(normalized, context=context, context_lens=None)
        normalized, x = block.cross_attn_residual_norm(x, attended, 1, cshift, cscale)
        normalized, x = normalized.to(dtype), x.to(dtype)
        return self._finish_block(index, block.ffn(normalized), cgate, x)

    def _prepare_qkv(self, q, k, v, index, write_only=False):
        if not k.is_cuda:
            return None  # CPU numerical reference: native norms/RoPE.
        if self._qk_norms[index] is None:
            raise ValueError('WP requires full-width RMS QK normalization')
        from .backend import _WAVE_STATE
        from .kernels import prepare_qkv
        frames, height, width, start, clean = self._active
        if self._axes is None:
            raise RuntimeError('Prepare WP request RoPE axes before CUDA inference')
        self._axes.check(frames, height, width, start)
        exchange = _WAVE_STATE['exchange_fn']
        target = getattr(exchange, 'prepare_targets', None)
        targets = target(index) if target is not None else None
        raw = (clean and start == 0) or getattr(exchange, 'requires_raw_key', False)
        return prepare_qkv(q, k, v, *self._qk_norms[index], self._axes, start,
                           targets=targets, raw=raw, kv_only=write_only)

    # The RF stage inspects this signature to filter optional conditioning
    # kwargs. A catch-all **kwargs incorrectly advertises support for masks
    # that neither CausalWan nor LocalWan accepts.
    def forward(self, hidden_states, encoder_hidden_states, timestep,
                encoder_hidden_states_image=None, kv_cache=None,
                crossattn_cache=None, current_start=0, cache_start=0,
                start_frame=0, updating_cache=False):
        try:
            return super().forward(
                hidden_states, encoder_hidden_states, timestep,
                encoder_hidden_states_image=encoder_hidden_states_image,
                kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=current_start, cache_start=cache_start,
                start_frame=start_frame, updating_cache=updating_cache)
        finally:
            self._step_hint = None
            self._next_normalized = None

    def rope(self, frames, height, width, start, device):
        key = (frames, height, width, start, device)
        if key not in self.rope_tables:
            from sglang.multimodal_gen.runtime.models.dits import causal_wanvideo as cw
            d = self.model.hidden_size // self.model.num_attention_heads
            # Match native WP's CPU construction, precision and placement exactly.
            cos, sin = cw.get_rotary_pos_embed(
                (frames, height, width), self.model.hidden_size,
                self.model.num_attention_heads,
                [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
                dtype=torch.float64 if cw.current_platform.is_float64_supported() else torch.float32,
                rope_theta=10000, start_frame=start)
            self.rope_tables[key] = (cos.to(device).float(), sin.to(device).float())
            if len(self.rope_tables) > 32:
                self.rope_tables.popitem(last=False)
        self.rope_tables.move_to_end(key)
        return self.rope_tables[key]


def optimization_metadata(enabled, *, paged=None):
    paged = enabled if paged is None else paged
    return dict(conditioning_table=enabled, text_kv_cache=enabled,
                packed_qkv_projection=enabled, clean_tail_pruning=enabled,
                clean_output_head_pruning=enabled,
                fused_qk_norm_rope_pack=enabled,
                final_address_kv=paged, fused_norm_modulation=enabled,
                fused_mlp_residual_next_norm=enabled,
                fixed_stage_modulation=enabled, axis_rope=enabled,
                fused_context_assembly=paged, batched_ready_waits=False,
                explicit_peer_kv_copy=False,
                paged_buffer_policy='one_slot_per_layer' if paged else None,
                cross_attention_backend='fa3' if enabled else 'torch_sdpa')


def runtime_metadata(cfg):
    from wave_rt.runtime.provenance import runtime_provenance
    enabled = cfg.uses_optimized_wan
    optimizations = optimization_metadata(enabled, paged=cfg.uses_paged_wan)
    if cfg.attention_backend in ('fa2', 'fa3') or (not enabled and cfg.attention_backend != 'sa'):
        optimizations['cross_attention_backend'] = cfg.attention_backend
    return dict(**runtime_provenance(),
                weights_mode='dummy' if cfg.dummy_weights or cfg.dit_scale not in (None, '1.3b') else 'checkpoint',
                dit_scale=cfg.dit_scale or '1.3b',
                full_warmup_runs=cfg.full_warmup_runs,
                warmup_frames=cfg.warmup_frames,
                local_compute='wan_runtime_v3' if enabled else 'native_wan', optimizations=optimizations,
                cross_attention_backend=optimizations['cross_attention_backend'])
