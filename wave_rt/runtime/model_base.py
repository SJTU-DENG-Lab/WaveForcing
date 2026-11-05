"""Local Wan forward and request caches for full-model Wave replicas.

The denoiser owns all cross-GPU KV communication. This module has no sequence
or tensor sharding and creates no process groups.
"""
from collections import OrderedDict
from types import MethodType
import torch
from torch import nn
from .attention import Attention
from .conditioning import ConditioningCache


class LocalWan(nn.Module):
    def __getattr__(self, name):
        # RF stages inspect config/dtype/device on their transformer when
        # preparing a fresh request. Preserve that surface after attachment.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)

    def __init__(self, model, backend="fa3"):
        super().__init__()
        if tuple(model.patch_size)[0] != 1:
            raise ValueError("Rolling forcing requires temporal patch size 1")
        if any(b.local_num_heads != model.num_attention_heads for b in model.blocks):
            raise ValueError("Wave requires unsharded sgldiff model weights")
        self.model = model
        self.cross_attention = Attention(backend)
        self.rope_tables = OrderedDict()
        self.text_kv = [None] * len(model.blocks)
        self.conditioning = ConditioningCache()
        self._active = None
        self._layout_key = None
        self._qk_norms = []
        # Instance-only hooks: do not change sgldiff classes or the WP backend.
        for index, block in enumerate(model.blocks):
            norms = (getattr(block, 'norm_q', None), getattr(block, 'norm_k', None))
            supported = all(norm is not None and hasattr(norm, 'weight')
                            and (hasattr(norm, 'variance_epsilon') or isinstance(norm, nn.RMSNorm))
                            and getattr(norm, 'variance_size_override', None) is None
                            and norm.weight.numel() == model.hidden_size for norm in norms)
            self._qk_norms.append(norms if supported else None)
            if supported:
                for norm in norms:
                    original = norm.forward
                    def defer_norm(module, x, _original=original):
                        # Normalization moves into the preparation kernel, after
                        # all three projections and before any communication.
                        return x if x.is_cuda else _original(x)
                    norm.forward = MethodType(defer_norm, norm)
            def self_attention(module, q, k, v, *args, _index=index, **kwargs):
                if self._active is None:
                    raise RuntimeError("Wave attention called outside its model forward")
                frames, height, width, start, clean = self._active
                prepared = self._prepare_qkv(q, k, v, _index)
                return self.joint(q, k, v, _index, frames, height, width, start,
                                  clean, None, **({'prepared': prepared} if prepared is not None else {}))
            block.attn1.forward = MethodType(self_attention, block.attn1)

            def cross_attention(module, x, context, context_lens=None,
                                crossattn_cache=None, _index=index):
                q = module.norm_q(module.to_q(x)[0]).unflatten(
                    2, (module.local_num_heads, module.head_dim))
                if self.text_kv[_index] is None:
                    k = module.norm_k(module.to_k(context)[0]).unflatten(
                        2, (module.local_num_heads, module.head_dim))
                    v = module.to_v(context)[0].unflatten(
                        2, (module.local_num_heads, module.head_dim))
                    self.text_kv[_index] = k, v
                k, v = self.text_kv[_index]
                return module.to_out(self.cross_attention(q, k, v).flatten(2))[0]
            block.attn2.forward = MethodType(cross_attention, block.attn2)

    def reset(self):
        self.joint.reset()
        self.conditioning.reset()
        self.text_kv = [None] * len(self.model.blocks)
        self._active = None

    @torch.no_grad()
    def forward(self, hidden_states, encoder_hidden_states, timestep,
                encoder_hidden_states_image=None, kv_cache=None,
                crossattn_cache=None, current_start=0, cache_start=0,
                start_frame=0, updating_cache=False):
        if self._active is not None:
            raise RuntimeError("Wave model does not permit overlapping forwards")
        if encoder_hidden_states_image is not None:
            raise ValueError("Wave adapter supports T2V only")
        model = self.model
        b, _, frames, height, width = hidden_states.shape
        pt, ph, pw = model.patch_size
        h, w = height//ph, width//pw
        if height % ph or width % pw:
            raise ValueError("Spatial dimensions must divide the patch size")
        if not 3 <= frames <= 96 or frames % 3 or timestep.shape != (b, frames):
            raise ValueError("Joint inference requires frame-varying timesteps for 1 through 32 three-frame chunks")
        if current_start != start_frame*h*w:
            raise ValueError("Token offset and absolute frame offset disagree")
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        layout_key = (b, h, w, hidden_states.dtype, hidden_states.device)
        if self._layout_key != layout_key:
            self.rope_tables.clear()
            self._layout_key = layout_key
        self._active = frames, h, w, start_frame, updating_cache
        try:
            x = model.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
            temb, proj, context = self._conditioning(timestep, encoder_hidden_states)
            proj = proj.unflatten(1, (6, model.hidden_size)).unflatten(0, timestep.shape)
            for index, block in enumerate(model.blocks):
                if updating_cache and index == len(model.blocks) - 1:
                    k, v = self._clean_projection(index, x, proj)
                    prepared = self._prepare_qkv(k, k, v, index, write_only=True)
                    self.joint(k, k, v, index, frames, h, w, start_frame, True,
                               None, write_only=True,
                               **({'prepared': prepared} if prepared is not None else {}))
                    # Clean-store callers consume cache writes only. No fabricated
                    # output tensor or output projection/gather is necessary.
                    return None
                x = self._run_block(index, x, context, proj, kv_cache=None,
                          crossattn_cache=None, current_start=current_start,
                          cache_start=cache_start, updating_cache=updating_cache,
                          post_patch_height=h, post_patch_width=w)
            shift, scale = self._head_modulation(temb, timestep)
            x = model.proj_out(model.norm_out(x, shift, scale))
            x = x.reshape(b, frames, h, w, pt, ph, pw, -1)
            return x.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        finally:
            self._active = None
            for block in model.blocks:
                if hasattr(block, '_wave_qkv'):
                    block._wave_qkv.reset()
