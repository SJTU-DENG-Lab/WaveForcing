"""Dummy larger-DiT support for system-level scaling tests.

We have no rolling-forcing checkpoint for a bigger Wan, but the *wavefront system*
(pipeline-parallel diffusion + streaming VAE + NCCL KV-exchange) can still be
exercised with a scaled-up transformer built from config with random weights.
Output is meaningless; the point is timing/memory/comm at scale.

Mechanism (all monkeypatches, sglang stays pristine):
  1. patch ``get_diffusers_component_config`` (in the transformer loader's
     namespace) to enlarge the DiT arch dims it reads from disk.
  2. patch ``maybe_load_fsdp_model`` to build the model from that scaled config
     and materialize it with random weights, SKIPPING the (shape-mismatched)
     safetensors load.
  3. the RF overlay is skipped automatically by pointing --gen-ckpt at a missing
     path (see WanRollingForcingPipeline.initialize_pipeline).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# arch presets: (num_attention_heads, attention_head_dim, num_layers, ffn_dim).
# 1.3b is the real Wan2.1-T2V-1.3B (baseline); 14b matches Wan2.1-T2V-14B.
_PRESETS = {
    "1.3b": dict(num_attention_heads=12, attention_head_dim=128, num_layers=30, ffn_dim=8960),
    "5b":   dict(num_attention_heads=24, attention_head_dim=128, num_layers=30, ffn_dim=13824),
    "14b":  dict(num_attention_heads=40, attention_head_dim=128, num_layers=40, ffn_dim=13824),
}


def scale_to_dims(scale: str) -> dict:
    if scale in _PRESETS:
        return dict(_PRESETS[scale])
    raise ValueError(f"unknown --dit-scale {scale!r}; choices: {list(_PRESETS)}")


def _random_materialize(model: nn.Module, device: torch.device,
                        dtype: torch.dtype) -> nn.Module:
    """Move a meta-device model to `device` with random params (skip real weights).
    Values are finite (params ~N(0,0.02), buffers zeroed) so kernels run at the
    right shapes without NaN/OOB -- correctness is irrelevant for a timing test."""
    model = model.to_empty(device=device)
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.normal_(0, 0.02)
            else:
                p.zero_()
        for b in model.buffers():
            b.zero_()  # zero is a safe index for any positional/gather buffers
    return model.to(dtype)


def install(scale: str, verbose: bool = False) -> None:
    """Patch the transformer loader to build a scaled random-init DiT."""
    from sglang.multimodal_gen.runtime.loader.component_loaders import (
        transformer_loader as tl,
    )
    from sglang.multimodal_gen.runtime.loader.fsdp_load import set_default_torch_dtype

    dims = scale_to_dims(scale)

    if not getattr(tl, "_wave_dummy_cfg_patched", False):
        orig_getcfg = tl.get_diffusers_component_config

        def patched_getcfg(component_path):
            c = orig_getcfg(component_path=component_path)
            # only enlarge the causal Wan DiT config
            if c.get("_class_name") == "CausalWanTransformer3DModel":
                c.update(dims)
            return c

        tl.get_diffusers_component_config = patched_getcfg
        tl._wave_dummy_cfg_patched = True

    if not getattr(tl, "_wave_dummy_load_patched", False):
        def patched_load(model_cls, init_params, weight_dir_list, device,
                         param_dtype, **kw):
            default_dtype = param_dtype if param_dtype else torch.bfloat16
            with set_default_torch_dtype(default_dtype), torch.device("meta"):
                model = model_cls(**init_params)
            model = _random_materialize(model, device, default_dtype)
            if verbose:
                n = sum(p.numel() for p in model.parameters())
                print(f"[wave-dummy] built {scale} DiT random-init: "
                      f"{n/1e9:.2f}B params (weight load skipped)", flush=True)
            return model

        tl.maybe_load_fsdp_model = patched_load
        tl._wave_dummy_load_patched = True
