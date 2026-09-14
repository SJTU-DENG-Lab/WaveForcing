"""CPU checkpoint helpers for loading weights before FSDP wrapping.

The caller owns rank coordination: call ``load_model_checkpoint`` only on
rank zero, validate and load its unwrapped model, then let FSDP synchronize
the parameters. These functions never initialize a process group or use CUDA.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


_WRAPPER_COMPONENTS = {
    "_fsdp_wrapped_module", "_checkpoint_wrapped_module", "_orig_mod",
}
_NATIVE_WAN_PREFIXES = (
    "patch_embedding.", "text_embedding.", "time_embedding.",
    "time_projection.", "blocks.", "head.",
)


def load_model_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """Read a trusted local checkpoint on CPU, mapping ZIP tensor storage.

    The old non-ZIP ``torch.save`` format cannot be memory mapped. Retry only
    that specific compatibility error; corrupt files and other loading errors
    must propagate. Mapping avoids eagerly duplicating the whole checkpoint in
    CPU RAM, but the referenced file must remain present while weights are used.
    """
    path = Path(path).expanduser()
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        if "mmap can only be used with files saved with" not in str(error):
            raise
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False, mmap=False)
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("model checkpoint must be a nonempty mapping")
    return checkpoint


def _tensor_mapping(state: Any, description: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{description} must be a nonempty tensor mapping")
    for key, value in state.items():
        if not isinstance(key, str) or not key or not torch.is_tensor(value):
            raise ValueError(f"{description} must contain only named tensors; invalid key {key!r}")
    return state


def generator_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Select raw generator weights and normalize legacy wrapper prefixes.

    Accept a raw state dict, ``{"generator": state}``, or ``{"model": state}``.
    Never silently substitute EMA for the raw generator. Reject ambiguous
    containers and key collisions rather than partially loading a wrong model.
    Returned tensors reference the original checkpoint storage without copies.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("model checkpoint must be a mapping")
    entries = [name for name in ("generator", "model") if name in checkpoint]
    if len(entries) > 1:
        raise ValueError("ambiguous checkpoint: contains both 'generator' and 'model'")
    if entries:
        selected = checkpoint[entries[0]]
    else:
        if any(name in checkpoint for name in ("generator_ema", "ema", "ema_state_dict")):
            raise ValueError("EMA-only checkpoint has no raw generator/model weights")
        selected = checkpoint
    selected = _tensor_mapping(selected, "generator state")

    normalized = {}
    for key, value in selected.items():
        clean_key = ".".join(
            component for component in key.split(".")
            if component not in _WRAPPER_COMPONENTS)
        if not clean_key:
            raise ValueError(f"empty parameter name after removing wrappers: {key!r}")
        if clean_key in normalized:
            raise ValueError(f"parameter name collision after removing wrappers: {clean_key!r}")
        normalized[clean_key] = value

    has_wrapper_prefix = any(key.startswith("model.") for key in normalized)
    has_native_prefix = any(key.startswith(_NATIVE_WAN_PREFIXES) for key in normalized)
    if has_wrapper_prefix and has_native_prefix:
        raise ValueError("generator state mixes native Wan and wrapper 'model.' names")
    if has_native_prefix:
        normalized = {f"model.{key}": value for key, value in normalized.items()}
    return normalized


def validate_load(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Check every key and tensor shape before mutating an unwrapped model.

    Meta parameters are supported. Dtype differences are allowed intentionally:
    loading BF16 initialization into FP32 training parameters is valid. The
    caller must still use ``module.load_state_dict(state, strict=True)``.
    """
    state = _tensor_mapping(state, "model state")
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = [
        f"{key}: checkpoint {tuple(state[key].shape)} != model {tuple(expected[key].shape)}"
        for key in sorted(set(expected) & set(state))
        if state[key].shape != expected[key].shape
    ]
    problems = []
    for label, items in (("missing keys", missing), ("unexpected keys", unexpected),
                         ("shape mismatches", mismatched)):
        if items:
            suffix = f" (and {len(items) - 8} more)" if len(items) > 8 else ""
            problems.append(f"{label}: {items[:8]}{suffix}")
    if problems:
        raise ValueError("checkpoint does not match model; " + "; ".join(problems))
