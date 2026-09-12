"""Explicit asset roots and runtime settings for reference training."""

from pathlib import Path
import os
import sys

_model_root = None


def configure_model_root(path):
    global _model_root
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model_root is not a directory: {root}")
    _model_root = root
    return root


def model_directory(name):
    if _model_root is None:
        raise RuntimeError("Configure model_root before constructing Wan models")
    if Path(name).name != name or name in (".", ".."):
        raise ValueError(f"Expected a model directory name, got {name!r}")
    path = _model_root / name
    if not path.is_dir():
        raise FileNotFoundError(f"Missing model directory: {path}")
    return str(path)


def configure_reference_runtime(config):
    """Set the reference module's knobs before its first heavy import.

    RF/CRF/SF still switch the module mask during forward/backward, exactly as
    in the reference. These settings specify the restored baseline and backend.
    """
    configure_model_root(config.model_root)
    overrides = [key for key in ("RF_LPIPS_W", "RF_LPIPS_SUBSET", "RF_OFFP_CTX_BLOCKS")
                 if os.environ.get(key)]
    if overrides:
        raise ValueError("Remove inherited algorithm overrides; use the resolved config: "
                         + ", ".join(overrides))
    mode = getattr(config, "attention_mode", "full")
    if mode not in ("full", "strict"):
        raise ValueError(f"Unsupported attention_mode: {mode}")
    flex = bool(getattr(config, "flex_attention", True))
    os.environ["RF_BLOCK_CAUSAL"] = "strict" if mode == "strict" else "0"
    os.environ["RF_FLEX_ATTN"] = "1" if flex else "0"
    os.environ["RF_FLEX_COMPILE"] = "1" if getattr(config, "flex_compile", True) else "0"
    module = sys.modules.get("wf_training.wan.modules.causal_model")
    if module is not None:
        module._RF_BLOCK_CAUSAL = mode == "strict"
        module._RF_FLEX_ATTN = flex
        module._RF_USE_FLEX = mode == "strict" and flex and module.FLEX_ATTENTION_AVAILABLE
