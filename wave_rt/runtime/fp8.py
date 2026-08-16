"""Post-load W8A8 quantization for DiT block linears.

Quantization happens after the Rolling Forcing checkpoint overlay and uses a
static weight scale plus a dynamic per-tensor activation scale.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from wave_rt.runtime.env import env_flag

_E4M3_MAX = 448.0
_FP8 = torch.float8_e4m3fn
_COMPILE_DEFAULT = env_flag("WAVE_FP8_COMPILE")


class WaveFp8Linear(nn.Module):
    """Degree-1 SGLang linear replacement with the same return contract."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        compile_fwd: bool | None = None,
    ) -> None:
        super().__init__()
        self.out_dtype = weight.dtype
        self.out_features = weight.shape[0]
        w = weight.detach().to(torch.float32)
        w_scale = (w.abs().amax() / _E4M3_MAX).clamp(min=1e-8)
        self.register_buffer("w_fp8", (w / w_scale).to(_FP8))
        self.register_buffer("w_scale", w_scale.reshape(()).to(torch.float32))
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(self.out_dtype))
        else:
            self.bias = None
        should_compile = _COMPILE_DEFAULT if compile_fwd is None else compile_fwd
        self._fwd = (
            torch.compile(self._raw, dynamic=False) if should_compile else self._raw
        )

    def _raw(self, x2: torch.Tensor) -> torch.Tensor:
        a_scale = (x2.abs().amax() / _E4M3_MAX).clamp(min=1e-8)
        x_fp8 = (x2 / a_scale).to(_FP8)
        return torch._scaled_mm(
            x_fp8,
            self.w_fp8.t(),
            scale_a=a_scale.reshape(()).to(torch.float32),
            scale_b=self.w_scale,
            bias=self.bias if self.bias is not None else None,
            out_dtype=self.out_dtype,
        )

    def forward(self, x: torch.Tensor):
        shape = x.shape
        y = self._fwd(x.reshape(-1, shape[-1]))
        return y.reshape(*shape[:-1], self.out_features), None


_TARGET_SUFFIXES = (
    "ffn.fc_in",
    "ffn.fc_out",
)


def _is_quantizable_linear(mod: nn.Module) -> bool:
    """Return whether a module is a supported two-dimensional linear."""
    if isinstance(mod, WaveFp8Linear):
        return False
    w = getattr(mod, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        return False
    return type(mod).__name__.endswith("Linear")


def _get_weight_bias(mod: nn.Module):
    """Extract an ``[out, in]`` weight and optional bias."""
    w = getattr(mod, "weight", None)
    if w is None:
        return None, None
    b = getattr(mod, "bias", None)
    if isinstance(b, nn.Parameter):
        b = b.data
    elif not isinstance(b, torch.Tensor):
        b = None
    return w.data, b


def quantize_transformer_fp8(
    transformer: nn.Module, verbose: bool = False, scope: str | None = None
) -> int:
    """Replace selected block linears in place and return the replacement count."""
    if scope is None:
        scope = os.environ.get("WAVE_FP8_SCOPE", "ffn")
    n = 0
    blocks = getattr(transformer, "blocks", None)
    if blocks is None:
        raise ValueError("transformer has no .blocks to quantize")
    for block in blocks:
        if scope == "all":
            targets = [
                name
                for name, m in block.named_modules()
                if name and _is_quantizable_linear(m)
            ]
        else:
            targets = list(_TARGET_SUFFIXES)
        for suffix in targets:
            parent = block
            *path, leaf = suffix.split(".")
            ok = True
            for p in path:
                parent = getattr(parent, p, None)
                if parent is None:
                    ok = False
                    break
            if not ok:
                continue
            mod = getattr(parent, leaf, None)
            if mod is None or isinstance(mod, WaveFp8Linear):
                continue
            w, b = _get_weight_bias(mod)
            if w is None or w.dim() != 2:
                continue
            fp8 = WaveFp8Linear(w, b).to(w.device)
            setattr(parent, leaf, fp8)
            n += 1
    if verbose:
        print(
            f"[wave-fp8] quantized {n} linears to W8A8 e4m3 (per-tensor, scope={scope})",
            flush=True,
        )
    return n
