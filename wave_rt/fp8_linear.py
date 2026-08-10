"""Post-load FP8 (W8A8) quantization of the DiT block linears.

Why here (not sglang's ``--quantization fp8``):
  1. sglang's online-fp8 path quantizes at *load* time, so its Linear layers own
     transposed fp8 params + scale tensors.  Rolling Forcing then overlays the
     ``longvideo.pt`` generator weights via a plain ``load_state_dict`` of bf16
     ``[out, in]`` tensors -> shape/layout mismatch -> crash.  Quantizing AFTER
     the full bf16 build + RF overlay side-steps that entirely.
  2. sglang's dynamic-fp8 uses *per-token (rowwise)* activation scaling, which on
     this box hits a slow ``_scaled_mm`` path (measured ~1.0x vs bf16).  We use
     *per-tensor* scaling -> the fast path (measured 1.5-1.7x on Wan1.3B GEMMs,
     1.8-1.9x on Wan14B).

Scheme: W8A8 e4m3, weight per-tensor static scale (computed once at swap time),
activation per-tensor *dynamic* scale (one amax reduction per forward -> no
calibration).  Near-lossless per step; long-video drift is model-inherent.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

_E4M3_MAX = 448.0
_FP8 = torch.float8_e4m3fn
# Fusing the activation quant via torch.compile helps the GEMM but, inside the
# lockstep wavefront, per-module compilation blocks the KV-exchange collective
# long enough to trip the NCCL watchdog.  Off by default; WAVE_FP8_COMPILE=1 to
# opt in (only safe with a pre-tick warmup that compiles outside the collective).
_COMPILE_DEFAULT = os.environ.get("WAVE_FP8_COMPILE", "0") not in ("", "0", "false")


class WaveFp8Linear(nn.Module):
    """Drop-in for a degree-1 sgl (Replicated/Column/Row)ParallelLinear.

    Matches their forward contract: takes x, returns ``(output, None)`` (bias is
    fused in, as when skip_bias_add is False -- the DiT default)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None,
                 compile_fwd: bool | None = None) -> None:
        super().__init__()
        self.out_dtype = weight.dtype
        self.out_features = weight.shape[0]
        w = weight.detach().to(torch.float32)
        w_scale = (w.abs().amax() / _E4M3_MAX).clamp(min=1e-8)
        # store weight as [out, in] fp8 contiguous; pass .t() (a [in, out]
        # column-major view) to _scaled_mm as required for the B operand.
        self.register_buffer("w_fp8", (w / w_scale).to(_FP8))
        self.register_buffer("w_scale", w_scale.reshape(()).to(torch.float32))
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(self.out_dtype))
        else:
            self.bias = None
        # Fuse the activation quant (amax + div + cast) with _scaled_mm.  Eager,
        # the extra elementwise/reduction kernels erase the GEMM win on this
        # launch-sensitive box; compiled, the big FFN GEMMs net ~1.1-1.35x.
        self._fwd = (torch.compile(self._raw, dynamic=False)
                     if (_COMPILE_DEFAULT if compile_fwd is None else compile_fwd)
                     else self._raw)

    def _raw(self, x2: torch.Tensor) -> torch.Tensor:
        a_scale = (x2.abs().amax() / _E4M3_MAX).clamp(min=1e-8)
        x_fp8 = (x2 / a_scale).to(_FP8)
        return torch._scaled_mm(
            x_fp8, self.w_fp8.t(),
            scale_a=a_scale.reshape(()).to(torch.float32),
            scale_b=self.w_scale,
            bias=self.bias if self.bias is not None else None,
            out_dtype=self.out_dtype,
        )

    def forward(self, x: torch.Tensor):
        shape = x.shape
        y = self._fwd(x.reshape(-1, shape[-1]))
        return y.reshape(*shape[:-1], self.out_features), None


# module attribute names to quantize inside each transformer block.
#
# scope="ffn" (legacy default): only the big FFN GEMMs (K/N=8960@1.3B, 13824@14B).
#   On 1.3B the attention projections (1536x1536) are too small -- the per-forward
#   activation-quant overhead dominates the tiny matmul and fp8 regresses (~0.4x
#   eager).  Kept as the safe default.
# scope="all": every 2D-weight linear inside each block -- attn1 q/k/v/out,
#   attn2 (cross) projections, and ffn.  Follows LongLive/fouroversix NVFP4 policy
#   for the Wan DiT: quantize all block linears, keep norms/embeds/head in high
#   precision (norms have 1D/no weight -> auto-excluded by the dim()==2 guard;
#   patch/time/text embeds + proj_out head live OUTSIDE .blocks -> never visited).
#   On 14B the attn projections are 5120x5120 -- big enough that fp8 (compiled)
#   nets a win, so "all" + WAVE_FP8_COMPILE=1 is the path toward the projection.
_TARGET_SUFFIXES = (
    "ffn.fc_in", "ffn.fc_out",
)


def _is_quantizable_linear(mod: nn.Module) -> bool:
    """A block leaf we can swap: sgl (Replicated/Column/Row/Merged/QKV)ParallelLinear
    or nn.Linear with a 2D weight.  Norms (1D/no weight) and containers are excluded."""
    if isinstance(mod, WaveFp8Linear):
        return False
    w = getattr(mod, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        return False
    return type(mod).__name__.endswith("Linear")


def _get_weight_bias(mod: nn.Module):
    """Extract [out,in] weight + optional bias from an sgl linear (bias may be a
    Parameter or None; to_out is wrapped as a container in some blocks)."""
    w = getattr(mod, "weight", None)
    if w is None:
        return None, None
    b = getattr(mod, "bias", None)
    if isinstance(b, nn.Parameter):
        b = b.data
    elif not isinstance(b, torch.Tensor):
        b = None
    return w.data, b


def quantize_transformer_fp8(transformer: nn.Module, verbose: bool = False,
                             scope: str | None = None) -> int:
    """Swap targeted linears in ``transformer.blocks`` to WaveFp8Linear in place.
    Returns the number of layers swapped.

    scope: "ffn" (default; WAVE_FP8_SCOPE env overrides) swaps only ffn.fc_in/fc_out;
    "all" swaps every 2D-weight linear in each block (attn1 q/k/v/out, attn2, ffn)."""
    if scope is None:
        scope = os.environ.get("WAVE_FP8_SCOPE", "ffn")
    n = 0
    blocks = getattr(transformer, "blocks", None)
    if blocks is None:
        raise ValueError("transformer has no .blocks to quantize")
    for block in blocks:
        if scope == "all":
            # walk every submodule; swap in place by resolving parent + leaf name.
            targets = [name for name, m in block.named_modules()
                       if name and _is_quantizable_linear(m)]
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
        print(f"[wave-fp8] quantized {n} linears to W8A8 e4m3 (per-tensor, scope={scope})",
              flush=True)
    return n
