"""Run actual Wave Sage and compiled W8A8 kernels before loading T5.

This is an approximate kernel smoke check, not a video-quality acceptance test.
"""
import torch
import torch.nn.functional as F

from wave_rt.runtime.attention import Attention
from wave_rt.runtime.sage import SagePrefixAttention
from sageattention import sageattn
from wave_rt.runtime.fp8 import WaveFp8Linear


def check(actual, expected, label):
    if not torch.isfinite(actual).all():
        raise AssertionError(f'{label}: nonfinite output')
    error = ((actual.float() - expected.float()).square().mean().sqrt()
             / expected.float().square().mean().sqrt().clamp_min(1e-6)).item()
    if error > .1:
        raise AssertionError(f'{label}: relative RMS error {error:.4f} exceeds 0.1')
    print(f'PASS {label}: relative RMS error {error:.4f}', flush=True)


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no reference fallback')
    torch.manual_seed(2026)
    sage, exact = SagePrefixAttention(sageattn), Attention('fa3')
    for heads in (12, 40):
        q = torch.randn(1, 96, heads, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 160, heads, 128, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        k[:, :64] += 1  # Include a nonzero key mean.
        expected = exact(q, k, v)
        check(sage(q, k, v), expected, f'Sage H={heads}')
    for width in (1536, 5120):
        w = torch.randn(3 * width, width, device='cuda', dtype=torch.bfloat16) * .02
        bias = torch.randn(3 * width, device='cuda', dtype=torch.bfloat16) * .02
        x = torch.randn(32, width, device='cuda', dtype=torch.bfloat16)
        linear = WaveFp8Linear(w, bias, compile_fwd=True)
        check(linear(x)[0], F.linear(x, w, bias), f'compiled FP8 QKV D={width}')
        check(linear.forward_rows(x, width)[0], F.linear(x, w[width:], bias[width:]),
              f'compiled FP8 KV-only D={width}')
        del w, bias, x, linear
    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
