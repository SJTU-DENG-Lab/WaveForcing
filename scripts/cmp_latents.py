"""Compare two saved latent tensors (latent-space PSNR), robust to BCTHW vs BTCHW.

Usage:
  .venv/bin/python scripts/cmp_latents.py A.pt B.pt [labelA labelB]
"""
import sys
import torch
from wave_rt import bench


def _to_bcthw(x: torch.Tensor) -> torch.Tensor:
    # normalize to (B, C=16, T, H, W): find the C=16 axis and the T axis (mult of 3)
    x = x.float().cpu()
    if x.dim() != 5:
        return x
    b, d1, d2 = x.shape[0], x.shape[1], x.shape[2]
    if d1 == 16:  # already (B, C, T, H, W)
        return x
    if d2 == 16:  # (B, T, C, H, W) -> (B, C, T, H, W)
        return x.permute(0, 2, 1, 3, 4).contiguous()
    return x


def main() -> None:
    a_path, b_path = sys.argv[1], sys.argv[2]
    la = sys.argv[3] if len(sys.argv) > 3 else a_path
    lb = sys.argv[4] if len(sys.argv) > 4 else b_path
    a = _to_bcthw(torch.load(a_path, map_location="cpu"))
    b = _to_bcthw(torch.load(b_path, map_location="cpu"))
    print(f"A={la} {tuple(a.shape)}  B={lb} {tuple(b.shape)}")
    n = min(a.shape[2], b.shape[2])
    if a.shape[2] != b.shape[2]:
        print(f"[warn] frame count differs ({a.shape[2]} vs {b.shape[2]}); comparing first {n}")
        a, b = a[:, :, :n], b[:, :, :n]
    m = bench.latent_psnr(a, b)
    print(f"PSNR = {m['psnr_db']:.2f} dB   (mse={m['mse']:.4e}, mean_abs={m['mean_abs']:.4e})")


if __name__ == "__main__":
    main()
