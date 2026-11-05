"""Actual sgldiff Wan2.1 decoder parity (no transport or kernel doubles).

Run from the repository root on the Linux sgldiff environment:
    python -m scripts.check_wave_vae_stream --device cuda --dtype bfloat16
The small model retains all three spatial and both temporal upsampling stages.
Missing sgldiff dependencies fail the command; there is no fallback decoder.
"""
import argparse

import torch

from wave_rt.pipelines.vae_stream import DecoderStage, balanced_partition, unit_flops, unscale_latents


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    args = parser.parse_args()
    from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEArchConfig, WanVAEConfig
    from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan

    torch.manual_seed(2026)
    config = WanVAEConfig(arch_config=WanVAEArchConfig(base_dim=8, num_res_blocks=1))
    config.load_encoder = False
    config.use_parallel_encode = False
    config.use_parallel_decode = False
    vae = AutoencoderKLWan(config).to(device=args.device, dtype=getattr(torch, args.dtype)).eval()
    units, costs = unit_flops(vae, (2, 3))
    cuts = balanced_partition(costs, 3)
    stages = [DecoderStage(units[a:b]) for a, b in zip(cuts, cuts[1:])]
    # Different request lengths catch stale per-stage first-frame flags/cache.
    for frames in (9, 3, 9):
        z = torch.randn(1, 16, frames, 2, 3, device=args.device, dtype=getattr(torch, args.dtype))
        scaled = unscale_latents(vae, z)
        expected = vae.decode(scaled)
        for stage in stages:
            stage.reset()
        outputs = []
        for start in range(0, frames, 3):
            x = vae.post_quant_conv(scaled[:, :, start:start + 3])
            for frame in range(x.shape[2]):
                h = x[:, :, frame:frame + 1]
                for stage in stages:
                    h = stage(h)
                outputs.append(h)
        actual = torch.cat(outputs, dim=2).float().clamp(-1, 1)
        assert actual.shape[2] == 4 * frames - 3, actual.shape
        # Same operators and ordering: exact equality, not a loose image PSNR.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f'PASS: {frames} latent frames -> {actual.shape[2]} pixels, cuts={cuts}', flush=True)


if __name__ == '__main__':
    main()
