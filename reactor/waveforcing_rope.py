"""Extend native absolute RoPE positions without growing a history table."""

from __future__ import annotations


def install() -> None:
    import torch
    from naive_rt.rolling_forcing.wan.modules import causal_model as native

    original = native.causal_rope_apply

    def apply(x, grid_sizes, freqs, start_frame=0):
        if all(
            0 <= start_frame and start_frame + int(g[0]) <= freqs.shape[0]
            for g in grid_sizes
        ):
            return original(x, grid_sizes, freqs, start_frame=start_frame)
        heads, half = x.shape[2], x.shape[3] // 2
        temporal_dim = half - 2 * (half // 3)
        _, fy, fx = freqs.split([temporal_dim, half // 3, half // 3], dim=1)
        result = []
        for i, (frames, height, width) in enumerate(grid_sizes.tolist()):
            # Match upstream rope_params' CPU float64 arithmetic and polar call.
            phase = torch.outer(
                torch.arange(start_frame, start_frame + frames),
                1.0
                / torch.pow(
                    10000,
                    torch.arange(0, 2 * temporal_dim, 2)
                    .to(torch.float64)
                    .div(2 * temporal_dim),
                ),
            )
            ft = torch.polar(torch.ones_like(phase), phase).to(x.device)
            angles = torch.cat(
                [
                    ft.view(frames, 1, 1, -1).expand(frames, height, width, -1),
                    fy[:height]
                    .view(1, height, 1, -1)
                    .expand(frames, height, width, -1),
                    fx[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
                ],
                dim=-1,
            ).reshape(frames * height * width, 1, -1)
            count = frames * height * width
            values = torch.view_as_complex(
                x[i, :count].to(torch.float64).reshape(count, heads, -1, 2)
            )
            rotated = torch.view_as_real(values * angles).flatten(2)
            result.append(torch.cat([rotated, x[i, count:]]))
        return torch.stack(result).type_as(x)

    native.causal_rope_apply = apply
