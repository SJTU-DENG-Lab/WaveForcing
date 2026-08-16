"""WaveRT backend smoke test (single process, wp_size==1).

Validates the whole sglang reuse path with NO wavefront:
  build_pipeline -> RF ckpt load -> prefix stages (text + latent) ->
  forward_chunk (falls through to the ORIGINAL attention since exchange_fn is
  None) -> VAE decode.  If this passes, the wavefront can be layered on top.

Run:
  cd .
  LD_LIBRARY_PATH=".venv/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH" \
    .venv/bin/python scripts/wrt_smoke_backend.py --num-frames 6
"""

import argparse
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-frames", type=int, default=6)  # 2 chunks of 3
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--decode", action="store_true", help="also run VAE decode")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29699")

    from wave_rt.config import WaveConfig
    from wave_rt.runtime.backend import WaveBackend

    cfg = WaveConfig(
        rf_step=0,
        wp_size=1,  # single process
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        seed=0,
    )
    be = WaveBackend(cfg, rank=0)
    t0 = time.perf_counter()
    be.init()
    print(f"[smoke] backend.init OK in {time.perf_counter() - t0:.1f}s", flush=True)

    npb = cfg.num_frames_per_block
    noise = be.full_noise_bcthw  # (B, C, nf, H, W)
    print(f"[smoke] full noise {tuple(noise.shape)}, dsl={be.dsl.tolist()}", flush=True)

    # run the first chunk through the ORIGINAL attention path at the highest step
    chunk0 = noise[:, :, 0:npb]
    step0 = float(be.dsl[0].item())
    t1 = time.perf_counter()
    x0 = be.forward_chunk(chunk0, step0, start_frame=0)
    torch.cuda.synchronize()
    print(
        f"[smoke] forward_chunk OK -> x0 {tuple(x0.shape)} in "
        f"{time.perf_counter() - t1:.2f}s (mean={x0.float().mean():.4f}, "
        f"nan={torch.isnan(x0).any().item()})",
        flush=True,
    )

    if args.decode:
        # decode the single chunk's x0 as a sanity check (BTCHW -> BCTHW)
        lat = x0.permute(0, 2, 1, 3, 4).contiguous()
        t2 = time.perf_counter()
        vid = be.decode(lat)
        torch.cuda.synchronize()
        print(
            f"[smoke] decode OK -> {tuple(vid.shape)} in "
            f"{time.perf_counter() - t2:.2f}s",
            flush=True,
        )

    be.shutdown()
    print("[smoke] DONE", flush=True)


if __name__ == "__main__":
    main()
