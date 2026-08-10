"""WaveRT VAE finalization.

Two paths:
  * finalize_on_store  -- Phase-1 SERIAL decode on the store rank (vae_stages==0).
  * vae_stage          -- n-stage FLOPs-balanced STREAMING pipeline (vae_stages>0):
    the store rank pushes each finalized chunk (numpy, BTCHW) into an mp.Queue as it
    drains off the diagonal; n independent VAE-stage procs (own NCCL world, disjoint
    from the diffusion world) decode frame-by-frame with per-stage feat_cache, so the
    VAE is overlapped with the ongoing diffusion instead of a serial tail.

The decode reuses naive's Wan VAE (WanVAEWrapper, bf16 -- matches the RF reference
and is faster than sgl's fp32 default) + naive_rt.wave.vae_pipe (split by per-layer
FLOPs).  naive VAE decoding sgl-produced latents is scale-compatible (same
latents_mean/std; verified ~40dB vs sgl self-decode).
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

from wave_rt import bench
from wave_rt.backend import WaveBackend
from wave_rt.config import WaveConfig

# WAVE_TIMELINE: dump per-stage per-frame run_seg intervals for the Gantt profiler
# (scripts/wp_timeline.py).  Launch timing (no per-frame sync -> doesn't perturb the
# streaming pipeline); shows the VAE pipeline flow vs the diffusion ticks.
import json as _json  # noqa: E402
_TIMELINE = os.environ.get("WAVE_TIMELINE", "") not in ("", "0", "false")
_TL_DIR = os.environ.get("WAVE_TIMELINE_DIR", "/tmp/wrt_timeline")


# --------------------------------------------------------------- p2p (VAE world)
def _vae_send(t: torch.Tensor, dst: int) -> None:
    t = t.contiguous()
    sh = torch.tensor(list(t.shape), dtype=torch.long, device=t.device)
    dist.send(sh, dst=dst)
    dist.send(t, dst=dst)


def _vae_stop(dst: int, dev) -> None:
    dist.send(torch.zeros(5, dtype=torch.long, device=dev), dst=dst)


def _vae_recv(src: int, dev, dtype=torch.bfloat16):
    sh = torch.empty(5, dtype=torch.long, device=dev)
    dist.recv(sh, src=src)
    if int(sh.sum().item()) == 0:
        return None
    t = torch.empty(torch.Size(sh.tolist()), dtype=dtype, device=dev)
    dist.recv(t, src=src)
    return t


# ------------------------------------------------------- n-stage streaming VAE
def vae_stage(grp_rank: int, gpu: int, q, cfg: WaveConfig, out_dir: str, meta_q,
              req_q=None) -> None:
    """One stage of the n-stage streaming VAE pipeline (naive Wan VAE, bf16).

    stage 0: pull finalized-chunk latents (BTCHW) from ``q`` -> conv2 -> run this
             stage's decoder units per frame -> p2p to stage 1.
    middle : recv -> run units -> p2p to next.
    last   : recv -> run units (incl. head) -> pixel -> collect -> save video + timing.
    Each stage owns its feat_cache; the split is FLOPs-balanced (vae_pipe)."""
    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"
    n = cfg.vae_stages
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{cfg.vae_port}",
        rank=grp_rank, world_size=n,
    )
    torch.set_grad_enabled(False)

    from naive_rt.rolling_forcing.utils.wan_wrapper import WanVAEWrapper
    from naive_rt.wave import vae_pipe as vp

    vae = WanVAEWrapper().to(dev, torch.bfloat16).eval()
    m = vae.model
    dec = m.decoder
    scale = [vae.mean.to(dev, torch.bfloat16), (1.0 / vae.std).to(dev, torch.bfloat16)]

    units = vp.flatten_decode_units(dec)
    _, flops = vp.layer_flops(vae, dev)
    if cfg.vae_partition == "time":
        # FLOPs mis-estimates this VAE (high-res upsample blocks: equal FLOPs, ~6x
        # slower).  rank0 times each unit under the real streaming decode and
        # broadcasts the cuts (timing noise would desync stages otherwise).
        if grp_rank == 0:
            _, tms = vp.measure_unit_times(
                vae, dev, npb=cfg.num_frames_per_block,
                latent_hw=(cfg.height // 8, cfg.width // 8))
            cuts = vp.balanced_partition(tms, n)
            cuts_t = torch.tensor(cuts, dtype=torch.long, device=dev)
        else:
            cuts_t = torch.empty(n + 1, dtype=torch.long, device=dev)
        dist.broadcast(cuts_t, src=0)
        cuts = cuts_t.tolist()
        if grp_rank == 0:
            print(f"[wave_rt/vae] time-balanced cuts={cuts} "
                  f"(per-unit ms/frame: {[round(t, 2) for t in tms]})", flush=True)
    else:
        cuts = vp.balanced_partition(flops, n)
    if grp_rank == 0:
        print(vp.report([nm for nm, _ in units], flops, cuts), flush=True)
    my_units = [L for _, L in units[cuts[grp_rank]:cuts[grp_rank + 1]]]
    is_first = grp_rank == 0
    is_last = grp_rank == n - 1

    def process_one_request(req_out_dir):
        """Decode one video: consume this request's chunk stream (stage0 from the
        mp.Queue, others from p2p) until the per-request None end-marker, then the
        last stage saves the video.  Fresh feat_cache per request (no cross-video leak)."""
        cache = [None] * 256

        def run_seg(x):
            idx = [0]
            for L in my_units:
                x = vp.apply_unit(L, x, cache, idx)
            return x

        outs = [] if is_last else None
        t0 = None
        nfr = 0
        tl = []              # WAVE_TIMELINE: (frame_idx, t_in, t_out) per run_seg
        if is_first:
            while True:
                item = q.get()
                if item is None:
                    if not is_last:
                        _vae_stop(grp_rank + 1, dev)
                    break
                _c, arr = item
                lat = torch.from_numpy(arr).to(dev, torch.bfloat16)   # BTCHW (1,npb,C,H,W)
                z = lat.permute(0, 2, 1, 3, 4)                        # -> BCTHW
                z = z / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)
                x = m.conv2(z)
                for f in range(x.shape[2]):
                    _ta = time.perf_counter() if _TIMELINE else 0.0
                    h = run_seg(x[:, :, f:f + 1])
                    if _TIMELINE:
                        tl.append((nfr, _ta, time.perf_counter()))
                    if is_last:
                        if t0 is None:
                            t0 = time.perf_counter()
                        outs.append(h); nfr += 1
                    else:
                        _vae_send(h, grp_rank + 1)
                        nfr += 1
        else:
            while True:
                h = _vae_recv(grp_rank - 1, dev)
                if h is None:
                    if not is_last:
                        _vae_stop(grp_rank + 1, dev)
                    break
                if is_last and t0 is None:
                    t0 = time.perf_counter()
                _ta = time.perf_counter() if _TIMELINE else 0.0
                h = run_seg(h)
                if _TIMELINE:
                    tl.append((nfr, _ta, time.perf_counter()))
                if is_last:
                    outs.append(h); nfr += 1
                else:
                    _vae_send(h, grp_rank + 1)
                    nfr += 1

        if is_last:
            import einops
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000 if t0 else 0.0
            t_end = time.perf_counter()
            npb = cfg.num_frames_per_block
            print(f"[wave_rt/vae] n={n} decoded {nfr} latent-frames, wall {wall:.0f}ms "
                  f"({wall/max(1,nfr):.1f}ms/frame, {wall/max(1,nfr//npb):.1f}ms/chunk)", flush=True)
            if outs and cfg.save_video and req_out_dir:
                pix = torch.cat(outs, dim=2).clamp(-1, 1)
                v = ((pix * 0.5 + 0.5).clamp(0, 1) * 255)
                vid = einops.rearrange(v, "b c t h w -> b t h w c")[0].byte().cpu()
                bench.save_video(vid, os.path.join(req_out_dir, "video.mp4"))
                print(f"[wave_rt/vae] saved {tuple(vid.shape)} -> {req_out_dir}/video.mp4", flush=True)
            if meta_q is not None:
                meta_q.put(("vae", wall, t_end))

        if _TIMELINE and tl:
            os.makedirs(_TL_DIR, exist_ok=True)
            with open(os.path.join(_TL_DIR, f"vae_s{grp_rank}.json"), "w") as _f:
                _json.dump(dict(stage=grp_rank, is_first=is_first, is_last=is_last,
                                nfr=nfr, events=tl), _f)

    if req_q is None:
        process_one_request(out_dir)                       # one-shot
    else:
        while True:                                        # serving: load once, many requests
            req = req_q.get()
            if req is None:
                break
            process_one_request(req.get("out") or out_dir)

    dist.barrier()
    dist.destroy_process_group()


def finalize_on_store(
    be: WaveBackend,
    cfg: WaveConfig,
    final_latents: dict[int, torch.Tensor],
    diffusion_ms: float,
    tick_ms: list[float],
) -> None:
    idx = sorted(final_latents.keys())
    latents_bcthw = torch.cat([final_latents[i] for i in idx], dim=2)  # (1,C,nf,H,W)
    print(
        f"[wave_rt/store] decoding {latents_bcthw.shape[2]} latent frames "
        f"({len(idx)} chunks) ...",
        flush=True,
    )

    t0 = time.perf_counter()
    video = be.decode(latents_bcthw)  # (B, C, T, H, W) float32 [0,1] on CPU
    torch.cuda.synchronize()
    vae_ms = (time.perf_counter() - t0) * 1000.0
    e2e_ms = diffusion_ms + vae_ms

    pixel_frames = video.shape[2]
    # DecodingStage returns (B, C, T, H, W); convert to (T, H, W, C) for imageio.
    video_thwc = video[0].permute(1, 2, 3, 0).contiguous()
    fps = pixel_frames / (e2e_ms / 1000.0) if e2e_ms else 0.0
    print(
        f"[wave_rt/store] VAE {vae_ms:.0f} ms, e2e {e2e_ms:.0f} ms, "
        f"{pixel_frames} pixel frames, {fps:.1f} fps",
        flush=True,
    )

    out_dir = bench.run_dir(cfg.out_root, cfg.task, cfg.run_tag)
    psnr = bench.finalize_run(
        out_dir,
        task=cfg.task,
        run_tag=cfg.run_tag,
        method="wave_rt",
        nstep=cfg.rf_step,
        num_output_frames=latents_bcthw.shape[2],
        num_gpus=cfg.wp_size,
        diffusion_ms=diffusion_ms,
        vae_ms=vae_ms,
        e2e_ms=e2e_ms,
        latents=latents_bcthw,
        video=video_thwc,
        ref_latents=cfg.ref_latents,
        baseline_e2e=cfg.baseline_e2e,
        per_tick_ms=tick_ms,
        seed=cfg.seed,
    )
    msg = f"[wave_rt/store] metrics -> {out_dir}"
    if psnr is not None:
        msg += f"  PSNR={psnr:.2f}dB"
    print(msg, flush=True)
