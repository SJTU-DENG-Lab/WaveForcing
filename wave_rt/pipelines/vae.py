"""WaveRT VAE finalization.

Two paths:
  * finalize_on_store  -- Phase-1 SERIAL decode on the store rank (vae_stages==0).
  * vae_stage          -- n-stage FLOPs-balanced STREAMING pipeline (vae_stages>0):
    the store rank pushes each finalized chunk (numpy, BTCHW) into an mp.Queue as it
    drains off the diagonal; n independent VAE-stage procs (own NCCL world, disjoint
    from the diffusion world) decode frame-by-frame with per-stage feat_cache, so the
    VAE is overlapped with the ongoing diffusion instead of a serial tail.

Decode uses the selected model's sgldiff Wan VAE in BF16. WaveRT partitions its
actual decoder modules and preserves sgldiff's causal feature-cache semantics.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

from wave_rt import bench
from wave_rt.runtime.backend import WaveBackend
from wave_rt.config import WaveConfig

# WAVE_TIMELINE: dump per-stage per-frame run_seg intervals for the Gantt profiler
# (scripts/wp_timeline.py).  Launch timing (no per-frame sync -> doesn't perturb the
# streaming pipeline); shows the VAE pipeline flow vs the diffusion ticks.
import json as _json  # noqa: E402
_TIMELINE = os.environ.get("WAVE_TIMELINE", "") not in ("", "0", "false")
_TL_DIR = os.environ.get("WAVE_TIMELINE_DIR", "/tmp/wrt_timeline")


# --------------------------------------------------------------- p2p (VAE world)
# ------------------------------------------------------- n-stage streaming VAE
def vae_stage(grp_rank: int, gpu: int, q, cfg: WaveConfig, out_dir: str, meta_q,
              req_q=None, phase_barrier=None, request_barrier=None) -> None:
    """One stage of the n-stage streaming sgldiff Wan VAE pipeline, BF16.

    stage 0: pull finalized-chunk latents (BTCHW) from ``q`` -> conv2 -> run this
             stage's decoder units per frame -> p2p to stage 1.
    middle : recv -> run units -> p2p to next.
    last   : recv -> run units (incl. head) -> pixel -> collect -> save video + timing.
    Each stage owns its feat_cache; the split is FLOPs-balanced (vae_pipe)."""
    torch.cuda.set_device(gpu)
    os.environ["LOCAL_RANK"] = str(gpu)
    dev = f"cuda:{gpu}"
    n = cfg.vae_stages
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{cfg.vae_port}",
        rank=grp_rank, world_size=n,
    )
    torch.set_grad_enabled(False)
    threads = int(os.environ.get('WAVE_THREADS', '1') or '1')
    if threads > 0:
        torch.set_num_threads(threads)

    from wave_rt.pipelines import vae_stream as vp

    vae = vp.load_vae(cfg, dev)
    if getattr(cfg, 'vae_cuts', None) is not None:
        units = vp.decoder_units(vae)
        cuts = list(cfg.vae_cuts)
        if len(cuts)!=n+1 or cuts[0]!=0 or cuts[-1]!=len(units) or any(a>=b for a,b in zip(cuts,cuts[1:])):
            raise ValueError('Fixed VAE cuts must strictly partition all decoder units')
    elif cfg.vae_partition == "time":
        units = vp.decoder_units(vae)
        # FLOPs mis-estimates this VAE (high-res upsample blocks: equal FLOPs, ~6x
        # slower).  rank0 times each unit under the real streaming decode and
        # broadcasts the cuts (timing noise would desync stages otherwise).
        if grp_rank == 0:
            _, tms = vp.unit_times(vae, (cfg.height // 8, cfg.width // 8))
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
        units, flops = vp.unit_flops(vae, (cfg.height // 8, cfg.width // 8))
        cuts = vp.balanced_partition(flops, n)
    if grp_rank == 0:
        print(f"[wave_rt/vae] sgldiff decoder partition cuts={cuts}", flush=True)
    segment = vp.DecoderStage(units[cuts[grp_rank]:cuts[grp_rank + 1]])
    is_first = grp_rank == 0
    is_last = grp_rank == n - 1
    # A stage owns only its decoder partition; stage 0 also needs normalization
    # metadata and post_quant_conv. Drop the full decoder and unused encoder.
    from types import SimpleNamespace
    frontend = (SimpleNamespace(config=vae.config, post_quant_conv=vae.post_quant_conv)
                if is_first else None)
    del units, vae
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    def process_one_request(req_out_dir, measured=True, phase=0, frames=None):
        """Decode one video: consume this request's chunk stream (stage0 from the
        mp.Queue, others from p2p) until the per-request None end-marker, then the
        last stage saves the video.  Fresh feat_cache per request (no cross-video leak)."""
        segment.reset()
        run_seg = segment
        save_video = measured and cfg.save_video and bool(req_out_dir)
        outs = [] if is_last and save_video else None
        from wave_rt.pipelines.vae_transport import FrameChannel
        incoming = FrameChannel(grp_rank - 1, dev) if not is_first else None
        outgoing = FrameChannel(grp_rank + 1, dev) if not is_last else None
        expected_frames = cfg.num_frames if frames is None else frames
        if request_barrier is not None:
            request_barrier.wait()
        torch.cuda.synchronize()
        profile = None
        if measured and getattr(cfg, 'runtime_profile', False):
            from wave_rt.denoiser.profile import RuntimeProfile
            profile = RuntimeProfile(f'vae_rank{grp_rank}', os.path.join(req_out_dir, 'profile'),
                start=cfg.profile_start, ticks=cfg.profile_ticks,
                stage=grp_rank, gpu=gpu, phase=phase, frames=expected_frames, cuts=cuts, role='vae')
            profile.attach(segment)
            profile.activate()
        from wave_rt.denoiser.output_trace import OutputTrace, enabled as output_trace_enabled
        trace = (OutputTrace(f'vae_stage{grp_rank}', stage=grp_rank, gpu=gpu, phase=phase,
                  frames=expected_frames, num_frames_per_block=cfg.num_frames_per_block,
                  cuts=cuts, stage_count=n) if measured and output_trace_enabled() else None)
        completion_events = []
        completion_anchor = None
        if measured and is_last:
            completion_anchor = torch.cuda.Event(enable_timing=True)
            anchor_lo = time.perf_counter()
            completion_anchor.record()
            completion_anchor.synchronize()
            anchor_hi = time.perf_counter()
        if phase_barrier is not None:
            phase_barrier.wait()
        torch.cuda.reset_peak_memory_stats(gpu)
        t0 = None
        nfr = 0
        tl = []              # WAVE_TIMELINE: (frame_idx, t_in, t_out) per run_seg
        if is_first:
            while True:
                if trace:
                    queue_wait = trace.begin('latent_queue_get')
                item = q.get()
                if trace:
                    queue_wait['chunk'] = item[0] if item is not None else None
                    trace.end(queue_wait)
                if item is None:
                    if nfr != expected_frames:
                        raise ValueError(f'VAE expected {expected_frames} latent frames, got {nfr}')
                    break
                _c, arr = item
                if trace:
                    prepare = trace.begin('latent_input_prepare', chunk=_c, gpu=True)
                from wave_rt.pipelines.latent_output import latent_from_payload
                lat = latent_from_payload(arr, dev)                  # BTCHW (1,npb,C,H,W)
                z = lat.permute(0, 2, 1, 3, 4)                        # -> BCTHW
                x = frontend.post_quant_conv(vp.unscale_latents(frontend, z))
                if trace:
                    trace.end(prepare)
                for f in range(x.shape[2]):
                    if profile is not None:
                        profile.at(nfr//cfg.num_frames_per_block, nfr//cfg.num_frames_per_block)
                    _ta = time.perf_counter() if _TIMELINE else 0.0
                    if trace:
                        compute = trace.begin('vae_compute', chunk=_c, frame=nfr, gpu=True)
                    h = run_seg(x[:, :, f:f + 1])
                    if trace:
                        trace.end(compute)
                    if _TIMELINE:
                        tl.append((nfr, _ta, time.perf_counter()))
                    if is_last:
                        if t0 is None:
                            t0 = time.perf_counter()
                        if outs is not None:
                            outs.append(((h.float().clamp(-1, 1)*0.5+0.5)*255).to(torch.uint8).cpu())
                        nfr += 1
                        if measured and nfr % cfg.num_frames_per_block == 0:
                            done = torch.cuda.Event(enable_timing=True); done.record()
                            completion_events.append(done)
                    else:
                        if trace:
                            send = trace.begin('vae_send', chunk=_c, frame=nfr, gpu=True)
                        outgoing.send(h)
                        if trace:
                            trace.end(send)
                        nfr += 1
        else:
            for _ in range(expected_frames):
                if profile is not None:
                    profile.at(nfr//cfg.num_frames_per_block, nfr//cfg.num_frames_per_block)
                if trace:
                    recv = trace.begin('vae_recv', chunk=nfr//cfg.num_frames_per_block, frame=nfr, gpu=True)
                h = incoming.recv()
                if trace:
                    trace.end(recv)
                if is_last and t0 is None:
                    t0 = time.perf_counter()
                _ta = time.perf_counter() if _TIMELINE else 0.0
                if trace:
                    compute = trace.begin('vae_compute', chunk=nfr//cfg.num_frames_per_block, frame=nfr, gpu=True)
                h = run_seg(h)
                if trace:
                    trace.end(compute)
                if _TIMELINE:
                    tl.append((nfr, _ta, time.perf_counter()))
                if is_last:
                    if outs is not None:
                        outs.append(((h.float().clamp(-1, 1)*0.5+0.5)*255).to(torch.uint8).cpu())
                    nfr += 1
                    if measured and nfr % cfg.num_frames_per_block == 0:
                        done = torch.cuda.Event(enable_timing=True); done.record()
                        completion_events.append(done)
                else:
                    if trace:
                        send = trace.begin('vae_send', chunk=nfr//cfg.num_frames_per_block, frame=nfr, gpu=True)
                    outgoing.send(h)
                    if trace:
                        trace.end(send)
                    nfr += 1

        torch.cuda.synchronize()
        t_end = time.perf_counter()
        peak = torch.tensor([torch.cuda.max_memory_allocated(gpu),
                             torch.cuda.max_memory_reserved(gpu)], dtype=torch.float64, device=dev)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if is_last:
            torch.cuda.synchronize()
            wall = (t_end - t0) * 1000 if t0 else 0.0
            npb = cfg.num_frames_per_block
            print(f"[wave_rt/vae] n={n} decoded {nfr} latent-frames, wall {wall:.0f}ms "
                  f"({wall/max(1,nfr):.1f}ms/frame, {wall/max(1,nfr//npb):.1f}ms/chunk)", flush=True)
            if outs and cfg.save_video and req_out_dir:
                # Encode the uint8 chunks without concatenating a second full video.
                bench.save_video_chunks(outs, os.path.join(req_out_dir, "video.mp4"))
                print(f"[wave_rt/vae] saved video -> {req_out_dir}/video.mp4", flush=True)
            if meta_q is not None and (measured or phase_barrier is None):
                if phase_barrier is None:
                    meta_q.put(("vae", wall, t_end))
                else:
                    meta_q.put(("vae_measure", phase, wall, t_end, *peak.tolist(), nfr))

        if trace:
            trace.dump(request_end_s=t_end, decoded_latent_frames=nfr)
        if measured and meta_q is not None and phase_barrier is not None:
            detail = dict(stage=grp_rank, gpu=gpu, cuts=cuts, frames=nfr,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(gpu),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(gpu))
            if is_last:
                detail.update(chunk_completion_ms=[completion_anchor.elapsed_time(e) for e in completion_events],
                              anchor_host_lower_s=anchor_lo, anchor_host_upper_s=anchor_hi,
                              completion_timing='CUDA events on final decoder stream; no per-chunk synchronize')
            meta_q.put(('vae_detail', detail))
        if profile is not None:
            profile.finish(request_end_s=t_end, decoded_latent_frames=nfr)

        if _TIMELINE and tl:
            os.makedirs(_TL_DIR, exist_ok=True)
            with open(os.path.join(_TL_DIR, f"vae_s{grp_rank}.json"), "w") as _f:
                _json.dump(dict(stage=grp_rank, is_first=is_first, is_last=is_last,
                                nfr=nfr, events=tl), _f)
        segment.reset()
        if request_barrier is not None:
            request_barrier.wait()
        if phase_barrier is not None:
            phase_barrier.wait()

    if req_q is None:
        process_one_request(out_dir)                       # one-shot
    else:
        while True:                                        # serving: load once, many requests
            req = req_q.get()
            if req is None:
                break
            process_one_request(req.get("out") or out_dir,
                                measured=not req.get("warmup", False), phase=req.get("phase", 0),
                                frames=int(req.get('num_frames', cfg.num_frames)))

    dist.barrier()
    dist.destroy_process_group()


def finalize_on_store(
    be: WaveBackend,
    cfg: WaveConfig,
    latents_bcthw: torch.Tensor,
    diffusion_ms: float,
    tick_ms: list[float],
    *, save: bool = True,
) -> None:
    print(
        f"[wave_rt/store] decoding {latents_bcthw.shape[2]} latent frames "
        f"({latents_bcthw.shape[2] // 3} chunks) ...",
        flush=True,
    )

    t0 = time.perf_counter()
    video = be.decode(latents_bcthw)  # B,C,T,H,W on the decoder's device
    if save and cfg.save_video:
        video = video.mul(255).clamp_(0, 255).to(torch.uint8).cpu()
    torch.cuda.synchronize()
    vae_ms = (time.perf_counter() - t0) * 1000.0
    e2e_ms = diffusion_ms + vae_ms

    pixel_frames = video.shape[2]
    # DecodingStage returns (B, C, T, H, W); convert to (T, H, W, C) for imageio.
    video_thwc = video[0].permute(1, 2, 3, 0) if save and cfg.save_video else None
    fps = pixel_frames / (e2e_ms / 1000.0) if e2e_ms else 0.0
    print(
        f"[wave_rt/store] VAE {vae_ms:.0f} ms, e2e {e2e_ms:.0f} ms, "
        f"{pixel_frames} pixel frames, {fps:.1f} fps",
        flush=True,
    )

    if not save:
        return
    out_dir = bench.run_dir(cfg.out_root, cfg.task, cfg.run_tag)
    from wave_rt.runtime.model import runtime_metadata
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
        extra=dict(runtime_metadata(cfg), decoded_frames=pixel_frames,
                   output_boundary='decoded_cpu_uint8' if cfg.save_video else 'decoded_gpu_tensor',
                   fps_with_vae=fps, pixel_frames=pixel_frames),
    )
    msg = f"[wave_rt/store] metrics -> {out_dir}"
    if psnr is not None:
        msg += f"  PSNR={psnr:.2f}dB"
    print(msg, flush=True)
