"""
s2_wave_dist.py -- Milestone 2: 5-rank NCCL wavefront, VAE still serial.

5 ranks form a systolic wavefront: ranks 0-3 are denoising stages 0-3,
rank 4 is the store rank. Per-layer all_gather lets the freshly finalized
chunk contribute its clean KV in real-time (eliminates 1-tick clean lag).
VAE decode runs serially on rank 4 after all chunks are finalized.

Expected numbers (252 latent frames, H200):
  - Diffusion: ~17.8 s (mean tick ~230 ms)
  - Total (including serial VAE): ~37 s
  - Speed-up vs s0 diffusion: ~3x, but VAE is blocking so end-to-end ~2x

Run:
  cd .
  python -m naive_rt.wave.history.s2_wave_dist --num_output_frames 252

Architecture:
  mp.spawn(diffusion_worker, nprocs=5), rank 4 collects latents and does
  serial VAE decode at the end.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from naive_rt.bench import harness
from collections import OrderedDict
import naive_rt.rolling_forcing.configs as _rf_configs_pkg
from omegaconf import OmegaConf

from naive_rt.rolling_forcing.pipeline.rolling_forcing_inference import (
    CausalInferencePipeline,
)
from naive_rt.wave.layer_pipeline import WaveLayerPipeline
from naive_rt.rolling_forcing.wan.modules.causal_model import causal_rope_apply

_CFG_DIR = os.path.dirname(_rf_configs_pkg.__file__)

DEFAULT_CKPT = "./ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt"
DEFAULT_PROMPT = (
    "A cinematic shot of a fluffy corgi running on a sunny beach, "
    "waves in the background."
)


def det_renoise(scheduler, x0, next_t, chunk, stage, device, B, npb):
    """Deterministic re-noising for reproducibility across ranks."""
    g = torch.Generator(device=device).manual_seed(1000 * chunk + stage)
    nz = torch.randn(
        x0.flatten(0, 1).shape, generator=g, device=device, dtype=x0.dtype
    )
    return scheduler.add_noise(
        x0.flatten(0, 1),
        nz,
        next_t * torch.ones([B * npb], device=device, dtype=torch.long),
    ).unflatten(0, (B, npb))


def diffusion_worker(rank, world, port, args):
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=world,
    )
    torch.set_grad_enabled(False)
    assert world == 5

    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(_CFG_DIR, "default_config.yaml")),
        OmegaConf.load(os.path.join(_CFG_DIR, "rolling_forcing_dmd.yaml")),
    )
    pipe = CausalInferencePipeline(cfg, device=dev)
    sd = torch.load(args.gen_ckpt, map_location="cpu", weights_only=False)["generator_ema"]
    fixed = OrderedDict()
    for k, v in sd.items():
        if k.startswith("model._fsdp_wrapped_module."):
            k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[k] = v
    pipe.generator.load_state_dict(fixed, strict=False)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(dev)
    pipe.generator.to(dev)
    pipe.vae.to(dev)

    wl = WaveLayerPipeline(pipe, context_noise=float(getattr(cfg, "context_noise", 0.0)))
    model = wl.model
    model.freqs = model.freqs.to(dev)
    npb, fsl, Ln = wl.npb, wl.fsl, wl.L
    dsl = wl.dsl.to(dev)

    B = 1
    nf = args.num_output_frames
    num_blocks = nf // npb
    C, Hh, Ww = 16, 60, 104
    max_attn_frames = wl.max_attn // fsl

    cond = pipe.text_encoder(text_prompts=[args.prompt])
    context_text = model.text_embedding(torch.stack([
        torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
        for u in cond["prompt_embeds"]
    ]))
    pipe._initialize_crossattn_cache(B, torch.bfloat16, dev)
    cc = pipe.crossattn_cache

    clean = [[] for _ in range(Ln)]

    gN = torch.Generator(device=dev).manual_seed(args.seed)
    noise_all = torch.randn(
        [B, nf, C, Hh, Ww], generator=gN, device=dev, dtype=torch.bfloat16
    )

    my = dict(latent=None)
    final_latents = {}
    num_ticks = num_blocks + world - 1
    KV_SHAPE = (2, npb * fsl, 12, 128)

    def act_denoise(n, t):
        c = t - n
        return (0 <= n <= 3) and (0 <= c < num_blocks)

    def act_store(t):
        return 0 <= (t - 4) < num_blocks

    def anchor_working(clean_L, grid1, csf, qf):
        if len(clean_L) == 0:
            return [], []
        wmax_frames = max_attn_frames - qf - npb
        wmax_chunks = max(0, wmax_frames // npb)
        anchor = clean_L[0]
        rest = clean_L[1:]
        working = rest[-wmax_chunks:] if wmax_chunks > 0 else []
        wflen = len(working) * npb
        rope_start = csf - wflen - npb
        ak = causal_rope_apply(
            anchor["ku"], grid1, model.freqs, start_frame=rope_start
        ).type_as(anchor["v"])
        return (
            [ak] + [w["kr"] for w in working],
            [anchor["v"]] + [w["v"] for w in working],
        )

    dist.barrier()
    t_start = time.perf_counter()
    tick_ms = []

    for t in range(num_ticks):
        tb = time.perf_counter()
        is_dn = act_denoise(rank, t)
        is_st = (rank == 4) and act_store(t)
        c = t - rank if is_dn else (t - 4 if is_st else -1)

        act_ranks = [n for n in range(4) if act_denoise(n, t)]
        oldest_c = t - max(act_ranks) if act_ranks else 0
        tick_csf = oldest_c * npb
        tick_qf = len(act_ranks) * npb

        # embed
        if is_dn:
            if rank == 0:
                my["latent"] = noise_all[:, c * npb : (c + 1) * npb]
            step = float(dsl[rank].item())
            t_row = torch.full([B, npb], step, device=dev, dtype=torch.float32)
            hx, e_time, e0, grid = wl._embed_chunk(my["latent"], t_row)
            state_x = hx
        elif is_st:
            t_row = torch.full(
                [B, npb], float(wl.context_noise), device=dev, dtype=torch.float32
            )
            hx, e_time, e0, grid = wl._embed_chunk(my["latent"], t_row)
            state_x = hx
        else:
            state_x = None

        dummy = torch.zeros(KV_SHAPE, device=dev, dtype=torch.bfloat16)
        store_kv = []
        grid1 = grid if (is_dn or is_st) else None

        for L in range(Ln):
            block = model.blocks[L]
            if is_dn or is_st:
                rq, rk, ku, v, e = wl._qkv(
                    block, state_x, e0, grid, start_frame=c * npb
                )
                mine = torch.stack([rk[0], v[0]])
            else:
                mine = dummy
            gathered = [
                torch.empty(KV_SHAPE, device=dev, dtype=torch.bfloat16)
                for _ in range(world)
            ]
            dist.all_gather(gathered, mine.contiguous())

            if is_dn:
                inflK = [
                    gathered[n][0].unsqueeze(0)
                    for n in range(4)
                    if act_denoise(n, t)
                ]
                inflV = [
                    gathered[n][1].unsqueeze(0)
                    for n in range(4)
                    if act_denoise(n, t)
                ]
                aK, aV = anchor_working(clean[L], grid1, tick_csf, tick_qf)
                liveK, liveV = [], []
                if act_store(t):
                    liveK = [gathered[4][0].unsqueeze(0)]
                    liveV = [gathered[4][1].unsqueeze(0)]
                state_x = wl._finish(
                    block, state_x, e, rq,
                    torch.cat(aK + liveK + inflK, dim=1),
                    torch.cat(aV + liveV + inflV, dim=1),
                    context_text, cc[L],
                )
            elif is_st:
                aK, aV = anchor_working(clean[L], grid1, c * npb, npb)
                state_x = wl._finish(
                    block, state_x, e, rq,
                    torch.cat(aK + [rk], dim=1),
                    torch.cat(aV + [v], dim=1),
                    context_text, cc[L],
                )
                store_kv.append((rk, ku, v))

        # denoise rank produces x0
        if is_dn:
            xh = model.head(
                state_x,
                e_time.unflatten(dim=0, sizes=(B, npb)).unsqueeze(2),
            )
            flow = torch.stack(model.unpatchify(xh, grid))
            my["x0"] = wl.gen._convert_flow_pred_to_x0(
                flow_pred=flow.permute(0, 2, 1, 3, 4).flatten(0, 1),
                xt=my["latent"].flatten(0, 1),
                timestep=torch.full(
                    [B, npb], step, device=dev
                ).flatten(0, 1),
            ).unflatten(0, (B, npb))

        if is_st:
            final_latents[c] = my["latent"].clone()

        # broadcast finalized chunk clean KV
        sf = torch.zeros(1, device=dev)
        if is_st:
            sf[0] = 1
        dist.all_reduce(sf, op=dist.ReduceOp.MAX)
        if sf[0] > 0:
            cid = t - 4
            KB = (B, npb * fsl, 12, 128)
            for L in range(Ln):
                if is_st:
                    rk, ku, v = store_kv[L]
                    kb = rk.contiguous()
                    kub = ku.contiguous()
                    vb = v.contiguous()
                else:
                    kb = torch.empty(KB, device=dev, dtype=torch.bfloat16)
                    kub = torch.empty(KB, device=dev, dtype=torch.bfloat16)
                    vb = torch.empty(KB, device=dev, dtype=torch.bfloat16)
                dist.broadcast(kb, src=4)
                dist.broadcast(kub, src=4)
                dist.broadcast(vb, src=4)
                clean[L].append(dict(kr=kb, ku=kub, v=vb, cid=cid))
                if len(clean[L]) > 8:
                    clean[L] = clean[L][:1] + clean[L][-7:]

        # migration: send/recv latents between ranks
        reqs = []
        recv_buf = torch.empty(
            [B, npb, C, Hh, Ww], device=dev, dtype=torch.bfloat16
        )
        will_recv = (
            (1 <= rank <= 3 and act_denoise(rank, t + 1))
            or (rank == 4 and act_store(t + 1))
        )
        will_send = is_dn and (rank <= 3)
        if will_recv:
            reqs.append(dist.irecv(recv_buf, src=rank - 1))
        if will_send:
            if rank <= 2:
                payload = det_renoise(
                    wl.scheduler, my["x0"], dsl[rank + 1],
                    c, rank, dev, B, npb,
                )
            else:
                payload = my["x0"]
            reqs.append(dist.isend(payload.contiguous(), dst=rank + 1))
        for r in reqs:
            r.wait()
        if will_recv:
            my["latent"] = recv_buf.clone()

        torch.cuda.synchronize()
        tick_ms.append((time.perf_counter() - tb) * 1000)

    dist.barrier()
    total = (time.perf_counter() - t_start) * 1000

    if rank == 0:
        print(
            f"[s2/dist5] {world} ranks, {num_blocks} chunks, {num_ticks} ticks, "
            f"diffusion {total:.0f} ms, mean tick {sum(tick_ms)/len(tick_ms):.1f} ms",
            flush=True,
        )

    # rank 4: serial VAE decode
    if rank == 4 and final_latents:
        idx = sorted(final_latents.keys())
        all_lat = torch.cat([final_latents[i] for i in idx], dim=1)
        print(f"[s2/rank4] Starting serial VAE decode ({all_lat.shape[1]} latent frames) ...", flush=True)
        vae_t0 = time.perf_counter()
        vid = pipe.vae.decode_to_pixel(all_lat, use_cache=False)
        vid = ((vid * 0.5 + 0.5).clamp(0, 1) * 255)
        torch.cuda.synchronize()
        vae_ms = (time.perf_counter() - vae_t0) * 1000
        total_with_vae = total + vae_ms
        pixel_frames = all_lat.shape[1] * 4
        fps = pixel_frames / (total_with_vae / 1000.0)
        print(
            f"[s2/rank4] VAE {vae_ms:.0f} ms, total {total_with_vae:.0f} ms, "
            f"{pixel_frames} pixel frames, {fps:.1f} fps",
            flush=True,
        )
        if args.save_video:
            from einops import rearrange
            vidhwc = rearrange(vid, "b t c h w -> b t h w c").byte().cpu()[0]
            harness.save_video(vidhwc, args.save_video)
            print(f"[s2/rank4] Saved -> {args.save_video}", flush=True)

        # --- metrics/latents/video (all AFTER timing; never affects timing) ---
        if args.out_root:
            out_dir = harness.run_dir(args.out_root, args.task, args.run_tag)
            psnr = harness.finalize_wave_run(
                out_dir, task=args.task, run_tag=args.run_tag, method="wave_dist",
                nstep=4, num_output_frames=args.num_output_frames, num_gpus=world,
                diffusion_ms=total, vae_ms=vae_ms, e2e_ms=total_with_vae,
                latents=all_lat, video=vid, ref_latents=args.ref_latents,
                baseline_e2e=args.baseline_e2e, per_tick_ms=tick_ms, seed=args.seed)
            print(f"[s2/rank4] metrics -> {out_dir}"
                  + (f" PSNR={psnr:.2f}dB" if psnr else ""), flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(
        description="s2: 5-rank NCCL wavefront, serial VAE"
    )
    ap.add_argument("--num_output_frames", type=int, default=252)
    ap.add_argument("--gen_ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=29650)
    ap.add_argument("--save_video", default="",
                    help="If set, save the output video to this path")
    ap.add_argument("--out_root", default="",
                    help="If set, write structured video+metrics.json under out_root/task/run_tag")
    ap.add_argument("--task", default="02_wave_dist_serialVAE")
    ap.add_argument("--run_tag", default="s2")
    ap.add_argument("--ref_latents", default="", help="baseline latents.pt for PSNR")
    ap.add_argument("--baseline_e2e", type=float, default=0.0, help="baseline e2e s for speedup")
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass

    procs = []
    for r in range(5):
        p = mp.Process(
            target=diffusion_worker, args=(r, 5, args.port, args)
        )
        procs.append(p)

    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wall = (time.perf_counter() - t0) * 1000
    print(f"[s2/main] end-to-end wall {wall:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
