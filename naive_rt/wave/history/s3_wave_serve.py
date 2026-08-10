"""
s3_wave_serve.py -- Milestone 3: 5-rank diffusion + independent VAE decode process.

Decouples VAE decode from diffusion via an mp.Queue numpy bridge: rank 4
(store) pushes finalized chunk latents as numpy arrays through the queue,
and a separate decode_server process on GPU 5 decodes them as they arrive.
This hides VAE latency behind diffusion, achieving ~3.3x speedup / 45 fps.

Expected numbers (252 latent frames, H200):
  - Diffusion: ~17.8 s
  - VAE (pipelined, hidden): ~15 s
  - End-to-end: ~22.4 s  ->  ~45 fps (1008 pixel frames / 22.4 s)

Run:
  cd .
  python -m naive_rt.wave.history.s3_wave_serve --num_output_frames 252

Architecture:
  - 5 diffusion workers (mp.Process, NCCL world=5, GPUs 0-4)
  - 1 decode_server (mp.Process, GPU 5, receives via mp.Queue)
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import OrderedDict
import naive_rt.rolling_forcing.configs as _rf_configs_pkg
from omegaconf import OmegaConf

from naive_rt.rolling_forcing.pipeline.rolling_forcing_inference import (
    CausalInferencePipeline,
)
from naive_rt.bench import harness
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


def diffusion_worker(rank, world, port, q, ns, meta_q=None):
    """5-rank systolic wavefront diffusion."""
    args = ns
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
    sd = torch.load(
        args.gen_ckpt, map_location="cpu", weights_only=False
    )["generator_ema"]
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

    wl = WaveLayerPipeline(
        pipe, context_noise=float(getattr(cfg, "context_noise", 0.0))
    )
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
            # Push to decode queue as numpy (mp.Queue bridge)
            q.put((c, my["latent"].detach().float().cpu().numpy()))

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
            f"[s3/dist5] {world} ranks, {num_blocks} chunks, {num_ticks} ticks, "
            f"diffusion {total:.0f} ms, "
            f"mean tick {sum(tick_ms)/len(tick_ms):.1f} ms",
            flush=True,
        )
        if meta_q is not None:
            meta_q.put(("diff", total, t_start, list(tick_ms)))
    if rank == 4:
        q.put(None)  # sentinel to stop decode_server
    dist.barrier()
    dist.destroy_process_group()


def decode_server(gpu, q, dump, out_dir="", meta_q=None):
    """Independent VAE decode process, receives latent chunks via mp.Queue."""
    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"
    torch.set_grad_enabled(False)

    from naive_rt.rolling_forcing.utils.wan_wrapper import WanVAEWrapper
    from naive_rt.bench import harness
    import einops

    vae = WanVAEWrapper().to(dev, torch.bfloat16).eval()
    vae.model.clear_cache()
    pix = []
    lats = []
    ms = []
    n = 0
    t0 = None

    while True:
        item = q.get()
        if item is None:
            break
        if t0 is None:
            t0 = time.perf_counter()
        c, arr = item
        lat = torch.from_numpy(arr).to(dev, dtype=torch.bfloat16)
        lats.append(lat.detach().cpu())  # for PSNR (outside the VAE timer below)
        _v = time.perf_counter()
        v = vae.decode_to_pixel(lat, use_cache=True)
        v = ((v * 0.5 + 0.5).clamp(0, 1) * 255)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - _v) * 1000)
        pix.append(v.to("cpu"))
        n += 1

    wall = (time.perf_counter() - t0) * 1000 if t0 else 0
    t_end = time.perf_counter()
    print(
        f"[s3/decode] decoded {n} chunks, wall {wall:.0f} ms, "
        f"mean/chunk {sum(ms)/max(1, len(ms)):.1f} ms",
        flush=True,
    )
    full = None
    if pix:
        full = einops.rearrange(torch.cat(pix, dim=1), "b t c h w -> b t h w c")[0]
    if dump and full is not None:
        harness.save_video(full, dump)
        print(f"[s3/decode] saved {tuple(full.shape)} -> {dump}", flush=True)
    # structured output (post-timing): video + latents to out_dir; timing to meta_q
    if out_dir and full is not None:
        harness.save_video(full, os.path.join(out_dir, "video.mp4"))
        if lats:
            torch.save(torch.cat(lats, dim=1), os.path.join(out_dir, "latents.pt"))
    if meta_q is not None:
        meta_q.put(("vae", wall, t_end))


def main():
    ap = argparse.ArgumentParser(
        description="s3: 5-rank diffusion + independent VAE decode process"
    )
    ap.add_argument("--num_output_frames", type=int, default=252)
    ap.add_argument("--gen_ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=29650)
    ap.add_argument("--save_video", default="",
                    help="Alias for --dump: save decoded video to this path")
    ap.add_argument("--dump", default="",
                    help="If set, save the decoded video to this path")
    ap.add_argument("--out_root", default="",
                    help="If set, write structured video+metrics.json under out_root/task/run_tag")
    ap.add_argument("--task", default="03_wave_decoupledVAE")
    ap.add_argument("--run_tag", default="s3")
    ap.add_argument("--ref_latents", default="", help="baseline latents.pt for PSNR")
    ap.add_argument("--baseline_e2e", type=float, default=0.0, help="baseline e2e s for speedup")
    args = ap.parse_args()

    dump_path = args.dump or args.save_video
    out_dir = harness.run_dir(args.out_root, args.task, args.run_tag) if args.out_root else ""

    mp.set_start_method("spawn", force=True)
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass

    q = mp.Queue(maxsize=64)
    meta_q = mp.Queue()
    procs = []
    for r in range(5):
        procs.append(
            mp.Process(target=diffusion_worker, args=(r, 5, args.port, q, args, meta_q))
        )
    procs.append(
        mp.Process(target=decode_server, args=(5, q, dump_path, out_dir, meta_q))
    )

    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wall = (time.perf_counter() - t0) * 1000
    pixel_frames = args.num_output_frames * 4
    fps = pixel_frames / (wall / 1000.0)
    print(
        f"[s3/main] end-to-end wall (incl. load) {wall:.0f} ms, "
        f"{pixel_frames} pixel frames, {fps:.1f} fps",
        flush=True,
    )

    # --- assemble structured metrics (post-run; system-monotonic perf_counter) ---
    if out_dir:
        meta = {}
        while not meta_q.empty():
            item = meta_q.get()
            meta[item[0]] = item[1:]
        diff_ms = meta.get("diff", [None])[0]
        t_start = meta.get("diff", [None, None])[1] if "diff" in meta else None
        tick_ms = meta.get("diff", [None, None, None])[2] if "diff" in meta else None
        vae_ms = meta.get("vae", [None])[0]
        t_end = meta.get("vae", [None, None])[1] if "vae" in meta else None
        e2e_ms = ((t_end - t_start) * 1000) if (t_start and t_end) else None
        ref_lat = os.path.join(out_dir, "latents.pt")
        psnr = None
        if args.ref_latents and os.path.isfile(args.ref_latents) and os.path.isfile(ref_lat):
            import torch as _t
            psnr = harness.latent_psnr(_t.load(ref_lat, map_location="cpu"),
                                       _t.load(args.ref_latents, map_location="cpu")).get("psnr_db")
        e2e_s = (e2e_ms / 1000.0) if e2e_ms else None
        harness.write_metrics(
            out_dir, task=args.task, run_tag=args.run_tag, method="wave_decoupled",
            nstep=4, num_output_frames=args.num_output_frames, num_gpus=6, seed=args.seed,
            diffusion_s=(diff_ms / 1000.0) if diff_ms else None,
            vae_s=(vae_ms / 1000.0) if vae_ms else None, end_to_end_s=e2e_s,
            per_tick_ms=tick_ms, psnr_db=psnr,
            speedup_vs_baseline=(args.baseline_e2e / e2e_s) if (args.baseline_e2e and e2e_s) else None,
        )
        print(f"[s3/main] metrics -> {out_dir}" + (f" PSNR={psnr:.2f}dB" if psnr else ""), flush=True)


if __name__ == "__main__":
    main()
