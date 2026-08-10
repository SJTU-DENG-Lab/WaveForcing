"""
wave_serve3.py -- generic (nstep+1) diffusion wavefront + n-stage VAE pipeline.

Two NCCL worlds, fully decoupled (disjoint rank sets, different TCP ports),
bridged only by mp.Queue (numpy): when diffusion rank S finalizes a chunk,
it q.put(numpy latent) -> VAE stage0 picks it up -> conv2 -> per-frame
p2p through the stages.  Avoids the deadlock of mixed in-job NCCL groups.

GPU layout: diffusion rank r -> cuda:r (0..S-1, S=store); VAE stage g ->
cuda:(S+1+g).  No torchrun needed.
"""
import os, sys, time, argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import naive_rt.rolling_forcing.configs as _rf_configs_pkg
from omegaconf import OmegaConf
from naive_rt.rolling_forcing.pipeline.rolling_forcing_inference import CausalInferencePipeline
from naive_rt.wave.layer_pipeline import WaveLayerPipeline
from naive_rt.rolling_forcing.wan.modules.causal_model import causal_rope_apply
from naive_rt.bench import harness

_CFG_DIR = os.path.dirname(_rf_configs_pkg.__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def det_renoise(scheduler, x0, next_t, chunk, stage, device, B, npb):
    g = torch.Generator(device=device).manual_seed(1000 * chunk + stage)
    nz = torch.randn(x0.flatten(0, 1).shape, generator=g, device=device, dtype=x0.dtype)
    return scheduler.add_noise(x0.flatten(0, 1), nz,
                               next_t * torch.ones([B * npb], device=device, dtype=torch.long)).unflatten(0, (B, npb))

def diffusion_worker(rank, world, port, q, ns, meta_q=None):
    os.chdir(_REPO_ROOT)
    args = ns
    S = args.nstep

    torch.cuda.set_device(rank); dev = f"cuda:{rank}"
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    torch.set_grad_enabled(False)
    assert world == S + 1

    cfg = OmegaConf.merge(OmegaConf.load(os.path.join(_CFG_DIR, "default_config.yaml")), OmegaConf.load(os.path.join(_CFG_DIR, "rolling_forcing_dmd.yaml")))
    cfg.denoising_step_list = [int(x) for x in args.steps.split(",")]
    pipe = CausalInferencePipeline(cfg, device=dev)
    sd = torch.load(args.gen_ckpt, map_location="cpu", weights_only=False)["generator_ema"]
    fixed = {(k.replace("model._fsdp_wrapped_module.", "model.", 1) if k.startswith("model._fsdp_wrapped_module.") else k): v for k, v in sd.items()}
    pipe.generator.load_state_dict(fixed, strict=False)
    pipe = pipe.to(dtype=torch.bfloat16) 
    pipe.text_encoder.to(dev); pipe.generator.to(dev); pipe.vae.to(dev)
    wl = WaveLayerPipeline(pipe, context_noise=float(getattr(cfg, "context_noise", 0.0)))
    model = wl.model; model.freqs = model.freqs.to(dev)
    npb, fsl, Ln = wl.npb, wl.fsl, wl.L
    dsl = wl.dsl.to(dev)
    B = 1; nf = args.num_output_frames; num_blocks = nf // npb; C, Hh, Ww = 16, 60, 104
    max_attn_frames = wl.max_attn // fsl   # 21

    cond = pipe.text_encoder(text_prompts=[args.prompt])
    context_text = model.text_embedding(torch.stack([
        torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))]) for u in cond["prompt_embeds"]]))
    pipe._initialize_crossattn_cache(B, torch.bfloat16, dev); cc = pipe.crossattn_cache

    # clean cache: per-layer list of dict(kr, ku, v, cid) -- replicated across ranks
    clean = [[] for _ in range(Ln)]

    gN = torch.Generator(device=dev).manual_seed(args.seed)
    noise_all = torch.randn([B, nf, C, Hh, Ww], generator=gN, device=dev, dtype=torch.bfloat16)

    my = dict(latent=None)
    final_latents = {}
    num_ticks = num_blocks + world - 1
    KV_SHAPE = (2, npb * fsl, 12, 128)

    def act_denoise(n, t):
        c = t - n
        return (0 <= n <= S - 1) and (0 <= c < num_blocks)

    def act_store(t):
        return 0 <= (t - S) < num_blocks

    def anchor_working(clean_L, grid1, csf, qf, has_live=False):
        """v3 chunk-list: [anchor(re-roped) + working(SWA)]."""
        if len(clean_L) == 0:
            return [], []
        wmax_frames = max_attn_frames - qf - npb - (npb if has_live else 0)
        wmax_chunks = max(0, wmax_frames // npb)
        anchor = clean_L[0]; rest = clean_L[1:]
        working = rest[-wmax_chunks:] if wmax_chunks > 0 else []
        wflen = len(working) * npb
        rope_start = csf - wflen - npb
        ak = causal_rope_apply(anchor['ku'], grid1, model.freqs, start_frame=rope_start).type_as(anchor['v'])
        return [ak] + [w['kr'] for w in working], [anchor['v']] + [w['v'] for w in working]

    dist.barrier(); t_start = time.perf_counter(); tick_ms = []

    for t in range(num_ticks):
        tb = time.perf_counter()
        is_dn = act_denoise(rank, t)
        is_st = (rank == S) and act_store(t)
        c = t - rank if is_dn else (t - S if is_st else -1)

        # oldest in-flight chunk in the window (denoise ranks)
        act_ranks = [n for n in range(S) if act_denoise(n, t)]
        oldest_c = t - max(act_ranks) if act_ranks else 0
        tick_csf = oldest_c * npb
        tick_qf = len(act_ranks) * npb

        # embed
        if is_dn:
            if rank == 0:
                my['latent'] = noise_all[:, c * npb:(c + 1) * npb]
            step = float(dsl[rank].item())
            t_row = torch.full([B, npb], step, device=dev, dtype=torch.float32)
            hx, e_time, e0, grid = wl._embed_chunk(my['latent'], t_row)
            state_x = hx
        elif is_st:
            t_row = torch.full([B, npb], float(wl.context_noise), device=dev, dtype=torch.float32)
            hx, e_time, e0, grid = wl._embed_chunk(my['latent'], t_row)   # clean x0 piped from rank S-1 last tick
            state_x = hx
        else:
            state_x = None
        dummy = torch.zeros(KV_SHAPE, device=dev, dtype=torch.bfloat16)
        store_kv = []   # store rank collects clean (rk, v, ku) per layer

        grid1 = grid if (is_dn or is_st) else None
        for L in range(Ln):
            block = model.blocks[L]
            if is_dn or is_st:
                rq, rk, ku, v, e = wl._qkv(block, state_x, e0, grid, start_frame=c * npb)
                # store rank needs ku (un-roped) for the anchor; recompute un-roped k
                mine = torch.stack([rk[0], v[0]])
            else:
                mine = dummy
            gathered = [torch.empty(KV_SHAPE, device=dev, dtype=torch.bfloat16) for _ in range(world)]
            dist.all_gather(gathered, mine.contiguous())
            if is_dn:
                inflK = [gathered[n][0].unsqueeze(0) for n in range(S) if act_denoise(n, t)]
                inflV = [gathered[n][1].unsqueeze(0) for n in range(S) if act_denoise(n, t)]
                _has_live = act_store(t)
                aK, aV = anchor_working(clean[L], grid1, tick_csf, tick_qf, has_live=_has_live)
                liveK, liveV = [], []
                if _has_live:   # store rank clean KV merged in real time (eliminates lag)
                    liveK = [gathered[S][0].unsqueeze(0)]; liveV = [gathered[S][1].unsqueeze(0)]
                state_x = wl._finish(block, state_x, e, rq, torch.cat(aK + liveK + inflK, dim=1),
                                     torch.cat(aV + liveV + inflV, dim=1), context_text, cc[L])
            elif is_st:
                aK, aV = anchor_working(clean[L], grid1, c * npb, npb)
                state_x = wl._finish(block, state_x, e, rq, torch.cat(aK + [rk], dim=1),
                                     torch.cat(aV + [v], dim=1), context_text, cc[L])
                store_kv.append((rk, ku, v))

        # denoise rank produces x0
        if is_dn:
            xh = model.head(state_x, e_time.unflatten(dim=0, sizes=(B, npb)).unsqueeze(2))
            flow = torch.stack(model.unpatchify(xh, grid))
            my['x0'] = wl.gen._convert_flow_pred_to_x0(
                flow_pred=flow.permute(0, 2, 1, 3, 4).flatten(0, 1), xt=my['latent'].flatten(0, 1),
                timestep=torch.full([B, npb], step, device=dev).flatten(0, 1)).unflatten(0, (B, npb))
        if is_st:
            final_latents[c] = my['latent'].clone()
            q.put((c, my['latent'].detach().float().cpu().numpy()))

        # store rank broadcasts this tick's finalized KV; all ranks append to clean cache
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
                    kb, kub, vb = rk.contiguous(), ku.contiguous(), v.contiguous()
                else:
                    kb = torch.empty(KB, device=dev, dtype=torch.bfloat16)
                    kub = torch.empty(KB, device=dev, dtype=torch.bfloat16)
                    vb = torch.empty(KB, device=dev, dtype=torch.bfloat16)
                dist.broadcast(kb, src=S); dist.broadcast(kub, src=S); dist.broadcast(vb, src=S)
                clean[L].append(dict(kr=kb, ku=kub, v=vb, cid=cid))
                if len(clean[L]) > 8:            # SWA: keep anchor + ~2 recent; prune to prevent OOM
                    clean[L] = clean[L][:1] + clean[L][-7:]

        # handoff: send noisy latent downstream
        reqs = []
        recv_buf = torch.empty([B, npb, C, Hh, Ww], device=dev, dtype=torch.bfloat16)
        will_recv = (1 <= rank <= S - 1 and act_denoise(rank, t + 1)) or (rank == S and act_store(t + 1))
        will_send = is_dn and (rank <= S - 1)
        if will_recv:
            reqs.append(dist.irecv(recv_buf, src=rank - 1))
        if will_send:
            payload = det_renoise(wl.scheduler, my['x0'], dsl[rank + 1], c, rank, dev, B, npb) if rank <= S - 2 else my['x0']
            reqs.append(dist.isend(payload.contiguous(), dst=rank + 1))
        for r in reqs:
            r.wait()
        if will_recv:
            my['latent'] = recv_buf.clone()

        torch.cuda.synchronize(); tick_ms.append((time.perf_counter() - tb) * 1000)

    dist.barrier(); total = (time.perf_counter() - t_start) * 1000
    if rank == 0:
        print(f"[dist5c] {world} ranks, {num_blocks} chunks, {num_ticks} ticks, total {total:.0f}ms, mean tick {sum(tick_ms)/len(tick_ms):.1f}ms", flush=True)
        if meta_q is not None:
            meta_q.put(("diff", total, t_start, list(tick_ms)))
    if rank == S:
        q.put(None)   # sentinel: signal VAE stage0 to stop
    dist.barrier(); dist.destroy_process_group()


# ===================== VAE n-stage pipeline (independent NCCL world) =====================

def _vae_send(t, dst):
    t = t.contiguous()
    sh = torch.tensor(list(t.shape), dtype=torch.long, device=t.device)
    dist.send(sh, dst=dst); dist.send(t, dst=dst)


def _vae_stop(dst, dev):
    dist.send(torch.zeros(5, dtype=torch.long, device=dev), dst=dst)


def _vae_recv(src, dev, dtype=torch.bfloat16):
    sh = torch.empty(5, dtype=torch.long, device=dev); dist.recv(sh, src=src)
    if int(sh.sum().item()) == 0:
        return None
    t = torch.empty(torch.Size(sh.tolist()), dtype=dtype, device=dev); dist.recv(t, src=src)
    return t


def vae_stage(grp_rank, gpu, q, dump, port2, world=3, out_dir="", meta_q=None):
    os.chdir(_REPO_ROOT)
    """n-stage VAE decode pipeline in an independent NCCL world (FLOPs-balanced split via vae_pipe).
    stage0:   pull latent from mp.Queue -> conv2 -> run assigned units per-frame -> p2p downstream
    mid:      recv -> run assigned units -> p2p downstream
    last:     recv -> run assigned units (incl. head) -> pixel -> collect -> save video
    Each stage owns a feat_cache (contiguous slots) that persists across frames and chunks;
    cut points are computed deterministically by every stage (identical result)."""
    torch.cuda.set_device(gpu); dev = f"cuda:{gpu}"
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port2}", rank=grp_rank, world_size=world)
    torch.set_grad_enabled(False)
    from naive_rt.rolling_forcing.utils.wan_wrapper import WanVAEWrapper
    from naive_rt.wave import vae_pipe as vp
    vae = WanVAEWrapper().to(dev, torch.bfloat16).eval()
    m = vae.model; dec = m.decoder
    scale = [vae.mean.to(dev, torch.bfloat16), (1.0 / vae.std).to(dev, torch.bfloat16)]
    n = world

    # FLOPs-balanced split (deterministic -> every stage computes identical cuts)
    units = vp.flatten_decode_units(dec)
    _, flops = vp.layer_flops(vae, dev)
    cuts = vp.balanced_partition(flops, n)
    if grp_rank == 0:
        print(vp.report([nm for nm, _ in units], flops, cuts), flush=True)
    my_units = [L for _, L in units[cuts[grp_rank]:cuts[grp_rank + 1]]]
    cache = [None] * 256

    def run_seg(x):
        idx = [0]
        for L in my_units:
            x = vp.apply_unit(L, x, cache, idx)
        return x

    is_first = (grp_rank == 0)
    is_last = (grp_rank == n - 1)
    outs = [] if is_last else None
    t0 = None
    nfr = 0

    if is_first:
        lats = []
        while True:
            item = q.get()
            if item is None:
                if not is_last:
                    _vae_stop(grp_rank + 1, dev)
                break
            c, arr = item
            lat = torch.from_numpy(arr).to(dev, torch.bfloat16)          # [1,npb,16,60,104]
            lats.append(torch.from_numpy(arr))                            # for PSNR (post-timing)
            z = lat.permute(0, 2, 1, 3, 4)
            z = z / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)
            x = m.conv2(z)
            for f in range(x.shape[2]):
                h = run_seg(x[:, :, f:f + 1])
                if is_last:                                               # n == 1: no pipeline
                    if t0 is None:
                        t0 = time.perf_counter()
                    outs.append(h); nfr += 1
                else:
                    _vae_send(h, grp_rank + 1)
        if out_dir and lats:
            torch.save(torch.cat(lats, dim=1), os.path.join(out_dir, "latents.pt"))
    else:
        while True:
            h = _vae_recv(grp_rank - 1, dev)
            if h is None:
                if not is_last:
                    _vae_stop(grp_rank + 1, dev)
                break
            if is_last and t0 is None:
                t0 = time.perf_counter()
            h = run_seg(h)
            if is_last:
                outs.append(h); nfr += 1
            else:
                _vae_send(h, grp_rank + 1)

    if is_last:
        import einops
        torch.cuda.synchronize(); wall = (time.perf_counter() - t0) * 1000 if t0 else 0
        t_end = time.perf_counter()
        print(f"[vae_pipe] n={n} decoded {nfr} latent-frames, wall {wall:.0f}ms "
              f"({wall/max(1,nfr):.1f}ms/frame, {wall/max(1,nfr//3):.1f}ms/chunk)", flush=True)
        vid = None
        if outs:
            pix = torch.cat(outs, dim=2).clamp(-1, 1)
            v = ((pix * 0.5 + 0.5).clamp(0, 1) * 255)
            vid = einops.rearrange(v, "b c t h w -> b t h w c")[0].byte().cpu()
        if dump and vid is not None:
            harness.save_video(vid, dump)                 # imageio (torchvision.io.write_video removed in 0.26)
            print(f"[vae_pipe] saved {tuple(vid.shape)} -> {dump}", flush=True)
        if out_dir and vid is not None:
            harness.save_video(vid, os.path.join(out_dir, "video.mp4"))
        if meta_q is not None:
            meta_q.put(("vae", wall, t_end))
    dist.barrier(); dist.destroy_process_group()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_output_frames", type=int, default=252)
    ap.add_argument("--prompt", default="A cinematic shot of a fluffy corgi running on a sunny beach, waves in the background.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--port", type=int, default=29650)
    ap.add_argument("--port2", type=int, default=29661)
    ap.add_argument("--nstep", type=int, default=4, help="denoising steps -> diffusion uses nstep+1 GPUs (nstep denoise + 1 store)")
    ap.add_argument("--vae-stages", dest="vae_stages", type=int, default=3,
                    help="VAE decode pipeline stages (FLOPs-balanced split); total GPUs = nstep+1+n")
    ap.add_argument("--gen_ckpt", default="./ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt")
    ap.add_argument("--steps", default="1000,750,500,250", help="denoising_step_list, comma-separated, length=nstep")
    ap.add_argument("--out_root", default="", help="structured output root directory out_root/task/run_tag")
    ap.add_argument("--task", default="04_wave_full_pipeline")
    ap.add_argument("--run_tag", default="serve")
    ap.add_argument("--ref_latents", default="", help="baseline latents.pt for PSNR")
    ap.add_argument("--baseline_e2e", type=float, default=0.0, help="baseline e2e s for speedup")
    ns = ap.parse_args()
    n_diff = ns.nstep + 1
    n_total = n_diff + ns.vae_stages
    ndev = torch.cuda.device_count()
    if n_total > ndev:
        raise SystemExit(
            f"[serve] need {n_total} GPUs (diffusion {n_diff} + vae {ns.vae_stages}) "
            f"but only {ndev} available. Lower --nstep or --vae-stages."
        )
    out_dir = harness.run_dir(ns.out_root, ns.task, ns.run_tag) if ns.out_root else ""
    mp.set_start_method("spawn", force=True)
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    q = mp.Queue(maxsize=64)
    meta_q = mp.Queue()
    procs = []
    for r in range(n_diff):
        procs.append(mp.Process(target=diffusion_worker, args=(r, n_diff, ns.port, q, ns, meta_q)))
    for g in range(ns.vae_stages):
        procs.append(mp.Process(target=vae_stage, args=(g, n_diff + g, q, ns.dump, ns.port2, ns.vae_stages, out_dir, meta_q)))
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"[main] end-to-end wall (incl. load) {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

    if out_dir:
        meta = {}
        while not meta_q.empty():
            it = meta_q.get(); meta[it[0]] = it[1:]
        diff_ms = meta.get("diff", [None])[0]
        t_start = meta["diff"][1] if "diff" in meta else None
        tick_ms = meta["diff"][2] if "diff" in meta else None
        vae_ms = meta.get("vae", [None])[0]
        t_end = meta["vae"][1] if "vae" in meta else None
        e2e_s = (t_end - t_start) if (t_start and t_end) else None
        this_lat = os.path.join(out_dir, "latents.pt")
        psnr = None
        if ns.ref_latents and os.path.isfile(ns.ref_latents) and os.path.isfile(this_lat):
            psnr = harness.latent_psnr(torch.load(this_lat, map_location="cpu"),
                                       torch.load(ns.ref_latents, map_location="cpu")).get("psnr_db")
        harness.write_metrics(
            out_dir, task=ns.task, run_tag=ns.run_tag, method="wave_full",
            nstep=ns.nstep, num_output_frames=ns.num_output_frames, num_gpus=n_total,
            seed=ns.seed, diffusion_s=(diff_ms / 1000.0) if diff_ms else None,
            vae_s=(vae_ms / 1000.0) if vae_ms else None, end_to_end_s=e2e_s,
            per_tick_ms=tick_ms, psnr_db=psnr,
            speedup_vs_baseline=(ns.baseline_e2e / e2e_s) if (ns.baseline_e2e and e2e_s) else None,
        )
        print(f"[main] metrics -> {out_dir}" + (f" PSNR={psnr:.2f}dB" if psnr else ""), flush=True)


if __name__ == "__main__":
    main()
