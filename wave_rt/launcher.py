"""Process launcher for WaveRT diffusion and streaming VAE ranks."""

from __future__ import annotations

import os
import time

import torch
import torch.multiprocessing as mp

from wave_rt import bench
from wave_rt.config import WaveConfig


def _worker(rank: int, cfg: WaveConfig, q=None, meta_q=None, req_q=None) -> None:
    # Worker imports must happen after CUDA visibility is configured.
    from wave_rt.denoiser import WaveDenoiser
    from wave_rt.runtime.backend import WaveBackend

    def run_denoiser(*, out_dir=None, warmup=True, save=True) -> None:
        denoiser = WaveDenoiser(backend, cfg, q=q, meta_q=meta_q)
        try:
            denoiser.run(out_dir=out_dir, warmup=warmup, save=save)
        except BaseException:
            # A failed rank may have left the distributed world, so never enter a
            # cleanup collective here.  Preserve the original exception even if
            # the now-broken CUDA context also rejects best-effort handle closes.
            try:
                denoiser.close(coordinated=False)
            except Exception as cleanup_exc:
                print(f"[wave_rt] rank {rank} cleanup after failure: "
                      f"{cleanup_exc!r}", flush=True)
            raise
        else:
            denoiser.close()

    backend = WaveBackend(cfg, rank)
    try:
        backend.init()
        if req_q is None:
            # one-shot: single generation with the config's prompt/seed/frames.
            run_denoiser()
        else:
            while True:
                req = req_q.get()
                if req is None:
                    break
                cfg.prompt = req.get("prompt", cfg.prompt)
                cfg.seed = int(req.get("seed", cfg.seed))
                cfg.num_frames = int(req.get("num_frames", cfg.num_frames))
                backend.prepare_request(
                    cfg.prompt, cfg.seed, cfg.num_frames,
                    req.get("height"), req.get("width"),
                )
                is_warmup = bool(req.get("warmup"))
                run_denoiser(out_dir=req.get("out"), warmup=False,
                             save=not is_warmup)
    except Exception:
        import traceback

        dbg_dir = os.environ.get("WAVE_DEBUG_DIR", "/tmp/wrt_dbg")
        try:
            os.makedirs(dbg_dir, exist_ok=True)
            with open(os.path.join(dbg_dir, f"EXC_r{rank}.log"), "w") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        print(f"[wave_rt] rank {rank} FAILED:\n{traceback.format_exc()}", flush=True)
        raise
    finally:
        backend.shutdown()


def _aggregate_metrics(cfg: WaveConfig, out_dir: str, meta_q) -> None:
    """Combine diffusion and VAE worker timings into the run metrics."""
    meta = {}
    while not meta_q.empty():
        it = meta_q.get()
        meta[it[0]] = it[1:]
    diff = meta.get("diff")
    vae = meta.get("vae")
    diffusion_ms = diff[0] if diff else None
    t_start = diff[1] if diff else None
    tick_ms = diff[2] if diff else None
    nlat = diff[3] if (diff and len(diff) > 3) else cfg.num_frames
    vae_ms = vae[0] if vae else None
    t_end = vae[1] if vae else None
    e2e_s = (t_end - t_start) if (t_start and t_end) else None
    lat_path = os.path.join(out_dir, "latents.pt")
    psnr = None
    if cfg.ref_latents and os.path.isfile(cfg.ref_latents) and os.path.isfile(lat_path):
        psnr = bench.latent_psnr(
            torch.load(lat_path, map_location="cpu"),
            torch.load(cfg.ref_latents, map_location="cpu"),
        ).get("psnr_db")
    speedup = (cfg.baseline_e2e / e2e_s) if (cfg.baseline_e2e and e2e_s) else None
    bench.write_metrics(
        out_dir, task=cfg.task, run_tag=cfg.run_tag, method="wave_rt_vae_pipe",
        nstep=cfg.rf_step, num_output_frames=nlat, num_gpus=cfg.n_total_gpus,
        seed=cfg.seed, diffusion_s=(diffusion_ms / 1000.0) if diffusion_ms else None,
        vae_s=(vae_ms / 1000.0) if vae_ms else None, end_to_end_s=e2e_s,
        per_tick_ms=tick_ms, psnr_db=psnr, speedup_vs_baseline=speedup,
    )
    msg = f"[wave_rt] metrics -> {out_dir}"
    if psnr is not None:
        msg += f"  PSNR={psnr:.2f}dB"
    if e2e_s:
        msg += f"  e2e={e2e_s:.2f}s  fps={nlat * 4 / e2e_s:.1f}"
    print(msg, flush=True)


def _serve(cfg: WaveConfig) -> None:
    """Serve queued generation requests with resident worker processes."""
    import itertools
    import queue as _queue
    import threading

    import uvicorn
    from fastapi import FastAPI

    if cfg.vae_stages <= 0:
        raise SystemExit("[wave_rt] --serve requires --vae-stages > 0 (streaming VAE)")

    n_diff, n_vae = cfg.wp_size, cfg.vae_stages
    req_qs = [mp.Queue() for _ in range(n_diff + n_vae)]
    q = mp.Queue(maxsize=64)
    meta_q = mp.Queue()

    from wave_rt.pipelines.vae import vae_stage
    procs = [mp.Process(target=_worker, args=(r, cfg, q, meta_q, req_qs[r]))
             for r in range(n_diff)]
    for g in range(n_vae):
        procs.append(mp.Process(
            target=vae_stage, args=(g, n_diff + g, q, cfg, "", meta_q, req_qs[n_diff + g])))
    for p in procs:
        p.start()

    job_queue: _queue.Queue = _queue.Queue()
    _SHUTDOWN = object()

    def _dispatch_req(req):
        """Broadcast one request to every worker, wait for its diff+vae timing."""
        for rq in req_qs:
            rq.put(req)
        meta = {}
        while not ("diff" in meta and "vae" in meta):
            it = meta_q.get()
            meta[it[0]] = it[1:]
        return meta

    def dispatcher():
        print(f"[wave_rt] warmup: dummy full-window generation "
              f"({cfg.warmup_frames} frames)...", flush=True)
        t_w = time.perf_counter()
        _dispatch_req({"prompt": cfg.prompt, "seed": 0,
                       "num_frames": cfg.warmup_frames, "out": None, "warmup": True})
        print(f"[wave_rt] warmup done in {time.perf_counter()-t_w:.1f}s; ready to serve",
              flush=True)
        while True:
            job = job_queue.get()
            if job is _SHUTDOWN:
                for rq in req_qs:
                    rq.put(None)
                break
            try:
                meta = _dispatch_req(job["req"])
                diff, vae = meta["diff"], meta["vae"]
                diffusion_ms, t_start, _tick, nlat = diff[0], diff[1], diff[2], diff[3]
                vae_ms, t_end = vae[0], vae[1]
                e2e_s = (t_end - t_start) if (t_start and t_end) else None
                job["result"] = {
                    "video": os.path.join(job["req"]["out"], "video.mp4"),
                    "num_output_frames": nlat,
                    "diffusion_s": round(diffusion_ms / 1000.0, 3) if diffusion_ms else None,
                    "vae_s": round(vae_ms / 1000.0, 3) if vae_ms else None,
                    "end_to_end_s": round(e2e_s, 3) if e2e_s else None,
                    "fps": round(nlat * 4 / e2e_s, 2) if e2e_s else None,
                }
            except Exception as e:
                job["result"] = {"error": repr(e)}
            job["event"].set()

    disp = threading.Thread(target=dispatcher, daemon=True)
    disp.start()

    app = FastAPI()
    _ids = itertools.count()
    _srv = {}

    @app.post("/generate")
    def generate(body: dict):
        nf = int(body.get("num_frames", cfg.num_frames))
        if nf % cfg.num_frames_per_block != 0:
            return {"error": f"num_frames must be divisible by {cfg.num_frames_per_block}"}
        out = body.get("out") or os.path.join(cfg.out_root, "serve", f"job_{next(_ids)}")
        os.makedirs(out, exist_ok=True)
        req = {
            "prompt": body.get("prompt", cfg.prompt),
            "seed": int(body.get("seed", cfg.seed)),
            "num_frames": nf, "out": out,
            "height": body.get("height"), "width": body.get("width"),
        }
        job = {"req": req, "event": threading.Event(), "result": None}
        job_queue.put(job)
        job["event"].wait()
        return job["result"]

    @app.get("/status")
    def status():
        return {"pending": job_queue.qsize()}

    @app.post("/shutdown")
    def shutdown():
        _srv["server"].should_exit = True
        return {"ok": True}

    server = uvicorn.Server(uvicorn.Config(
        app, host=cfg.serve_host, port=cfg.serve_port, log_level="warning"))
    _srv["server"] = server
    print(f"[wave_rt] serving on http://{cfg.serve_host}:{cfg.serve_port} "
          f"(POST /generate | GET /status | POST /shutdown); model loading in workers...",
          flush=True)
    server.run()

    job_queue.put(_SHUTDOWN)
    disp.join(timeout=60)
    for p in procs:
        p.join(timeout=120)
    # Escalate if a worker is still trapped in a collective after SIGTERM.
    for p in procs:
        if p.is_alive():
            p.terminate()
    time.sleep(3)
    for p in procs:
        if p.is_alive():
            p.kill()
    print("[wave_rt] serving stopped.", flush=True)


def _configure_environment(cfg: WaveConfig) -> None:
    if cfg.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices

    wave_threads = os.environ.get("WAVE_THREADS", "")
    if not wave_threads:
        try:
            core_count = len(os.sched_getaffinity(0))
        except Exception:
            core_count = os.cpu_count() or cfg.wp_size
        wave_threads = str(
            max(1, min(16, core_count // max(1, cfg.n_total_gpus)))
        )
        os.environ["WAVE_THREADS"] = wave_threads
        print(
            f"[wave_rt] WAVE_THREADS defaulted to {wave_threads} "
            f"({core_count} cores / {cfg.n_total_gpus} procs)",
            flush=True,
        )
    if wave_threads not in ("", "0"):
        thread_vars = (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        for name in thread_vars:
            os.environ.setdefault(name, wave_threads)

    os.environ.setdefault("MASTER_ADDR", cfg.master_addr)
    os.environ.setdefault("MASTER_PORT", str(cfg.master_port))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Full replicas do not need RunAI's load-time WORLD rendezvous.
    os.environ.setdefault("SGLANG_USE_RUNAI_MODEL_STREAMER", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def launch(cfg: WaveConfig) -> None:
    _configure_environment(cfg)

    print(
        f"[wave_rt] launching {cfg.wp_size} diffusion + {cfg.vae_stages} VAE ranks "
        f"(rf_step={cfg.rf_step}, num_frames={cfg.num_frames}, "
        f"backend={cfg.attention_backend})",
        flush=True,
    )
    if cfg.vae_stages > 0:
        ndev = torch.cuda.device_count() if torch.cuda.is_available() else cfg.n_total_gpus
        if cfg.n_total_gpus > ndev:
            raise SystemExit(
                f"[wave_rt] need {cfg.n_total_gpus} GPUs (diffusion {cfg.wp_size} + "
                f"vae {cfg.vae_stages}) but only {ndev} visible. Lower --wp-size / --vae-stages."
            )
    # Patch the shared file once, before workers can race on it.
    try:
        from wave_rt.distributed.compat import ensure_causal_transformer_config
        ensure_causal_transformer_config(cfg.model_path)
    except Exception as e:
        print(f"[wave_rt] config pre-patch skipped: {e!r}", flush=True)
    mp.set_start_method("spawn", force=True)
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass

    if cfg.serve:
        _serve(cfg)
        return

    streaming = cfg.vae_stages > 0
    q = mp.Queue(maxsize=64) if streaming else None
    meta_q = mp.Queue() if streaming else None
    out_dir = bench.run_dir(cfg.out_root, cfg.task, cfg.run_tag) if streaming else ""

    procs = [mp.Process(target=_worker, args=(r, cfg, q, meta_q))
             for r in range(cfg.wp_size)]
    if streaming:
        from wave_rt.pipelines.vae import vae_stage
        for g in range(cfg.vae_stages):
            procs.append(mp.Process(
                target=vae_stage,
                args=(g, cfg.wp_size + g, q, cfg, out_dir, meta_q)))

    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0
    print(f"[wave_rt] end-to-end wall {wall:.1f}s", flush=True)

    if streaming and meta_q is not None:
        _aggregate_metrics(cfg, out_dir, meta_q)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("wave_rt.launcher")
    WaveConfig.add_cli_args(parser)
    launch(WaveConfig.from_args(parser.parse_args()))
