"""WaveRT launcher: bring up ``wp_size`` worker processes (one GPU each) that
together form the systolic wavefront.  Each worker owns exactly one CUDA device
and runs sglang with all parallel degrees == 1 (zero collective comm inside the
model); WaveRT itself owns the NCCL world used by the wavefront.
"""

from __future__ import annotations

import os
import time

import torch
import torch.multiprocessing as mp

from wave_rt import bench
from wave_rt.config import WaveConfig


def _worker(rank: int, cfg: WaveConfig, q=None, meta_q=None, req_q=None,
            request_barrier=None) -> None:
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
                cfg.height = int(req.get("height") or cfg.height)
                cfg.width = int(req.get("width") or cfg.width)
                backend.prepare_request(
                    cfg.prompt, cfg.seed, cfg.num_frames,
                    cfg.height, cfg.width,
                )
                is_warmup = bool(req.get("warmup"))
                if request_barrier is not None:
                    request_barrier.wait()
                run_denoiser(out_dir=req.get("out"), warmup=False,
                             save=not is_warmup)
                if cfg.vae_stages == 0 and rank == cfg.store_rank and meta_q is not None:
                    meta_q.put(("serial_done",))
                if request_barrier is not None:
                    request_barrier.wait()
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
    """Combine store-rank (diffusion) + last-VAE-stage timings into metrics.json
    (t_start/t_end from time.perf_counter are CLOCK_MONOTONIC -> comparable across
    procs on the same host)."""
    meta = {}
    while not meta_q.empty():
        it = meta_q.get()
        meta[it[0]] = it[1:]
    diff = meta.get("diff")
    vae = meta.get("vae")
    if diff is None or vae is None:
        raise RuntimeError('Missing diffusion/VAE completion; refusing partial metrics')
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
    from wave_rt.runtime.model import runtime_metadata
    bench.write_metrics(
        out_dir, task=cfg.task, run_tag=cfg.run_tag, method="wave_rt_vae_pipe",
        nstep=cfg.rf_step, num_output_frames=nlat, num_gpus=cfg.n_total_gpus,
        seed=cfg.seed, diffusion_s=(diffusion_ms / 1000.0) if diffusion_ms else None,
        vae_s=(vae_ms / 1000.0) if vae_ms else None, end_to_end_s=e2e_s,
        per_tick_ms=tick_ms, psnr_db=psnr, speedup_vs_baseline=speedup,
        extra=runtime_metadata(cfg),
    )
    msg = f"[wave_rt] metrics -> {out_dir}"
    if psnr is not None:
        msg += f"  PSNR={psnr:.2f}dB"
    if e2e_s:
        msg += f"  e2e={e2e_s:.2f}s  fps={(nlat * 4 - 3) / e2e_s:.1f}"
    print(msg, flush=True)


def _run_warmed(cfg: WaveConfig) -> None:
    """Full warmup then measurement in the same workers, without an HTTP server."""
    import queue
    from wave_rt.distributed.supervisor import receive_results, join_workers, stop_workers
    from wave_rt.pipelines.vae import vae_stage

    timeout = float(os.environ.get('WAVE_REQUEST_TIMEOUT', '3600'))
    streaming = cfg.vae_stages > 0
    chunks = mp.Queue(maxsize=64) if streaming else None
    messages = mp.Queue()
    requests = [mp.Queue(maxsize=1) for _ in range(cfg.n_total_gpus)]
    barrier = mp.Barrier(cfg.n_total_gpus, timeout=timeout)
    out = bench.run_dir(cfg.out_root, cfg.task, cfg.run_tag)
    processes = [mp.Process(target=_worker,
        args=(rank, cfg, chunks, messages, requests[rank], barrier))
        for rank in range(cfg.wp_size)]
    for stage in range(cfg.vae_stages):
        processes.append(mp.Process(target=vae_stage,
            args=(stage, cfg.wp_size+stage, chunks, cfg, out, messages,
                  requests[cfg.wp_size+stage]), kwargs={'request_barrier': barrier}))
    try:
        for process in processes:
            process.start()
        kinds = ('diff', 'vae') if streaming else ('serial_done',)
        for phase in range(cfg.full_warmup_runs + 1):
            warmup = phase < cfg.full_warmup_runs
            frames = cfg.warmup_frames if warmup else cfg.num_frames
            print(f"[wave_rt] {'full warmup' if warmup else 'measurement'} "
                  f"phase={phase}, latent_frames={frames}, resident workers", flush=True)
            request = dict(prompt=cfg.prompt, seed=cfg.seed, num_frames=frames,
                           height=cfg.height, width=cfg.width,
                           warmup=warmup, out=None if warmup else out)
            for channel in requests:
                channel.put(request)
            # Consume every phase's completion before sending the next request.
            result = receive_results(messages, processes, kinds, timeout)
            if not warmup and streaming:
                completed = queue.Queue()
                for kind in kinds:
                    completed.put((kind, *result[kind]))
                _aggregate_metrics(cfg, out, completed)
        for channel in requests:
            channel.put(None)
        join_workers(processes, timeout=timeout)
    except BaseException:
        stop_workers(processes)
        raise
    finally:
        for channel in [chunks, messages, *requests]:
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()


def _serve(cfg: WaveConfig) -> None:
    """Resident serving: load the model once, then process an HTTP-fed FIFO job
    queue -- clients POST /generate concurrently (accepted + queued instantly), a
    single dispatcher thread runs them one at a time (the wavefront owns all GPUs)."""
    import itertools
    import os
    import queue as _queue
    import threading

    import uvicorn
    from fastapi import FastAPI

    if cfg.vae_stages <= 0:
        raise SystemExit("[wave_rt] --serve requires --vae-stages > 0 (streaming VAE)")

    n_diff, n_vae = cfg.wp_size, cfg.vae_stages
    req_qs = [mp.Queue(maxsize=1) for _ in range(n_diff + n_vae)]
    q = mp.Queue(maxsize=64)     # diffusion store -> VAE stage0 chunk stream
    meta_q = mp.Queue()          # per-request (diff, vae) timing back to the dispatcher

    from wave_rt.pipelines.vae import vae_stage
    procs = [mp.Process(target=_worker, args=(r, cfg, q, meta_q, req_qs[r]))
             for r in range(n_diff)]
    for g in range(n_vae):
        procs.append(mp.Process(
            target=vae_stage, args=(g, n_diff + g, q, cfg, "", meta_q, req_qs[n_diff + g])))
    for p in procs:
        p.start()

    job_queue: _queue.Queue = _queue.Queue(maxsize=int(os.environ.get('WAVE_MAX_PENDING', '32')))
    request_timeout = float(os.environ.get('WAVE_REQUEST_TIMEOUT', '3600'))
    fatal = {}
    from wave_rt.distributed.supervisor import receive_results, stop_workers
    _SHUTDOWN = object()

    def _dispatch_req(req):
        """Broadcast one request to every worker, wait for its diff+vae timing."""
        for rq in req_qs:
            rq.put(req)
        return receive_results(meta_q, procs, ('diff', 'vae'), request_timeout)

    def dispatcher():
        # One-time startup warmup: a full dummy generation through the real path so
        # steady-state full-window kernels/NCCL/VAE (+ compile if on) are hot before
        # any client request.  Output discarded (out=None, save=False).
        print(f"[wave_rt] warmup: dummy full-window generation "
              f"({cfg.warmup_frames} frames)...", flush=True)
        t_w = time.perf_counter()
        try:
            _dispatch_req({"prompt": cfg.prompt, "seed": 0,
                           "num_frames": cfg.warmup_frames, "out": None, "warmup": True})
        except Exception as exc:
            fatal['error'] = repr(exc)
            stop_workers(procs)
        print(f"[wave_rt] warmup done in {time.perf_counter()-t_w:.1f}s; ready to serve",
              flush=True)
        while True:
            job = job_queue.get()
            if job is _SHUTDOWN:
                for rq in req_qs:
                    rq.put(None)
                break
            try:
                if fatal:
                    raise RuntimeError(fatal['error'])
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
                    "fps": round((nlat * 4 - 3) / e2e_s, 2) if e2e_s else None,
                }
            except Exception as e:
                job["result"] = {"error": repr(e)}
                if not fatal:
                    fatal['error'] = repr(e)
                    stop_workers(procs)
            job["event"].set()

    disp = threading.Thread(target=dispatcher, daemon=True)
    disp.start()

    app = FastAPI()
    _ids = itertools.count()
    _srv = {}

    @app.post("/generate")
    def generate(body: dict):  # sync -> FastAPI threadpool; blocking wait is safe here
        if fatal:
            return dict(fatal)
        try:
            nf = int(body.get("num_frames", cfg.num_frames))
            height = int(body.get('height') or cfg.height)
            width = int(body.get('width') or cfg.width)
            seed = int(body.get('seed', cfg.seed))
        except (TypeError, ValueError):
            return {'error': 'Invalid integer request parameter'}
        if nf < 3 or nf % cfg.num_frames_per_block != 0:
            return {"error": f"num_frames must be divisible by {cfg.num_frames_per_block}"}
        if min(height, width) < 16 or height % 16 or width % 16:
            return {'error': 'Height and width must be positive multiples of 16'}
        out = body.get("out") or os.path.join(cfg.out_root, "serve", f"job_{next(_ids)}")
        os.makedirs(out, exist_ok=True)
        req = {
            "prompt": body.get("prompt", cfg.prompt),
            "seed": seed,
            "num_frames": nf, "out": out,
            "height": height, "width": width,
        }
        job = {"req": req, "event": threading.Event(), "result": None}
        try:
            job_queue.put_nowait(job)
        except _queue.Full:
            return {'error': 'Request queue is full'}
        job["event"].wait()        # this client waits for its own result
        return job["result"]

    @app.get("/status")
    def status():
        return {"pending": job_queue.qsize(), 'error': fatal.get('error')}

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
    server.run()  # blocks until /shutdown sets should_exit

    job_queue.put(_SHUTDOWN)       # dispatcher broadcasts None -> workers exit
    disp.join(timeout=60)
    for p in procs:
        p.join(timeout=120)
    # A worker stuck in a NCCL collective ignores SIGTERM and would orphan (pinning
    # its GPU + the serve port on the next launch), so escalate to SIGKILL.
    for p in procs:
        if p.is_alive():
            p.terminate()
    time.sleep(3)
    for p in procs:
        if p.is_alive():
            p.kill()
    print("[wave_rt] serving stopped.", flush=True)


def launch(cfg: WaveConfig) -> None:
    if cfg.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    # Cap CPU threads per rank.  torch defaults intra-op threads to the physical
    # core count; with wp_size processes each spinning ~ncores OMP threads that is
    # a massive oversubscription (e.g. 6 x 128 = 768 busy-wait threads on 128
    # cores) that CRUSHES the launch-bound sgl DiT forward -- measured 1050ms vs
    # 180ms/forward, tick 760ms vs 148ms.  So we DEFAULT to an even split of cores
    # across ranks (capped at 16; beyond that these small ops don't benefit and OMP
    # busy-wait only hurts).  Override with WAVE_THREADS=N (N=0 disables the cap).
    import os as _os
    _wt = _os.environ.get("WAVE_THREADS", "")
    if _wt == "":
        try:
            _ncores = len(_os.sched_getaffinity(0))
        except Exception:
            _ncores = _os.cpu_count() or cfg.wp_size
        _wt = str(max(1, min(16, _ncores // max(1, cfg.n_total_gpus))))
        os.environ["WAVE_THREADS"] = _wt  # propagate to spawned workers (backend reads it)
        print(f"[wave_rt] WAVE_THREADS defaulted to {_wt} "
              f"({_ncores} cores / {cfg.n_total_gpus} procs)", flush=True)
    if _wt not in ("", "0"):
        for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(_v, _wt)
    os.environ.setdefault("MASTER_ADDR", cfg.master_addr)
    os.environ.setdefault("MASTER_PORT", str(cfg.master_port))
    # Tokenizers fork-safety + deterministic-ish cuBLAS are nice-to-haves.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Disable the RunAI model streamer: its find_local_ranks() does a distributed
    # all_gather_object over the WORLD group during weight load.  With the extra VAE
    # procs contending at startup, one diffusion rank reaches that load-time
    # rendezvous late -> TCPStore timeout -> crash cascade.  wrt loads a full replica
    # per rank, so RunAI's cross-rank coordination is pointless; the plain safetensors
    # loader avoids the rendezvous entirely.
    os.environ.setdefault("SGLANG_USE_RUNAI_MODEL_STREAMER", "0")
    # Make the NCCL watchdog abort a hung collective (see backend dist_timeout)
    # so a wavefront deadlock tears the process down instead of pinning GPUs.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

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
    # Pre-patch the shared transformer config ONCE (single process) so the
    # wp_size ranks don't race read-modify-writing it during load.
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
    if cfg.full_warmup_runs:
        _run_warmed(cfg)
        return

    # Diffusion procs (wp_size) + optional streaming VAE procs (vae_stages), sharing
    # an mp.Queue bridge + meta_q for timing.  Two DISJOINT NCCL worlds (diffusion:
    # master_port; VAE: vae_port) linked only by the CPU queue -> no in-job NCCL mix.
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
    from wave_rt.distributed.supervisor import join_workers
    join_workers(procs, timeout=float(os.environ.get('WAVE_REQUEST_TIMEOUT', '3600')))
    wall = time.perf_counter() - t0
    print(f"[wave_rt] end-to-end wall {wall:.1f}s", flush=True)

    if streaming and meta_q is not None:
        _aggregate_metrics(cfg, out_dir, meta_q)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("wave_rt.launcher")
    WaveConfig.add_cli_args(parser)
    launch(WaveConfig.from_args(parser.parse_args()))
