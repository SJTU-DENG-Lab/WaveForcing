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
    from wave_rt.serving.protocol import CommandKind, WorkerCommand

    backend = WaveBackend(cfg, rank)
    try:
        backend.init()
        if req_q is None:
            WaveDenoiser(backend, cfg, q=q, meta_q=meta_q).run(
                request_id="oneshot"
            )
        else:
            while True:
                command = req_q.get()
                if not isinstance(command, WorkerCommand):
                    raise TypeError(
                        f"expected WorkerCommand, got {type(command).__name__}"
                    )
                if command.kind is CommandKind.SHUTDOWN:
                    break
                req = command.request
                assert req is not None
                cfg.prompt = req.prompt
                cfg.seed = req.seed
                cfg.num_frames = req.num_frames
                backend.prepare_request(
                    req.prompt,
                    req.seed,
                    req.num_frames,
                    req.height,
                    req.width,
                )
                WaveDenoiser(backend, cfg, q=q, meta_q=meta_q).run(
                    out_dir=req.out_dir or None,
                    warmup=False,
                    save=not req.warmup,
                    request_id=req.request_id,
                )
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
    from wave_rt.serving.protocol import WorkerEvent

    meta = {}
    while not meta_q.empty():
        it = meta_q.get()
        if isinstance(it, WorkerEvent):
            meta[it.source] = it.payload
        else:
            meta[it[0]] = it[1:]
    diff = meta.get("diffusion") or meta.get("diff")
    vae = meta.get("vae")
    if isinstance(diff, dict):
        diffusion_ms = diff.get("diffusion_ms")
        t_start = diff.get("started_monotonic_s")
        tick_ms = diff.get("tick_ms")
        nlat = diff.get("num_latent_frames", cfg.num_frames)
    else:
        diffusion_ms = diff[0] if diff else None
        t_start = diff[1] if diff else None
        tick_ms = diff[2] if diff else None
        nlat = diff[3] if (diff and len(diff) > 3) else cfg.num_frames
    if isinstance(vae, dict):
        vae_ms = vae.get("vae_ms")
        t_end = vae.get("completed_monotonic_s")
    else:
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
    """Run the resident WaveRT control plane."""
    from wave_rt.pipelines.vae import vae_stage
    from wave_rt.serving.http import require_http_dependencies, run_http_server
    from wave_rt.serving.runtime import WaveServingRuntime

    require_http_dependencies()
    runtime = WaveServingRuntime(cfg, _worker, vae_stage)
    runtime.start()
    try:
        run_http_server(runtime, cfg)
    finally:
        runtime.shutdown(drain=True)
    print("[wave_rt/serve] stopped", flush=True)


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
