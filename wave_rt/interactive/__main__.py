import argparse
import json
import os
from pathlib import Path
import hashlib
import subprocess
import sys
import time


def main():
    from wave_rt.config import WaveConfig
    from .state import Limits
    from .runtime import WaveRuntime
    from .server import Controller, Trace, make_app
    from .metrics import GpuMonitor
    import uvicorn
    import wave_rt

    parser = argparse.ArgumentParser(description="Experimental playback-paced WaveRT")
    WaveConfig.add_cli_args(parser)
    parser.set_defaults(serve=True, kv_context="causal", exchange_mode="paged", save_video=False)
    parser.add_argument("--manual-start", action="store_true", help="Wait for a browser-supplied initial prompt")
    parser.add_argument("--playback-fps", type=int, default=16, help="The model generates 16 FPS content")
    parser.add_argument("--buffer", type=int, default=Limits.buffer,
                        help="Target chunks ahead of playback and initial buffering threshold")
    parser.add_argument("--delivery-slack", type=int, default=Limits.delivery_slack,
                        help="Extra chunks allowed for delayed playback ACKs; bounds the issued frontier")
    parser.add_argument("--video-kbps", type=int,
                        help="Bound the H.264 bitrate for a slow viewer link (default: 2.5 Mbit/s peak)")
    parser.add_argument("--encoder-workers", type=int, default=2,
                        help="Independent CPU encoders (0: synchronous baseline)")
    parser.add_argument("--encoder-capacity", type=int, default=3,
                        help="Maximum encoding chunks in flight, including completed output awaiting order")
    parser.add_argument("--trace-dir", help="Append server and browser timing rows to a JSONL file here")
    args = parser.parse_args()
    if args.playback_fps <= 0:
        parser.error("Playback FPS must be positive")
    cfg = WaveConfig.from_args(args)
    limits = Limits(args.buffer, args.delivery_slack)
    root = Path(wave_rt.__file__).resolve().parents[1]
    provenance = {"source": str(root), "python": sys.executable,
                  "base_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                  "checkpoint": str(Path(cfg.gen_ckpt).resolve()),
                  "feature_sources": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).parent.glob("*") if p.is_file()}}
    print(json.dumps(provenance, indent=2), flush=True)
    trace_path = None
    if args.trace_dir:
        os.makedirs(args.trace_dir, exist_ok=True)
        trace_path = os.path.join(args.trace_dir, f"trace-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}.jsonl")
    runtime = WaveRuntime(cfg, fps=args.playback_fps, kbps=args.video_kbps,
                          encoder_workers=args.encoder_workers, encoder_capacity=args.encoder_capacity)
    controller = Controller(runtime, cfg.prompt, limits, autostart=not args.manual_start,
                            trace=Trace(trace_path))
    devices = cfg.cuda_visible_devices or ",".join(str(i) for i in range(cfg.n_total_gpus))
    roles = [f"去噪 {i}" for i in range(cfg.store_rank)] + ["KV Store"] + [f"VAE {i}" for i in range(cfg.vae_stages)]
    monitor = GpuMonitor(devices, roles)
    try:
        # The player polls several times per second; the trace carries the timeline instead.
        uvicorn.run(make_app(controller, args.playback_fps, monitor), host=cfg.serve_host,
                    port=cfg.serve_port, access_log=False)
    finally:
        controller.close()
        monitor.close()


if __name__ == "__main__":
    main()
