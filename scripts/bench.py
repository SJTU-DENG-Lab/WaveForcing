"""
Sweep benchmark: run naive-rt-serve across nstep=1/2/4 and dump results to outputs/.

Usage:
    python scripts/bench.py --num_output_frames 252 --gen_ckpt <path>
    python scripts/bench.py --num_output_frames 63   # quick smoke
"""
import argparse
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description="Sweep nstep=1/2/4 benchmark")
    ap.add_argument("--num_output_frames", type=int, default=252)
    ap.add_argument("--gen_ckpt", default="./ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt")
    ap.add_argument("--gen_ckpt_2step", default="./ckpts/zhuhz22/Causal-Forcing/causal-forcing++/framewise-2step.pt")
    ap.add_argument("--gen_ckpt_1step", default="./ckpts/zhuhz22/Causal-Forcing/causal-forcing++/framewise-1step.pt")
    ap.add_argument("--prompt", default="A cinematic shot of a fluffy corgi running on a sunny beach, waves in the background.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=29650)
    args = ap.parse_args()

    configs = [
        {"nstep": 4, "steps": "1000,750,500,250", "ckpt": args.gen_ckpt, "label": "4step"},
        {"nstep": 2, "steps": "1000,500", "ckpt": args.gen_ckpt_2step, "label": "2step"},
        {"nstep": 1, "steps": "1000", "ckpt": args.gen_ckpt_1step, "label": "1step"},
    ]

    py = sys.executable
    results = []
    for cfg in configs:
        tag = f"{cfg['label']}_nf{args.num_output_frames}"
        dump = f"outputs/{tag}.mp4"
        cmd = [
            py, "-m", "naive_rt.wave.serve",
            "--nstep", str(cfg["nstep"]),
            "--steps", cfg["steps"],
            "--gen_ckpt", cfg["ckpt"],
            "--num_output_frames", str(args.num_output_frames),
            "--prompt", args.prompt,
            "--seed", str(args.seed),
            "--dump", dump,
            "--port", str(args.port + cfg["nstep"] * 10),
        ]
        print(f"\n{'='*60}")
        print(f"[bench] Running {cfg['label']} ({cfg['nstep']+1} diffusion + 3 VAE = {cfg['nstep']+4} GPUs)")
        print(f"{'='*60}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=False)
        wall = time.perf_counter() - t0
        results.append({"label": cfg["label"], "nstep": cfg["nstep"], "wall_s": round(wall, 1), "exit": proc.returncode})
        print(f"[bench] {cfg['label']} done in {wall:.1f}s (exit {proc.returncode})", flush=True)

    print(f"\n{'='*60}")
    print("[bench] Summary:")
    for r in results:
        status = "OK" if r["exit"] == 0 else f"FAIL({r['exit']})"
        print(f"  {r['label']:8s}  nstep={r['nstep']}  wall={r['wall_s']:8.1f}s  {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
