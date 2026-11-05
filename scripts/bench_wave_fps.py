"""Repeated real/dummy-weight Wave FPS runs; loading is outside runtime timing."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def summarize(metrics, frames, steps):
    pixels = 4 * frames - 3
    rows = []
    for m in metrics:
        diffusion, e2e = m['diffusion_s'], m['end_to_end_s']
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
               for v in (diffusion, e2e)):
            raise ValueError('Missing or invalid completed runtime timings')
        if m['num_output_frames'] != frames:
            raise ValueError('Output latent frame count does not match request')
        ticks = m.get('per_tick_ms') or []
        # Explicit window excludes fill, context growth, and drain.
        steady = ticks[max(10, steps + 1):frames // 3]
        tick = statistics.median(steady) if steady else None
        rows.append(dict(diffusion_s=diffusion, end_to_end_s=e2e,
                         dit_fps=pixels/diffusion, e2e_fps=pixels/e2e,
                         steady_tick_p50_ms=tick,
                         steady_fps=12000/tick if tick else None))
    return dict(pixel_frames=pixels, runs=rows,
                median_dit_fps=statistics.median(r['dit_fps'] for r in rows),
                median_e2e_fps=statistics.median(r['e2e_fps'] for r in rows),
                timing='Runtime generation only; excludes model loading and file encoding. '
                       'See warmup_protocol for the per-repetition warmup policy.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True, type=Path)
    weights = p.add_mutually_exclusive_group(required=True)
    weights.add_argument('--gen-ckpt', type=Path)
    weights.add_argument('--dummy', action='store_true', help='random DiT; throughput only')
    p.add_argument('--dit-scale', choices=('1.3b','5b','14b'), default='1.3b')
    p.add_argument('--weight-seed', type=int, default=2026)
    p.add_argument('--steps', default='1000,750,500,250')
    p.add_argument('--vae-stages', type=int, default=3)
    p.add_argument('--frames', type=int, default=399)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=832)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup-runs', type=int, default=1)
    p.add_argument('--warmup-frames', type=int, default=None,
                   help='full warmup latent frames; defaults to --frames')
    p.add_argument('--attention-backend', choices=('fa3','fa2','sa','torch_sdpa'), default='fa3')
    p.add_argument('--exchange-mode', choices=('sync','overlap','onesided','staggered','paged'), default='paged')
    p.add_argument('--quantization', choices=('bf16','fp8'), default='bf16')
    p.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=ROOT/'outputs/fps')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.warmup_frames is None:
        args.warmup_frames = args.frames
    if args.warmup_runs < 0 or args.warmup_frames < 3 or args.warmup_frames % 3:
        p.error('Warmup runs must be nonnegative; warmup frames must be positive multiples of three')
    steps = [int(x) for x in args.steps.split(',')]
    if args.repeats < 1 or args.frames < 3 or args.frames % 3 or not steps or args.vae_stages < 0:
        p.error('Positive repeats/steps, frames divisible by three, and nonnegative VAE stages required')
    needed = len(steps) + 1 + args.vae_stages
    if len(args.devices.split(',')) < needed:
        p.error(f'This topology needs {needed} visible GPUs')
    if not args.dummy and args.dit_scale != '1.3b':
        p.error('--dit-scale requires --dummy for larger synthetic models')
    if not args.dry_run and (not args.model_path.is_dir() or
                            (not args.dummy and not args.gen_ckpt.is_file())):
        p.error('Model directory and real generator checkpoint must exist')
    args.out.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='wave-', dir=args.out)).resolve()
    metrics, commands = [], []
    for repeat in range(args.repeats):
        tag = f'repeat_{repeat:02d}'
        command = [sys.executable,'-m','wave_rt','--model-path',str(args.model_path.resolve()),
                   '--rf-step',str(len(steps)),
                   '--wp-size',str(len(steps)+1),'--denoising-step-list',args.steps,
                   '--vae-stages',str(args.vae_stages),'--vae-partition','time',
                   '--full-warmup-runs',str(args.warmup_runs),'--warmup-frames',str(args.warmup_frames),
                   '--num-frames',str(args.frames),'--height',str(args.height),'--width',str(args.width),
                   '--attention-backend',args.attention_backend,'--kv-context','causal',
                   '--exchange-mode',args.exchange_mode,'--quantization',args.quantization,
                   '--cuda-visible-devices',args.devices,'--seed',str(args.seed),
                   '--no-save-video','--out-root',str(output),'--task','runs','--run-tag',tag]
        if args.dummy:
            command += ['--dummy-weights','--dit-scale',args.dit_scale,
                        '--weight-seed',str(args.weight_seed)]
        else:
            command += ['--gen-ckpt',str(args.gen_ckpt.resolve())]
        commands.append(command)
    (output/'commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
    weights_mode = 'dummy' if args.dummy else 'checkpoint'
    print(f'{weights_mode} {args.dit_scale}; {needed} GPUs; {args.frames} latent / {4*args.frames-3} pixel frames. Results: {output}',flush=True)
    if args.dry_run:
        print('Dry run: commands written; no inference launched.')
        return
    for i,command in enumerate(commands):
        log_path = output/f'repeat_{i:02d}.log'
        print(f'Run {i+1}/{args.repeats}: {log_path}',flush=True)
        with log_path.open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        path = output/'runs'/f'repeat_{i:02d}'/'metrics.json'
        metrics.append(json.loads(path.read_text(encoding='utf-8')))
        report = summarize(metrics,args.frames,len(steps))
        report.update(completed=len(metrics),requested=args.repeats,complete=len(metrics)==args.repeats)
        report.update(weights_mode=weights_mode,dit_scale=args.dit_scale,
                      weight_seed=args.weight_seed if args.dummy else None)
        report.update(warmup_runs=args.warmup_runs, warmup_frames=args.warmup_frames,
                      warmup_protocol=('full DiT+VAE requests in the same resident workers before measurement; '
                                       'new workers for each repetition' if args.warmup_runs else
                                       'legacy kernel-only warmup; no full request warmup'))
        (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        row = report['runs'][-1]
        print(f"  DiT {row['dit_fps']:.2f} FPS | E2E {row['e2e_fps']:.2f} FPS | "
              f"E2E {row['end_to_end_s']:.3f}s",flush=True)
    print(f"Median: DiT {report['median_dit_fps']:.2f} FPS | E2E {report['median_e2e_fps']:.2f} FPS")


if __name__ == '__main__':
    main()
