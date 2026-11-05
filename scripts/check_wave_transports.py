"""Compare real Wave trajectories across transports on a multi-GPU host."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--gen-ckpt', required=True)
    p.add_argument('--steps', default='1000,750,500,250')
    p.add_argument('--frames', type=int, default=42)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=832)
    p.add_argument('--vae-stages', type=int, default=3)
    p.add_argument('--attention-backend', choices=('fa2','fa3','sa','torch_sdpa'), default='fa3')
    p.add_argument('--transports', nargs='+', choices=('sync','overlap','onesided','staggered','paged'),
                   default=['sync','overlap','onesided','staggered','paged'])
    p.add_argument('--out', type=Path, default=Path('outputs/transport-validation'))
    p.add_argument('--atol', type=float, default=.02)
    p.add_argument('--rtol', type=float, default=.02)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA inference and weights are required')
    if args.frames < 42 or args.frames % 3:
        p.error('Use at least 42 latent frames, divisible by three, to cross history eviction')
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='run-',dir=args.out)).resolve()
    reference, report = None, {}
    steps = len(args.steps.split(','))
    for mode in dict.fromkeys(['sync',*args.transports]):
        command = [sys.executable,'-m','wave_rt','--model-path',args.model_path,
                   '--gen-ckpt',args.gen_ckpt,'--rf-step',str(steps),'--wp-size',str(steps+1),
                   '--denoising-step-list',args.steps,'--num-frames',str(args.frames),
                   '--height',str(args.height),'--width',str(args.width),
                   '--vae-stages',str(args.vae_stages),'--kv-context','causal',
                   '--exchange-mode',mode,'--attention-backend',args.attention_backend,
                   '--seed','817','--no-save-video','--out-root',str(root),
                   '--task','transports','--run-tag',mode]
        try:
            with (root/f'{mode}.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            actual = torch.load(root/'transports'/mode/'latents.pt',map_location='cpu',weights_only=True)
            if not torch.isfinite(actual).all():
                raise AssertionError('Nonfinite generated latents')
            if reference is None:
                reference = actual
            torch.testing.assert_close(actual,reference,atol=args.atol,rtol=args.rtol)
            report[mode] = dict(passed=True,max_abs=float((actual.float()-reference.float()).abs().max()))
            print(f'PASS {mode}',flush=True)
        except Exception as exc:
            report[mode] = dict(passed=False,error=repr(exc))
            raise
        finally:
            (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Results: {root}')


if __name__ == '__main__':
    main()
