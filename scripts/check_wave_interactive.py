"""Equal-seed prompt regression, including equal-refill A->A / A->B controls."""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import time

import numpy as np

PROMPTS = {
    'sailboat': 'A red sailboat glides across a calm lake at sunrise, snowy mountains in the distance.',
    'robot': 'A silver humanoid robot walks through a large industrial warehouse, cinematic tracking shot.',
    'sunflowers': 'A vast field of golden sunflowers sways gently in the breeze under a bright blue sky.',
}
SWITCH = 'A heavy snowstorm sweeps over the lake, the red sailboat rocks in dark waves.'


def main():
    import wave_rt
    from wave_rt.config import WaveConfig
    from wave_rt.interactive.runtime import WaveRuntime
    parser = argparse.ArgumentParser()
    WaveConfig.add_cli_args(parser)
    parser.set_defaults(serve=True, kv_context='causal', exchange_mode='paged', save_video=False)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cfg = WaveConfig.from_args(args)
    root = Path(__file__).resolve().parents[1]
    source = Path(wave_rt.__file__).resolve()
    if not source.is_relative_to(root):
        raise RuntimeError(f'wave_rt imported from another checkout: {source}')
    provenance = {
        'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'source': str(root), 'python': sys.executable, 'wave_rt': str(source),
        'model': str(Path(cfg.model_path).resolve()),
        'checkpoint': str(Path(cfg.gen_ckpt).resolve()),
        'attention': cfg.attention_backend,
        'runtime_sha256': {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in sorted((root / 'wave_rt').rglob('*.py'))},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2), flush=True)
    runtime = WaveRuntime(cfg, fps=16, capture=True, encoder_workers=2)
    events = queue.Queue(); runtime.start(events.put)
    report = {'seed': 7, 'warmup_prompt': cfg.prompt, 'prompts': PROMPTS, 'switch_prompt': SWITCH, 'runs': {}}
    def wait(run, last):
        while last not in run['latent'] or last not in run['encoded']:
            event = events.get(timeout=180)
            if event['kind']=='error': raise RuntimeError(event['error'])
            if event.get('epoch')!=run['epoch']: continue
            if event['kind']=='latent': run['latent'][event['chunk']]=event['latent']
            if event['kind']=='chunk':
                c=event['chunk'];run['encoded'][c]=time.time()
                (run['dir']/f'{c:03}.mp4').write_bytes(event['data'])
    def generate(name, prompt, n=16, change=None):
        d=out/name;d.mkdir(exist_ok=True)
        began=time.time()
        run={'dir':d, 'epoch':runtime.reset(7,prompt),'latent':{},'encoded':{}}
        if change is not None:
            runtime.advance(8);wait(run,7)
            switched=time.time();runtime.switch(change,8)
        runtime.advance(n);wait(run,n-1)
        info={'prompt':prompt,'switch':change,'chunks':n,'seconds':time.time()-began,
              'first_chunk_ms':(run['encoded'][0]-began)*1000,
              'latent_sha256':{str(c):hashlib.sha256(x.tobytes()).hexdigest() for c,x in run['latent'].items()}}
        if change is not None: info['switch_first_chunk_ms']=(run['encoded'][8]-switched)*1000
        if n==40:info['e2e_fps']=477/info['seconds']
        report['runs'][name]=info
        np.savez_compressed(d/'latent_endpoints.npz',first=run['latent'][0],last=run['latent'][n-1])
        print(json.dumps({'run':name,**{k:v for k,v in info.items() if k!='latent_sha256'}}),flush=True)
        return run
    def diff(a,b,indices):
        return {'identical':all(np.array_equal(a['latent'][c],b['latent'][c]) for c in indices),
                'max_abs_diff':max(float(np.abs(a['latent'][c]-b['latent'][c]).max()) for c in indices)}
    try:
        a=generate('sailboat',PROMPTS['sailboat'],40)
        b=generate('robot',PROMPTS['robot'])
        c=generate('sunflowers',PROMPTS['sunflowers'])
        again=generate('sailboat_again',PROMPTS['sailboat'])
        noop=generate('same_prompt_refill',PROMPTS['sailboat'],change=PROMPTS['sailboat'])
        switched=generate('changed_prompt_refill',PROMPTS['sailboat'],change=SWITCH)
        repeated=generate('changed_prompt_repeat',PROMPTS['sailboat'],change=SWITCH)
        checks={
            'different_initial_prompts_a_b':not diff(a,b,range(16))['identical'],
            'different_initial_prompts_a_c':not diff(a,c,range(16))['identical'],
            'same_initial_prompt_after_other_sessions':diff(a,again,range(16))['identical'],
            'switch_preserves_prefix':diff(a,switched,range(8))['identical'],
            'text_effect_vs_identical_refill_control':not diff(noop,switched,range(8,16))['identical'],
            'switch_sequence_repeatable':diff(switched,repeated,range(16))['identical'],
        }
        report['checks']=checks
        report['same_prompt_refill_vs_free']=diff(a,noop,range(8,16))
        report['changed_prompt_vs_same_refill']=diff(noop,switched,range(8,16))
        report['passed']=all(checks.values())
    finally:
        runtime.close()
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2),flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
