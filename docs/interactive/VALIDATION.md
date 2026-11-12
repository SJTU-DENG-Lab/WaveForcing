# Interactive validation

This record describes the single-session implementation validated on
2026-11-12. All performance numbers below are observations, not service SLOs.

## CPU and browser checks

Run from the repository root in the existing inference environment:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_interactive*.py' -v
```

The 38 checks cover pacing, lifecycle, source-adapter compatibility, stale
session isolation, HTTP controls, GPU metric parsing, independent video
fragments, bitrate bounds, encoding order/capacity/error/reset fences, and
optimized prompt-cache invalidation. The cache regression fails against the
pre-fix prompt handler and passes against the corrected handler.

For browser tests, install Playwright and Chromium in a prepared Node test
environment. Node must resolve `playwright`, or set `PLAYWRIGHT_MODULE` to its
installed module path. Start each synthetic source in a separate terminal:

```bash
# Manual-start controls, regeneration, stale requests, reload, memory UI:
.venv/bin/python tests/interactive_synthetic_server.py --port 8892 --tick-seconds .10
# Another terminal:
WAVERT_TEST_URL=http://127.0.0.1:8892/ node tests/test_interactive_controls.cjs

# Autostart continuous playback, prompt colour boundaries, pause and reload:
.venv/bin/python tests/interactive_synthetic_server.py --port 8893 --autostart --tick-seconds .10
# Another terminal:
WAVERT_TEST_URL=http://127.0.0.1:8893/ node tests/test_interactive_player.cjs
```

Synthetic colours validate transport and playback ordering, not model prompt
following. Actual-model browser checks separately regenerated a sailboat and
then a warehouse robot, visually inspected both player frames, and exercised
a prompt change with zero recorded underruns/page errors in that short run.
The browser was on a CPU pod through the notebook gateway; this does not
establish performance on every external viewer's network.

## GPU regression

Use the same prepared weights, native VAE working-directory requirement and
inference environment as the [launch instructions](README.md). The harness
writes provenance, SHA-256 source hashes, encoded chunks, endpoint latents and
JSON results beneath the specified output directory:

```bash
.venv/bin/python -m scripts.check_wave_interactive \
  --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --gen-ckpt /path/to/four-step/longvideo.pt \
  --attention-backend fa3 --cuda-visible-devices 0,1,2,3,4,5,6,7 \
  --output outputs/interactive-validation
```

Before launching, verify device ownership/availability and source provenance.
The script checks its import origin and records source hashes; recording a
hash does not grant permission to run a dirty checkout on a shared worker.

Six GPU checks passed with seed 7:

1. Sailboat and robot initial prompts produce different latents.
2. Sailboat and sunflower initial prompts produce different latents.
3. Repeating sailboat after other sessions reproduces the original latents.
4. A switch at chunk 8 preserves the committed prefix.
5. A→B differs from A→A at the same refill position, isolating the text effect.
6. Repeating A→B reproduces the same complete sequence.

Frames from the three initial prompts were also inspected and showed the
requested subjects. These checks establish conditioning changes, not a general
score for semantic adherence or natural transitions.

## Measured configuration and results

8×H200; 4 DiT + Store + 3 VAE; FA3/BF16; 480×832; native attention window;
steps `[1000, 750, 500, 250]`; two bounded CPU encoders, capacity three;
16 FPS playback. The generator was the workspace's chunkwise four-step
`longvideo.pt`, not an assertion that arbitrary checkpoints reproduce the
same figures. Python 3.12, torch 2.11.0+cu128, PyAV 16.1.0, FastAPI 0.139.0,
Uvicorn 0.51.0 and HTTPX 0.28.1 were installed in the tested environment.

| Post-fix measurement | Observed value |
| --- | ---: |
| Reset request → final encoded block, 40 chunks / 477 frames | 4.232 s |
| Encoded end-to-end throughput, including fill | 112.70 FPS |
| Reset request → first encoded block, sailboat | 470.85 ms |
| Prompt request → first new encoded block, two repeats | 637.54 / 635.77 ms |

This is one run, without a multi-run confidence interval or P95/P99 claim.
Server encoding completion excludes network and browser display latency.

Earlier matched encoding runs measured 48.34 → 114.31 FPS after overlapping
encoding with VAE and avoiding redundant RGB copies. They both retained the
warmup corgi condition because optimized text caches were not invalidated.
That comparison remains an encoding-path observation under that condition;
it does not establish correct prompt switching. The cache fix introduced the
six checks above. A→A refill itself differs from free running (max latent
absolute difference 4.421875), which explains the false assurance from the old
free-versus-switch test.

## Reproducibility records

The original workspace records remain under the sibling Auto-Research
repository (paths relative to the WaveForcing repository root); they are
deliberately not required files in a source checkout:

- `../Auto-Research/auto_experiments/2026-11-12-wavert-interactive-h264/`
- `../Auto-Research/auto_experiments/2026-11-12-wavert-interactive-prompt-cache/`

They preserve original/failed tests, full source manifests, diffs, model/import
provenance, raw timing, frames and browser traces. The tested tree used base
`fde159c` plus the core FA3 files from `437832b` and the uncommitted interactive
implementation, with explicit source-hash authorization. A commit hash alone
therefore does not identify those historical runs. The source checkout keeps
the reusable regression harness; outputs and full historical logs stay outside
the proposed commit.

## Intermediate checkpoint verification — 2026-11-13 CST

The interactive branch now starts from `437832b`. Before committing the
single-session checkpoint, all 62 runtime files listed in the prompt-cache
experiment manifest were checked and matched their recorded SHA-256 hashes.
The 38 CPU tests above were rerun and passed. GPU and actual-model browser
results remain the historical validations described above; they were not
rerun for this commit. This checkpoint retains the experimental source-rewrite
adapter. Explicit runtime session state, multi-user scheduling and CPU offload
remain follow-up work.
