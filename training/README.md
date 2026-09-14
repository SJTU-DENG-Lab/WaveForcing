# WaveForcing training

Train a Wan2.1-T2V-1.3B WaveForcing student from the Wan2.1-T2V-14B teacher.
The recipe has three stages (S1 / S2 / S3). Training uses the same Python 3.12
and PyTorch 2.11.0+cu128 environment as
[WaveRT](../README.md#quick-start-13b-5-step-preview).

Weights, prompts, and teacher pairs are external assets. None are bundled
with this package.

This directory is released under a separate [academic-use license](LICENSE).
The repository root Apache-2.0 license does not apply here.

## Scope

The current release supports:

- Student and critic: `Wan2.1-T2V-1.3B`. Teacher: `Wan2.1-T2V-14B`.
- Text-to-video at the 1.3B 480p latent shape `[21, 16, 60, 104]`, three
latent frames per block.
- Any strictly decreasing `denoising_step_list` whose length is smaller than
the number of teacher blocks (at most six steps for 21-frame pairs).
- Fixed RF/CRF/SF mix probabilities, `batch_size = 1` per rank, effective
batch equal to the rank count.
- A single-node `torchrun` launch; `--world-size` is the local GPU count.

Larger students, other resolutions, image-to-video, and multi-node launches are not part of this release and require future progress.

Experimental single-node 14B FSDP8 recipes are documented in
[README_14B.md](README_14B.md). The FSDP8/SP4 implementation passed
an eight-H200 S1/S2/S3 smoke and S2 restart check. Long-run stability
and generation-quality acceptance remain pending.

## Recipe

The default five-step schedule starts from the official RollingForcing
checkpoint and runs S2, then S3. S1 is optional, for runs that start from
an ODE initialization instead.


| Stage | Initialization                                            | Generator objective                                                         | Iterations | Generator / critic updates |
| ----- | --------------------------------------------------------- | --------------------------------------------------------------------------- | ---------- | -------------------------- |
| S1    | Raw ODE initialization                                    | On-policy DMD, RF/CRF/SF probabilities `0.5/0/0.5`                          | 3000       | 600 / 3000                 |
| S2    | Official RF checkpoint, or the preceding S1 raw generator | Paired-only off-policy DMD + `0.25 × LPIPS`, coupled on the same prediction | 1000       | 1000 / 1000                |
| S3    | Raw S2 generator                                          | On-policy DMD, RF/CRF/SF each `1/3`; no paired loss or LPIPS                | 2000       | 400 / 2000                 |


RF is the rolling path, CRF is its strict block-causal variant, and SF is
the sequential self-rollout path. Probabilities are always listed in
RF/CRF/SF order.

S2 re-noises a clean teacher window into a staircase, predicts it once, and
applies both DMD and LPIPS to that prediction. Prefix KV comes from clean
teacher blocks. S3 uses student rollouts and the student's own history.

Defaults are eight ranks, one sample per rank, and **effective batch 8 per
optimizer update**, with no gradient accumulation. Changing `--world-size`
changes the effective batch and is a different recipe. On S1 and S3 the
generator is updated every fifth iteration; a training iteration is not
always a generator update.

The default `denoising_step_list` in `configs/base.yaml` is
`[1000, 800, 600, 400, 200]`. Edit that list (or pass `--set`) for another
step count. A common four-step table is `[1000, 750, 500, 250]`. Each list
is warped once with shift 5. Four-step training is not a truncated five-step
list. If you change the step list, also set `max_steps` in the stage YAML
when you want a different iteration budget.

Each block is three latent frames. An S2 window therefore contains **3 latent
frames per step** (12 for four-step, 15 for five-step), taken from a
21-latent-frame teacher pair. The teacher must have more blocks than the
window, so a 21-frame pair supports at most six steps. S2 casts warped
timesteps to integer indices and evaluates LPIPS after half-resolution
latent decoding on every third decoded frame.

Starting S2 from the official RollingForcing checkpoint and starting from
an S1 run are different experiments. Stage budgets live in
`configs/s1.yaml`, `s2.yaml`, and `s3.yaml`.

## Installation

Use the WaveRT environment from the repository root. Do not create a
second Python or PyTorch installation. From that root, with WaveRT already
installed in `.venv`:

```bash
uv sync --extra training
uv pip install --no-deps -e training
source .venv/bin/activate
python -m wf_training --help
```

`--extra training` adds `lpips`, `tensorboard`, and `wandb`. `--no-deps`
keeps the existing torch stack. The default `full` attention path also
needs FlashAttention-2 for this PyTorch and CUDA. The CRF path uses
FlexAttention, which compiles on first use and needs a C++ compiler.

## Assets

Copy [assets.example.yaml](wf_training/configs/assets.example.yaml) and
replace the placeholders with local paths:

```bash
cp wf_training/configs/assets.example.yaml assets.yaml
```

Keep the filled `assets.yaml` on the machine that runs training. It is not
part of the source tree. Relative paths resolve from the YAML file's
directory.


| Key                          | Contents                                                                                                                                      |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_root`                 | Native Wan directories named `Wan2.1-T2V-1.3B` and `Wan2.1-T2V-14B`, including model configuration and all DiT shards.                        |
| `prompts`                    | Prompt text, one nonempty line per prompt. Comparisons that reuse a published split should keep the original file bytes and order.            |
| `rf_init`                    | Official RollingForcing `rolling_forcing_dmd.pt`, used by the default five-step S2 start.                                                     |
| `ode_init`                   | ODE initialization for S1. Self-Forcing `ode_init.pt` and Causal-Forcing `causal_ode.pt` are different checkpoints; record which one you use. |
| `paired_train`, `paired_val` | Verified JSONL manifests for 2048 training pairs and 128 held-out pairs, plus every referenced tensor file.                                   |
| `s3_init`                    | Raw S2 generator for a standalone S3 launch. Not required when S2 and S3 run in one command.                                                  |


Each native Wan directory must also contain the UMT5 encoder checkpoint,
`google/umt5-xxl` tokenizer files, and `Wan2.1_VAE.pth`. Pair generation
needs the 14B teacher's text encoder and tokenizer as well. A
Diffusers-layout directory is not a substitute for this native layout.

Before S2, stage the LPIPS calibration weights and torchvision's VGG16
checkpoint (`vgg16-397923af.pth` under `$TORCH_HOME/hub/checkpoints/`).
Keep caches in the normal per-user local cache root. 

## Teacher pairs

Each pair stores the initial noise `z_ref`, the teacher endpoint `y_ref`,
the prompt, the seed, and metadata. Both tensors have shape
`[21, 16, 60, 104]`. An existing package can be reused after you verify
prompts, seeds, teacher recipe, tensor hashes, and the train/validation
split.

To build a new package, generate with Wan2.1-T2V-14B, UniPC 50 steps,
CFG 3, shift 5, and fixed prompt/noise seeds. The command below runs on
one GPU. Independent workers can split the same index set with `--start`
and `--stride`. Verify only after every shard is finished.

```bash
python -m wf_training.prepare_pairs \
  --model-root /path/to/models \
  --prompt-path /path/to/vidprom_filtered_extended.txt \
  --output-dir /path/to/pairs \
  --count 2176 --train-count 2048 \
  --base-seed 1504386 --prompt-seed 1504386 \
  --teacher-name Wan2.1-T2V-14B \
  --sampling-steps 50 --guidance-scale 3 --shift 5

python -m wf_training.prepare_pairs \
  --prompt-path /path/to/vidprom_filtered_extended.txt \
  --output-dir /path/to/pairs \
  --count 2176 --train-count 2048 --verify-only
```

Verification writes `manifest_train.jsonl`, `manifest_val.jsonl`, and
`verification.json`. New manifests store paths relative to their
directory, so the package can be moved as a unit. Re-run verification to
rewrite older absolute paths. To replay one held-out teacher sample, repeat
the generation command with `--replay-index 2048`. That check requires a
GPU and the same teacher assets.

## Launch

The default is S2 then S3, eight ranks, and the step list in
`configs/base.yaml`. Inspect the resolved configuration before starting
workers:

```bash
python -m wf_training show-config \
  --assets assets.yaml --output runs/wf5

python -m wf_training preflight \
  --assets assets.yaml --output runs/wf5

python -m wf_training run \
  --assets assets.yaml --output runs/wf5 --dry-run
```

`run --dry-run` prints the plan and writes nothing. `show-config` and
`preflight` are CPU checks and do not import the training models.

A real launch writes the resolved configuration, asset inventory, and
installed package versions under each stage directory. Workers also record
the imported trainer path, GPU, attention settings, and the warped
timestep list. Worker logs are under `torchrun_logs/`; TensorBoard
scalars are under `tensorboard/`.

```bash
python -m wf_training run \
  --assets assets.yaml --output runs/wf5 \
  --stages s2,s3 --world-size 8
```

Change the step list and budgets in the YAML files, or on the command line:

```bash
--set denoising_step_list=[1000,750,500,250]
--set s3.max_steps=1000
```

Four-step training from an ODE initialization:

```bash
python -m wf_training run \
  --assets assets.yaml --output runs/ode4 \
  --stages s1,s2,s3 --world-size 8 \
  --set denoising_step_list=[1000,750,500,250] \
  --set s3.max_steps=1000
```

`--stage s1`, `--stage s2`, or `--stage s3` runs one stage. The default
start assets are `ode_init`, `rf_init`, and `s3_init`. Add another path to
the asset file and select it with `--init-key NAME`.
`--set s3.max_steps=1000` applies only to S3; `--set max_steps=1000`
applies to every selected stage. Overrides become part of the resolved
configuration.

Model weights are saved every `log_iters`. Full trainer state is saved
every `resume_save_iters`, which must be a multiple of `log_iters`. The
final step always writes a complete recovery checkpoint. For a short
execution check, override `max_steps`; that does not evaluate the default
recipe.

## Initialization and resume

A new stage loads the previous checkpoint's **raw** `generator`. It
creates its own optimizer, fake score, and EMA. It does not initialize
from `generator_ema` and does not import the previous optimizer. Sequential
`--stages` passes that raw endpoint automatically.

Resuming an interrupted stage restores the saved configuration and the
full trainer state: generator, critic, optimizer, EMA, RNG, and
data-loader progress. S2 also restores its separate generator/critic
paired-batch counters and text-loader progress. Pass a complete checkpoint
directory:

```bash
python -m wf_training resume \
  --run runs/wf5/s3 \
  --checkpoint runs/wf5/s3/checkpoint_model_001000 \
  --dry-run

python -m wf_training resume \
  --run runs/wf5/s3 \
  --checkpoint runs/wf5/s3/checkpoint_model_001000
```

Resume uses the configuration stored with that run. It does not merge a
new asset file or a new recipe into old state. A `model.pt` file is enough
to initialize the next stage; it is not enough to resume. Keep
`resume_complete.json` and every state file it lists together.

## Export

Choose the weight key explicitly:

```bash
wf-export \
  --checkpoint runs/wf5/s3/checkpoint_model_002000/model.pt \
  --weights ema \
  --output exports/wf5-s3-2000-ema
```

`--weights raw` exports the raw generator. The command writes native Wan
safetensors and a `model_manifest.json` that records the source checkpoint
and the selected key. It does not launch WaveRT, generate video, or convert
to a Diffusers layout.

## Citation

Please cite WaveForcing when using this training code; see the
[repository README](../README.md#citation) for the current entry.

## Acknowledgements

This trainer started from the
[RollingForcing](https://github.com/TencentARC/RollingForcing) training
code and modifies its recipe, pipeline, and launcher. RollingForcing in
turn builds on [Self Forcing](https://github.com/guandeh17/Self-Forcing)
and [Wan2.1](https://github.com/Wan-Video/Wan2.1). We thank the authors of
all three projects. The RollingForcing-derived portions keep their original
license terms; see [LICENSE](LICENSE).