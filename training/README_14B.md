# Experimental 14B training on H200 GPUs

This guide covers both the single-node FSDP recipes and the multi-node HSDP
recipes. The 2×8 H200 HSDP8×2/SP4 smoke completed S1/S2/S3 (6/2/6 steps)
and a native S2 step 1→2 restart on 2026-09-14. The first table below remains
the historical single-node FSDP8/SP4 result; the multi-node launch, recovery,
and measurements are documented later in this file.

The `14b-fsdp8` and `14b-fsdp8-smoke` recipes target one node with
eight H200 GPUs. Select spatial sequence parallelism independently with
`--sp 1`, `--sp 2`, `--sp 4`, or `--sp 8` (`-sp` is an alias).
Generator, critic, and frozen real score are all Wan2.1-T2V-14B.
The **FSDP8/SP4 main GPU smoke passed on 2026-09-14**: all S1/S2/S3
budgets of 6/2/6 iterations completed, including BF16 training, stage
handoffs, S2 paired DMD/LPIPS, EMA, and three complete final checkpoints.
The independent S2 restart check (load final step 2 and continue to step 3)
also passed, exiting successfully at 06:01:43 UTC. All eight ranks restored
saved state exactly and completed one generator, critic, and EMA update. This short smoke
does not establish long-run stability or generation quality.

| Stage | Generator peak allocated | Critic peak allocated | Final checkpoint |
|---|---:|---:|---|
| S1 | 122.514 GiB | 70.870 GiB | Complete, 425.838 GiB |
| S2 | 104.736 GiB | 69.135 GiB | Complete, 425.838 GiB |
| S3 | 121.190 GiB | 70.873 GiB | Complete, 425.838 GiB |

S3 exercised RF/CRF/SF in 2/1/5 generator microbatches, respectively.
All recorded generator/critic nonfinite flags were zero, with no skipped
updates. Peaks are scoped PyTorch allocations across ranks, not total
GPU occupancy; S1 peak reserved memory was 136.494 GiB.

The earlier **FSDP8/SP1** run reached S1 step 5 and ran out of memory in
the second generator backward, after one generator and five critic updates.
The SP4 result above includes both spatial activation sharding and separate
head-sharded history KV; it is not an ablation isolating either saving.
See the SP4 smoke report (`2026-09-14-wf-14b-fsdp8-sp4-smoke/REPORT.md` in external experiment storage) for immutable logs and provenance,
and the earlier SP1 report (`2026-09-14-wf-14b-fsdp8-smoke/REPORT.md` in external experiment storage)
for its OOM record.

The default `--recipe reference` retains the existing 1.3B student/critic,
14B teacher, and S2 → S3 plan. The `14b-fsdp8*` recipes select S1 → S2 → S3
by default and require exactly eight ranks. They use full FSDP, rank-0
initialization, sharded EMA, BF16 mixed precision, and gradient checkpointing.
The `14b-hsdp*` recipes use the same model and stage settings with explicit
per-node shard and cross-node replica groups. Without `--sp`, each packaged
YAML supplies its own sequence-parallel default.
The option takes one degree per launch; it does not launch a sweep.

FSDP spans all eight ranks for every SP degree. Each SP group processes
one sample. `gradient_accumulation_steps: auto` resolves to the SP degree,
preserving effective batch 8:

| Option | Independent samples per microbatch | Default accumulation | Effective batch |
|---|---:|---:|---:|
| `--sp 1` | 8 | 1 | 8 |
| `--sp 2` | 4 | 2 | 8 |
| `--sp 4` | 2 | 4 | 8 |
| `--sp 8` | 1 | 8 | 8 |

Accumulation is independently configurable: `--sp 4 --set gradient_accumulation_steps=2`
produces effective batch 4. The resolved configuration always records integer
SP/accumulation, data-parallel size, and the resulting effective batch.
`--set sequence_parallel_size=4` also works; a conflicting `--sp` and
applicable `--set sequence_parallel_size=...` is rejected. SP must divide
WORLD size, spatial tokens per frame, and every model's attention heads.
14B has 40 heads and supports all four choices; the `reference` recipe's
1.3B student/critic have 12 heads, so SP8 is rejected before model loading.

For example, SP4 groups are `[0,1,2,3]` and `[4,5,6,7]`. Every microbatch performs
a normal FSDP backward; `no_sync()` is not used, so accumulated gradients
remain sharded. Adam, clipping, and EMA run once per optimizer update.

With SP4, each frame's 1560 spatial tokens are split into four sets of 390. Before
self-attention, differentiable all-to-all exchanges these for the complete
frame sequence and 10 of the 40 heads per rank. RoPE, temporal masks, sink
positions, and cache indices retain their global units. Q/K normalization
occurs before distributing heads. Only the final small latent prediction is
gathered; its backward sums gradients across the SP group, which together
with WORLD FSDP averaging gives the mean over distinct samples. Historical
self-KV is separately stored by local heads, reducing the 27-frame cache
from approximately 32.1 to 8.0 GiB per GPU. Cross-attention text KV stays
replicated. The total GPU peak is not divided by four.

Matching-dtype Q/K/V now share one packed all-to-all, after Q/K RMSNorm.
Together with the output head-to-sequence exchange, this reduces each
self-attention forward from four all-to-alls to two; network payload bytes
are unchanged. Backward also packs the three gradients into one inverse
exchange. Send buffers are filled directly without an intermediate QKV
concatenation and released before unpacking the outputs. Mixed-dtype Q/K/V
use separate exchanges to preserve precision (possible with FP32 norm weights
under ordinary autocast); the FSDP BF16 path uses the fused exchange.
The historical GPU timings and memory peaks above predate this optimization;
its H200 timing and allocation peaks have not been remeasured.

Prompts, teacher pairs, timesteps, and noise agree within each SP group.
Choices affecting model/collective order (RF/CRF/SF path, rollout length,
exit schedule, and S2 prefix length) remain synchronized across all eight
ranks. Resume checkpoints record SP size, accumulation, and data-replica
identity; a checkpoint from a different topology is rejected. Raw model
weights retain the existing layout for stage handoff.

Use `--recipe 14b-fsdp8-smoke --sp 4` with the same asset YAML to select the
smoke; its S1/S2/S3 budgets remain 6/2/6 training iterations. The completed
main smoke and the separate restart check are recorded in the
SP4 experiment (`2026-09-14-wf-14b-fsdp8-sp4-smoke/REPORT.md` in external experiment storage).

Initialization constructs and loads one model at a time on rank 0. Other
ranks construct meta parameters; FSDP materializes and synchronizes them
before the next model is constructed. EMA keeps only each rank's local
FP32 parameter shards on CPU during training. Full `generator_ema` weights
are gathered only for checkpoint export, using a temporary EMA substitution
that restores the raw generator afterward.

## Packaged recipes

The full and smoke YAMLs contain the algorithm settings, default stage order,
initial-asset keys, and overridable parallelism defaults:

- [14b-fsdp8.yaml](wf_training/configs/14b-fsdp8.yaml): full experimental budgets; choose SP with `--sp`.
- [14b-fsdp8-smoke.yaml](wf_training/configs/14b-fsdp8-smoke.yaml): smoke budgets; choose SP with `--sp`.
- [14b-hsdp.yaml](wf_training/configs/14b-hsdp.yaml): full multi-node
  HSDP budgets; node count comes from `torchrun`.
- [14b-hsdp-smoke.yaml](wf_training/configs/14b-hsdp-smoke.yaml):
  multi-node smoke budgets; validated on 2×8 H200 with SP4.
- [assets.14b.example.yaml](wf_training/configs/assets.14b.example.yaml): local asset template.

| Stage | Initialization | Objective | Full iterations | Smoke iterations |
|---|---|---|---:|---:|
| S1 | Candidate four-step distilled 14B weights | On-policy DMD; RF/CRF/SF = 0.5/0/0.5 | 3000 | 6 |
| S2 | Raw S1 generator | Paired off-policy DMD + 0.25 LPIPS on the same prediction | 1000 | 2 |
| S3 | Raw S2 generator | On-policy DMD; RF/CRF/SF = 1/3 each | 2000 | 6 |

All stages use the native four-step table `[1000, 750, 500, 250]`, warped
once with shift 5. S1 and S3 update the generator every fifth iteration;
S2 updates it every iteration. The smoke budgets exercise both generator
and critic updates. Smoke explicitly sets `ema_start_step=0` to exercise
sharded EMA; the full recipe starts EMA at iteration 200. These differences
make smoke an execution check, not an evaluation of the full recipe.

Each fresh stage loads the previous **raw generator** and creates a fresh
critic, optimizer, and EMA. It does not initialize from `generator_ema`.
This three-stage recipe differs from the historical 14B pure-RF → direct-Mix
two-stage experiment.

## Assets and initialization identity

Use the repository's existing Python 3.12 / PyTorch 2.11+cu128 `.venv`,
with its training extra and editable training package installed. Keep
model data, dependencies, and caches available before allocating GPU nodes.

Copy the new template outside the source tree and fill every placeholder:

```bash
cp training/wf_training/configs/assets.14b.example.yaml /path/to/run-assets.14b.yaml
```

`model_root` must contain a native `Wan2.1-T2V-14B` directory with its
`config.json`, all DiT safetensor shards, UMT5 encoder and tokenizer, and
`Wan2.1_VAE.pth`. Preflight checks the native model metadata for dimension
5120, FFN dimension 13824, 40 layers, 40 heads, T2V, 16 latent channels,
and the `[1,2,2]` patch contract. Native Wan may omit patch/text-dimension
entries; preflight then uses the model constructor defaults. It records
large files by size and modification time, without loading their tensors.

Use `distill_init_14b` for the first S1 initialization. One historically
staged candidate is named `self_forcing_plus_14b_distill.pt`, sourced from
`lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill`, revision
`b9baaa9a0c29226dea39043db647d1ced950bbea`, file `distill_model.pt`.
The recorded file size is 28,577,333,801 bytes and SHA-256 is
`13fec11400e094785dd9dae25a928a3acfb3cfc61168ab1f388344f06bef4db7`.
This candidate is a **bidirectional four-step step/CFG-distilled model**.
It is not a verified 14B causal ODE initialization or a trained 14B
WaveForcing checkpoint. Its loading and execution with this adaptation recipe passed the SP4 smoke;
its resulting generation quality remains unvalidated. Preflight file
existence and native model metadata cannot verify this initialization's
tensor values, key coverage, or training provenance.

Standalone S2 uses `s2_init_14b`; standalone S3 uses `s3_init_14b`.
Sequential plans supply the preceding raw endpoint automatically, so
these two keys are unnecessary for a complete S1 → S2 → S3 launch.
`--init-key NAME` selects an explicit alternative for the first stage.

The teacher-pair contract remains `[21,16,60,104]` with 2048 training
pairs and 128 validation pairs. Existing pairs can be reused after checking
their Wan2.1-T2V-14B teacher recipe (UniPC 50 steps, CFG 3, shift 5), prompt
and seed identity, tensor hashes, shape, and split. Scaling the student to
14B alone does not change this latent contract. S2 uses a 12-latent-frame
four-step window, clean teacher prefix KV, and LPIPS on decoded frames.
Stage the LPIPS calibration and VGG16 weights in pod-local caches too.

## Inspect and launch

These commands run from the canonical WaveForcing repository. `show-config`
and `run --dry-run` need only the asset YAML and import no training models;
they neither launch GPUs nor create output directories. `preflight` also
checks the referenced local files and native model metadata.

```bash
.venv/bin/python -m wf_training show-config \
  --recipe 14b-fsdp8-smoke --sp 4 \
  --assets /path/to/run-assets.14b.yaml --output /path/to/runs/14b-smoke

.venv/bin/python -m wf_training preflight \
  --recipe 14b-fsdp8-smoke --sp 4 \
  --assets /path/to/run-assets.14b.yaml --output /path/to/runs/14b-smoke

.venv/bin/python -m wf_training run \
  --recipe 14b-fsdp8-smoke --sp 4 \
  --assets /path/to/run-assets.14b.yaml --output /path/to/runs/14b-smoke \
  --world-size 8 --dry-run
```

After the assets and execution environment are ready, remove `--dry-run`
to run the smoke plan. Use a new output root and `--recipe 14b-fsdp8 --sp 4`
for the full experimental budgets when moving beyond the smoke. Explicit
`--stage` / `--stages` and stage-scoped `--set` remain available. All
single-node 14B plans retain the native four-step schedule, eight-rank topology,
full FSDP, rank-0 initialization, and sharded EMA constraints.

Launch the SP4 smoke with the same assets and a separate output directory:

```bash
.venv/bin/python -m wf_training run \
  --recipe 14b-fsdp8-smoke --sp 4 \
  --assets /path/to/run-assets.14b.yaml --output /path/to/runs/14b-sp4-smoke \
  --world-size 8
```

Memory profiling is enabled by default. Each rank writes
`memory_rankNN.jsonl`; sections include `initialize/<role>`,
`train/generator`, `train/critic`, `checkpoint/save`, and
`resume/trainer_state`. Inspect initialization, generator
and critic backward, S2 paired/VAE/LPIPS work, EMA updates, and save/export
peaks. Check host memory during model loading and state export as well as
CUDA memory. Full exported weights and recovery states remain large; a
successful forward does not prove that save or resume will fit.

In particular, saving currently gathers the FP32 generator, critic, and
EMA export on rank 0 together: **about 160 GiB of host memory for those
three full weight copies alone**, plus local EMA, process overhead, and
other live state. Local EMA storage during training does not turn the full
export into a sharded save or reduce its total disk footprint. Measure the
actual peak when changing the training workload. The completed SP4 smoke
measured rank-0 lifetime peak RSS of approximately 176–177 GiB during
full exports, with 425.838 GiB written per final checkpoint.

### Multi-node HSDP launch

Each node contributes eight contiguous ranks. HSDP uses `HYBRID_SHARD`:
parameters are sharded over the eight GPUs within a node, while the same
local-rank shard is replicated across nodes. SP groups must remain within a
node. Parameter all-gather, gradient reduce-scatter, and QKV exchange are
intra-node; the gradient-shard all-reduce crosses nodes. Every rank records
its actual shard, replica, and SP groups in the run manifest.

Each SP group processes one independent sample:

`effective_batch = WORLD_SIZE / SP × gradient_accumulation_steps`

| Topology | Global microbatch | Accumulation | Effective batch |
|---|---:|---:|---:|
| 8 GPUs, SP4 | 2 | 4 (`auto`) | 8 |
| 16 GPUs, SP4, validated | 4 | 4 (`auto`) | 16 |
| 64 GPUs, SP4 | 16 | 4 (`auto`) | 64 |
| 64 GPUs, SP8 | 8 | 1 (explicit) | 8 |

`auto` follows the SP degree; it does not preserve a fixed global batch as
the node count changes. Use the same source package, Python environment,
asset paths, and shared output path on every node. GPU workers are offline,
so prepare dependencies and model assets before launch.

Preview the validated two-node configuration without initializing distributed
training or creating the output directory:

```bash
.venv/bin/python -m wf_training show-config \
  --recipe 14b-hsdp-smoke --world-size 16 --gpus-per-node 8 \
  --sp 4 --assets /shared/configs/assets.14b.yaml \
  --output /shared/runs/wf14b-hsdp-smoke-001
```

Replace `show-config` with `preflight` to check assets and native model
metadata. Launch this command once on every node, changing only
`WF_NODE_RANK` in `0..WF_NNODES-1`:

```bash
export WF_NNODES=2
export WF_NODE_RANK=0
export WF_MASTER_ADDR=10.0.0.1
export WF_MASTER_PORT=29501

.venv/bin/python -m torch.distributed.run \
  --nnodes="$WF_NNODES" --nproc-per-node=8 \
  --node-rank="$WF_NODE_RANK" \
  --master-addr="$WF_MASTER_ADDR" --master-port="$WF_MASTER_PORT" \
  --max-restarts=0 \
  -m wf_training distributed-run \
  --recipe 14b-hsdp-smoke --sp 4 \
  --assets /shared/configs/assets.14b.yaml \
  --output /shared/runs/wf14b-hsdp-smoke-001
```

The launcher reads topology from `WORLD_SIZE` and `LOCAL_WORLD_SIZE`; an
explicit `--world-size` must agree with `torchrun`. Use `--recipe 14b-hsdp`
and a new output root for full budgets. NCCL network variables must match the
cluster topology. The validated two-node run used eth0 TCP Socket; it did not
validate RDMA throughput.

After each successfully trained and saved stage, the process synchronizes
CUDA, verifies that all FSDP modules and handles are idle, releases FSDP1
saved parameter views, then destroys the trainer, runs GC, and clears the
allocator cache. Failed stages do not take this success-only cleanup path.
This explicit release is required because frozen/no-grad forward can leave
live view-reference cycles in `flat_param._tensors`; Python GC and
`empty_cache()` cannot release live tensor storage by themselves. Before the
fix, S2 and S3 began with about 16.019 and 31.974 GiB allocated per GPU and S3
ran out of memory on its sixth generator update. After the fix, both stages
began at about 66 MiB and the previously failing update completed.

## Checkpoints and resume acceptance

The single-node 14B recipes save a complete final checkpoint. They do not bypass saving
or disable recovery to make the smoke run fit. Default periodic save
intervals remain 500 iterations, so each short smoke stage normally saves
only its final state. Keep `model.pt`, `resume_complete.json`, and all eight
`trainer_state_rank*.pt` files together. With an unchanged maximum-step budget, resuming an already completed
final step is rejected by the CLI.

The completed independent restart diagnostic preserved the original S2 run
and used a separate output with maximum step 3, loading its final step-2
checkpoint. All eight ranks exactly verified both optimizer states, EMA,
Python/NumPy/CPU/CUDA RNG, and cursors; the first resumed text and paired
batches also matched. Each rank then executed exactly one generator step,
one critic step, and one EMA update without skipping. The text cursor
advanced 8→12 and the paired cursor 16→24.

The new step-3 checkpoint contains `model.pt`, eight rank state files, and
the complete marker (425.838 GiB). Reloaded rank-state optimizer/EMA tensors
matched the live final state, and original input identities remained
unchanged. Large tensor files were checked by before/after stat identity,
not newly computed full-file hashes. This verifies restart continuation;
it does not simulate killing a process or claim GPU bitwise equality with
uninterrupted training. See the restart result (`2026-09-14-wf-14b-fsdp8-sp4-smoke/resume_checks/01/s2/probe_complete.json` in external experiment storage).

For a separate intentional interruption/resume procedure, launch S1 with a
checkpoint at iteration 5, interrupt after that complete checkpoint is
written and before the final step, then resume from it:

```bash
.venv/bin/python -m wf_training run \
  --recipe 14b-fsdp8-smoke --sp 4 --stage s1 \
  --assets /path/to/run-assets.14b.yaml --output /path/to/runs/14b-resume \
  --set s1.log_iters=5 --set s1.resume_save_iters=5

.venv/bin/python -m wf_training resume \
  --run /path/to/runs/14b-resume/s1 \
  --checkpoint /path/to/runs/14b-resume/s1/checkpoint_model_000005 --dry-run

.venv/bin/python -m wf_training resume \
  --run /path/to/runs/14b-resume/s1 \
  --checkpoint /path/to/runs/14b-resume/s1/checkpoint_model_000005
```

The periodic checkpoint adds substantial disk use and host-memory work.
Record the training commit, resolved config, imported training source,
Python executable, explicit asset paths, package versions, memory logs,
and checkpoint completeness with each GPU acceptance run. Source must be
preserved before launch; these instructions do not launch inference or
authorize changing inference branches.

Each stage's `run_manifest.json` includes the read-only Git commit and
branch when available, plus SHA-256 hashes for the actual installed
`wf_training` Python and YAML files, including untracked new source.
Without Git, the Git fields are null and file hashes are still recorded.
Hidden paths and symlinks are excluded; the manifest records skipped file
symlinks. It does not record a Git diff or inherited private state.

Multi-node HSDP checkpoints use format version 2. Every rank writes
`trainer_state_rankNN.pt` with its RNG, data cursors, counters, and topology.
Only ranks 0–7 in the first replica write optimizer and sharded EMA tensor
payloads; corresponding ranks in later replicas record the owner rank.
`resume_complete.json` lists all rank files and records world size, GPUs per
node, SP, accumulation, shard groups, and replica groups. Keep the complete
checkpoint directory. Native resume requires exactly the saved topology;
changing world size, SP, accumulation, or the HSDP layout requires starting a
new run from raw generator weights instead of restoring optimizer/RNG state.

For same-topology HSDP resume, run the same `torchrun` prefix on every node:

```bash
.venv/bin/python -m torch.distributed.run \
  --nnodes="$WF_NNODES" --nproc-per-node=8 \
  --node-rank="$WF_NODE_RANK" \
  --master-addr="$WF_MASTER_ADDR" --master-port="$WF_MASTER_PORT" \
  --max-restarts=0 \
  -m wf_training distributed-resume \
  --run /shared/runs/wf14b-hsdp-full-001/s2 \
  --checkpoint /shared/runs/wf14b-hsdp-full-001/s2/checkpoint_model_000500
```

## Validation checks

Use the project environment to run the configuration, initialization,
checkpoint, cache-lifetime, EMA, and source-provenance tests:

```bash
.venv/bin/python -m unittest discover -s training/tests -v
```

Experiment reports, logs, provenance, and checkpoints are retained outside
this repository. Record paths below are relative to that experiment store;
they are identifiers, not public download links.

As of 2026-09-14 06:01 UTC, the SP4 validation record contains:

- 90 CPU tests passed, covering configuration, accumulation, cache, resume,
  and source identity; a separate eight-process GLOO collective suite passed
  all three cases.
- 12 small-model SP1/SP4 cases passed on four-rank GLOO, covering
  bidirectional, causal, teacher-forcing, RF, CRF, and SF paths, each with
  activation checkpointing enabled and disabled.
- The same 12 cases passed on eight-GPU NCCL with real FULL_SHARD FSDP
  on both the SP1 and SP4 models. This numerical comparison used FP32,
  mathematical SDPA, and disabled TF32. Maximum absolute errors were
  `2.9802322387695312e-08` for parameter gradients and
  `2.3469328880310059e-07` after the Adam update.

The full 14B SP4 attempt 01 **completed successfully at 05:30:13 UTC**,
with all 6/2/6 iterations and three complete endpoint checkpoints. This
adds actual production BF16 attention, 14B memory, S2 VAE/LPIPS, stage
handoff, and save/export evidence to the small-model comparisons. The
S2 step-2-to-3 restart diagnostic also passed, with exact state restoration,
one continued update, a verified new checkpoint, and original inputs preserved.

Only the test reference and its diagnostics changed during the NCCL
investigation; no tolerance was widened and the production Python/YAML
snapshot stayed unchanged throughout GPU execution. All three stages and
the completed restart used
source SHA-256
`3e1ae02b21b071eb19f8d936efceb4847de21a3ad3a28f07cbf12c59094c01fb`,
based on commit `43cf63e9b9f5071599b353ea44b51e63c588930b` on
`train/14b-fsdp8` in the canonical checkout. The uncommitted source snapshot,
not that base commit alone, identifies this run. Small-model comparisons
use FP32/mathematical SDPA/block wrapping; the production run uses
BF16/actual attention kernels/size wrapping, so they do not prove BF16
bitwise equivalence.
See the SP4 validation record (`2026-09-14-wf-14b-fsdp8-sp4-smoke/REPORT.md` in external experiment storage)
for exact provenance, completed stage records, and the successful restart.

Immediately after that GPU execution, only acceptance comments in the two
SP4 YAMLs were updated. Python files were unchanged and both parsed configurations
remained identical. That earlier delivery package hash was
`510aeeec33b19be299adc52d49fb6c4b1cac05207b122ef74b4e0e212321d166`,
recorded separately in delivery provenance (`2026-09-14-wf-14b-fsdp8-sp4-smoke/source/provenance02.json` in external experiment storage).
The actual executed hash remained `3e1ae02b…c01fb`; the delivery hash is
not presented as a separately GPU-tested snapshot.

The later CLI refactor separates `--sp` from the recipe and removes the
duplicate SP4 recipe names and YAMLs. After removing those names, all 25
configuration/CLI regression tests passed, covering SP1/2/4/8 and rejection
of removed recipe names. This does not extend the historical 14B GPU smoke
result beyond SP4.
Saved resolved configurations using the supported recipe names retain their
recorded values during resume; changing SP or accumulation requires a new run
rather than resuming optimizer/RNG state from a different topology.

After the CLI and explicit teacher-forcing flag refactors and packed QKV
exchange, all 104 CPU tests and 12 four-rank GLOO model comparisons passed.
These include SP2/4/8 exchange ordering, exact reference outputs and gradients,
unused-gradient semantics, second derivatives, and checkpoint recomputation.
The later multi-node package passed 138 CPU tests, including a real four-process
CPU/GLOO nested-FSDP HSDP2×2/SP2 lifecycle regression. Its 2×8 H200 smoke
passed S1/S2/S3 and native S2 step 1→2 resume. Maximum per-rank allocated
memory was 122.514/104.736/121.190 GiB for S1/S2/S3 and 104.736 GiB for the
resume; all 16 ranks ended each stage at 66.000–68.973 MiB. Complete markers,
rank topology, optimizer/EMA metadata, finite loss/gradient scalars, and zero
nonfinite/skip signals passed acceptance. The immutable record is
`Auto-Research/auto_experiments/2026-09-14-wf-14b-hsdp16-sp4-smoke/REPORT.md`
in external experiment storage. This short smoke does not establish
convergence, video quality, long-run stability, RDMA performance, or
cross-rank tensor numerical equivalence.
