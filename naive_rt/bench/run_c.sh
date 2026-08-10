#!/usr/bin/env bash
# Group C (+ A1 baseline) at the full 252 latent frames.
# tp=1 = naive baseline (A1) -> produces reference latents + baseline e2e.
# tp=2/4/6 = naive + Tensor Parallel, compared to the baseline (PSNR + speedup).
set -u
cd .
PY=./.venv/bin/python
ROOT=./outputs/experiments
NF=${NF:-252}
export TOKENIZERS_PARALLELISM=false

echo "=================== A1 / C0: naive tp=1 (nf=$NF) ==================="
CUDA_VISIBLE_DEVICES=0 $PY -m naive_rt.wave.tp_naive --tp 1 --num_output_frames $NF \
  --seed 1 --out_root $ROOT --task 00_naive_baseline --run_tag naive_nf${NF}

REF=$ROOT/00_naive_baseline/naive_nf${NF}/latents.pt
BASE_E2E=$($PY -c "import json;print(json.load(open('$ROOT/00_naive_baseline/naive_nf${NF}/metrics.json'))['end_to_end_s'])" 2>/dev/null)
echo "baseline e2e=${BASE_E2E}s  ref=$REF"

run_tp () {  # $1=tp  $2=gpus  $3=task
  echo "=================== $3: naive+TP tp=$1 gpus=$2 (nf=$NF) ==================="
  CUDA_VISIBLE_DEVICES=$2 $PY -m naive_rt.wave.tp_naive --tp $1 --num_output_frames $NF \
    --seed 1 --out_root $ROOT --task $3 --run_tag tp$1_nf${NF} \
    --ref_latents $REF --baseline_e2e ${BASE_E2E:-0}
}

run_tp 2 0,1       20_naive_tp2
run_tp 4 0,1,2,3   21_naive_tp4
# NOTE: tp must divide BOTH 12 heads AND 8960 FFN -> only {2,4} are valid
# (tp=3/6 leave FFN 8960 non-divisible). So the TP scaling stops at tp=4.

echo "=================== Group C done ==================="
for t in 00_naive_baseline 20_naive_tp2 21_naive_tp4 22_naive_tp6; do
  f=$(ls $ROOT/$t/*/metrics.json 2>/dev/null | head -1)
  [ -n "$f" ] && $PY -c "import json;d=json.load(open('$f'));print(f\"{d['task']:22s} tp={d['tp_size']} diff={d['diffusion_s']}s vae={d['vae_s']}s e2e={d['end_to_end_s']}s fps={d['fps_with_vae']} fps_noVAE={d['fps_without_vae']} PSNR={d.get('psnr_db')} speedup={d.get('speedup_vs_baseline')}\")"
done
