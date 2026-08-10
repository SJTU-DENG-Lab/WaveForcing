#!/usr/bin/env bash
# Groups A3/A4/A5 (wave-parallel evolution @ 4-step) + B3/B4 (wave 1/2-step),
# all @ 252 latent frames, compared to the A1 naive baseline (PSNR + speedup).
set -u
cd .
PY=./.venv/bin/python
ROOT=./outputs/experiments
NF=252
REF=$ROOT/00_naive_baseline/naive_nf${NF}/latents.pt
BASE=$($PY -c "import json;print(json.load(open('$ROOT/00_naive_baseline/naive_nf${NF}/metrics.json'))['end_to_end_s'])" 2>/dev/null)
echo "baseline e2e=${BASE}s ref=$REF"
export TOKENIZERS_PARALLELISM=false

echo "########## A3: s2 5-rank wavefront, serial VAE ##########"
$PY -m naive_rt.wave.history.s2_wave_dist --num_output_frames $NF --seed 1 \
  --out_root $ROOT --task 02_wave_dist_serialVAE --run_tag s2_nf${NF} \
  --ref_latents $REF --baseline_e2e ${BASE:-0} --port 29661

echo "########## A4: s3 5-rank diffusion + decoupled VAE ##########"
$PY -m naive_rt.wave.history.s3_wave_serve --num_output_frames $NF --seed 1 \
  --out_root $ROOT --task 03_wave_decoupledVAE --run_tag s3_nf${NF} \
  --ref_latents $REF --baseline_e2e ${BASE:-0} --port 29662

echo "########## A5: serve full pipeline (4-step, 8 GPU) ##########"
$PY -m naive_rt.wave.serve --num_output_frames $NF --seed 1 --nstep 4 --steps 1000,750,500,250 \
  --out_root $ROOT --task 04_wave_full_pipeline --run_tag serve4_nf${NF} \
  --ref_latents $REF --baseline_e2e ${BASE:-0} --port 29663 --port2 29673

echo "########## B3: wave 2-step (serve, 6 GPU) ##########"
$PY -m naive_rt.wave.serve --num_output_frames $NF --seed 1 --nstep 2 --steps 1000,500 \
  --out_root $ROOT --task 12_wave_2step --run_tag serve2_nf${NF} \
  --baseline_e2e ${BASE:-0} --port 29664 --port2 29674

echo "########## B4: wave 1-step (serve, 5 GPU) ##########"
$PY -m naive_rt.wave.serve --num_output_frames $NF --seed 1 --nstep 1 --steps 1000 \
  --out_root $ROOT --task 13_wave_1step --run_tag serve1_nf${NF} \
  --baseline_e2e ${BASE:-0} --port 29665 --port2 29675

echo "########## done ##########"
for t in 02_wave_dist_serialVAE 03_wave_decoupledVAE 04_wave_full_pipeline 12_wave_2step 13_wave_1step; do
  f=$(ls $ROOT/$t/*/metrics.json 2>/dev/null|head -1)
  [ -n "$f" ] && $PY -c "import json;d=json.load(open('$f'));print(f\"{d['task']:24s} gpu={d['num_gpus']} diff={d['diffusion_s']}s vae={d['vae_s']}s e2e={d['end_to_end_s']}s fps={d['fps_with_vae']} speedup={d.get('speedup_vs_baseline')} PSNR={d.get('psnr_db')}\")"
done
