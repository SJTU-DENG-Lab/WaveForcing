#!/bin/zsh
# 14B dummy, 399 latent frames (~1596 pixel frames): sustained FPS across the main
# configs.  Long video -> steady state dominates -> real sustainable fps.
# Reliable pattern: poll metrics.json, then kill.  Do not interfere while running.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs/w8a8 outputs/w8a8_eval
NF=399

run() {  # $1=tag $2..=extra args
  local tag=$1; shift
  echo "=== $tag ==="
  rm -f outputs/w8a8_eval/$tag/metrics.json
  WAVE_PREFETCH=1 WAVE_FP8_COMPILE=1 WAVE_FP8_SCOPE=all timeout 1400 $PY -m wave_rt \
    --wp-size 5 --rf-step 4 --num-frames $NF --vae-stages 3 --dit-scale 14b \
    --cuda-visible-devices 0,1,2,3,4,5,6,7 \
    --task w8a8_eval --run-tag $tag --no-save-video "$@" \
    > logs/w8a8/$tag.log 2>&1 &
  local pid=$!
  for i in $(seq 1 320); do
    [[ -f outputs/w8a8_eval/$tag/metrics.json ]] && break
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  sleep 3; kill -9 $pid 2>/dev/null; pkill -9 -f "run-tag $tag" 2>/dev/null; sleep 6
}

run L400_base        --kv-context joint  --exchange-mode sync
run L400_sage        --kv-context joint  --exchange-mode sync --attention-backend sa
run L400_sage_w8a8   --kv-context joint  --exchange-mode sync --attention-backend sa --quantization fp8
run L400_full        --kv-context causal --exchange-mode overlap --attention-backend sa --quantization fp8

echo "=== SUMMARY (399 latent / 1596 pixel frames) ==="
$PY - <<'PY'
import json, statistics as s, os
rows=[("L400_base","bf16+SDPA (base)"),("L400_sage","+Sage"),
      ("L400_sage_w8a8","Sage+W8A8(all,cc)"),("L400_full","Sage+W8A8+overlap (full)")]
for tag,lab in rows:
    p=f"outputs/w8a8_eval/{tag}/metrics.json"
    if not os.path.exists(p): print(f"{lab:28s} (missing)"); continue
    m=json.load(open(p)); t=m["per_tick_ms"]; st=s.median(t[10:-3])
    print(f"{lab:28s} e2e={m['end_to_end_s']:6.2f}s  fps_e2e={m['fps_with_vae']:5.1f}  "
          f"fps_diff-only={m['fps_without_vae']:5.1f}  steady_tick={st:.0f}ms")
PY
echo "ALL DONE"
