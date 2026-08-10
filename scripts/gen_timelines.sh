#!/bin/zsh
# Generate WAVE_TIMELINE Gantt SVGs for 2 models x 3 modes (diffusion + VAE lanes).
# The wave_rt process hangs in post-metrics teardown, so we advance once the
# "metrics ->" line appears (timeline JSONs are already flushed by then), kill it,
# and plot.
cd "$(dirname "$0")/.."
OUT=outputs
PY=.venv/bin/python
mkdir -p logs/w8a8 "$OUT"

killall_wave() { pkill -9 -f "wave_rt" 2>/dev/null; pgrep -f spawn_main | xargs -r kill -9 2>/dev/null; sleep 4; }

run() {  # $1=tag $2=title  $3..=extra wave_rt args
  local tag=$1 title=$2; shift 2
  echo "=== $tag ==="
  killall_wave
  rm -rf /tmp/tl_$tag
  WAVE_TIMELINE=1 WAVE_TIMELINE_DIR=/tmp/tl_$tag timeout 600 $PY -m wave_rt \
    --wp-size 5 --rf-step 4 --num-frames 24 --vae-stages 3 \
    --cuda-visible-devices 0,1,2,3,4,5,6,7 "$@" \
    --task w8a8_eval --run-tag $tag --no-save-video > logs/w8a8/tl_$tag.log 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 130); do
    if grep -q "metrics ->" logs/w8a8/tl_$tag.log 2>/dev/null; then ok=1; break; fi
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  sleep 4    # let VAE stage JSONs flush
  killall_wave
  if [[ $ok == 1 ]]; then
    $PY scripts/wp_timeline.py --dir /tmp/tl_$tag --out $OUT/timeline_$tag.svg --title "$title"
  else
    echo "  $tag FAILED (no metrics line)"
  fi
}

run g1_joint_sync   "1.3B joint-sync"                --kv-context joint  --exchange-mode sync
run g1_causal_sync  "1.3B causal-sync"               --kv-context causal --exchange-mode sync
run g1_causal_ovlp  "1.3B causal-overlap(prefetch)"  --kv-context causal --exchange-mode overlap
run g14_joint_sync  "14B joint-sync"                 --dit-scale 14b --kv-context joint  --exchange-mode sync
run g14_causal_sync "14B causal-sync"                --dit-scale 14b --kv-context causal --exchange-mode sync
run g14_causal_ovlp "14B causal-overlap(prefetch)"   --dit-scale 14b --kv-context causal --exchange-mode overlap

killall_wave
echo "ALL DONE"
ls -la $OUT/timeline_g*.svg