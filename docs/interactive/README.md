# Experimental single-session interactive streaming

This entry adds a resident, playback-paced demo on top of WaveRT's FA3 runtime.
It targets one viewer using 4 DiT ranks, one KV Store rank and 3 streaming VAE
ranks on eight GPUs. It does not change the fixed-length `python -m wave_rt`
entry. Both paths currently require batch size one.

## Run

Prepare the normal WaveRT Python 3.12 environment, editable SGLang backend,
FA3 installation and model weights first. The HTTP layer additionally needs
`fastapi` and `uvicorn`; CPU HTTP tests need `httpx`. Prepare dependencies on a
network-enabled host before using an offline GPU worker. The validated versions
are recorded in [VALIDATION.md](VALIDATION.md).

The inherited VAE wrapper loads
`wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` relative to the working directory.
Run from a directory containing that asset; pass absolute model/checkpoint
paths and use this repository's interpreter/imports. From the repository root,
with the VAE asset prepared there:

```bash
.venv/bin/python -m wave_rt.interactive \
  --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --gen-ckpt /path/to/four-step/longvideo.pt \
  --rf-step 4 --wp-size 5 --denoising-step-list 1000,750,500,250 \
  --attention-backend fa3 --kv-context causal --exchange-mode paged \
  --vae-stages 3 --cuda-visible-devices 0,1,2,3,4,5,6,7 \
  --encoder-workers 2 --encoder-capacity 3 \
  --playback-fps 16 --buffer 6 --delivery-slack 2 --video-kbps 1200 \
  --manual-start --serve-host 127.0.0.1 --serve-port 8890 \
  --trace-dir outputs/interactive/trace
```

Open `http://127.0.0.1:8890/` after loading and warmup finish. Enter an initial
prompt and start generation. Regenerate starts a new timeline with a fresh
seed while keeping models loaded. The second prompt box changes subsequent
content; pause/resume controls playback. Memory usage and utilization are
sampled once per second per GPU. Use a prepared port forward or notebook proxy
for a remote server; opening `player.html` as a standalone file does not run
the API. Only one playback client is accepted at a time.

## Runtime and switch semantics

- The systolic loop stays resident. A command tape parks all ranks at tick
  boundaries with in-flight latents and the native working KV window retained.
  Resuming ordinary generation commits one additional chunk per tick.
- A prompt change starts at `issued`, the exclusive frontier of work already
  authorized for output. The old committed prefix plays first; in-flight old
  work is discarded, and the pipeline refills at that frontier. Video history
  and streaming VAE state continue across this boundary.
- Each prompt restart refreshes T5 conditioning and invalidates both native
  and optimized model text caches, including projected text, fixed-step
  conditioning and per-layer text K/V. New sessions also clear video history
  and VAE state. Epoch/session IDs reject late results and stale requests.
- The model attention window remains 21 latent frames. No historical KV/VAE
  archive for rollback is maintained. `Pages` currently stores encoded output;
  its prefix-sharing methods are not a deployed GPU KV paging system.

Generation follows the greater of the playback and delivery cursors, subject
to a hard cap:

```text
target = min(max(played, pace) + 1 + buffer,
             played + 1 + buffer + delivery_slack)
```

`played` means the last fully displayed chunk. `pace` means bytes handed to
the live response, not confirmed browser playback. Encoded blocks are pruned
by `played`. Paused/stalled playback eventually fills the bounded lead and
parks generation. Outstanding work includes additional in-flight pipeline
state; this is not a configurable GPU-memory-byte budget.

At 16 FPS an ordinary chunk contains 12 frames (0.75 s); the first contains 9.
The default buffer is 3 chunks and slack is 2. The example above retains the
more conservative tested buffer of 6 chunks. A larger buffer tolerates more
jitter but delays visible prompt changes. Server first-new-chunk latency and
client visible interaction latency must be measured separately.

## Encoding and transport

The last VAE stage submits RGB frames to a bounded CPU pool. Independent
H.264 encoders overlap VAE work and emit chunks in order. Capacity counts
queued, running and completed-but-not-emitted work; the producer may retain
one additional decoded block while blocked. A reset fence drains old encoding
work before acknowledging a new epoch. Background failures reach the
controller even when GPU workers are parked.

Each block begins with an IDR and uses an independent fMP4 container.
`GET /live` sends `uint32 chunk_id`, `uint32 byte_length`, then MP4 bytes
(big-endian integers) over one HTTP response. The browser appends them to a
single MSE timeline. This is a custom framed HTTP stream, not HLS or WebRTC.
`GET /chunk/{branch}/{chunk}` remains available for diagnostic clients.
`--encoder-workers 0` selects synchronous encoding with the current pixel path.

## Observability and source layout

| File | Responsibility |
| --- | --- |
| `wave_rt/interactive/runtime.py` | Resident DiT/Store/VAE workers and lifecycle |
| `wave_rt/interactive/native.py` | Tick tape, scheduler adaptation, prompt reset |
| `wave_rt/interactive/state.py` | Playback/delivery pacing and output pages |
| `wave_rt/interactive/server.py` | Session controller, HTTP stream, trace |
| `wave_rt/interactive/video.py`, `encoding.py` | Codec, pixel packing, bounded ordered encoding |
| `wave_rt/interactive/player.html` | Initial prompt, regenerate, switch, playback and diagnostics |
| `wave_rt/interactive/metrics.py` | Background device-level GPU sampling |

Server trace distinguishes diffusion, VAE, encoding queue/service and delivery
intervals. Browser trace records requests, appends, frame callbacks and
underruns. Some intervals include queueing; device-level memory includes more
than live model/KV tensors. Traces are appended as JSONL when configured.

## Limits and follow-up

This is a single-session research demo. It has no multi-user scheduler,
admission control, context offload, arbitrary playback-position rollback, or
trained semantic-transition guarantee. A viewer token provides session
ownership, not a production authentication system.

The resident adapter rewrites specific native scheduler source fragments and
fails if those fragments change; it needs review when the underlying runtime
changes. Geometry is currently prepared for 768 latent frames; unbounded
session duration has not been validated. Worker errors require restarting the
service. Refilling even with the same prompt can differ from uninterrupted
generation, so prompt tests must use an equal-refill control.

See [validation and tests](VALIDATION.md) and the
[next-stage multi-user scheduling design](SLO_DESIGN.md).
