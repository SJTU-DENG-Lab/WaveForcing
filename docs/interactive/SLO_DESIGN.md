# Next stage: multiple sessions sharing a resident pipeline

This is a design proposal, not implemented functionality. The current
single-session demo and its validation are described in [README.md](README.md)
and [VALIDATION.md](VALIDATION.md).

## Session state and switching

Keep one shared copy of model weights and independent per-session state:
prompt conditioning/text K/V, clean video KV window, in-flight latents,
absolute chunk/tick counters, VAE temporal state, encoding epoch, playback
progress and pending commands. Suspend at a safe tick boundary only after
outstanding exchanges are complete. Restore all participating ranks together.

First verify A→B→A scheduling against isolated runs with the same prompts and
seeds. Measure save/restore bytes, latency and refill cost before adding an
admission policy. Merely pausing session A does not currently make its GPU
state available for a second session.

## Playback and interaction budgets

Track issued, latent-ready, pixel-ready, encoded, handed-to-network, appended
and actually displayed frontiers separately. Browser feedback includes its
monotonic sampling time, epoch and contiguous playable range; stale feedback
must not be treated as current buffer. Device clocks cannot be subtracted
without synchronization or a bounded mapping.

Use a lower buffer threshold to request work, a target to stop replenishing,
and an upper issued lead to bound interaction delay. Prepare new content while
the viewer consumes old content. In the current linear timeline, visible
interaction cannot precede the previously committed content boundary. A 5 s
irrevocable backlog is incompatible with a 2 s visible-interaction target,
even if server generation takes under 1 s.

Rollback-based future replacement is a separate option: it requires preserving
both DiT history and VAE snapshots at the target playback boundary. Retaining
additional history must not enlarge the model attention window.

## Scheduler and admission

For each session estimate the next-block playback deadline from fresh client
buffer feedback. Subtract estimated GPU work, context switching, VAE,
encoding, delivery and a jitter margin to obtain remaining slack. Prefer work
with less slack, with fairness/aging to avoid starvation. Run bounded groups
of ticks, stopping once the session reaches its buffer target or a more urgent
session needs service.

Prompt changes add refill work to the same scheduling problem. They must not
silently starve ordinary playback for other admitted users. Bound both CPU
encoding tasks/bytes and GPU context memory; a full downstream queue applies
backpressure rather than accumulating unlimited latent or RGB output.

Admission considers measured service distributions and switching overhead
across all stages, plus per-rank memory. Do not infer supported user count by
simply dividing peak single-session FPS by playback FPS. Report admitted and
rejected sessions alongside SLO satisfaction.

## CPU offload

After correctness and on-GPU multi-session scheduling work, separate history
storage from the active attention window. Keep upcoming working sets on the
GPU, use bounded pinned CPU storage for colder session state, and prefetch
before a scheduled restore. Measure PCIe/NVLink copies, synchronization and
contention as part of context-switch cost. Eviction/offload follows session
liveness and playback policy, with explicit pinned-page/reference lifetimes.

## Acceptance measurements

Measure warm request-to-first-display, active playback stall ratio and longest
stall, prompt-to-first-new-display, context-switch cost, per-stage queue/service
distributions, memory by rank, and fairness under mixed viewers. Semantic
response and visual transition quality remain separate model-quality checks.
Start with two sessions, then vary session count, pause/resume patterns,
prompt-change frequency and link jitter. The target is useful simultaneous
session time meeting declared SLOs, not peak unconstrained generation FPS.
