"""Resident 4-step WaveRT + streaming VAE workers behind one parked pipeline."""

import os
import queue
import threading
import time
import traceback

WARMUP_CHUNKS = 12  # Past the point where the attention window stops growing.


class LatentTap:
    """Stamps each committed latent on its way from the Store rank to the VAE."""

    def __init__(self, latents, events, capture):
        self.latents, self.events, self.capture = latents, events, capture
        self.epoch = None

    def put(self, item):
        chunk, array = item
        self.latents.put(item)
        event = {"kind": "latent", "epoch": self.epoch, "chunk": chunk, "at": time.time()}
        if self.capture:  # Validation only: lets a harness compare runs value by value.
            event["latent"] = array
        self.events.put(event)


def diffusion_worker(rank, cfg, commands, events, latents, capture):
    backend = denoiser = None
    try:
        import torch
        from wave_rt.denoiser import WaveDenoiser
        from wave_rt.runtime.backend import WaveBackend
        from .native import (ChunkNoise, Tape, continuation_function, install_bf16_text,
                             paged_continuation_methods, switch_prompt)

        install_bf16_text()
        backend = WaveBackend(cfg, rank)
        backend.init()
        systolic = continuation_function()

        # TODO: move this class out of this function for readability
        class Denoiser(WaveDenoiser):
            def _act_denoise(self, n, t):
                return super()._act_denoise(n, t) and t - n >= self.session_start

            def _act_store(self, t):
                return super()._act_store(t) and t - self.store_rank >= self.session_start

            def _anchor_rope(self, *args):
                # The native table cache gains one entry per tick, which a
                # request-scoped denoiser never notices and a resident one must bound.
                if len(self._rope_cache) > 16:
                    self._rope_cache.clear()
                return super()._anchor_rope(*args)

        for name, method in paged_continuation_methods().items():
            setattr(Denoiser, name, method)
        tap = LatentTap(latents, events, capture)
        denoiser = Denoiser(backend, cfg, q=tap)
        # Neither model attention geometry nor its native working list
        # (anchor + 7) may grow with the session length.
        if denoiser.max_attn_frames != 21:
            raise ValueError("Interactive v1 requires the native 21-latent attention window")
        denoiser.num_blocks = float("inf")
        denoiser.session_start = 0
        denoiser.session_clean = [[] for _ in range(denoiser.Ln)]

        def restart(command):
            # Every rank arrives here on the same tick boundary, so these
            # collectives pair up. In-flight chunks are simply never resumed.
            denoiser._drain_tick_exchange()
            torch.cuda.synchronize()
            backend.barrier()
            if "seed" in command:
                cfg.seed = command["seed"]
                tap.epoch = command["epoch"]
                for layer in denoiser.session_clean:
                    layer.clear()
            # Also rebuilds the cross-attention cache.
            switch_prompt(backend, command["prompt"])
            denoiser.session_start = command["start"]
            # Ticks resume below the last one issued; a stale release flag
            # would satisfy the device-side waits before the payload arrives.
            if denoiser._os_flags is not None:
                denoiser._os_flags.zero_()
            torch.cuda.synchronize()
            backend.barrier()
            events.put({"kind": "reset" if "seed" in command else "switched", "rank": rank,
                        "epoch": tap.epoch, "start": command["start"]})
            return command["start"]

        def report(tick, ms, waited_ms):
            events.put({"kind": "tick", "epoch": tap.epoch, "tick": tick,
                        "ms": ms, "waited_ms": waited_ms})

        # The Store rank copies each latent to the host, so only its tick
        # times include the GPU work.
        denoiser.session_tape = Tape(commands, cfg.store_rank, restart,
                                     report if rank == cfg.store_rank else None)
        events.put({"kind": "ready", "rank": rank})
        systolic(denoiser, ChunkNoise(backend), warmup_pass=False, num_ticks=None)
        denoiser.close()
        denoiser = None
    except BaseException:
        events.put({"kind": "error", "rank": rank, "error": traceback.format_exc()})
        if denoiser is not None:
            try:
                denoiser.close(coordinated=False)
            except Exception:
                pass
        raise
    finally:
        if backend is not None:
            backend.shutdown()


def vae_worker(stage, cfg, commands, events, latents, fps, kbps, encode=True,
               encoder_workers=2, encoder_capacity=3):
    rank = cfg.wp_size + stage
    encoder = None
    try:
        from .video import encode_chunk
        from .encoding import OrderedVideoEncoder
        import torch
        import torch.distributed as dist
        from naive_rt.rolling_forcing.utils.wan_wrapper import WanVAEWrapper
        from naive_rt.wave import vae_pipe as vp
        from wave_rt.pipelines.vae_transport import FrameChannel

        torch.cuda.set_device(rank)
        dev = f"cuda:{rank}"
        torch.set_grad_enabled(False)
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{cfg.vae_port}",
                                rank=stage, world_size=cfg.vae_stages)

        def open_channels():
            # FA3 replaced the per-call shape handshake with a channel that
            # remembers the first frame and the steady shape.
            incoming = FrameChannel(stage - 1, dev) if stage else None
            outgoing = FrameChannel(stage + 1, dev) if stage + 1 < cfg.vae_stages else None
            return incoming, outgoing

        incoming, outgoing = open_channels()
        vae = WanVAEWrapper().to(dev, torch.bfloat16).eval()
        model = vae.model
        units, flops = vp.layer_flops(vae, dev)
        if cfg.vae_partition == "time":
            if stage == 0:
                _, timings = vp.measure_unit_times(vae, dev, npb=3,
                                                   latent_hw=(cfg.height // 8, cfg.width // 8))
                cuts = torch.tensor(vp.balanced_partition(timings, cfg.vae_stages), device=dev)
            else:
                cuts = torch.empty(cfg.vae_stages + 1, dtype=torch.long, device=dev)
            dist.broadcast(cuts, src=0)
            cuts = cuts.tolist()
        else:
            cuts = vp.balanced_partition(flops, cfg.vae_stages)
        layers = [unit for _, unit in units[cuts[stage]:cuts[stage + 1]]]
        scale = [vae.mean.to(dev, torch.bfloat16), (1 / vae.std).to(dev, torch.bfloat16)]
        events.put({"kind": "ready", "rank": rank})
        if encode and encoder_workers and stage + 1 == cfg.vae_stages:
            encoder = OrderedVideoEncoder(
                events.put, fps, kbps, workers=encoder_workers, capacity=encoder_capacity,
                on_error=lambda exc: events.put({"kind": "error", "rank": rank,
                    "error": "".join(traceback.format_exception(exc))}))
        cache, chunk, epoch = [None] * 256, 0, None
        while True:
            command = commands.get()
            if command is None:
                break
            if "seed" in command:
                # Encoding runs ahead independently; fence its ordered output
                # before acknowledging a reset or assigning the next epoch.
                if encoder is not None:
                    encoder.drain()
                # Latents are decoded in commit order, so every chunk of the
                # previous session has already left this stage.
                cache, chunk, epoch = [None] * 256, 0, command["epoch"]
                incoming, outgoing = open_channels()
                torch.cuda.synchronize()
                events.put({"kind": "reset", "rank": rank, "epoch": epoch, "start": 0})
                continue
            while chunk < command["end"]:
                if stage == 0:
                    received, array = latents.get()
                    if received != chunk:
                        raise RuntimeError("Latent transport crossed a session boundary")
                    z = torch.from_numpy(array).to(dev, torch.bfloat16).permute(0, 2, 1, 3, 4)
                    z = z / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)
                    x = model.conv2(z)
                frames = []
                for f in range(3):
                    h = x[:, :, f:f + 1] if stage == 0 else incoming.recv()
                    index = [0]
                    for layer in layers:
                        h = vp.apply_unit(layer, h, cache, index)
                    if outgoing is not None:
                        outgoing.send(h)
                    else:
                        pixels = ((h[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).cpu().numpy()
                        frames.extend(pixels)
                if stage + 1 == cfg.vae_stages:
                    expected = 9 if chunk == 0 else 12
                    if len(frames) != expected:
                        raise RuntimeError(f"Unexpected VAE continuation: {len(frames)} frames")
                    decoded_at = time.time()
                    if encoder is not None:
                        encoder.submit(frames, epoch=epoch, chunk=chunk, decoded_at=decoded_at)
                    else:
                        data = encode_chunk(frames, fps, kbps) if encode else b""
                        events.put({"kind": "chunk", "epoch": epoch, "chunk": chunk, "data": data,
                                    "size": len(data), "decoded_at": decoded_at,
                                    "encode_started_at": decoded_at,
                                    "encoded_at": time.time() if encode else decoded_at})
                chunk += 1
        if encoder is not None:
            encoder.close()
        dist.destroy_process_group()
    except BaseException:
        events.put({"kind": "error", "rank": rank, "error": traceback.format_exc()})
        raise


def configure_environment(cfg):
    """Same process environment the FA3 launcher sets before spawning ranks.

    `_configure_environment` was folded into `launch()`; interactive spawns its
    own workers, so it keeps this copy. WAVE_THREADS defaults to a capped share
    of the cores — without it each rank oversubscribes OpenMP and the DiT tick
    collapses.
    """
    if cfg.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    threads = os.environ.get("WAVE_THREADS", "")
    if threads == "":
        try:
            cores = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cores = os.cpu_count() or cfg.wp_size
        threads = str(max(1, min(16, cores // max(1, cfg.n_total_gpus))))
        os.environ["WAVE_THREADS"] = threads
    if threads not in ("", "0"):
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(name, threads)
    os.environ.setdefault("MASTER_ADDR", cfg.master_addr)
    os.environ.setdefault("MASTER_PORT", str(cfg.master_port))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("SGLANG_USE_RUNAI_MODEL_STREAMER", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


class WaveRuntime:
    """One resident GPU group. Commands queue up; nothing waits for a drain."""

    def __init__(self, cfg, fps=16, timeout=900, capture=False, kbps=None, encode=True,
                 encoder_workers=2, encoder_capacity=3):
        import torch.multiprocessing as mp
        from wave_rt.distributed.compat import ensure_causal_transformer_config

        if cfg.kv_context != "causal" or cfg.exchange_mode != "paged" or cfg.vae_stages < 1:
            raise ValueError("Interactive v1 requires causal / paged / streaming VAE")
        if cfg.num_frames_per_block != 3 or cfg.height != 480 or cfg.width != 832:
            raise ValueError("Interactive v1 is validated for 3 latent chunks at 480x832")
        if os.environ.get("WAVE_MAX_ATTN_FRAMES"):
            raise ValueError("Do not override the model attention window")
        if encoder_workers < 0 or encoder_capacity < max(1, encoder_workers):
            raise ValueError("Require encoder capacity >= workers >= 0")
        cfg.serve = True
        cfg.save_video = False
        # FA3 builds the RoPE axes for this many latent frames and refuses a
        # later position. 768 frames is 256 chunks, about 3 minutes at 16 fps.
        cfg.num_frames = 768
        self.cfg, self.timeout = cfg, timeout
        configure_environment(cfg)
        ensure_causal_transformer_config(cfg.model_path)
        ctx = mp.get_context("spawn")
        # Unbounded: issuing a command must never block the control plane.
        # Outstanding work is bounded by the session policy instead.
        self.events = ctx.Queue()
        latents = ctx.Queue()
        self.commands = [ctx.Queue() for _ in range(cfg.n_total_gpus)]
        self.processes = []
        self.epoch = 0
        self.thread = None
        self.closed = False
        for rank in range(cfg.n_total_gpus):
            if rank < cfg.wp_size:
                target = diffusion_worker
                args = (rank, cfg, self.commands[rank], self.events, latents, capture)
            else:
                target = vae_worker
                args = (rank - cfg.wp_size, cfg, self.commands[rank], self.events, latents,
                        fps, kbps, encode, encoder_workers, encoder_capacity)
            process = ctx.Process(target=target, args=args)
            process.start()
            self.processes.append(process)
        try:
            deadline = time.monotonic() + timeout
            ready = 0
            while ready < len(self.processes):
                if self._next(deadline)["kind"] == "ready":
                    ready += 1
            self._warm()
        except BaseException:
            self.close()
            raise

    def _next(self, deadline=None):
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("WaveRT workers stopped responding")
            try:
                event = self.events.get(timeout=1)
            except queue.Empty:
                if any(p.exitcode is not None for p in self.processes):
                    raise RuntimeError("A WaveRT worker exited") from None
                continue
            if event["kind"] == "error":
                raise RuntimeError(event["error"])
            return event

    def _warm(self):
        """Throwaway session, so the first viewer does not pay lazy CUDA/codec setup."""
        began = time.monotonic()
        epoch = self.reset(self.cfg.seed, self.cfg.prompt)
        self.advance(WARMUP_CHUNKS)
        first = None
        while True:
            event = self._next(began + self.timeout)
            if event["kind"] == "chunk" and event["epoch"] == epoch:
                first = first or time.monotonic()
                if event["chunk"] == WARMUP_CHUNKS - 1:
                    break
        print(f"[interactive] warm-up: first chunk after {first - began:.1f} s, "
              f"{WARMUP_CHUNKS} chunks after {time.monotonic() - began:.1f} s", flush=True)

    def start(self, on_event):
        """Deliver worker events to `on_event` from one background thread."""
        def pump():
            # A reset/prompt change is reported once, after every rank did it.
            acks = {}
            while not self.closed:
                try:
                    event = self._next()
                except Exception as exc:
                    if not self.closed:
                        on_event({"kind": "error", "error": str(exc)})
                    return
                if event["kind"] in ("reset", "switched"):
                    key = (event["kind"], event["epoch"], event["start"])
                    acks[key] = acks.get(key, 0) + 1
                    expected = len(self.processes) if event["kind"] == "reset" else self.cfg.wp_size
                    if acks[key] < expected:
                        continue
                    del acks[key]
                    event = {k: v for k, v in event.items() if k != "rank"}
                on_event(event)

        self.thread = threading.Thread(target=pump, daemon=True, name="wavert-events")
        self.thread.start()

    def reset(self, seed, prompt):
        """Start a new session from chunk zero; its events carry the returned epoch."""
        self.epoch += 1
        for commands in self.commands:
            commands.put({"seed": seed, "prompt": prompt, "start": 0, "epoch": self.epoch})
        return self.epoch

    def switch(self, prompt, start):
        """Change the prompt from chunk `start`, which must be the committed frontier."""
        for commands in self.commands[:self.cfg.wp_size]:
            commands.put({"prompt": prompt, "start": start})

    def advance(self, end):
        """Commit chunks below `end`; the pipeline parks again once they are stored."""
        for commands in self.commands:
            commands.put({"end": end})

    def close(self):
        self.closed = True
        # Only children owned by this runtime; never terminate external workers.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=5)
