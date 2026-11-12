"""Small, fail-closed adapter that keeps the existing systolic scheduler resident.

The native loop is entered once per worker and never drains: its tick source
blocks between ticks, so every rank keeps its in-flight latent and clean pages
and one more tick commits exactly one more chunk. The native attention
assembly, sampling, communication and window remain unchanged. Keeping this
experimental entry separate lets the existing batch runtime remain untouched.
"""

import inspect
import textwrap
import time


class Discard:
    """Sink for the per-request accumulators of a loop that never returns."""

    def __setitem__(self, key, value):
        pass

    def append(self, value):
        pass


def continuation_function():
    from wave_rt.denoiser import engine

    source = textwrap.dedent(inspect.getsource(engine.WaveDenoiser._systolic))
    for before, after in [
        ("clean = [[] for _ in range(Ln)]", "clean = self.session_clean"),
        ("for t in range(num_ticks):", "for t in self.session_tape:"),
        ("final_latents: dict[int, torch.Tensor] = {}", "final_latents = Discard()"),
        ("tick_ms: list[float] = []", "tick_ms = Discard()"),
        # FA3 sizes the whole-clip noise and output tensors up front. A resident
        # session is unbounded, so those become one chunk (ChunkNoise.shape).
        ("torch.empty_like(full_noise)",
         "torch.empty(full_noise.shape, device=full_noise.device, dtype=full_noise.dtype)"),
        ("my_latent = full_noise[:, :, c * npb : (c + 1) * npb].contiguous()",
         "my_latent = full_noise[c]"),
        ("target = self._final_output[:, :, c*npb:(c+1)*npb]",
         "target = self._final_output"),
    ]:
        if source.count(before) != 1:
            raise RuntimeError("Native scheduler changed; review the continuation adapter")
        source = source.replace(before, after)
    namespace = dict(vars(engine), Discard=Discard)
    exec(compile(source, __file__ + ":native_systolic", "exec"), namespace)
    return namespace["_systolic"]


class Tape:
    """Tick source shared by every diffusion rank.

    Chunk c sits on rank n at tick c + n, so committing chunks below `end`
    takes ticks below `end + depth`. The tape parks there with the pipeline
    full. Each rank reads the same command sequence only while parked, hence
    all ranks stop, restart and resume on the same tick without negotiation.
    """

    def __init__(self, commands, depth, restart, report=None):
        self.commands, self.depth = commands, depth
        self.restart, self.report = restart, report

    def __iter__(self):
        tick = end = None
        while True:
            waited = time.perf_counter()
            while end is None or tick >= end + self.depth:
                command = self.commands.get()
                if command is None:
                    return
                if "end" in command:
                    if end is None:
                        raise RuntimeError("Tape advanced before a session started")
                    end = max(end, command["end"])
                    continue
                # A new session or prompt drops the in-flight chunks and
                # refills from `start`; only a new session may move backwards.
                if "seed" not in command and command["start"] != end:
                    raise RuntimeError("Prompt change does not match the committed frontier")
                tick = end = self.restart(command)
            started = time.perf_counter()
            yield tick
            if self.report is not None:
                self.report(tick, (time.perf_counter() - started) * 1000,
                            (started - waited) * 1000)
            tick += 1


def switch_prompt(backend, prompt):
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from wave_rt.runtime.text_encoder import encode_local_text

    cls = type(backend.pipeline).sampling_params_cls
    batch = Req(sampling_params=cls(prompt=prompt, seed=backend.cfg.seed,
                num_frames=backend.cfg.num_frames, height=backend.cfg.height,
                width=backend.cfg.width))
    # Validation on every rank. Text encoding only where an encoder is loaded:
    # the FA3 backend keeps it on rank 0 and broadcasts when conditioning is shared.
    batch = backend.pipeline.stages[0](batch, backend.server_args)
    if backend.shared_conditioning:
        from wave_rt.distributed.conditioning import broadcast_tree
        from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import TextEncodingStage
        payload = None
        if backend.rank == 0:
            batch = encode_local_text(backend.pipeline.stages[1], batch, backend.server_args, backend.device)
            payload = {name: getattr(batch, name, None)
                       for name in TextEncodingStage.deduplicated_output_fields}
        payload = broadcast_tree(payload, backend.device)
        for name, value in payload.items():
            setattr(batch, name, value)
    else:
        batch = encode_local_text(backend.pipeline.stages[1], batch, backend.server_args, backend.device)
    for field in backend.pipeline.stages[1].deduplicated_output_fields:
        setattr(backend.batch, field, getattr(batch, field))
    backend.batch.prompt = prompt
    backend.ctx.prompt_embeds = backend.rf_stage._prepare_causal_dmd_prompt_embeds(
        backend.batch, backend.server_args, backend.ctx.target_dtype)
    backend.ctx.pos_cond_kwargs = backend.rf_stage._prepare_causal_dmd_pos_cond_kwargs(
        backend.batch, backend.server_args, backend.ctx.target_dtype)
    # FA3 leaves these None: the optimized path does not allocate the native
    # full-window cross-attention caches.
    for cache in backend.rf_stage.crossattn_cache:
        if cache is not None:
            cache.reset()
    # The optimized Wan path owns its own projected text, fixed-step
    # conditioning and per-layer text K/V. Updating ctx alone leaves those
    # pointing at the warm-up prompt. Reset all adapter request caches on
    # every restart, including a new session; WaveDenoiser still owns the
    # retained video history, which WaveExchange.reset() does not touch.
    if backend.wp_adapter is not None:
        backend.wp_adapter.reset()


def install_bf16_text():
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    original = ServerArgs.from_dict.__func__

    def from_dict(cls, values):
        args = original(cls, values)
        args.pipeline_config.text_encoder_precisions = ("bf16",)
        return args

    ServerArgs.from_dict = classmethod(from_dict)


class ChunkNoise:
    """Deterministic absolute-position noise without a whole-session allocation."""

    def __init__(self, backend):
        self.backend = backend
        self.shape = (1, backend.latent_channels, backend.cfg.num_frames_per_block,
                      backend.ctx.height, backend.ctx.width)
        self.device = backend.device
        self.dtype = backend.ctx.target_dtype
        self.is_cuda = True

    def __getitem__(self, index):
        import torch

        block = self.backend.cfg.num_frames_per_block
        if isinstance(index, int):
            chunk = index
        else:
            temporal = index[2]
            assert temporal.start % block == 0 and temporal.stop - temporal.start == block
            chunk = temporal.start // block
        generator = torch.Generator(device=self.backend.device)
        generator.manual_seed((self.backend.cfg.seed + chunk * 1000003) % (2**63 - 1))
        return torch.randn(self.shape, device=self.backend.device,
                           dtype=self.backend.ctx.target_dtype, generator=generator)


def paged_continuation_methods():
    """Native fixed-slot exchange with explicit retained-history geometry."""
    from wave_rt.denoiser.exchange import paged

    result = {}
    for name, expression in [
        ("_make_denoise_exch_paged", "min(8, max(0, tick - store_rank))"),
        ("_make_store_exch_paged", "min(8, max(0, tick - self.store_rank))"),
    ]:
        source = textwrap.dedent(inspect.getsource(getattr(paged.PagedExchangeMixin, name)))
        before = "clean_len = " + expression
        if source.count(before) != 1:
            raise RuntimeError("Native paged exchange changed; review continuation geometry")
        # After a prompt change the tape refills from a later chunk, where the
        # retained history holds more pages than a fresh pipeline at that
        # tick. The native assembly already reads the real clean list.
        source = source.replace(before, "clean_len = len(clean[0])")
        namespace = dict(vars(paged))
        exec(compile(source, __file__ + ":" + name, "exec"), namespace)
        result[name] = namespace[name]
    return result
