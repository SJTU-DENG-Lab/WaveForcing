"""WaveForcing client policy and explicit continuous generation."""

from pathlib import Path

from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    connected,
    event,
    get_weights_path,
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner

from waveforcing_assets import prepare
from waveforcing_model import (
    Anchor,
    WaveForcingInput,
    WaveForcingModel,
    WaveForcingResult,
)
from waveforcing_types import StateUpdate, WaveForcingOutput, WaveForcingState


class WaveForcing(ReactorApp):
    """Generate a continuous silent video from text and update its prompt live."""

    state: WaveForcingState
    buffer_size = 12

    def load(self, config_path: Path | None) -> None:
        settings = prepare(
            config_path or Path(__file__).with_suffix(".yaml"), get_weights_path()
        )
        self.engine = DistributedRunner(
            WaveForcingModel,
            world_size=1,
            # One isolated CUDA worker; native inference uses no collectives.
            init_process_group=False,
            load_kwargs={"settings": settings},
            call_timeout=30,
            start_timeout=600,
        )
        self.engine.start()

    async def process_input(self) -> WaveForcingInput:
        state = self.state
        if state._anchor is None or state.paused:
            raise ApplicationError("Press Start or resume generation")
        if not state.prompt.strip():
            raise ApplicationError("Enter a scene description")
        return WaveForcingInput(
            world_id=state._world_id,
            prompt=state.prompt,
            anchor=state._anchor
            if state._applied_world_id != state._world_id
            else None,
        )

    def generate(self, input: WaveForcingInput) -> WaveForcingResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> WaveForcingOutput | None:
        if outcome.error is not None:
            # Native failures may leave partially mutated KV/RNG/decoder state;
            # ending the service preserves the error instead of silently resetting.
            raise outcome.error
        result: WaveForcingResult = outcome.result
        state = self.state
        if result.world_id != state._world_id or state._anchor is None:
            return None
        state._applied_world_id = result.world_id
        state._completed = result.chunk_index
        state._frames += len(result.frames)
        state._last_seconds = outcome.elapsed
        await self.send(self._status())
        return WaveForcingOutput(main_video=result.frames)

    def _status(self) -> StateUpdate:
        s = self.state
        return StateUpdate(
            prompt=s.prompt,
            seed=s._anchor.seed if s._anchor else None,
            active=s._anchor is not None,
            paused=s.paused,
            world_id=s._world_id,
            completed_chunks=s._completed,
            output_frames=s._frames,
            last_chunk_seconds=s._last_seconds,
        )

    async def _broadcast(self) -> StateUpdate:
        status = self._status()
        await self.send(status)
        return status

    def _fresh(self, seed: int) -> None:
        s = self.state
        s._world_id += 1
        s._anchor = Anchor(seed=seed)
        s._applied_world_id = None
        s._completed = s._frames = 0
        s._last_seconds = None
        s.paused = False
        self.output.flush()

    @event(
        name="start",
        description="Start a fresh text-generated video and replace previous playback. Generation continues until Pause, Stop, or session end. Fails with `invalid_prompt` when neither supplied nor staged text contains a scene description.",
    )
    async def start(
        self,
        prompt: str = InputField(
            default="",
            moderate=True,
            max_length=4096,
            description="Scene and motion for the new video; empty uses the staged prompt.",
        ),
        seed: int = InputField(
            default=0, ge=0, le=2147483647, description="Random seed for this video."
        ),
    ) -> StateUpdate:
        prompt = prompt.strip() or self.state.prompt.strip()
        if not prompt:
            raise CommandError("invalid_prompt", "Enter a scene description")
        self.state.prompt = prompt
        self._fresh(seed)
        return await self._broadcast()

    @event(
        name="set_prompt",
        description="Guide upcoming chunks while keeping the current video and pause state. Buffered frames retain their earlier prompt. Can stage text before Start. Fails with `invalid_prompt` for empty text.",
    )
    async def set_prompt(
        self, prompt: str = WaveForcingState._public_fields["prompt"]
    ) -> StateUpdate:
        if not prompt.strip():
            raise CommandError("invalid_prompt", "Enter a scene description")
        self.state.prompt = prompt.strip()
        return await self._broadcast()

    @event(
        name="set_paused",
        description="Pause or resume generation between chunks without restarting the video. Buffered frames may still play. Fails with `no_video` before Start or after Stop.",
    )
    async def set_paused(
        self, paused: bool = WaveForcingState._public_fields["paused"]
    ) -> StateUpdate:
        if self.state._anchor is None:
            raise CommandError("no_video", "Start a video first")
        self.state.paused = paused
        return await self._broadcast()

    @event(
        name="reset",
        description="Restart the current video with the accepted prompt and an optional new seed. Fails with `no_video` before Start or after Stop.",
    )
    async def reset(
        self,
        seed: int | None = InputField(
            default=None,
            ge=0,
            le=2147483647,
            description="New seed; omit to retain the current video's seed.",
        ),
    ) -> StateUpdate:
        if self.state._anchor is None:
            raise CommandError("no_video", "Start a video first")
        self._fresh(self.state._anchor.seed if seed is None else seed)
        return await self._broadcast()

    @event(
        name="stop",
        description="End the current video and clear buffered playback. Keep the prompt staged for a new Start.",
    )
    async def stop_video(self) -> StateUpdate:
        s = self.state
        s._anchor = None
        s._applied_world_id = None
        s.paused = False
        s._completed = s._frames = 0
        s._last_seconds = None
        self.engine.reset()
        self.output.flush()
        return await self._broadcast()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._status())

    @event(
        name="get_state",
        description="Read the current video status without changing generation or notifying other viewers.",
    )
    def get_state(self) -> StateUpdate:
        return self._status()

    @session_ended
    def on_session_ended(self) -> None:
        if self.engine.healthy:
            self.engine.reset()
        self.output.flush()
