"""Client inputs, generated video and complete shared status."""

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

from waveforcing_model import Anchor


class WaveForcingState(InputState):
    prompt: str = InputField(
        default="",
        moderate=True,
        max_length=4096,
        description="Describe the scene and motion for the next generated chunk.",
    )
    paused: bool = InputField(
        default=False,
        description="Pause generation between chunks; buffered frames can still play.",
    )
    _world_id: int = 0
    _applied_world_id: int | None = None
    _anchor: Anchor | None = None
    _completed: int = 0
    _frames: int = 0
    _last_seconds: float | None = None


class WaveForcingOutput(Output):
    """Silent generated video, at native 832 by 480 resolution."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Shared video status after changes and completed chunks; joining snapshot is caller-only."""

    prompt: str = MessageField(description="Accepted text for upcoming chunks.")
    seed: int | None = MessageField(
        description="Current video's seed, null before Start."
    )
    active: bool = MessageField(description="Whether a video has been started.")
    paused: bool = MessageField(description="Whether future generation is paused.")
    world_id: int = MessageField(
        description="Current video identity, zero before Start."
    )
    completed_chunks: int = MessageField(description="Chunks generated in this video.")
    output_frames: int = MessageField(description="Frames published in this video.")
    last_chunk_seconds: float | None = MessageField(
        description="Runtime-measured time of the latest chunk, null before its first output."
    )
