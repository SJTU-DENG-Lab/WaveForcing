"""Runtime-independent state and step contracts for WaveForcing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from waveforcing_backend import WaveForcingBackend

WIDTH, HEIGHT = 832, 480
BLOCK_FRAMES = 3


@dataclass(frozen=True)
class Settings:
    """Pinned source/asset locations and the bounded initial-noise page size.

    ``noise_page_frames`` counts latent frames; it is internal, not a video
    limit. Each page uses the upstream BCTHW draw order before conversion to
    BTCHW. A finite native comparison must use its exact noise horizon here.
    """

    source: Path
    weights: Path
    noise_page_frames: int = 399


@dataclass(frozen=True)
class Anchor:
    """Explicit random seed for a fresh video; never inferred by the model."""

    seed: int


@dataclass(frozen=True)
class WaveForcingInput:
    """Immutable request for one finalized video chunk.

    ``world_id`` identifies the video. ``anchor`` is required for an unseen
    id and is carried until acknowledged by a result. ``prompt`` is the text
    in effect for this step, including history-preserving live updates.
    """

    world_id: int
    prompt: str
    anchor: Anchor | None = None


@dataclass(frozen=True)
class WaveForcingResult:
    """Acknowledged world, one-based chunk count, and CPU uint8 THWC RGB.

    Frames have shape [T,480,832,3]: 9 for the first chunk, then 12. Timing
    belongs to the caller; this result carries only completed model output.
    """

    world_id: int
    chunk_index: int
    frames: np.ndarray


class NoAnchor(ValueError):
    """A fresh world requires its explicit seed."""


class WaveForcingModel:
    """Own resident weights and causal state independently of Reactor."""

    def __init__(self) -> None:
        self._backend: WaveForcingBackend | None = None
        self._world_id: int | None = None
        self._chunk_index = 0

    def load(self, settings: Settings) -> None:
        """Load once, warm first/continuation shapes, then discard warmup state."""
        from waveforcing_backend import WaveForcingBackend

        self._backend = WaveForcingBackend(settings)
        self._backend.start("A quiet landscape in daylight.", seed=0)
        self._backend.step("A quiet landscape in daylight.")
        self._backend.step("A quiet landscape in daylight.")
        self.reset()

    def generate(self, input: WaveForcingInput) -> WaveForcingResult:
        """Finalize one chunk, raising NoAnchor for an unseen unanchored world."""
        if self._backend is None:
            raise RuntimeError("Load the model before generating")
        if not input.prompt.strip():
            raise ValueError("A prompt is required")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise NoAnchor("A fresh world requires its seed")
            self._backend.start(input.prompt, seed=input.anchor.seed)
            self._world_id = input.world_id
            self._chunk_index = 0
        frames = self._backend.step(input.prompt)
        if (
            frames.ndim != 4
            or frames.shape[1:] != (HEIGHT, WIDTH, 3)
            or frames.dtype != np.uint8
        ):
            raise RuntimeError("Unexpected native video shape or dtype")
        self._chunk_index += 1
        return WaveForcingResult(
            world_id=input.world_id, chunk_index=self._chunk_index, frames=frames
        )

    def reset(self) -> None:
        """Release video history and random state while retaining loaded weights."""
        if self._backend is not None:
            self._backend.reset()
        self._world_id = None
        self._chunk_index = 0
