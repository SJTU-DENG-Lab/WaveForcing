"""CPU-only session policy and reference-counted chunk pages."""

from dataclasses import dataclass, field
from typing import Any


class CapacityError(RuntimeError):
    pass


class Pages:
    """One physical payload per page; forks share immutable prefix pages.

    Payloads can be GPU tensors, decoder checkpoints or encoded RGB. No model
    attention indices are changed by this storage policy.
    """

    def __init__(self, budget: int):
        self.budget = budget
        self.branches: dict[int, dict[int, tuple[Any, int]]] = {0: {}} # branch_id -> chunk_id -> (payload, size)

    @property
    def used(self) -> int:
        unique = {id(page): page for b in self.branches.values() for page in b.values()}
        return sum(page[1] for page in unique.values())

    def put(self, branch: int, chunk: int, payload: Any, size: int) -> None:
        if chunk in self.branches[branch]:
            raise ValueError("A committed page is immutable")
        if size < 0 or self.used + size > self.budget:
            raise CapacityError("Session page budget exhausted")
        self.branches[branch][chunk] = (payload, size)

    def get(self, branch: int, chunk: int) -> Any:
        return self.branches[branch][chunk][0]

    def fork(self, parent: int, branch: int, before: int) -> None:
        if branch in self.branches:
            raise ValueError("Branch already exists")
        self.branches[branch] = {
            c: page for c, page in self.branches[parent].items() if c < before
        }

    def prune(self, floor: int, keep: tuple[int, ...] = (0,)) -> None:
        for branch in self.branches.values():
            for chunk in list(branch):
                if chunk < floor and chunk not in keep:
                    del branch[chunk]

    def retain_branches(self, branches: set[int]) -> None:
        for branch in list(self.branches):
            if branch not in branches:
                del self.branches[branch]


@dataclass(frozen=True)
class Limits:
    # Chunks kept ahead of the playback cursor. The resident pipeline commits
    # one chunk per tick, so this only has to cover delivery jitter; it is
    # also how long an already committed prompt keeps playing after a change.
    buffer: int = 3
    # Allow brief playback-ACK delays without letting /live recursively grow
    # the committed timeline. At 16 FPS, two extra chunks cover 1.5 seconds.
    delivery_slack: int = 2

    def __post_init__(self):
        if self.buffer < 1:
            raise ValueError("Require at least one buffered chunk")
        if self.delivery_slack < 0:
            raise ValueError("Delivery slack cannot be negative")

    @property
    def max_ahead(self):
        return self.buffer + self.delivery_slack


@dataclass
class Session:
    """One linear timeline paced by playback.

    `issued` is the frontier the pipeline has been told to commit. Delivery
    can advance it briefly without playback ACKs, up to max_ahead chunks.
    """

    prompt: str
    limits: Limits = field(default_factory=Limits)
    played: int = -1  # Last fully displayed chunk, never a download ACK.
    pace: int = -1  # Delivery cursor (live stream); generation follows max(played, pace).
    end: int = 0  # Exclusive frontier of encoded chunks.
    issued: int = 0  # Exclusive frontier of committed chunks.
    boundary: int = 0  # First chunk of the current prompt.
    pending_prompt: str | None = None

    def interact(self, prompt: str) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be nonempty")
        self.pending_prompt = prompt

    def acknowledge(self, chunk: int) -> None:
        if not isinstance(chunk, int) or isinstance(chunk, bool) or not -1 <= chunk < self.end:
            raise ValueError("Playback acknowledgement is outside generated content")
        # Delayed/repeated ACKs cannot move the playback cursor back.
        self.played = max(self.played, chunk)

    def advance_pace(self, chunk: int) -> None:
        """How far the live stream has handed bytes to the viewer."""
        if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < -1:
            raise ValueError("Delivery pace is outside the timeline")
        self.pace = max(self.pace, chunk)

    def schedule(self) -> list[dict]:
        """Pipeline commands owed since the last call, in execution order."""
        commands = []
        if self.pending_prompt is not None:
            # Committed chunks keep the old prompt; the new one starts right
            # behind them, so the change needs no rollback of model state.
            self.prompt, self.pending_prompt = self.pending_prompt, None
            self.boundary = self.issued
            commands.append({"prompt": self.prompt, "start": self.boundary})
        # Pace on delivery when the live stream is ahead of /played. A stalled
        # playback ACK must not park generation while the browser still reads.
        target = max(self.played, self.pace) + 1 + self.limits.buffer
        target = min(target, self.played + 1 + self.limits.max_ahead)
        if target > self.issued:
            self.issued = target
            commands.append({"end": target})
        return commands

    def produced(self, chunk: int) -> None:
        if chunk != self.end or chunk >= self.issued:
            raise RuntimeError("Encoded chunk is outside the committed timeline")
        self.end = chunk + 1

    def view(self) -> dict:
        # `active`/`offer` are constant: the browser player still speaks the
        # earlier branch protocol and reads a single branch with no offers.
        switching = self.pending_prompt is not None or (
            self.boundary > 0 and self.end <= self.boundary)
        return {"active": 0, "played": self.played, "end": self.end, "offer": None,
                "pending": switching, "paused": self.end == self.issued,
                "busy": self.end < self.issued, "issued": self.issued,
                "max_ahead": self.limits.max_ahead}
