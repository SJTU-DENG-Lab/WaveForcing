"""Typed messages and lifecycle rules for resident WaveRT workers.

The module intentionally has no torch imports. Every object sent through a
multiprocessing queue remains cheap to validate and pickle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
import re
from typing import Any, Mapping
from uuid import uuid4


class ProtocolError(ValueError):
    """A request or worker event violates the serving contract."""


class StaleEventError(ProtocolError):
    """A worker event crossed a request boundary."""


class WorkerSetPoisoned(RuntimeError):
    """The distributed worker set must be replaced before reuse."""


class RestartBudgetExhausted(RuntimeError):
    """No full worker-set restart remains."""


class QueueFullError(RuntimeError):
    """The serving admission queue has reached its configured bound."""


class CommandKind(str, Enum):
    GENERATE = "generate"
    SHUTDOWN = "shutdown"


class EventKind(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


class WorkerSetState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    POISONED = "poisoned"
    STOPPED = "stopped"


class RequestStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: str
    seed: int
    num_frames: int
    out_dir: str
    height: int
    width: int
    warmup: bool = False

    def __post_init__(self) -> None:
        if not _REQUEST_ID.fullmatch(self.request_id):
            raise ProtocolError(
                "request_id must contain only letters, digits, '.', '_' or '-' "
                "and be at most 128 characters"
            )
        if not self.prompt.strip():
            raise ProtocolError("prompt must be non-empty")
        if self.num_frames <= 0:
            raise ProtocolError("num_frames must be positive")
        if self.height <= 0 or self.width <= 0:
            raise ProtocolError("height and width must be positive")
        if not self.warmup and not self.out_dir:
            raise ProtocolError("out_dir must be non-empty")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        cfg: Any,
        *,
        request_id: str | None = None,
        warmup: bool = False,
    ) -> "GenerationRequest":
        if not isinstance(payload, Mapping):
            raise ProtocolError("request body must be a JSON object")
        rid = str(request_id or payload.get("request_id") or uuid4().hex)
        try:
            num_frames = int(payload.get("num_frames", cfg.num_frames))
            seed = int(payload.get("seed", cfg.seed))
            height = int(payload.get("height", cfg.height))
            width = int(payload.get("width", cfg.width))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"invalid numeric request field: {exc}") from exc
        if num_frames % cfg.num_frames_per_block:
            raise ProtocolError(
                f"num_frames must be divisible by {cfg.num_frames_per_block}"
            )
        # VAE partition profiling and WaveRT communication buffers are built for
        # the launch shape. A shape change therefore requires a new worker set.
        if height != cfg.height or width != cfg.width:
            raise ProtocolError(
                f"resident workers use the launch resolution {cfg.height}x{cfg.width}; "
                "restart WaveRT to serve a different resolution"
            )
        out_dir = "" if warmup else str(
            payload.get("out")
            or os.path.join(cfg.out_root, "serve", rid)
        )
        return cls(
            request_id=rid,
            prompt=str(payload.get("prompt", cfg.prompt)),
            seed=seed,
            num_frames=num_frames,
            out_dir=out_dir,
            height=height,
            width=width,
            warmup=warmup,
        )


@dataclass(frozen=True)
class WorkerCommand:
    kind: CommandKind
    request: GenerationRequest | None = None

    def __post_init__(self) -> None:
        if self.kind is CommandKind.GENERATE and self.request is None:
            raise ProtocolError("generate command requires a request")
        if self.kind is CommandKind.SHUTDOWN and self.request is not None:
            raise ProtocolError("shutdown command cannot carry a request")

    @classmethod
    def generate(cls, request: GenerationRequest) -> "WorkerCommand":
        return cls(CommandKind.GENERATE, request)

    @classmethod
    def shutdown(cls) -> "WorkerCommand":
        return cls(CommandKind.SHUTDOWN)


@dataclass(frozen=True)
class WorkerEvent:
    request_id: str
    source: str
    kind: EventKind
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ProtocolError("event request_id must be non-empty")
        if not self.source:
            raise ProtocolError("event source must be non-empty")
        object.__setattr__(self, "payload", dict(self.payload))


class RequestEventCollector:
    """Collect one terminal event from each data-plane component."""

    def __init__(self, request_id: str, expected_sources: set[str]) -> None:
        if not request_id or not expected_sources:
            raise ProtocolError("collector needs a request id and expected sources")
        self.request_id = request_id
        self.expected_sources = frozenset(expected_sources)
        self._events: dict[str, WorkerEvent] = {}

    def ingest(self, event: WorkerEvent) -> None:
        if event.request_id != self.request_id:
            raise StaleEventError(
                f"event for {event.request_id!r} cannot satisfy {self.request_id!r}"
            )
        if event.source not in self.expected_sources:
            raise ProtocolError(f"unexpected event source: {event.source}")
        if event.source in self._events:
            raise ProtocolError(f"duplicate terminal event from {event.source}")
        self._events[event.source] = event

    @property
    def complete(self) -> bool:
        return self._events.keys() == self.expected_sources

    @property
    def failed(self) -> bool:
        return any(event.kind is EventKind.FAILED for event in self._events.values())

    def result(self) -> dict[str, WorkerEvent]:
        missing = self.expected_sources.difference(self._events)
        if missing:
            raise ProtocolError(f"missing terminal events: {sorted(missing)}")
        return dict(self._events)


class WorkerSetLifecycle:
    """Explicit lifecycle for one generation of distributed workers."""

    def __init__(self) -> None:
        self.state = WorkerSetState.STARTING
        self.reason: str | None = None

    def mark_ready(self) -> None:
        self._require(WorkerSetState.STARTING)
        self.state = WorkerSetState.READY

    def begin_request(self) -> None:
        if self.state is WorkerSetState.POISONED:
            raise WorkerSetPoisoned(self.reason or "worker set is poisoned")
        self._require(WorkerSetState.READY)
        self.state = WorkerSetState.BUSY

    def finish_request(self) -> None:
        self._require(WorkerSetState.BUSY)
        self.state = WorkerSetState.READY

    def poison(self, reason: str) -> None:
        if self.state is WorkerSetState.STOPPED:
            raise ProtocolError("cannot poison a stopped worker set")
        self.reason = reason or "unspecified worker failure"
        self.state = WorkerSetState.POISONED

    def stop(self) -> None:
        self.state = WorkerSetState.STOPPED

    def _require(self, expected: WorkerSetState) -> None:
        if self.state is not expected:
            raise ProtocolError(
                f"expected worker state {expected.value}, got {self.state.value}"
            )


class WorkerSetRecovery:
    """Bound full-set replacements; failed requests are never retried."""

    def __init__(self, max_restarts: int) -> None:
        if max_restarts < 0:
            raise ProtocolError("max_restarts must be non-negative")
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.reasons: list[str] = []

    @property
    def can_restart(self) -> bool:
        return self.restart_count < self.max_restarts

    def replace(self, lifecycle: WorkerSetLifecycle, reason: str) -> int:
        if lifecycle.state is not WorkerSetState.POISONED:
            raise ProtocolError("only a poisoned worker set may be replaced")
        if not self.can_restart:
            raise RestartBudgetExhausted(
                f"restart budget exhausted after {self.restart_count} replacements"
            )
        lifecycle.stop()
        self.restart_count += 1
        self.reasons.append(reason or "worker failure")
        return self.restart_count
