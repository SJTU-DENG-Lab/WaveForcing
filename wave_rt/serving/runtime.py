"""Resident worker supervision and FIFO request scheduling for WaveRT."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
import os
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Callable

from wave_rt.config import WaveConfig
from wave_rt.serving.protocol import (
    EventKind,
    GenerationRequest,
    ProtocolError,
    QueueFullError,
    RequestEventCollector,
    RequestStatus,
    RestartBudgetExhausted,
    WorkerCommand,
    WorkerEvent,
    WorkerSetLifecycle,
    WorkerSetRecovery,
    WorkerSetState,
)


class WorkerProcessError(RuntimeError):
    """One or more ranks exited or stopped producing terminal events."""


@dataclass
class RequestSession:
    """Thread-safe state for one admitted generation request."""

    request: GenerationRequest
    status: RequestStatus = RequestStatus.QUEUED
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark_running(self) -> None:
        with self._lock:
            if self.status is not RequestStatus.QUEUED:
                raise ProtocolError(f"cannot run request in {self.status.value} state")
            self.status = RequestStatus.RUNNING
            self.started_at = time.time()

    def succeed(self, result: dict[str, Any]) -> None:
        with self._lock:
            if self.status is not RequestStatus.RUNNING:
                raise ProtocolError(f"cannot finish request in {self.status.value} state")
            self.status = RequestStatus.SUCCEEDED
            self.result = dict(result)
            self.completed_at = time.time()
            self._done.set()

    def fail(self, error: str) -> None:
        with self._lock:
            if self.status not in (RequestStatus.QUEUED, RequestStatus.RUNNING):
                return
            self.status = RequestStatus.FAILED
            self.error = error
            self.completed_at = time.time()
            self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            row: dict[str, Any] = {
                "request_id": self.request.request_id,
                "status": self.status.value,
                "submitted_at": self.submitted_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            }
            if self.started_at is not None:
                row["queue_s"] = round(self.started_at - self.submitted_at, 3)
            if self.completed_at is not None and self.started_at is not None:
                row["service_s"] = round(self.completed_at - self.started_at, 3)
            if self.result is not None:
                row.update(self.result)
            if self.error is not None:
                row["error"] = self.error
            return row


@dataclass(frozen=True)
class _Job:
    session: RequestSession


class WaveServingRuntime:
    """Keep all ranks resident and serialize requests onto one wavefront.

    HTTP clients may submit concurrently, but the data plane stays FIFO because
    every WaveRT rank must execute collectives in the same request order.
    """

    _STOP = object()

    def __init__(
        self,
        cfg: WaveConfig,
        diffusion_worker: Callable[..., None],
        vae_worker: Callable[..., None],
    ) -> None:
        if cfg.vae_stages <= 0:
            raise ProtocolError("resident serving requires vae_stages > 0")
        self.cfg = cfg
        self.diffusion_worker = diffusion_worker
        self.vae_worker = vae_worker
        self.jobs: Queue[_Job | object] = Queue(maxsize=cfg.serve_queue_size)
        self.sessions: OrderedDict[str, RequestSession] = OrderedDict()
        self.lifecycle = WorkerSetLifecycle()
        self.recovery = WorkerSetRecovery(cfg.serve_max_restarts)
        self._sessions_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._dispatcher: threading.Thread | None = None
        self._accepting = False
        self._active_request_id: str | None = None
        self._fatal_error: str | None = None
        self._processes: list[Any] = []
        self._request_queues: list[Any] = []
        self._latent_queue: Any = None
        self._event_queue: Any = None
        self._worker_cfg = cfg

    def start(self) -> None:
        with self._state_lock:
            if self._dispatcher is not None:
                raise ProtocolError("serving runtime has already been started")
            self._accepting = True
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name="wave-rt-dispatcher",
                daemon=True,
            )
            self._dispatcher.start()

    def submit(
        self,
        payload: dict[str, Any],
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        with self._state_lock:
            if not self._accepting:
                raise ProtocolError("serving runtime is not accepting requests")
        request = GenerationRequest.from_payload(payload, self.cfg)
        session = RequestSession(request)
        with self._sessions_lock:
            if request.request_id in self.sessions:
                raise ProtocolError(f"duplicate request_id: {request.request_id}")
            self.sessions[request.request_id] = session
        try:
            self.jobs.put_nowait(_Job(session))
        except Full as exc:
            with self._sessions_lock:
                self.sessions.pop(request.request_id, None)
            raise QueueFullError(
                f"serving queue is full (capacity={self.cfg.serve_queue_size})"
            ) from exc
        if wait:
            finished = session.wait(timeout)
            result = session.snapshot()
            if not finished:
                result["wait_timed_out"] = True
            return result
        return session.snapshot()

    def request_status(self, request_id: str) -> dict[str, Any] | None:
        with self._sessions_lock:
            session = self.sessions.get(request_id)
        return session.snapshot() if session is not None else None

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            accepting = self._accepting
            active = self._active_request_id
            fatal_error = self._fatal_error
        return {
            "status": self.lifecycle.state.value,
            "accepting": accepting,
            "pending": self.jobs.qsize(),
            "active_request_id": active,
            "worker_generation": self.recovery.restart_count,
            "restart_count": self.recovery.restart_count,
            "restart_reasons": list(self.recovery.reasons),
            "fatal_error": fatal_error,
        }

    def shutdown(self, *, drain: bool = True, timeout: float = 180.0) -> None:
        with self._state_lock:
            self._accepting = False
        if not drain:
            self._fail_pending("service stopped before request execution")
        dispatcher = self._dispatcher
        if dispatcher is None or not dispatcher.is_alive():
            return
        self.jobs.put(self._STOP)
        dispatcher.join(timeout=timeout)
        if dispatcher.is_alive():
            self._stop_worker_set(graceful=False)
            raise TimeoutError("serving dispatcher did not stop cleanly")

    def _dispatch_loop(self) -> None:
        try:
            if not self._boot_worker_set():
                self._fail_pending(self._fatal_error or "worker startup failed")
                return
            while True:
                item = self.jobs.get()
                if item is self._STOP:
                    break
                assert isinstance(item, _Job)
                session = item.session
                request_id = session.request.request_id
                with self._state_lock:
                    self._active_request_id = request_id
                session.mark_running()
                self.lifecycle.begin_request()
                try:
                    result = self._execute(session.request)
                except BaseException as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    self.lifecycle.poison(reason)
                    session.fail(reason)
                    if not self._recover_worker_set(reason):
                        self._set_fatal(reason)
                        self._fail_pending(reason)
                        break
                else:
                    self.lifecycle.finish_request()
                    session.succeed(result)
                finally:
                    with self._state_lock:
                        self._active_request_id = None
                    self._trim_history()
        except BaseException as exc:
            reason = f"dispatcher failed: {type(exc).__name__}: {exc}"
            self._set_fatal(reason)
            self._fail_pending(reason)
        finally:
            self._stop_worker_set(graceful=self.lifecycle.state is not WorkerSetState.POISONED)

    def _boot_worker_set(self) -> bool:
        while True:
            try:
                self._start_worker_set()
                self._warmup()
                self.lifecycle.mark_ready()
                return True
            except BaseException as exc:
                reason = f"worker startup failed: {type(exc).__name__}: {exc}"
                self.lifecycle.poison(reason)
                if not self._recover_worker_set(reason):
                    self._set_fatal(reason)
                    return False

    def _recover_worker_set(self, reason: str) -> bool:
        try:
            generation = self.recovery.replace(self.lifecycle, reason)
        except RestartBudgetExhausted:
            return False
        self._stop_worker_set(graceful=False)
        self.lifecycle = WorkerSetLifecycle()
        print(
            f"[wave_rt/serve] restarting full worker set: generation={generation} "
            f"reason={reason}",
            flush=True,
        )
        try:
            self._start_worker_set()
            self._warmup()
            self.lifecycle.mark_ready()
            return True
        except BaseException as exc:
            retry_reason = f"replacement startup failed: {type(exc).__name__}: {exc}"
            self.lifecycle.poison(retry_reason)
            return self._recover_worker_set(retry_reason)

    def _start_worker_set(self) -> None:
        import torch.multiprocessing as mp

        generation = self.recovery.restart_count
        port_offset = generation * 10
        self._worker_cfg = replace(
            self.cfg,
            master_port=self.cfg.master_port + port_offset,
            vae_port=self.cfg.vae_port + port_offset,
        )
        os.environ["MASTER_PORT"] = str(self._worker_cfg.master_port)
        n_diff, n_vae = self.cfg.wp_size, self.cfg.vae_stages
        self._request_queues = [mp.Queue() for _ in range(n_diff + n_vae)]
        self._latent_queue = mp.Queue(maxsize=64)
        self._event_queue = mp.Queue()
        self._processes = [
            mp.Process(
                name=f"wave-diffusion-{rank}",
                target=self.diffusion_worker,
                args=(
                    rank,
                    self._worker_cfg,
                    self._latent_queue,
                    self._event_queue,
                    self._request_queues[rank],
                ),
            )
            for rank in range(n_diff)
        ]
        for stage in range(n_vae):
            self._processes.append(
                mp.Process(
                    name=f"wave-vae-{stage}",
                    target=self.vae_worker,
                    args=(
                        stage,
                        n_diff + stage,
                        self._latent_queue,
                        self._worker_cfg,
                        "",
                        self._event_queue,
                        self._request_queues[n_diff + stage],
                    ),
                )
            )
        for process in self._processes:
            process.start()

    def _warmup(self) -> None:
        request = GenerationRequest.from_payload(
            {
                "prompt": self.cfg.prompt,
                "seed": 0,
                "num_frames": self.cfg.warmup_frames,
            },
            self._worker_cfg,
            request_id=f"warmup-{self.recovery.restart_count}",
            warmup=True,
        )
        print(
            f"[wave_rt/serve] warming resident workers with "
            f"{request.num_frames} latent frames...",
            flush=True,
        )
        started = time.perf_counter()
        self._execute(request)
        print(
            f"[wave_rt/serve] workers ready in {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    def _execute(self, request: GenerationRequest) -> dict[str, Any]:
        if not request.warmup:
            os.makedirs(request.out_dir, exist_ok=True)
        command = WorkerCommand.generate(request)
        for request_queue in self._request_queues:
            request_queue.put(command)
        collector = RequestEventCollector(request.request_id, {"diffusion", "vae"})
        timeout_s = self.cfg.serve_request_timeout_s
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        while not collector.complete:
            self._assert_workers_healthy()
            wait_s = 0.25
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"request {request.request_id} exceeded {timeout_s:.1f}s"
                    )
                wait_s = min(wait_s, remaining)
            try:
                event = self._event_queue.get(timeout=wait_s)
            except Empty:
                continue
            if not isinstance(event, WorkerEvent):
                raise ProtocolError(f"unexpected worker event: {type(event).__name__}")
            collector.ingest(event)
        events = collector.result()
        if collector.failed:
            errors = [
                str(event.payload.get("error", "worker failure"))
                for event in events.values()
                if event.kind is EventKind.FAILED
            ]
            raise WorkerProcessError("; ".join(errors))
        return self._build_result(request, events)

    def _build_result(
        self,
        request: GenerationRequest,
        events: dict[str, WorkerEvent],
    ) -> dict[str, Any]:
        diff = events["diffusion"].payload
        vae = events["vae"].payload
        diffusion_ms = float(diff["diffusion_ms"])
        vae_ms = float(vae["vae_ms"])
        start = float(diff["started_monotonic_s"])
        end = float(vae["completed_monotonic_s"])
        latent_frames = int(diff["num_latent_frames"])
        e2e_s = max(0.0, end - start)
        output_frames = latent_frames * 4
        return {
            "request_id": request.request_id,
            "video": (
                os.path.join(request.out_dir, "video.mp4")
                if self.cfg.save_video and not request.warmup
                else None
            ),
            "latents": (
                os.path.join(request.out_dir, "latents.pt")
                if not request.warmup
                else None
            ),
            "num_latent_frames": latent_frames,
            "num_output_frames": output_frames,
            "diffusion_s": round(diffusion_ms / 1000.0, 3),
            "vae_s": round(vae_ms / 1000.0, 3),
            "end_to_end_s": round(e2e_s, 3),
            "fps": round(output_frames / e2e_s, 2) if e2e_s else None,
            "worker_generation": self.recovery.restart_count,
        }

    def _assert_workers_healthy(self) -> None:
        dead = [
            f"{process.name}(exit={process.exitcode})"
            for process in self._processes
            if process.exitcode is not None
        ]
        if dead:
            raise WorkerProcessError("worker processes exited: " + ", ".join(dead))

    def _stop_worker_set(self, *, graceful: bool) -> None:
        processes = self._processes
        if not processes:
            return
        if graceful and self._request_queues:
            command = WorkerCommand.shutdown()
            for request_queue, process in zip(self._request_queues, processes):
                if process.is_alive():
                    request_queue.put(command)
        join_deadline = time.monotonic() + (120.0 if graceful else 3.0)
        for process in processes:
            process.join(timeout=max(0.0, join_deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=3.0)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=3.0)
        for queue in [*self._request_queues, self._latent_queue, self._event_queue]:
            if queue is None:
                continue
            try:
                queue.close()
                queue.cancel_join_thread()
            except Exception:
                pass
        self._processes = []
        self._request_queues = []
        self._latent_queue = None
        self._event_queue = None
        if self.lifecycle.state is not WorkerSetState.STOPPED:
            self.lifecycle.stop()

    def _fail_pending(self, reason: str) -> None:
        while True:
            try:
                item = self.jobs.get_nowait()
            except Empty:
                return
            if isinstance(item, _Job):
                item.session.fail(reason)

    def _set_fatal(self, reason: str) -> None:
        with self._state_lock:
            self._fatal_error = reason
            self._accepting = False

    def _trim_history(self) -> None:
        limit = self.cfg.serve_history_size
        with self._sessions_lock:
            while len(self.sessions) > limit:
                request_id, session = next(iter(self.sessions.items()))
                if session.status in (RequestStatus.QUEUED, RequestStatus.RUNNING):
                    break
                self.sessions.pop(request_id)
