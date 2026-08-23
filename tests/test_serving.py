from __future__ import annotations

from dataclasses import replace
import pickle
import time
import unittest

from wave_rt.config import WaveConfig
from wave_rt.serving.protocol import (
    CommandKind,
    EventKind,
    GenerationRequest,
    ProtocolError,
    RequestEventCollector,
    RequestStatus,
    RestartBudgetExhausted,
    StaleEventError,
    WorkerCommand,
    WorkerEvent,
    WorkerSetLifecycle,
    WorkerSetRecovery,
    WorkerSetState,
)
from wave_rt.serving.runtime import RequestSession, WaveServingRuntime


class ServingProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = WaveConfig()

    def test_request_defaults_and_fixed_shape(self) -> None:
        request = GenerationRequest.from_payload(
            {"request_id": "demo-1", "prompt": "a wave", "num_frames": 24},
            self.cfg,
        )
        self.assertEqual(request.out_dir, f"{self.cfg.out_root}/serve/demo-1")
        self.assertEqual((request.height, request.width), (480, 832))
        with self.assertRaises(ProtocolError):
            GenerationRequest.from_payload(
                {"prompt": "a wave", "num_frames": 25}, self.cfg
            )
        with self.assertRaises(ProtocolError):
            GenerationRequest.from_payload(
                {"prompt": "a wave", "height": 720}, self.cfg
            )

    def test_request_id_is_safe_for_default_output_path(self) -> None:
        with self.assertRaises(ProtocolError):
            GenerationRequest.from_payload(
                {"request_id": "../escape", "prompt": "a wave"}, self.cfg
            )

    def test_worker_messages_are_picklable(self) -> None:
        request = GenerationRequest.from_payload(
            {"request_id": "pickle-1", "prompt": "a wave"}, self.cfg
        )
        messages = [
            WorkerCommand.generate(request),
            WorkerCommand.shutdown(),
            WorkerEvent(
                request.request_id,
                "diffusion",
                EventKind.PASSED,
                {"diffusion_ms": 1.0},
            ),
        ]
        self.assertEqual(
            [pickle.loads(pickle.dumps(item)) for item in messages], messages
        )
        self.assertIs(messages[1].kind, CommandKind.SHUTDOWN)

    def test_terminal_events_cannot_cross_requests(self) -> None:
        collector = RequestEventCollector("a", {"diffusion", "vae"})
        collector.ingest(WorkerEvent("a", "diffusion", EventKind.PASSED))
        with self.assertRaises(StaleEventError):
            collector.ingest(WorkerEvent("b", "vae", EventKind.PASSED))
        collector.ingest(WorkerEvent("a", "vae", EventKind.PASSED))
        self.assertTrue(collector.complete)

    def test_failure_poisons_and_replaces_the_full_worker_set(self) -> None:
        lifecycle = WorkerSetLifecycle()
        lifecycle.mark_ready()
        lifecycle.begin_request()
        lifecycle.poison("rank 2 exited")
        recovery = WorkerSetRecovery(max_restarts=1)
        self.assertEqual(recovery.replace(lifecycle, "rank 2 exited"), 1)
        self.assertIs(lifecycle.state, WorkerSetState.STOPPED)
        failed_again = WorkerSetLifecycle()
        failed_again.poison("rank 4 exited")
        with self.assertRaises(RestartBudgetExhausted):
            recovery.replace(failed_again, "rank 4 exited")

    def test_request_session_state_and_snapshot(self) -> None:
        request = GenerationRequest.from_payload(
            {"request_id": "state-1", "prompt": "a wave"}, self.cfg
        )
        session = RequestSession(request)
        self.assertIs(session.status, RequestStatus.QUEUED)
        session.mark_running()
        session.succeed({"fps": 117.7})
        snapshot = session.snapshot()
        self.assertEqual(snapshot["status"], "succeeded")
        self.assertEqual(snapshot["fps"], 117.7)
        self.assertTrue(session.wait(0))

    def test_result_reports_output_frames_and_e2e_fps(self) -> None:
        runtime = WaveServingRuntime(self.cfg, lambda: None, lambda: None)
        request = GenerationRequest.from_payload(
            {"request_id": "metrics-1", "prompt": "a wave"}, self.cfg
        )
        events = {
            "diffusion": WorkerEvent(
                "metrics-1",
                "diffusion",
                EventKind.PASSED,
                {
                    "diffusion_ms": 800.0,
                    "started_monotonic_s": 10.0,
                    "num_latent_frames": 24,
                },
            ),
            "vae": WorkerEvent(
                "metrics-1",
                "vae",
                EventKind.PASSED,
                {"vae_ms": 200.0, "completed_monotonic_s": 11.0},
            ),
        }
        result = runtime._build_result(request, events)
        self.assertEqual(result["num_output_frames"], 96)
        self.assertEqual(result["fps"], 96.0)

    def test_dispatcher_preserves_fifo_and_recovers_for_queued_work(self) -> None:
        class FakeRuntime(WaveServingRuntime):
            def __init__(self, cfg):
                super().__init__(cfg, lambda: None, lambda: None)
                self.executed = []

            def _start_worker_set(self):
                return None

            def _warmup(self):
                return None

            def _stop_worker_set(self, *, graceful):
                self.lifecycle.stop()

            def _execute(self, request):
                self.executed.append(request.request_id)
                if request.request_id == "bad":
                    raise RuntimeError("injected failure")
                return {"request_id": request.request_id, "fps": 1.0}

        runtime = FakeRuntime(replace(self.cfg, serve_max_restarts=1))
        runtime.start()
        deadline = time.time() + 1
        while runtime.status()["status"] == "starting" and time.time() < deadline:
            time.sleep(0.001)
        bad = runtime.submit(
            {"request_id": "bad", "prompt": "a wave"}, wait=True
        )
        good = runtime.submit(
            {"request_id": "good", "prompt": "another wave"}, wait=True
        )
        runtime.shutdown()
        self.assertEqual(bad["status"], "failed")
        self.assertEqual(good["status"], "succeeded")
        self.assertEqual(runtime.executed, ["bad", "good"])
        self.assertEqual(runtime.recovery.restart_count, 1)


if __name__ == "__main__":
    unittest.main()
