"""Generation isolation and memory-monitor failure reporting."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from interactive_fakes import FakeRuntime, wait_until
from wave_rt.interactive.server import Controller, Trace, make_app
from wave_rt.interactive.state import Limits
from wave_rt.interactive.metrics import GpuMonitor, parse_gpu_rows


class RegenerateTest(unittest.TestCase):
    def test_restart_isolates_the_old_generation(self):
        runtime = FakeRuntime()
        controller = Controller(runtime, "old", Limits())
        with TestClient(make_app(controller)) as client:
            token = client.post('/attach').json()['client']
            old = {'X-Client-Id': token, 'X-Session-Id': controller.session_id}
            wait_until(lambda: controller.session.end == 3)
            self.assertEqual(client.get('/chunk/0/0', headers=old).content, b'old')
            runtime.gate.clear()  # Hold the workers inside the restart.
            body = {'prompt': 'new beginning', 'request_id': 'restart-1'}
            started = client.post('/start', headers=old, json=body)
            self.assertEqual(started.status_code, 200)
            fresh = started.json()
            self.assertTrue(fresh['resetting'])
            self.assertEqual((fresh['played'], fresh['end']), (-1, 0))
            new = {**old, 'X-Session-Id': fresh['session_id']}
            self.assertNotEqual(old['X-Session-Id'], new['X-Session-Id'])
            self.assertEqual(client.post('/start', headers=old, json=body).json()['session_id'], fresh['session_id'])
            for endpoint, payload in [('/played', {'branch': 0, 'chunk': 0}), ('/prompt', {'prompt': 'stale'})]:
                self.assertEqual(client.post(endpoint, headers=old, json=payload).status_code, 409)
            self.assertEqual(client.get('/chunk/0/0', headers=old).status_code, 409)
            self.assertEqual(client.post('/prompt', headers=new, json={'prompt': 'early'}).status_code, 409)
            self.assertEqual(client.post('/start', headers=new, json={**body, 'request_id': 'restart-2'}).status_code, 409)
            self.assertEqual(client.get('/chunk/0/0', headers=new).status_code, 404)
            runtime.gate.set()
            wait_until(lambda: controller.session.end == 3 and not controller.resetting)
            self.assertEqual(runtime.calls, [('reset', 'old'), ('advance', 3), ('reset', 'new beginning'), ('advance', 3)])
            self.assertEqual(client.get('/chunk/0/0', headers=new).content, b'new beginning')
            self.assertEqual(controller.session.played, -1)
            self.assertNotEqual(fresh['seed'], None)

    def test_events_of_a_replaced_generation_are_dropped(self):
        runtime = FakeRuntime()
        controller = Controller(runtime, 'old', Limits(), autostart=False)
        with controller.lock:
            controller.start('first', 'a', controller.session_id)
        wait_until(lambda: controller.session.end == 3)
        stale = controller.epoch
        with controller.lock:
            controller.start('second', 'b', controller.session_id)
        now = time.time()
        runtime.on_event({'kind': 'chunk', 'epoch': stale, 'chunk': 3, 'data': b'late', 'size': 4,
                          'decoded_at': now, 'encoded_at': now})
        self.assertIsNone(controller.error)
        wait_until(lambda: controller.session.end == 3 and not controller.resetting)
        self.assertEqual(controller.frames.get(0, 0), b'second')
        controller.close()

    def test_manual_start_is_idle_and_requires_prompt(self):
        runtime = FakeRuntime()
        controller = Controller(runtime, 'default', Limits(), autostart=False)
        with TestClient(make_app(controller)) as client:
            token = client.post('/attach').json()['client']
            headers = {'X-Client-Id': token, 'X-Session-Id': controller.session_id}
            time.sleep(.1)
            self.assertEqual((runtime.calls, runtime.ticks), ([], []))
            state = client.get('/state', headers=headers).json()
            self.assertFalse(state['has_session'])
            self.assertEqual(state['initial_prompt'], '')
            self.assertEqual(client.post('/start', headers=headers, json={'prompt': '  ', 'request_id': 'bad'}).status_code, 409)
            self.assertFalse(controller.has_session)
            self.assertEqual(client.post('/prompt', headers=headers, json={'prompt': 'later'}).status_code, 409)
            self.assertEqual(client.get('/metrics').status_code, 409)
            self.assertEqual(client.get('/metrics', headers=headers).json()['status'], 'unavailable')


class TraceTest(unittest.TestCase):
    def test_rows_are_bounded_in_memory_and_appended_to_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            trace = Trace(str(path), keep=2)
            for n in range(3):
                trace.add('server', 'tick', tick=n)
            trace.add('client', 'sample', row={'seq': 1})
            self.assertEqual([row['tick'] for row in trace.rows['server']], [1, 2])
            wait_until(lambda: path.read_text().count('\n') == 4)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows], ['tick', 'tick', 'tick', 'sample'])
            self.assertIsNone(trace.error)


class MetricsTest(unittest.TestCase):
    def test_parse_and_do_not_report_missing_memory_as_zero(self):
        rows = parse_gpu_rows('0, NVIDIA H200, 1024, 143771, 25\n1, NVIDIA H200, N/A, 143771, N/A', ['去噪', 'VAE'])
        self.assertEqual(rows[0]['used_mib'], 1024)
        self.assertIsNone(rows[1]['used_mib'])
        self.assertIsNone(rows[1]['utilization_percent'])
        self.assertEqual(rows[1]['role'], 'VAE')

    def test_failed_sampler_preserves_readings_and_marks_stale(self):
        fail = threading.Event()
        def query():
            if fail.is_set():
                raise TimeoutError()
            return '0, NVIDIA H200, 1024, 143771, 25'
        monitor = GpuMonitor('0', ['KV Store'], interval=.02, query=query)
        try:
            deadline = time.monotonic() + 2
            while monitor.snapshot()['status'] != 'ok':
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            fail.set()
            while monitor.snapshot()['status'] != 'stale':
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            snapshot = monitor.snapshot()
            self.assertEqual(snapshot['used_mib'], 1024)
            self.assertEqual(snapshot['error'], 'TimeoutError')
            self.assertGreaterEqual(snapshot['age_seconds'], 0)
        finally:
            monitor.close()
