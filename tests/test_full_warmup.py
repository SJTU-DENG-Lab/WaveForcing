"""Full benchmark warmup shares worker lifetime with the measured request."""
import tempfile
import unittest
from unittest.mock import Mock, patch

from wave_rt.config import WaveConfig
from wave_rt.launcher import _run_warmed


class RecordingQueue:
    def __init__(self, *args, **kwargs):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def cancel_join_thread(self):
        pass

    def close(self):
        pass


class FullWarmupTest(unittest.TestCase):
    def test_full_requests_share_processes_and_discard_warmup_metrics(self):
        for stages in (0, 3):
            with self.subTest(vae_stages=stages), tempfile.TemporaryDirectory() as out:
                cfg = WaveConfig(full_warmup_runs=1, warmup_frames=399,
                                 num_frames=399, vae_stages=stages, out_root=out)
                queues, processes = [], []
                def queue_factory(*args, **kwargs):
                    q = RecordingQueue()
                    queues.append(q)
                    return q
                def process_factory(*args, **kwargs):
                    process = Mock()
                    process.specification = kwargs
                    processes.append(process)
                    return process
                result = {'diff': (10000., 1., [78.]*137, 399), 'vae': (11000., 12.)}
                def receive(*args):
                    # Every worker receives warmup before measurement; same
                    # processes are already alive for both phases.
                    request_queues = queues[-cfg.n_total_gpus:]
                    phase = len(request_queues[0].items)-1
                    self.assertEqual(phase, receive.calls)
                    self.assertEqual(args[2], ('diff','vae') if stages else ('serial_done',))
                    for q in request_queues:
                        self.assertEqual(q.items[phase]['num_frames'], 399)
                        self.assertEqual(q.items[phase]['warmup'], phase == 0)
                    receive.calls += 1
                    return result if stages else {'serial_done': ()}
                receive.calls = 0
                with patch('wave_rt.launcher.mp.Queue', side_effect=queue_factory), \
                     patch('wave_rt.launcher.mp.Process', side_effect=process_factory), \
                     patch('wave_rt.launcher.mp.Barrier') as barrier, \
                     patch('wave_rt.distributed.supervisor.receive_results', side_effect=receive), \
                     patch('wave_rt.distributed.supervisor.join_workers') as join, \
                     patch('wave_rt.distributed.supervisor.stop_workers') as stop, \
                     patch('wave_rt.launcher._aggregate_metrics') as aggregate:
                    _run_warmed(cfg)
                    self.assertEqual(receive.calls, 2)
                    self.assertEqual(len(processes), cfg.n_total_gpus)
                    for process in processes:
                        process.start.assert_called_once()
                        spec = process.specification
                        if 'kwargs' in spec:
                            self.assertIs(spec['kwargs']['request_barrier'], barrier.return_value)
                        else:
                            self.assertIs(spec['args'][-1], barrier.return_value)
                    for q in queues[-cfg.n_total_gpus:]:
                        self.assertEqual(len(q.items), 3)
                        self.assertIsNone(q.items[-1])
                        self.assertIsNone(q.items[0]['out'])
                        self.assertIsNotNone(q.items[1]['out'])
                    self.assertEqual(aggregate.call_count, 1 if stages else 0)
                    join.assert_called_once()
                    stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
