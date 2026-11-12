"""Encoding overlap must preserve order, bounded memory and generation fences."""
import threading
import time
import unittest

from wave_rt.interactive.encoding import OrderedVideoEncoder


class EncoderTest(unittest.TestCase):
    def test_out_of_order_completion_stays_bounded_and_emits_while_producer_idle(self):
        slow, second_done, submitted, delivered = [threading.Event() for _ in range(4)]
        rows = []

        def encode(value, fps, kbps):
            if value == 0:
                self.assertTrue(slow.wait(3))
            elif value == 1:
                second_done.set()
            return bytes([value])

        def emit(row):
            rows.append(row)
            if len(rows) == 3:
                delivered.set()

        pool = OrderedVideoEncoder(emit, workers=2, capacity=2, encode_fn=encode)
        producer = None
        try:
            for chunk in (0, 1):
                pool.submit(chunk, epoch=7, chunk=chunk, decoded_at=time.time())
            self.assertTrue(second_done.wait(2))
            self.assertEqual(rows, [])

            def third():
                pool.submit(2, epoch=7, chunk=2, decoded_at=time.time())
                submitted.set()

            producer = threading.Thread(target=third)
            producer.start()
            self.assertFalse(submitted.wait(.1))  # Completed chunk 1 still occupies a slot.
            slow.set()
            self.assertTrue(submitted.wait(2))
            self.assertTrue(delivered.wait(2))  # No fourth submit/drain needed to flush chunk 2.
            self.assertEqual([row['chunk'] for row in rows], [0, 1, 2])
            self.assertEqual([row['data'] for row in rows], [b'\0', b'\1', b'\2'])
        finally:
            slow.set()
            if producer is not None:
                producer.join(3)
            pool.close()

    def test_reset_fence_and_epoch_tags(self):
        gate, finished = threading.Event(), threading.Event()
        rows = []

        def encode(value, *_):
            if value == 'old':
                self.assertTrue(gate.wait(3))
            return value.encode()

        pool = OrderedVideoEncoder(rows.append, encode_fn=encode)
        pool.submit('old', epoch=1, chunk=8, decoded_at=time.time())
        def reset():
            pool.drain()
            rows.append({'epoch': 2, 'kind': 'reset'})
            pool.submit('new', epoch=2, chunk=0, decoded_at=time.time())
            pool.drain()
            finished.set()
        thread = threading.Thread(target=reset)
        thread.start()
        try:
            self.assertFalse(finished.wait(.1))
            gate.set()
            self.assertTrue(finished.wait(2))
            self.assertEqual([(r['epoch'], r['kind']) for r in rows],
                             [(1, 'chunk'), (2, 'reset'), (2, 'chunk')])
        finally:
            gate.set()
            thread.join(3)
            pool.close()

    def test_failure_is_reported_even_without_another_submission(self):
        failed = threading.Event()
        rows = []
        def encode(*_):
            raise ValueError('codec failed')
        pool = OrderedVideoEncoder(rows.append, encode_fn=encode, on_error=lambda _: failed.set())
        pool.submit([], epoch=1, chunk=0, decoded_at=time.time())
        self.assertTrue(failed.wait(2))
        with self.assertRaisesRegex(RuntimeError, 'encoding failed'):
            pool.drain()
        with self.assertRaisesRegex(RuntimeError, 'encoding failed'):
            pool.submit([], epoch=1, chunk=1, decoded_at=time.time())
        with self.assertRaisesRegex(RuntimeError, 'encoding failed'):
            pool.close()
        self.assertEqual(rows, [])
        self.assertFalse(pool._consumer.is_alive())


if __name__ == '__main__':
    unittest.main()
