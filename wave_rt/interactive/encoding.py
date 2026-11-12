"""Bounded CPU encoding, overlapped with VAE work and emitted in chunk order."""

from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time

from .video import encode_chunk


class OrderedVideoEncoder:
    """One producer, independent encoders, one ordered completion thread.

    Capacity counts running, queued and completed-but-not-emitted chunks. The
    producer retains at most one additional decoded chunk while waiting for a
    slot. Each task owns its codec/container, preserving independent IDRs and
    existing quality settings. drain() is a fence before a generation reset.
    """

    def __init__(self, emit, fps=16, kbps=None, *, workers=2, capacity=3,
                 encode_fn=encode_chunk, on_error=None):
        if workers < 1 or capacity < workers:
            raise ValueError("Require capacity >= workers >= 1")
        self.emit, self.fps, self.kbps = emit, fps, kbps
        self.encode_fn, self.on_error = encode_fn, on_error
        self._slots = threading.BoundedSemaphore(capacity)
        self._pending = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wavert-encode")
        self._error = None
        self._closed = False
        self._consumer = threading.Thread(target=self._deliver, name="wavert-encoded", daemon=True)
        self._consumer.start()

    def _check(self):
        if self._error is not None:
            raise RuntimeError("Video encoding failed") from self._error

    def submit(self, frames, *, epoch, chunk, decoded_at):
        if self._closed:
            raise RuntimeError("Video encoder is closed")
        self._check()
        while not self._slots.acquire(timeout=0.1):
            self._check()
        try:
            self._check()
            future = self._pool.submit(self._encode, frames, epoch, chunk, decoded_at)
            self._pending.put(future)
        except BaseException:
            self._slots.release()
            raise

    def _encode(self, frames, epoch, chunk, decoded_at):
        started_at = time.time()
        data = self.encode_fn(frames, self.fps, self.kbps)
        return {"kind": "chunk", "epoch": epoch, "chunk": chunk,
                "data": data, "size": len(data), "decoded_at": decoded_at,
                "encode_started_at": started_at, "encoded_at": time.time()}

    def _deliver(self):
        while True:
            future = self._pending.get()
            try:
                if future is None:
                    return
                event = future.result()
                if self._error is None:
                    self.emit(event)
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
                    # Report even when the GPU worker is parked on commands.
                    if self.on_error is not None:
                        try:
                            self.on_error(exc)
                        except BaseException:
                            pass  # Preserve the original error for drain/submit.
            finally:
                if future is not None:
                    self._slots.release()
                self._pending.task_done()

    def drain(self):
        """Wait for every earlier submission to be emitted, or raise its error."""
        self._pending.join()
        self._check()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.drain()
        finally:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pending.put(None)
            self._consumer.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
