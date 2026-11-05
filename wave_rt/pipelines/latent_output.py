"""Bounded asynchronous D2H staging for the diffusion-to-VAE queue."""
import queue
import threading
import torch


class LatentQueueWriter:
    def __init__(self, output, device, slots=2, trace=None):
        if slots < 1:
            raise ValueError('At least one latent staging slot is required')
        self.output, self.device = output, device
        self.trace = trace
        self.stream = torch.cuda.Stream(device=device)
        self.free = queue.Queue(maxsize=slots)
        self.pending = queue.Queue(maxsize=slots)
        for _ in range(slots):
            self.free.put(None)
        self.error = None
        self.closed = False
        self.worker = threading.Thread(target=self._drain, daemon=True)
        self.worker.start()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('Latent queue writer failed') from self.error

    def _enqueue(self, item):
        while True:
            self._check()
            try:
                self.pending.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def submit(self, chunk, latent):
        self._check()
        if self.closed:
            raise RuntimeError('Latent writer is closed')
        if self.trace:
            self.trace.point('latent_ready', chunk=chunk)
            slot_wait = self.trace.begin('staging_slot_wait', chunk=chunk)
        while True:
            self._check()
            try:
                host = self.free.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        if self.trace:
            self.trace.end(slot_wait)
        self._check()
        value = latent.permute(0, 2, 1, 3, 4).contiguous()
        if value.dtype != torch.bfloat16:
            raise ValueError('Latent transport expects BF16')
        if host is None or host.shape != value.shape:
            host = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
        ready, complete = torch.cuda.Event(), torch.cuda.Event()
        ready.record()
        self.stream.wait_event(ready)
        with torch.cuda.stream(self.stream):
            if self.trace:
                transfer = self.trace.begin('latent_d2h', chunk=chunk, gpu=True, stream=self.stream)
            host.copy_(value, non_blocking=True)
            complete.record()
            if self.trace:
                self.trace.end(transfer)
        value.record_stream(self.stream)
        if self.trace:
            pending = self.trace.begin('staging_enqueue_wait', chunk=chunk)
        self._enqueue((chunk, host, complete))
        if self.trace:
            self.trace.end(pending)

    def _drain(self):
        try:
            while True:
                item = self.pending.get()
                if item is None:
                    return
                chunk, host, complete = item
                if self.trace:
                    copy_wait = self.trace.begin('d2h_completion_wait', chunk=chunk)
                complete.synchronize()
                if self.trace:
                    self.trace.end(copy_wait)
                # NumPy has no BF16 dtype. Preserve bits as uint16 and copy before
                # returning the slot: mp.Queue serializes in a separate feeder.
                payload = host.view(torch.uint16).numpy().copy()
                if self.trace:
                    publish = self.trace.begin('latent_queue_put', chunk=chunk)
                self.output.put((chunk, payload))
                if self.trace:
                    self.trace.end(publish)
                self.free.put(host)
        except BaseException as exc:
            self.error = exc
            # Wake a producer blocked waiting for a staging slot.
            try:
                self.free.put_nowait(None)
            except queue.Full:
                pass

    def close(self):
        if self.closed:
            self._check()
            return
        self.closed = True
        self._check()
        self._enqueue(None)
        self.worker.join()
        self._check()


def latent_from_payload(array, device):
    tensor = torch.from_numpy(array)
    if tensor.dtype == torch.uint16:
        tensor = tensor.view(torch.bfloat16)
    return tensor.to(device=device, dtype=torch.bfloat16)
