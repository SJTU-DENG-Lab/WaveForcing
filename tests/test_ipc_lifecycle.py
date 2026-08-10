import unittest
from unittest import mock

from wave_rt.denoiser.engine import WaveDenoiser
import wave_rt.denoiser.engine as engine_mod


class _FakeBackend:
    def __init__(self, events):
        self.events = events

    def barrier(self):
        self.events.append("barrier")


class _FakeP2P:
    def __init__(self, events):
        self.events = events
        self.next_ptr = 0x1000

    def ipc_open_handle(self, _handle_bytes):
        ptr = self.next_ptr
        self.next_ptr += 0x1000
        self.events.append(f"open:{ptr:#x}")
        return ptr

    def ipc_close_handle(self, ptr):
        self.events.append(f"close:{ptr:#x}")


class _FakeStream:
    def __init__(self, events):
        self.events = events

    def synchronize(self):
        self.events.append("copy-sync")


def _make_denoiser(events):
    denoiser = object.__new__(WaveDenoiser)
    denoiser.be = _FakeBackend(events)
    denoiser.device = "cuda:0"
    denoiser._p2p = _FakeP2P(events)
    denoiser._ipc_handles = []
    denoiser._closed = False
    denoiser._os_scopy = _FakeStream(events)
    denoiser._os_dst = {0: object()}
    denoiser._pg_dst = {0: object()}
    denoiser._rl_down = object()
    denoiser._os_pending = [object()]
    denoiser._st_prefetch = object()
    denoiser._st_ev = {0: object()}
    denoiser._os_kv = object()
    denoiser._os_flags = object()
    denoiser._pg_k = object()
    denoiser._pg_v = object()
    denoiser._rl_kv = object()
    denoiser._rl_flags = object()
    return denoiser


class IpcLifecycleTest(unittest.TestCase):
    def test_coordinated_close_pairs_every_open_in_reverse_order(self):
        events = []
        denoiser = _make_denoiser(events)
        first = denoiser._ipc_open(b"first")
        second = denoiser._ipc_open(b"second")

        with mock.patch.object(
            engine_mod.torch.cuda,
            "synchronize",
            side_effect=lambda device: events.append(f"device-sync:{device}"),
        ):
            denoiser.close()
            denoiser.close()  # idempotent: no second close or barrier

        self.assertEqual((first, second), (0x1000, 0x2000))
        self.assertEqual(
            events,
            [
                "open:0x1000",
                "open:0x2000",
                "copy-sync",
                "device-sync:cuda:0",
                "barrier",
                "close:0x2000",
                "close:0x1000",
                "barrier",
            ],
        )
        self.assertEqual(denoiser._ipc_handles, [])
        self.assertEqual(denoiser._os_dst, {})
        self.assertEqual(denoiser._pg_dst, {})
        self.assertIsNone(denoiser._rl_down)
        self.assertEqual(denoiser._os_pending, [])
        self.assertIsNone(denoiser._os_kv)
        self.assertIsNone(denoiser._os_flags)
        self.assertIsNone(denoiser._pg_k)
        self.assertIsNone(denoiser._pg_v)
        self.assertIsNone(denoiser._rl_kv)
        self.assertIsNone(denoiser._rl_flags)

    def test_exception_path_cleanup_never_enters_distributed_barrier(self):
        events = []
        denoiser = _make_denoiser(events)
        denoiser._ipc_open(b"peer")

        with mock.patch.object(
            engine_mod.torch.cuda,
            "synchronize",
            side_effect=lambda device: events.append(f"device-sync:{device}"),
        ):
            denoiser.close(coordinated=False)

        self.assertNotIn("barrier", events)
        self.assertEqual(events.count("close:0x1000"), 1)


if __name__ == "__main__":
    unittest.main()
