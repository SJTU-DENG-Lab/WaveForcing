import queue
import time
import unittest

from interactive_fakes import FakeRuntime, wait_until
from wave_rt.interactive.native import Tape
from wave_rt.interactive.state import CapacityError, Limits, Pages, Session
from wave_rt.interactive.server import Controller


def produce(session):
    while session.end < session.issued:
        session.produced(session.end)


class PagesTest(unittest.TestCase):
    def test_fork_shares_prefix_and_excludes_old_future(self):
        pages = Pages(40)
        for c in range(4):
            pages.put(0, c, object(), 10)
        pages.fork(0, 1, 2)
        self.assertEqual(pages.used, 40)
        self.assertIs(pages.get(0, 1), pages.get(1, 1))
        self.assertNotIn(2, pages.branches[1])
        with self.assertRaises(CapacityError):
            pages.put(1, 2, object(), 10)
        pages.retain_branches({1})
        pages.put(1, 2, object(), 10)
        self.assertEqual(pages.used, 30)

    def test_eviction_obeys_playback_floor_and_pins_anchor(self):
        pages = Pages(100)
        for c in range(10):
            pages.put(0, c, c, 10)
        pages.prune(4)
        self.assertEqual(set(pages.branches[0]), {0, 4, 5, 6, 7, 8, 9})
        with self.assertRaises(ValueError):
            pages.put(0, 4, "overwrite", 0)

    def test_decoder_fork_restores_state_before_boundary(self):
        pages = Pages(100)
        for c in range(-1, 8):
            pages.put(0, c, {"after": c}, 10)
        pages.fork(0, 1, 5)
        self.assertEqual(pages.get(1, 4), {"after": 4})
        self.assertNotIn(5, pages.branches[1])
        pages.prune(4, (-1,))
        self.assertEqual(set(pages.branches[1]), {-1, 4})


class SessionTest(unittest.TestCase):
    def test_generation_follows_the_playback_cursor(self):
        session = Session("old")
        self.assertEqual(session.schedule(), [{"end": 3}])
        self.assertEqual(session.schedule(), [])
        produce(session)
        self.assertTrue(session.view()["paused"])
        session.acknowledge(0)
        self.assertEqual(session.schedule(), [{"end": 4}])
        self.assertTrue(session.view()["busy"])
        session.acknowledge(0)  # A repeated ACK asks for nothing new.
        self.assertEqual(session.schedule(), [])

    def test_live_pace_keeps_generation_moving_without_played(self):
        session = Session("old")
        session.schedule()
        produce(session)
        # The live stream has handed chunk 0; /played is still silent.
        session.advance_pace(0)
        self.assertEqual(session.schedule(), [{"end": 4}])
        produce(session)
        self.assertEqual(session.played, -1)
        self.assertEqual(session.pace, 0)
        self.assertEqual(session.end, 4)

    def test_prompt_change_starts_behind_committed_chunks(self):
        session = Session("old")
        session.schedule()
        produce(session)
        session.interact("new")
        session.acknowledge(0)
        self.assertEqual(session.schedule(), [{"prompt": "new", "start": 3}, {"end": 4}])
        self.assertEqual(session.prompt, "new")
        self.assertTrue(session.view()["pending"])
        session.produced(3)
        self.assertFalse(session.view()["pending"])
        self.assertEqual(session.schedule(), [])

    def test_fast_live_reader_cannot_extend_prompt_backlog_without_playback(self):
        session = Session("old", Limits(buffer=3, delivery_slack=2))
        session.schedule()
        produce(session)
        for chunk in range(30):
            session.advance_pace(chunk)
            session.schedule()
            produce(session)
            self.assertLessEqual(session.issued - session.played - 1, 5)
        self.assertEqual(session.issued, 5)
        session.interact("new")
        self.assertEqual(session.schedule(), [{"prompt": "new", "start": 5}])
        session.acknowledge(0)
        self.assertEqual(session.schedule(), [{"end": 6}])

    def test_prompt_change_does_not_extend_a_paused_buffer(self):
        session = Session("old")
        session.schedule()
        session.interact("new")
        self.assertEqual(session.schedule(), [{"prompt": "new", "start": 3}])
        session.interact("newest")
        self.assertEqual(session.schedule(), [{"prompt": "newest", "start": 3}])
        with self.assertRaises(ValueError):
            session.interact("  ")

    def test_bad_or_late_ack_does_not_move_cursor(self):
        session = Session("old")
        session.schedule()
        produce(session)
        for bad in (3, 999, -2, True, "1", None):
            with self.assertRaises(ValueError):
                session.acknowledge(bad)
        session.acknowledge(2)
        session.acknowledge(1)
        self.assertEqual(session.played, 2)

    def test_chunks_arrive_in_order_inside_committed_range(self):
        session = Session("old")
        session.schedule()
        with self.assertRaises(RuntimeError):
            session.produced(1)
        produce(session)
        with self.assertRaises(RuntimeError):
            session.produced(3)

    def test_long_session_keeps_frontier_bounded(self):
        session = Session("old", Limits(5))
        for _ in range(500):
            session.schedule()
            produce(session)
            self.assertEqual(session.end - session.played - 1, 5)
            session.acknowledge(session.played + 1)
        with self.assertRaises(ValueError):
            Limits(0)


def run_tape(commands, depth=4):
    source, restarts, reports = queue.Queue(), [], []
    for command in commands:
        source.put(command)

    def restart(command):
        restarts.append(command)
        return command["start"]

    ticks = list(Tape(source, depth, restart, lambda tick, ms, waited: reports.append(tick)))
    return ticks, restarts, reports


RESET = {"seed": 1, "prompt": "old", "start": 0, "epoch": 1}


class TapeTest(unittest.TestCase):
    def test_parks_with_the_pipeline_full(self):
        # Four fill ticks run before anything is committed; each committed
        # chunk then costs exactly one more tick.
        ticks, restarts, reports = run_tape([RESET, {"end": 1}, {"end": 3}, None])
        self.assertEqual(ticks, [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(restarts, [RESET])
        self.assertEqual(reports, ticks)

    def test_prompt_change_refills_from_the_frontier(self):
        change = {"prompt": "new", "start": 3}
        ticks, restarts, _ = run_tape([RESET, {"end": 3}, change, {"end": 5}, None])
        self.assertEqual(ticks, [0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8])
        self.assertEqual(restarts, [RESET, change])

    def test_new_session_restarts_from_zero(self):
        ticks, _, _ = run_tape([RESET, {"end": 2}, dict(RESET, epoch=2), None])
        self.assertEqual(ticks, [0, 1, 2, 3, 4, 5, 0, 1, 2, 3])

    def test_stale_frontier_never_moves_back(self):
        ticks, _, _ = run_tape([RESET, {"end": 2}, {"end": 1}, None])
        self.assertEqual(ticks, [0, 1, 2, 3, 4, 5])

    def test_inconsistent_commands_fail_closed(self):
        with self.assertRaises(RuntimeError):
            run_tape([{"end": 1}])
        with self.assertRaises(RuntimeError):
            run_tape([RESET, {"end": 3}, {"prompt": "new", "start": 2}])
        with self.assertRaises(RuntimeError):
            run_tape([{"prompt": "new", "start": 0}])


class ContinuationTest(unittest.TestCase):
    def test_native_scheduler_still_accepts_the_adapter(self):
        from wave_rt.interactive.native import continuation_function
        self.assertEqual(continuation_function().__name__, "_systolic")

    def test_retained_history_controls_both_slot_layouts(self):
        from inspect import getclosurevars
        from types import SimpleNamespace
        from wave_rt.interactive.native import paged_continuation_methods
        methods = paged_continuation_methods()
        for start in (4, 8, 12):
            class _Page:
                def unsqueeze(self, _axis):
                    return self

            class _Pages:
                def __getitem__(self, _key):
                    return _Page()

            # FA3 builds the per-layer target list before returning the exchange.
            fake = SimpleNamespace(store_rank=4, rank=0, _cur_tick=start,
                _os_gt=3, tokens_per_chunk=1, npb=3, max_attn_frames=21, _pg_wmax=8,
                Ln=1, _pg_k=_Pages(), _pg_v=_Pages())
            clean = [[{}] * min(8, start)]
            for ranks in ([0], [0, 1], [0, 1, 2, 3]):
                expected = 1 + min(6 - len(ranks), len(clean[0]) - 1)
                dn = methods["_make_denoise_exch_paged"](fake, clean, ranks, start * 3, len(ranks) * 3, False)
                st = methods["_make_store_exch_paged"](fake, clean, {}, start, ranks)
                self.assertEqual(getclosurevars(dn).nonlocals["naw"], expected)
                self.assertEqual(getclosurevars(st).nonlocals["send_slots"], [(r, expected) for r in ranks])
                self.assertLessEqual(expected + len(ranks), 7)


class ControllerTest(unittest.TestCase):
    def test_failed_runtime_is_reported_and_owned_workers_closed(self):
        runtime = FakeRuntime()
        controller = Controller(runtime, "old", Limits())
        runtime.fail("worker failed")
        self.assertEqual(controller.error, "worker failed")
        self.assertTrue(runtime.closed)
        with controller.lock:
            controller._dispatch()  # A failed service issues nothing further.
        controller.close()

    def test_idle_viewer_parks_generation_and_second_viewer_rejected(self):
        runtime = FakeRuntime()
        controller = Controller(runtime, "old", Limits())
        try:
            wait_until(lambda: controller.session.end == 3 and not controller.resetting)
            time.sleep(.05)
            self.assertEqual(controller.session.end, 3)
            self.assertEqual(runtime.calls, [("reset", "old"), ("advance", 3)])
            self.assertEqual(runtime.ticks, list(range(7)))
            token = controller.attach()["client"]
            controller.validate(token)
            with self.assertRaises(ValueError):
                controller.attach()
        finally:
            controller.close()

    def test_each_played_chunk_costs_one_tick_and_prompt_changes_at_frontier(self):
        runtime = FakeRuntime()
        controller = Controller(runtime, "old", Limits())
        try:
            wait_until(lambda: controller.session.end == 3)
            history = controller.view()["prompt_history"]
            self.assertEqual([(row["kind"], row["text"], row["start"]) for row in history],
                             [("start", "old", 0)])
            for chunk in range(4):
                with controller.lock:
                    controller.played(0, chunk)
                wait_until(lambda: controller.session.end == chunk + 4)
            self.assertEqual(runtime.ticks, list(range(11)))
            with controller.lock:
                controller.prompt("new")
                self.assertTrue(controller.view()["pending"])
                controller.played(0, 4)
            wait_until(lambda: controller.session.end == 8)
            self.assertEqual(runtime.calls[-2:], [("switch", "new", 7), ("advance", 8)])
            # The four in-flight chunks are dropped and refilled; nothing committed is redone.
            self.assertEqual(runtime.ticks[11:], [7, 8, 9, 10, 11])
            self.assertEqual(controller.frames.get(0, 6), b"old")
            self.assertEqual(controller.frames.get(0, 7), b"new")
            self.assertFalse(controller.view()["pending"])
            history = controller.view()["prompt_history"]
            self.assertEqual([(row["kind"], row["text"], row["start"]) for row in history],
                             [("start", "old", 0), ("switch", "new", 7)])
            self.assertTrue(all("wall" in row for row in history))
            with self.assertRaises(KeyError):
                controller.frames.get(0, 4)  # Played output is released.
            rows = [row for row in controller.trace.rows["server"] if row["event"] == "chunk"]
            self.assertEqual([row["chunk"] for row in rows], list(range(8)))
            self.assertIn("since_prompt_ms", rows[7])
            self.assertIn("total_ms", rows[7])
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
