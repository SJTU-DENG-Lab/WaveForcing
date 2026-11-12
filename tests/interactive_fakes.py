"""CPU stand-in for the GPU group, driven by the real tape."""
import queue
import threading
import time

from wave_rt.interactive.native import Tape


class FakeRuntime:
    """One fake chunk per committing tick; the payload defaults to the prompt."""

    depth = 4

    def __init__(self, payload=None, tick_seconds=0.0):
        self.payload = payload or (lambda prompt, chunk: prompt.encode())
        self.tick_seconds = tick_seconds
        self.commands = queue.Queue()
        self.gate = threading.Event()  # Clear to hold a reset/prompt change mid-flight.
        self.gate.set()
        self.epoch = 0
        self.calls = []
        self.ticks = []
        self.closed = False

    def start(self, on_event):
        self.on_event = on_event
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        session = {}

        def restart(command):
            self.gate.wait()
            if "seed" in command:
                session["epoch"] = command["epoch"]
            session["prompt"], session["start"] = command["prompt"], command["start"]
            self.on_event({"kind": "reset" if "seed" in command else "switched",
                           "epoch": session["epoch"], "start": command["start"]})
            return command["start"]

        for tick in Tape(self.commands, self.depth, restart):
            self.ticks.append(tick)
            time.sleep(self.tick_seconds)
            chunk = tick - self.depth
            if chunk >= session["start"]:
                data, now = self.payload(session["prompt"], chunk), time.time()
                self.on_event({"kind": "latent", "epoch": session["epoch"], "chunk": chunk, "at": now})
                self.on_event({"kind": "chunk", "epoch": session["epoch"], "chunk": chunk,
                               "data": data, "size": len(data), "decoded_at": now, "encoded_at": now})

    def fail(self, error):
        self.on_event({"kind": "error", "error": error})

    def reset(self, seed, prompt):
        self.epoch += 1
        self.calls.append(("reset", prompt))
        self.commands.put({"seed": seed, "prompt": prompt, "start": 0, "epoch": self.epoch})
        return self.epoch

    def switch(self, prompt, start):
        self.calls.append(("switch", prompt, start))
        self.commands.put({"prompt": prompt, "start": start})

    def advance(self, end):
        self.calls.append(("advance", end))
        self.commands.put({"end": end})

    def close(self):
        self.closed = True
        self.commands.put(None)


def wait_until(condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("Condition was not reached in time")
        time.sleep(.005)
