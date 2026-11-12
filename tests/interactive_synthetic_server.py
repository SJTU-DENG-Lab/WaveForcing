"""CPU-only synthetic source: exercise the real player, controller and tape.

Every new session or prompt change shows the next of three colours, so the
browser tests can tell exactly which frames belong to which request.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))
from interactive_fakes import FakeRuntime
from wave_rt.interactive.server import Controller, make_app
from wave_rt.interactive.state import Limits
from wave_rt.interactive.video import encode_chunk

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8892)
parser.add_argument("--autostart", action="store_true")
parser.add_argument("--fps", type=int, default=16)
parser.add_argument("--buffer", type=int, default=Limits.buffer)
parser.add_argument("--tick-seconds", type=float, default=.14)
args = parser.parse_args()

encoded = {}
for colour in range(3):
    for count in (9, 12):
        frames = []
        for n in range(count):
            rgb = np.zeros((480, 832, 3), dtype=np.uint8)
            rgb[:, :, colour] = 160
            rgb[40:200, n * 25:n * 25 + 100, :] = 230
            frames.append(rgb)
        encoded[colour, count] = encode_chunk(frames, args.fps)


class Runtime(FakeRuntime):
    def __init__(self):
        super().__init__(lambda scene, chunk: encoded[int(scene), 9 if chunk == 0 else 12],
                         args.tick_seconds)
        self.scenes = -1

    def reset(self, seed, prompt):
        self.scenes += 1
        return super().reset(seed, str(self.scenes % 3))

    def switch(self, prompt, start):
        self.scenes += 1
        super().switch(str(self.scenes % 3), start)


class Monitor:
    def snapshot(self):
        used = 10240 + 64 * (int(time.time()) % 20)  # Visibly different every second.
        return dict(status="ok", sampled_at=time.time(), age_seconds=0, used_mib=used * 8,
                    total_mib=143771 * 8,
                    gpus=[dict(index=i, name="Synthetic H200", role="Test", used_mib=used,
                               total_mib=143771, utilization_percent=25) for i in range(8)])

    def close(self):
        pass


controller = Controller(Runtime(), "old", Limits(args.buffer), autostart=args.autostart)
uvicorn.run(make_app(controller, args.fps, Monitor()), host="127.0.0.1", port=args.port,
            log_level="warning")
