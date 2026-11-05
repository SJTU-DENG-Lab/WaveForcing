"""Opt-in output-pipeline diagnostics, with deferred CUDA event resolution.

Never synchronize a per-frame event. Host times share perf_counter on one host;
GPU clocks are local to each device and only approximately aligned to host time.
The anchor bracket and end-anchor residual are retained, not hidden as precision.
"""
import json
import os
from pathlib import Path
import time
import threading


def enabled():
    return os.environ.get('WAVE_OUTPUT_TRACE', '').lower() not in ('', '0', 'false')


class OutputTrace:
    def __init__(self, name, **metadata):
        import torch
        self.cuda = torch.cuda
        self.name, self.metadata = name, metadata
        self.rows = []
        self.origin, self.anchor_lo, self.anchor_hi = self._anchor()

    def _anchor(self):
        event = self.cuda.Event(enable_timing=True)
        lo = time.perf_counter()
        event.record()
        event.synchronize()  # only before/after request measurement
        return event, lo, time.perf_counter()

    def begin(self, kind, *, chunk=None, frame=None, gpu=False, stream=None):
        row = dict(kind=kind, chunk=chunk, frame=frame, host_start_s=time.perf_counter(),
                   host_thread=threading.current_thread().name)
        if gpu:
            begin, end = self.cuda.Event(enable_timing=True), self.cuda.Event(enable_timing=True)
            begin.record(stream)
            row['_events'] = begin, end
            row['_stream'] = stream
        self.rows.append(row)  # producer + staging thread; each owns its own row
        return row

    def end(self, row):
        if '_events' in row:
            row['_events'][1].record(row['_stream'])
        row['host_end_s'] = time.perf_counter()

    def point(self, kind, *, chunk=None):
        event = self.cuda.Event(enable_timing=True)
        row = dict(kind=kind, chunk=chunk, frame=None, host_start_s=time.perf_counter(),
                   host_thread=threading.current_thread().name)
        event.record()
        row['host_end_s'] = time.perf_counter()
        row['_events'] = event, event
        self.rows.append(row)

    def dump(self, **metadata):
        # Call only after the request's existing synchronization and writer join.
        self.cuda.synchronize()  # also finish diagnostic events on the copy stream
        end, lo, hi = self._anchor()
        elapsed = self.origin.elapsed_time(end) / 1000
        midpoint = (self.anchor_lo + self.anchor_hi) / 2
        residual = (lo+hi)/2 - (midpoint+elapsed)
        uncertainty = (self.anchor_hi-self.anchor_lo)/2 + (hi-lo)/2 + abs(residual)
        rows = []
        for value in self.rows:
            row = {k:v for k,v in value.items() if not k.startswith('_')}
            if 'host_end_s' not in row:
                raise RuntimeError(f'Unfinished output trace span: {row}')
            row['host_ms'] = (row['host_end_s']-row['host_start_s'])*1000
            if '_events' in value:
                a,b = value['_events']
                start_ms, end_ms = self.origin.elapsed_time(a), self.origin.elapsed_time(b)
                row.update(gpu_start_ms=start_ms, gpu_end_ms=end_ms, gpu_ms=a.elapsed_time(b),
                    gpu_start_host_s=midpoint+start_ms/1000, gpu_end_host_s=midpoint+end_ms/1000)
            rows.append(row)
        result = dict(version=1, name=self.name, diagnostic_only=True, timing_valid=False,
            metadata=dict(self.metadata, **metadata), anchor_host_lower_s=self.anchor_lo,
            anchor_host_upper_s=self.anchor_hi, end_anchor_host_lower_s=lo, end_anchor_host_upper_s=hi,
            end_anchor_gpu_ms=elapsed*1000, end_anchor_residual_ms=residual*1000,
            estimated_alignment_uncertainty_ms=uncertainty*1000, events=rows,
            scope='GPU stream intervals include launch gaps/dependencies, not sums of kernel-active time; '
                  'cross-device host alignment is approximate; no per-frame event synchronization')
        root = Path(os.environ['WAVE_OUTPUT_TRACE_DIR']); root.mkdir(parents=True,exist_ok=True)
        path = root/(self.name+'.json')
        temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(result,indent=2),encoding='utf-8')
        temp.replace(path)
