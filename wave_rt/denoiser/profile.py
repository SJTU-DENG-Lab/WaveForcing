"""Opt-in, measured-request-only profiling. Never synchronizes individual spans.

Stream intervals include launch gaps/dependencies. FLOPs are explicit analytical
counts for observed Linear and self-attention calls, not hardware counters.
"""
from contextlib import contextmanager
from pathlib import Path
import json

from wave_rt.denoiser.output_trace import OutputTrace

_current = None


def current():
    return _current


class RuntimeProfile(OutputTrace):
    def __init__(self, name, out, start=12, ticks=4, **metadata):
        super().__init__(name, **metadata)
        self.out = Path(out)
        self.start, self.stop = start, start+ticks
        self.tick = -1
        self.chunk = None
        self.profiler = None
        self.kernel_saved = False
        self.handles = []
        self.pending_attention = {}

    @property
    def detailed(self):
        return self.start <= self.tick < self.stop

    def activate(self):
        global _current
        _current = self

    def at(self, tick, chunk=None):
        import torch
        self.tick, self.chunk = tick, chunk
        if tick >= self.stop and self.profiler is not None:
            self._finish_kernels()
        if self.start <= tick < self.stop and self.profiler is None and not self.kernel_saved:
            self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_flops=True)
            self.profiler.start()

    def _finish_kernels(self):
        self.profiler.stop()  # diagnostic boundary only; can perturb this request
        self.out.mkdir(parents=True, exist_ok=True)
        self.profiler.export_chrome_trace(str(self.out / f'{self.name}.kernels.json'))
        ops = []
        for op in self.profiler.key_averages(group_by_input_shape=True):
            ops.append(dict(name=op.key, count=op.count, input_shapes=op.input_shapes,
                cpu_time_total_us=op.cpu_time_total,
                device_time_total_us=getattr(op, 'device_time_total', 0),
                self_device_time_total_us=getattr(op, 'self_device_time_total', 0),
                supported_operator_flops=getattr(op, 'flops', 0)))
        (self.out / f'{self.name}.operators.json').write_text(json.dumps(ops, indent=2), encoding='utf-8')
        self.profiler = None
        self.kernel_saved = True

    @contextmanager
    def span(self, kind, *, detail=True, stream=None, **fields):
        if detail and not self.detailed:
            yield
            return
        import torch
        row = self.begin(kind, chunk=self.chunk, gpu=True, stream=stream)
        row.update(tick=self.tick, stream_id=int((stream or torch.cuda.current_stream()).cuda_stream), **fields)
        # record_function associates semantic scopes with CPU launches in the
        # kernel trace; GPU event intervals provide the actual stream boundaries.
        with torch.profiler.record_function(f'WaveRT::{kind}::tick={self.tick}'):
            try:
                yield
            finally:
                self.end(row)

    def mark(self, kind, **fields):
        if self.detailed:
            self.point(kind, chunk=self.chunk)
            self.rows[-1].update(tick=self.tick, **fields)

    def attention(self, phase, layer, **details):
        if not self.detailed:
            return
        if phase == 'attention_enter':
            q, k = details['tensors']['q']['shape'], details['tensors']['k']['shape']
            row = self.begin('self_attention', chunk=self.chunk, gpu=True)
            row.update(tick=self.tick, layer=layer, q_shape=q, k_shape=k,
                       analytic_flops=4*q[0]*q[1]*q[2]*q[3]*k[1],
                       flop_scope='QK and AV only; excludes softmax and data movement')
            self.pending_attention[layer] = row
        elif phase == 'attention_returned':
            self.end(self.pending_attention.pop(layer))

    def attach(self, model):
        import torch
        # Called-module coverage is intentional: fused kernels bypassing these
        # modules are visible in the raw kernel trace, not invented in FLOP totals.
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            stack = []
            def before(mod, inputs, name=name, stack=stack):
                if not self.detailed:
                    stack.append(None)
                    return
                x = inputs[0]
                row = self.begin('linear', chunk=self.chunk, gpu=True)
                row.update(tick=self.tick, module=name, input_shape=list(x.shape),
                           analytic_flops=2*x.numel()*mod.out_features,
                           logical_operand_bytes=x.numel()*x.element_size()+mod.weight.numel()*mod.weight.element_size(),
                           byte_scope='logical inputs and weights, NOT measured HBM traffic')
                stack.append(row)
            def after(mod, inputs, output, stack=stack):
                row = stack.pop()
                if row is not None:
                    self.end(row)
            self.handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]

    def finish(self, **metadata):
        global _current
        _current = None
        for handle in self.handles:
            handle.remove()
        if self.profiler is not None:
            self._finish_kernels()
        if self.pending_attention:
            raise RuntimeError('Unfinished attention trace')
        self.dump(profile_start=self.start, profile_stop=self.stop,
                  flop_coverage='observed nn.Linear calls and self-attention QK/AV; partial, never total model FLOPs',
                  hardware_counters_collected=False, **metadata)


@contextmanager
def span(kind, **fields):
    if _current is None:
        yield
    else:
        with _current.span(kind, **fields):
            yield
