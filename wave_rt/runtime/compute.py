"""Request constants for the fixed-timestep WP stages (also CPU-testable)."""
import torch


class AxisRoPE:
    """Native-precision axes; the CUDA consumers index them without full grids."""
    def __init__(self, frames, height, width, head_dim, device, compute_device='cpu',
                 dtype=torch.float64):
        self.shape = frames, height, width
        self.dims = (head_dim - 4*(head_dim//6), 2*(head_dim//6), 2*(head_dim//6))
        self.tables = []
        for size, dim in zip(self.shape, self.dims):
            positions = torch.arange(size, dtype=dtype, device=compute_device)
            frequencies = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=dtype,
                                                       device=compute_device) / dim))
            angles = torch.outer(positions, frequencies)
            self.tables.extend((angles.cos().float().to(device), angles.sin().float().to(device)))
        self.tables = tuple(self.tables)

    def check(self, frames, height, width, start):
        if (height, width) != self.shape[1:] or max(0, start) + frames > self.shape[0]:
            raise ValueError('WP RoPE request geometry/position exceeds prepared axes')

    def grid(self, frames, start):
        # Reference/diagnostic surface only. Production kernels read the axes.
        _, h, w = self.shape
        start = max(0, start)
        return tuple(torch.cat((
            self.tables[c][start:start+frames, None, None].expand(frames, h, w, -1),
            self.tables[2+c][None, :, None].expand(frames, h, w, -1),
            self.tables[4+c][None, None, :].expand(frames, h, w, -1)), -1).reshape(frames*h*w, -1)
            for c in (0, 1))


def first_norm(block, hidden, shift, scale):
    if hidden.is_cuda:
        from .kernels import norm_modulate
        return norm_modulate(hidden, shift, scale, block.norm1.eps)
    frames = shift.shape[1]
    return ((block.norm1(hidden.float()).unflatten(1, (frames, -1)) * (1 + scale)
             + shift).flatten(1, 2).to(hidden.dtype))


class FixedConditioning:
    def __init__(self):
        self.reset()

    def reset(self):
        self.entries = {}
        self.steps = None

    def get(self, model, cache, timestep, context, step):
        if step is None:
            # Direct model oracles lack the scheduler's host timestep. Production
            # always supplies it through WaveBackend; no GPU scalar read there.
            values = timestep.detach().cpu().flatten().tolist()
            if len(set(values)) != 1:
                raise ValueError('A WP stage requires one fixed timestep')
            step = int(values[0])
        b, frames = timestep.shape
        key = int(step), b, frames
        if key not in self.entries:
            if self.steps is None:
                self.steps = cache.timesteps.detach().cpu().tolist()
            if step not in self.steps:
                raise ValueError('WP timestep is not in the configured schedule')
            # One lazy initialization inside the original autocast context.
            cache.prepare(model.condition_embedder, context, timestep.device)
            i = self.steps.index(step)
            temb = cache.temb[i:i+1].expand(b*frames, -1).contiguous()
            proj = cache.proj[i:i+1].expand(b*frames, -1).contiguous()
            p = proj.reshape(b, frames, 6, model.hidden_size)
            modulation = [tuple(v.contiguous() for v in
                          (block.scale_shift_table + p.float()).chunk(6, 2))
                          for block in model.blocks]
            head = tuple(v.contiguous() for v in
                         (model.scale_shift_table.unsqueeze(1) +
                          temb.reshape(b, frames, 1, -1)).chunk(2, 2))
            zero = torch.zeros(1, device=timestep.device, dtype=proj.dtype)
            self.entries[key] = temb, proj, cache.text, modulation, head, zero
        return self.entries[key]
