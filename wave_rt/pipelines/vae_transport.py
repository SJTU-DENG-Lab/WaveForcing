"""Request-scoped VAE frames: negotiate first/steady shapes only once."""
import torch
import torch.distributed as dist


class FrameChannel:
    def __init__(self, peer, device, dtype=torch.bfloat16):
        self.peer, self.device, self.dtype = peer, device, dtype
        self.index = 0
        self.shapes = {}
        self.buffers = {}

    def send(self, value):
        value = value.contiguous()
        key = min(self.index, 1)
        if value.ndim != 5 or value.dtype != self.dtype:
            raise ValueError('VAE transport requires 5D tensors with the agreed dtype')
        if key not in self.shapes:
            self.shapes[key] = tuple(value.shape)
            dist.send(torch.tensor(value.shape, dtype=torch.long, device=value.device), dst=self.peer)
        elif tuple(value.shape) != self.shapes[key]:
            raise ValueError('VAE steady frame shape changed within a request')
        dist.send(value, dst=self.peer)
        self.index += 1

    def recv(self):
        key = min(self.index, 1)
        if key not in self.buffers:
            shape = torch.empty(5, dtype=torch.long, device=self.device)
            dist.recv(shape, src=self.peer)
            dimensions = shape.tolist()
            if min(dimensions) <= 0:
                raise ValueError('Invalid VAE frame shape')
            self.buffers[key] = torch.empty(dimensions, dtype=self.dtype, device=self.device)
        value = self.buffers[key]
        dist.recv(value, src=self.peer)
        self.index += 1
        return value
