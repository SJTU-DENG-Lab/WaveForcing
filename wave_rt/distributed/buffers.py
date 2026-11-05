"""Reusable bounded tensor allocations for Wave exchanges."""
from collections import OrderedDict
import torch

class Buffers:
    """One high-water allocation per purpose/dtype/device, reused across shapes.

    Callers own tensor references while operations are in flight; removing a
    pool entry never invalidates those references. Sequential layer use only.
    """
    def __init__(self, capacity=32):
        self.tensors = OrderedDict()
        self.capacity = capacity

    def clear(self):
        self.tensors.clear()

    def get(self, name, shape, like, *, dtype=None):
        import math
        dtype = like.dtype if dtype is None else dtype
        key = (name, dtype, like.device)
        size = math.prod(shape)
        if key not in self.tensors or self.tensors[key].numel() < size:
            self.tensors[key] = torch.empty(size, dtype=dtype, device=like.device)
        self.tensors.move_to_end(key)
        value = self.tensors[key][:size].view(shape)
        while len(self.tensors) > self.capacity:
            self.tensors.popitem(last=False)
        return value

    def visible(self, shape, like, ring=False):
        if ring:
            return self.get('ring_a', (2, *shape), like).unbind(0)
        return self.get('wp_visible_k', shape, like), self.get('wp_visible_v', shape, like)
