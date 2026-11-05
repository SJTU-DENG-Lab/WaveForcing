"""One sgldiff merged QKV GEMM behind the native block's three-call API."""
from types import SimpleNamespace
import weakref

import torch
from torch import nn


class PackedProjection(nn.Module):
    def __init__(self, q, k, v):
        super().__init__()
        from sglang.multimodal_gen.runtime.layers.linear import MergedColumnParallelLinear
        inputs = (q, k, v)
        self.widths = [layer.weight.shape[0] for layer in inputs]
        weight = q.weight
        if any(layer.weight.shape[1] != weight.shape[1] for layer in inputs):
            raise ValueError('Packed QKV input dimensions disagree')
        has_bias = q.bias is not None
        if any((layer.bias is not None) != has_bias for layer in inputs):
            raise ValueError('Packed QKV requires consistent bias configuration')
        with torch.device(weight.device):
            self.linear = MergedColumnParallelLinear(weight.shape[1], self.widths,
                bias=has_bias, params_dtype=weight.dtype, quant_config=None)
        if self.linear.tp_size != 1:
            raise ValueError('Pack local QKV shards only with singleton sgldiff groups')
        with torch.no_grad():
            self.linear.weight.copy_(torch.cat([layer.weight for layer in inputs]))
            if has_bias:
                self.linear.bias.copy_(torch.cat([layer.bias for layer in inputs]))
        self.reset()

    def reset(self):
        self._input = self._parts = self._next = None

    def part(self, index, x):
        if self._parts is None:
            if index == 0:
                projected = self.linear(x)[0]
                self._parts = projected.split(self.widths, dim=-1)
            elif index == 1:
                # Last clean block: one KV GEMM using a contiguous weight view,
                # through the same sgldiff linear method. No duplicate KV weights.
                start = self.widths[0]
                from wave_rt.runtime.fp8 import WaveFp8Linear
                if isinstance(self.linear, WaveFp8Linear):
                    projected = self.linear.forward_rows(x, start)[0]
                else:
                    view = SimpleNamespace(weight=self.linear.weight[start:])
                    bias = None if self.linear.bias is None else self.linear.bias[start:]
                    projected = self.linear.quant_method.apply(view, x, bias)
                self._parts = (None, *projected.split(self.widths[1:], dim=-1))
            else:
                raise RuntimeError('V projection called before Q/K')
            self._input, self._next = x, index
        if self._input is not x or self._next != index:
            self.reset()
            raise RuntimeError('Native block QKV call order/input changed')
        result = self._parts[index]
        self._next = index + 1
        if index == 2:
            self.reset()
        return result, None


class ProjectionPart(nn.Module):
    def __init__(self, owner, index):
        super().__init__()
        # Registered once on the block, not once per part; avoids module cycles
        # and duplicate weight ownership in the model state dict.
        self.owner = weakref.ref(owner)
        self.index = index

    def forward(self, x):
        return self.owner().part(self.index, x)


def pack_wan_qkv(model):
    """Call after checkpoint loading and, for TP, after actual weight sharding."""
    for block in model.blocks:
        if hasattr(block, '_wave_qkv'):
            raise ValueError('Wan projections are already packed')
        owner = PackedProjection(block.to_q, block.to_k, block.to_v)
        block._wave_qkv = owner
        for index, name in enumerate(('to_q', 'to_k', 'to_v')):
            setattr(block, name, ProjectionPart(owner, index))
