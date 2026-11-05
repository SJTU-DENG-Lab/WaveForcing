"""Shared prefix preparation for WP and CP, retaining Sage's SM90 kernel.

The private quantization ABI is SageAttention 2.2.0 / SM90 / D=128. Other
architectures and layouts continue through the installed public Sage backend.
There are no runtime optimization switches and no cross-tick KV-value cache.
"""
from collections import OrderedDict

import torch

from .sage_support import supported


def prefix_segments(query_tokens, key_tokens, chunk_tokens, prefix_tokens=0, kv_start=0):
    """Disjoint Q spans and their visible KV lengths; future-only spans are absent."""
    if (query_tokens < 1 or key_tokens < 1 or chunk_tokens < 1 or
            query_tokens % chunk_tokens or prefix_tokens < 0 or kv_start < 0):
        raise ValueError('Invalid Sage prefix geometry')
    spans = []
    for start in range(0, query_tokens, chunk_tokens):
        end = start+chunk_tokens
        visible = min(key_tokens, max(0, prefix_tokens+end-kv_start))
        if not visible:
            continue
        if visible == key_tokens:
            end = query_tokens
        spans.append((start, end-start, visible))
        if end == query_tokens:
            break
    return tuple(spans)


class SagePrefixAttention:
    def __init__(self, public):
        self.public = public
        self.workspaces = OrderedDict()

    @staticmethod
    def accepts(q, k, v):
        return (supported(q) and k.ndim == v.ndim == 4 and
                k.shape == v.shape and q.shape[0] == k.shape[0] and
                q.shape[2:] == k.shape[2:] and min(q.shape[:3]) > 0 and k.shape[1] > 0 and
                q.dtype == k.dtype == v.dtype and q.device == k.device == v.device and
                k.stride(-1) == v.stride(-1) == 1)

    @classmethod
    def can_promote(cls, q, k, v):
        return (q is not None and q.ndim == 4 and q.shape[0] == 1 and q.shape[2:] == (12, 128)
                and q.dtype == torch.bfloat16 and cls.accepts(q, k, v))

    def _run(self, q, k, v, segments, lse, kv_promotion=None):
        from .sage_kernels import PrefixWorkspace
        # Pools cannot be overwritten by another stream while kernels still read
        # them. Each entry is stream-local; allocations also belong to that stream.
        with torch.cuda.device(q.device):
            key = (q.device, q.dtype, torch.cuda.current_stream(q.device).cuda_stream)
            if key not in self.workspaces:
                self.workspaces[key] = PrefixWorkspace()
            self.workspaces.move_to_end(key)
            while len(self.workspaces) > 4:
                self.workspaces.popitem(last=False)
            return self.workspaces[key].attention(q, k, v, segments, lse=lse,
                                                   kv_promotion=kv_promotion)

    def prefix(self, q, k, v, chunk_tokens, *, prefix_tokens=0, kv_start=0, lse=False):
        if not self.accepts(q, k, v):
            return None
        spans = prefix_segments(q.shape[1], k.shape[1], chunk_tokens, prefix_tokens, kv_start)
        return self._run(q, k, v, spans, lse)

    def __call__(self, q, k, v, *, tensor_layout='NHD', is_causal=False,
                 return_lse=False, kv_promotion=None, **kwargs):
        if tensor_layout == 'NHD' and not is_causal and not kwargs and self.accepts(q, k, v):
            return self._run(q, k, v, ((0, q.shape[1], k.shape[1]),), return_lse, kv_promotion)
        if kv_promotion is not None:
            raise ValueError('Deferred KV promotion requires the supported Sage preparation path')
        return self.public(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal,
                           return_lse=return_lse, **kwargs)
