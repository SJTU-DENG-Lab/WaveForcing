"""Explicit dense FlashAttention adapters (B,S,H,D), without fallback."""
import importlib
import torch
import torch.nn.functional as F


class Attention:
    def __init__(self, backend="fa3"):
        self.name = backend
        if backend == "fa3":
            self.fn = importlib.import_module("flash_attn_interface").flash_attn_func
        elif backend == "fa2":
            self.fn = importlib.import_module("flash_attn").flash_attn_func
        elif backend == "sdpa":
            self.fn = None  # explicit CPU numerical oracle
        else:
            raise ValueError("Expected fa2, fa3 or the CPU-only sdpa test backend")

    def __call__(self, q, k, v):
        if self.name == "sdpa":
            if q.device.type != "cpu":
                raise RuntimeError("The sdpa oracle is CPU-only")
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                v.transpose(1, 2), dropout_p=0., is_causal=False).transpose(1, 2)
        return self.fn(q, k, v, causal=False, return_attn_probs=False)


class FlashAttentionModule(torch.nn.Module):
    def __init__(self, backend="fa3"):
        super().__init__()
        self.attention = Attention(backend)

    def forward(self, q, k, v):
        return self.attention(q, k, v)
