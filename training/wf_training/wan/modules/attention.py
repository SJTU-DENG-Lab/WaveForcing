# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention_raw
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import os
import warnings

_flex_attention_compiled = None


def _get_flex_attention():
    """Compile flex_attention lazily (first call), matching the Wan 1.3B note in
    wan/modules/causal_model.py: its channel/head config needs max-autotune for
    flex_attention to work (https://github.com/pytorch/pytorch/issues/133254).
    RF_FLEX_COMPILE=0 disables compilation (eager fallback, used by tests)."""
    global _flex_attention_compiled
    if _flex_attention_compiled is None:
        if os.environ.get("RF_FLEX_COMPILE", "1") not in ("0", "false"):
            # flex recompiles per (lq, lk) shape; a training rollout produces
            # far more shapes than dynamo's default per-frame cache_size_limit
            # (8). Hitting the limit flips the frame between compiled and eager
            # execution, which breaks non-reentrant gradient checkpointing
            # (saved-vs-recomputed metadata mismatch) and silently degrades to
            # the dense eager backward. Raise the limit so every shape stays
            # compiled in the reference training implementation.
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 256)
            _flex_attention_compiled = torch.compile(
                _flex_attention_raw, dynamic=False, mode="max-autotune-no-cudagraphs")
        else:
            _flex_attention_compiled = _flex_attention_raw
    return _flex_attention_compiled

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    attn_mask=None,
    block_mask=None,
):
    # A flex BlockMask forces the flex_attention path (block-sparse kernel).
    if block_mask is not None:
        assert FLEX_ATTENTION_AVAILABLE, "flex_attention requires torch >= 2.5"
        assert attn_mask is None, "pass either block_mask or attn_mask, not both"
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using flex_attention. It can have a significant impact on performance.'
            )
        qh = q.transpose(1, 2).to(dtype)
        kh = k.transpose(1, 2).to(dtype)
        vh = v.transpose(1, 2).to(dtype)
        if q_scale is not None:
            qh = qh * q_scale
        out = _get_flex_attention()(
            qh, kh, vh, block_mask=block_mask, scale=softmax_scale)
        return out.transpose(1, 2).contiguous().type(q.dtype)

    # Custom attn_mask forces SDPA (FA2/FA3 do not take arbitrary block masks).
    if attn_mask is not None or not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        qh = q.transpose(1, 2).to(dtype)
        kh = k.transpose(1, 2).to(dtype)
        vh = v.transpose(1, 2).to(dtype)
        if q_scale is not None:
            qh = qh * q_scale
        out = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=attn_mask, is_causal=causal,
            dropout_p=dropout_p, scale=softmax_scale)
        return out.transpose(1, 2).contiguous().type(q.dtype)

    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
