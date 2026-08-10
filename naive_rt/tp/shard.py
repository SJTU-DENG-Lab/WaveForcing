"""Head-parallel tensor parallelism for the Rolling-Forcing CausalWanModel.

The residual stream stays full-dim and replicated on every rank; only the
attention core (q/k/v/o) and the FFN are sharded by head / hidden, with an
all-reduce after `o` and after `ffn[2]` (Megatron pattern). The one Wan-specific
subtlety is that `norm_q/norm_k` are RMSNorm over the *full* model dim (all heads
mixed) applied before the head reshape — so under head-sharding we all-reduce the
per-token sum-of-squares to keep the RMS denominator exact.

12 heads -> TP degree must divide 12 (2, 3, 4, 6).
"""
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

FULL_DIM = 1536
NUM_HEADS = 12
HEAD_DIM = 128


def init_distributed(rank: int, world: int, port: int = 29711):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)


def _slice(rank: int, world: int, total: int):
    assert total % world == 0, f"{total} not divisible by tp={world}"
    per = total // world
    return slice(rank * per, (rank + 1) * per), per


class TPRMSNorm(nn.Module):
    """WanRMSNorm over the full model dim, but this rank only holds a channel
    slice of the input. RMS mean is computed over the FULL dim via all-reduce."""

    def __init__(self, weight_slice: torch.Tensor, full_dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight_slice.clone())
        self.full_dim = full_dim
        self.eps = eps

    def forward(self, x):
        # x: [..., dim_local]
        ss = x.float().pow(2).sum(-1, keepdim=True)  # [..., 1]
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(ss)  # sum-of-squares over all heads/ranks
        x = x * torch.rsqrt(ss / self.full_dim + self.eps)
        return (x * self.weight).type_as(self.weight)


class RowParallelLinear(nn.Module):
    """Input is sharded across ranks; matmul then all-reduce; bias added once
    (each rank adds to its own identical post-all-reduce copy)."""

    def __init__(self, weight_in_slice: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.weight = nn.Parameter(weight_in_slice.clone())  # [out, in_local]
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    def forward(self, x):  # x: [..., in_local]
        out = F.linear(x, self.weight)  # [..., out]
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out


def _col_shard_linear(lin: nn.Linear, out_slice: slice) -> nn.Linear:
    """Shard a Linear along its output dim (column-parallel); no communication."""
    w = lin.weight.data[out_slice, :].contiguous()
    b = lin.bias.data[out_slice].contiguous() if lin.bias is not None else None
    new = nn.Linear(w.shape[1], w.shape[0], bias=b is not None,
                    device=w.device, dtype=w.dtype)
    new.weight.data.copy_(w)
    if b is not None:
        new.bias.data.copy_(b)
    return new


def _shard_attn(attn, rank: int, world: int):
    """Shard a WanSelfAttention / WanT2VCrossAttention module by head."""
    hsl, n_local = _slice(rank, world, NUM_HEADS)          # head slice
    csl, _ = _slice(rank, world, FULL_DIM)                 # channel slice (heads*head_dim)
    eps = attn.eps
    # q/k/v: column-parallel (output = local heads' channels)
    attn.q = _col_shard_linear(attn.q, csl)
    attn.k = _col_shard_linear(attn.k, csl)
    attn.v = _col_shard_linear(attn.v, csl)
    # o: row-parallel (input = local heads' channels) + all-reduce
    o = attn.o
    attn.o = RowParallelLinear(o.weight.data[:, csl].contiguous(),
                               o.bias.data if o.bias is not None else None)
    # qk RMSNorm over full dim -> TP-aware
    if not isinstance(attn.norm_q, nn.Identity):
        attn.norm_q = TPRMSNorm(attn.norm_q.weight.data[csl], FULL_DIM, eps)
        attn.norm_k = TPRMSNorm(attn.norm_k.weight.data[csl], FULL_DIM, eps)
    attn.num_heads = n_local
    # head_dim stays HEAD_DIM (unchanged)


def _shard_ffn(block, rank: int, world: int):
    ffn = block.ffn  # nn.Sequential(Linear(dim,ffn), GELU, Linear(ffn,dim))
    ffn_dim = ffn[0].out_features
    fsl, _ = _slice(rank, world, ffn_dim)
    ffn[0] = _col_shard_linear(ffn[0], fsl)                 # column-parallel
    down = ffn[2]
    ffn[2] = RowParallelLinear(down.weight.data[:, fsl].contiguous(),
                               down.bias.data if down.bias is not None else None)


def shard_model_for_tp(model, rank: int, world: int):
    """In-place shard every transformer block of the (already weight-loaded)
    CausalWanModel. Call AFTER load_state_dict + .to(device)."""
    if world == 1:
        return model
    for block in model.blocks:
        _shard_attn(block.self_attn, rank, world)
        _shard_attn(block.cross_attn, rank, world)
        _shard_ffn(block, rank, world)
    return model


def patch_cache_init(pipe, world: int):
    """Make the pipeline allocate per-rank KV / cross-attn caches with the local
    number of heads (originally hardcoded to 12)."""
    if world == 1:
        return
    n_local = NUM_HEADS // world

    def _kv(self, batch_size, dtype, device):
        kv_cache_size = 1560 * 24
        self.kv_cache_clean = [{
            "k": torch.zeros([batch_size, kv_cache_size, n_local, HEAD_DIM], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, n_local, HEAD_DIM], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        } for _ in range(self.num_transformer_blocks)]

    def _cross(self, batch_size, dtype, device):
        self.crossattn_cache = [{
            "k": torch.zeros([batch_size, 512, n_local, HEAD_DIM], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 512, n_local, HEAD_DIM], dtype=dtype, device=device),
            "is_init": False,
        } for _ in range(self.num_transformer_blocks)]

    import types
    pipe._initialize_kv_cache = types.MethodType(_kv, pipe)
    pipe._initialize_crossattn_cache = types.MethodType(_cross, pipe)
