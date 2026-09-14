"""Spatial-token Ulysses collectives for overlapping WORLD FSDP and local SP.

Every sequence is frame-major: ``[frame, spatial_token]``. Communication must
concatenate spatial shards *inside* each frame, never concatenate flattened
rank sequences. FSDP continues to reduce gradients across WORLD; each SP group
evaluates the same full loss, so output gathering sums gradients in backward.
"""

import torch
import torch.distributed as dist


_SP_GROUP = None
_SP_SIZE = 1
_SP_RANK = 0
_SP_SRC_RANK = 0
_SP_WORLD_GROUP = None


def initialize_sequence_parallel(size):
    """Create contiguous SP groups collectively on every WORLD rank.

    With WORLD=8 and size=4, groups are [0,1,2,3] and [4,5,6,7]. No
    orthogonal DP process group is needed: FSDP owns the WORLD gradient
    reduction. Repeated initialization with the same topology is harmless.
    """
    global _SP_GROUP, _SP_SIZE, _SP_RANK, _SP_SRC_RANK, _SP_WORLD_GROUP
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("sequence parallel size must be a positive integer")
    if not dist.is_initialized():
        if size != 1:
            raise RuntimeError("initialize torch.distributed before sequence parallelism")
        _SP_GROUP, _SP_SIZE, _SP_RANK, _SP_SRC_RANK = None, 1, 0, 0
        _SP_WORLD_GROUP = None
        return

    world_size = dist.get_world_size()
    if world_size % size:
        raise ValueError(f"WORLD size {world_size} is not divisible by SP size {size}")
    if _SP_WORLD_GROUP is dist.group.WORLD:
        if _SP_SIZE != size:
            raise RuntimeError("sequence parallel topology is already initialized")
        return

    rank = dist.get_rank()
    group = None
    if size > 1:
        # All WORLD ranks create all groups in the same order, including
        # groups they do not belong to, as required by new_group.
        for first_rank in range(0, world_size, size):
            ranks = list(range(first_rank, first_rank + size))
            candidate = dist.new_group(ranks=ranks)
            if first_rank <= rank < first_rank + size:
                group = candidate
    _SP_GROUP = group
    _SP_SIZE = size
    _SP_RANK = rank % size
    _SP_SRC_RANK = rank - _SP_RANK
    _SP_WORLD_GROUP = dist.group.WORLD


def get_sp_world_size():
    return _SP_SIZE


def get_sp_rank():
    return _SP_RANK


def get_sp_group():
    return _SP_GROUP


def get_sp_src_rank():
    if _SP_SIZE == 1:
        return dist.get_rank() if dist.is_initialized() else 0
    return _SP_SRC_RANK


def get_data_parallel_rank():
    return dist.get_rank() // _SP_SIZE if dist.is_initialized() else 0


def get_data_parallel_world_size():
    return dist.get_world_size() // _SP_SIZE if dist.is_initialized() else 1


def sp_broadcast(tensor):
    """Return sampled values synchronized from the group's first rank.

    This is for non-differentiable inputs such as noise and timesteps, not
    model activations. Activations use the differentiable collectives below.
    Noncontiguous inputs are copied first; callers must use the return value.
    """
    if _SP_SIZE > 1:
        tensor = tensor.contiguous()
        with torch.no_grad():
            dist.broadcast(tensor, src=_SP_SRC_RANK, group=_SP_GROUP)
    return tensor


def _spatial_size(x, num_frames, ndim):
    if x.ndim != ndim:
        raise ValueError(f"expected a {ndim}-D tensor, got shape {tuple(x.shape)}")
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
        raise ValueError("num_frames must be a positive integer")
    if x.shape[1] % num_frames:
        raise ValueError("sequence length must be divisible by num_frames")
    return x.shape[1] // num_frames


def split_spatial(x, num_frames):
    """Slice [B,F*P,C] into [B,F*(P/SP),C] using ordinary autograd.

    The slice backward restores zeros at other ranks' spatial positions. It
    deliberately performs no gradient communication: FSDP reduces parameter
    gradients, while gather_spatial supplies the replicated-loss SP factor.
    """
    if _SP_SIZE == 1:
        return x
    spatial_size = _spatial_size(x, num_frames, 3)
    if spatial_size % _SP_SIZE:
        raise ValueError("spatial tokens per frame must be divisible by SP size")
    local_size = spatial_size // _SP_SIZE
    frames = x.reshape(x.shape[0], num_frames, spatial_size, x.shape[2])
    return frames[:, :, _SP_RANK * local_size:(_SP_RANK + 1) * local_size].reshape(
        x.shape[0], num_frames * local_size, x.shape[2]
    ).contiguous()


class _GatherSpatial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_frames, group, size):
        batch, local_sequence, channels = x.shape
        local_spatial = local_sequence // num_frames
        gathered = x.new_empty((size * batch, local_sequence, channels))
        dist.all_gather_into_tensor(gathered, x.contiguous(), group=group)
        ctx.num_frames, ctx.group, ctx.size = num_frames, group, size
        return gathered.reshape(size, batch, num_frames, local_spatial, channels).permute(
            1, 2, 0, 3, 4
        ).reshape(batch, num_frames * size * local_spatial, channels)

    @staticmethod
    def backward(ctx, grad_output):
        batch, sequence, channels = grad_output.shape
        local_spatial = sequence // ctx.num_frames // ctx.size
        # reduce_scatter expects rank-major chunks. Undo the frame-major
        # permutation before summing every replica's full-loss gradient.
        packed = grad_output.reshape(
            batch, ctx.num_frames, ctx.size, local_spatial, channels
        ).permute(2, 0, 1, 3, 4).contiguous().reshape(
            ctx.size * batch, ctx.num_frames * local_spatial, channels
        )
        local_grad = grad_output.new_empty((batch, ctx.num_frames * local_spatial, channels))
        dist.reduce_scatter_tensor(local_grad, packed, op=dist.ReduceOp.SUM, group=ctx.group)
        return local_grad, None, None, None


def gather_spatial(x, num_frames):
    """Gather per-frame spatial shards; backward is reduce-scatter SUM.

    All ranks in an SP group must evaluate the full gathered loss. Together
    with WORLD FSDP's gradient average, this gives the mean over distinct DP
    samples without an additional SP scale factor.
    """
    if _SP_SIZE == 1:
        return x
    _spatial_size(x, num_frames, 3)
    return _GatherSpatial.apply(x, num_frames, _SP_GROUP, _SP_SIZE)


def _exchange(x, num_frames, group, size, to_head):
    batch, sequence, heads, head_dim = x.shape
    if to_head:
        local_spatial, local_heads = sequence // num_frames, heads // size
        packed = x.reshape(
            batch, num_frames, local_spatial, size, local_heads, head_dim
        ).permute(3, 0, 1, 2, 4, 5).contiguous()
    else:
        local_spatial, local_heads = sequence // num_frames // size, heads
        packed = x.reshape(
            batch, num_frames, size, local_spatial, local_heads, head_dim
        ).permute(2, 0, 1, 3, 4, 5).contiguous()
    received = torch.empty_like(packed)
    # Leading dimension is the destination/source rank. The packed buffers
    # have equal contiguous chunks, supported by both NCCL and GLOO.
    dist.all_to_all_single(received, packed, group=group)
    if to_head:
        return received.permute(1, 2, 0, 3, 4, 5).reshape(
            batch, num_frames * size * local_spatial, local_heads, head_dim
        )
    return received.permute(1, 2, 3, 0, 4, 5).reshape(
        batch, num_frames * local_spatial, size * local_heads, head_dim
    )


class _SpatialHeadExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_frames, group, size, to_head):
        ctx.num_frames, ctx.group, ctx.size, ctx.to_head = num_frames, group, size, to_head
        return _exchange(x, num_frames, group, size, to_head)

    @staticmethod
    def backward(ctx, grad_output):
        return _SpatialHeadExchange.apply(
            grad_output, ctx.num_frames, ctx.group, ctx.size, not ctx.to_head
        ), None, None, None, None


def sequence_to_head(x, num_frames):
    """[B,F*P_local,H,D] -> [B,F*P,H/SP,D], preserving global token order."""
    if _SP_SIZE == 1:
        return x
    _spatial_size(x, num_frames, 4)
    if x.shape[2] % _SP_SIZE:
        raise ValueError("attention heads must be divisible by SP size")
    return _SpatialHeadExchange.apply(x, num_frames, _SP_GROUP, _SP_SIZE, True)


def head_to_sequence(x, num_frames):
    """Inverse of sequence_to_head, with the corresponding inverse backward."""
    if _SP_SIZE == 1:
        return x
    spatial_size = _spatial_size(x, num_frames, 4)
    if spatial_size % _SP_SIZE:
        raise ValueError("spatial tokens per frame must be divisible by SP size")
    return _SpatialHeadExchange.apply(x, num_frames, _SP_GROUP, _SP_SIZE, False)
