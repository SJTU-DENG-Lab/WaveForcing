"""Broadcast a small nested conditioning payload without pickling CUDA tensors."""
import torch
import torch.distributed as dist


def broadcast_tree(value, device, src=0):
    tensors = []

    def pack(item):
        if isinstance(item, torch.Tensor):
            tensors.append(item.to(device).contiguous())
            return ('tensor', tuple(item.shape), item.dtype)
        if isinstance(item, dict):
            return ('dict', [(key, pack(v)) for key, v in item.items()])
        if isinstance(item, (list, tuple)):
            return ('tuple' if isinstance(item, tuple) else 'list', [pack(v) for v in item])
        return ('value', item)

    sender = dist.get_rank() == src
    metadata = [pack(value) if sender else None]
    dist.broadcast_object_list(metadata, src=src)
    index = 0

    def unpack(item):
        nonlocal index
        kind = item[0]
        if kind == 'tensor':
            tensor = tensors[index] if sender else torch.empty(item[1], dtype=item[2], device=device)
            index += 1
            dist.broadcast(tensor, src=src)
            return tensor
        if kind == 'dict':
            return {key: unpack(v) for key, v in item[1]}
        if kind in ('list', 'tuple'):
            values = [unpack(v) for v in item[1]]
            return tuple(values) if kind == 'tuple' else values
        return item[1]

    return unpack(metadata[0])
