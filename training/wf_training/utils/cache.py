"""Reusable rollout caches with an explicit backward lifetime boundary."""

import torch


def offpolicy_cache_frames(total_frames, context_blocks, window_blocks, block_frames):
    """Bound capacity by the actual teacher sequence without shortening context."""
    return min(total_frames, (context_blocks + window_blocks) * block_frames)


class TrainingKVCache:
    """Own cache storage across predictions, including checkpoint recomputation.

    A prediction with gradients reserves its caches until the trainer calls
    ``release_after_backward``. Releasing only changes ownership; tensors are
    detached and reset at the start of the next prediction. The caller must not
    release a graph that it intends to backward again with ``retain_graph=True``.
    """

    def __init__(self):
        self.kv = None
        self.crossattn = None
        self.pending_backward = False

    def reserve_for_backward(self, output):
        self.pending_backward |= output.requires_grad

    def release_after_backward(self):
        # Do not mutate storage here: this also runs during exception cleanup.
        self.pending_backward = False

    def _check_reusable(self):
        if self.pending_backward:
            raise RuntimeError(
                "Cannot reset rollout caches before the previous prediction's "
                "backward completes; call release_caches() after backward."
            )

    def reset_kv(self, num_layers, shape, dtype, device):
        self._check_reusable()
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        compatible = self.kv is not None and len(self.kv) == num_layers and all(
            tuple(layer[key].shape) == tuple(shape)
            and layer[key].dtype == dtype
            and layer[key].device == device
            for layer in self.kv for key in ("k", "v")
        )
        if not compatible:
            # Release the old list before allocating a replacement. Constructing
            # a new full cache and only then assigning it doubles peak memory.
            self.kv = None
            self.kv = [
                {
                    "k": torch.zeros(shape, dtype=dtype, device=device),
                    "v": torch.zeros(shape, dtype=dtype, device=device),
                    "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
                    "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
                }
                for _ in range(num_layers)
            ]
            return

        for layer in self.kv:
            for key in ("k", "v"):
                # Cache writes during a grad window can attach CopySlices nodes.
                # Drop those graphs before reusing the underlying allocation.
                layer[key] = layer[key].detach()
                layer[key].zero_()
            layer["global_end_index"].zero_()
            layer["local_end_index"].zero_()

    def reset_crossattn(self, num_layers):
        self._check_reusable()
        if self.crossattn is None or len(self.crossattn) != num_layers:
            self.crossattn = None
            self.crossattn = [{} for _ in range(num_layers)]
        for layer in self.crossattn:
            # WanT2VCrossAttention replaces k/v on its first call. Empty slots
            # avoid allocating placeholders and release the previous prediction.
            layer.update(k=None, v=None, is_init=False)
