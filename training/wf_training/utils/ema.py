"""FP32 EMA over local FSDP original-parameter shards, between train steps."""
from contextlib import contextmanager

import torch


def local_parameters(module):
    result = {}
    for name, parameter in module.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "")
        if name in result:
            raise ValueError(f"Duplicate local EMA parameter: {name}")
        result[name] = parameter
    return result


class ShardedEMA:
    """No full-parameter gather in initialization or update.

    FSDP ``use_orig_params=True`` exposes rank-local 1D shards (including
    zero-sized parameters) outside forward/backward. The topology is fixed for
    resume; serialized shapes protect against a different sharding layout.
    """
    def __init__(self, module, decay=0.999, *, initialize=True):
        self.decay = float(decay)
        if not 0 <= self.decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.shadow = {}
        if initialize:
            self.shadow = {
                name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
                for name, parameter in local_parameters(module).items()
            }

    def _validate(self, module):
        parameters = local_parameters(module)
        if parameters.keys() != self.shadow.keys():
            raise ValueError("EMA shard parameter names differ from the model")
        for name, parameter in parameters.items():
            if parameter.shape != self.shadow[name].shape:
                raise ValueError(f"EMA shard shape mismatch for {name}; keep the FSDP topology fixed")
        return parameters

    @torch.no_grad()
    def update(self, module):
        for name, parameter in self._validate(module).items():
            self.shadow[name].mul_(self.decay).add_(
                parameter.detach().to(device="cpu", dtype=torch.float32), alpha=1 - self.decay)

    def state_dict(self):
        return {"format": "wf_ema_local_v1", "decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state, module):
        if not isinstance(state, dict) or state.get("format") != "wf_ema_local_v1":
            raise ValueError("Expected a local sharded EMA checkpoint")
        if state.get("decay") != self.decay:
            raise ValueError("EMA decay differs from the saved configuration")
        shadow = state.get("shadow")
        if not isinstance(shadow, dict) or any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            for value in shadow.values()
        ):
            raise ValueError("EMA checkpoint must contain FP32 tensor shards")
        self.shadow = {name: value.detach().cpu().clone() for name, value in shadow.items()}
        self._validate(module)

    @contextmanager
    @torch.no_grad()
    def applied_to(self, module):
        """Temporarily install local EMA shards for an explicit full export.

        A full state dict gather happens only in the caller's save operation.
        Backups are rank-local CPU shards; raw weights are restored on failure.
        """
        parameters = self._validate(module)
        backup = {}
        try:
            for name, parameter in parameters.items():
                backup[name] = parameter.detach().cpu().clone()
                parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))
            yield
        finally:
            # FSDP state_dict may refresh the local parameter views.
            current = local_parameters(module)
            for name, value in backup.items():
                current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))

