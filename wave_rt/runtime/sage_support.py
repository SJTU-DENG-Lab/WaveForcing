"""Versioned SM90 Sage preparation eligibility."""
from functools import lru_cache
from importlib.metadata import version
import torch

@lru_cache(maxsize=16)
def _sm90_sage_220(device):
    # Private quantization layouts are versioned. Do not apply this contract
    # to a different Sage release or silently switch its attention algorithm.
    return (torch.cuda.get_device_capability(device) == (9, 0)
            and version('sageattention').split('+')[0] == '2.2.0')

def supported(q):
    return (q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
            and q.ndim == 4 and q.shape[-1] == 128 and q.stride(-1) == 1
            and _sm90_sage_220(q.device))
