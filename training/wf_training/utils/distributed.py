from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

from wf_training.utils.parallel_topology import get_fsdp_topology


def fsdp_state_dict(model):
    """Export full weights from only the first HSDP replica's shard group.

    PyTorch's HSDP rank0_only means *shard-group* rank zero. Calling this on
    every replica would materialize the full CPU checkpoint once per node.
    The pinned FSDP FULL_STATE_DICT path (including lazy initialization) only
    communicates through its shard groups; it does not reduce replica grads.
    All shard ranks on node zero therefore participate, while other nodes
    return early. The trainer's WORLD barrier follows the completed export.
    """
    topology = get_fsdp_topology()
    if topology is not None and topology.hybrid:
        for module in FSDP.fsdp_modules(model):
            if (module.sharding_strategy not in (
                    ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2)
                    or module.process_group is not topology.shard_group
                    or module._inter_node_pg is not topology.replica_group):
                raise ValueError("full export requires the same explicit HSDP groups on every module")
        if not topology.export_replica:
            return {}
    elif getattr(model, "sharding_strategy", None) in (
            ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        raise RuntimeError("initialize explicit HSDP topology before full export")
    # CPU/GLOO diagnostics already hold unsharded tensors on CPU. Asking
    # FSDP to offload CPU storage to itself can invalidate its storage views.
    compute_device = getattr(model, "compute_device", torch.device("cuda"))
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=compute_device.type != "cpu", rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def clear_completed_fsdp_saved_views(model):
    """Release FSDP1 forward views before discarding a completed stage.

    With use_orig_params=True, torch 2.11 keeps parameter views in _tensors
    until backward. Forward-only models can retain a Tensor-base reference
    cycle that Python GC cannot collect. Call only after successful training
    and device synchronization, when no subsequent backward will use them.
    These private fields are covered by pinned-version HSDP teardown tests.
    """
    from torch.distributed.fsdp._common_utils import HandleTrainingState, TrainingState

    handles = []
    for state in FSDP.fsdp_modules(model):
        handle = state._handle
        if handle is None or not state._use_orig_params:
            continue
        if (state.training_state != TrainingState.IDLE
                or handle._training_state != HandleTrainingState.IDLE):
            raise RuntimeError("cannot clear FSDP saved views outside completed idle training")
        if getattr(state, "_post_backward_callback_queued", False):
            raise RuntimeError("cannot clear FSDP saved views with a queued backward callback")
        handles.append(handle)
    cleared = 0
    for handle in handles:
        saved_views = handle.flat_param._tensors
        if saved_views is not None:
            cleared += sum(view is not None for view in saved_views)
            saved_views[:] = [None] * len(saved_views)
    return {"handles": len(handles), "cleared_saved_views": cleared}


@torch.no_grad()
def sync_module_buffers_from_rank0(module, device):
    """Synchronize buffers across WORLD before hybrid FSDP's two broadcasts.

    Torch 2.11 marks buffers as synced during its first, shard-group broadcast
    and skips them in the replica broadcast. Parameters do participate in both
    broadcasts. Synchronizing buffers explicitly preserves checkpoint-loaded
    values when only global rank zero constructs and loads real parameters.
    No complete parameter copy or model checkpoint is loaded on other ranks.
    """
    buffers = list(module.named_buffers(remove_duplicate=False))
    metadata = [(name, tuple(buffer.shape), str(buffer.dtype), str(buffer.layout))
                for name, buffer in buffers]
    error = None
    if any(buffer.layout != torch.strided for _, buffer in buffers):
        error = "only dense strided model buffers are supported"
    if dist.get_rank() == 0 and any(buffer.is_meta for _, buffer in buffers):
        error = "rank zero must initialize model buffers before synchronization"
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, (metadata, error))
    if any(record[1] for record in records):
        raise ValueError("HSDP buffer synchronization failed: "
                         + "; ".join(record[1] for record in records if record[1]))
    if any(record[0] != metadata for record in records):
        raise ValueError("HSDP model buffer names, shapes, or dtypes differ between ranks")
    for name, buffer in buffers:
        value = (buffer.detach().to(device=device).contiguous() if dist.get_rank() == 0
                 else torch.empty(buffer.shape, dtype=buffer.dtype, device=device))
        dist.broadcast(value, src=0)
        if buffer.is_meta:
            parent_name, _, leaf = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            parent._buffers[leaf] = value
        else:
            buffer.copy_(value.to(device=buffer.device))


def materialize_meta_parameters(module, device):
    """Materialize this module only; FSDP visits children separately.

    Real buffers (and Wan's unregistered RoPE tensors) must retain their
    initialized values. Only parameters/buffers actually on meta are empty.
    """
    for name, parameter in module.named_parameters(recurse=False):
        if parameter.is_meta:
            module._parameters[name] = torch.nn.Parameter(
                torch.empty_like(parameter, device=device),
                requires_grad=parameter.requires_grad,
            )
    for name, buffer in module.named_buffers(recurse=False):
        if buffer.is_meta:
            module._buffers[name] = torch.empty_like(buffer, device=device)


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False, sync_module_states=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    topology = get_fsdp_topology()
    process_group = None
    if sharding_strategy in ("hybrid_full", "hybrid_zero2"):
        if topology is None or topology.sharding_strategy != sharding_strategy:
            raise RuntimeError("initialize the matching explicit HSDP topology before wrapping")
        process_group = (topology.shard_group, topology.replica_group)
        if sync_module_states:
            sync_module_buffers_from_rank0(
                module, torch.device("cuda", torch.cuda.current_device()))
    elif topology is not None and topology.sharding_strategy != sharding_strategy:
        raise ValueError("model sharding strategy differs from initialized FSDP topology")
    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        process_group=process_group,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=sync_module_states,
        param_init_fn=(partial(materialize_meta_parameters,
                               device=torch.device("cuda", torch.cuda.current_device()))
                       if sync_module_states else None),
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("invalid RANK/WORLD_SIZE/LOCAL_RANK")
    if dist.is_initialized():
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise ValueError("torchrun environment differs from initialized WORLD")
        if dist.get_backend() != backend:
            raise ValueError("distributed backend differs from initialized WORLD")
        if backend == "nccl" and torch.cuda.current_device() != local_rank:
            raise ValueError("current CUDA device differs from LOCAL_RANK")
        return
    if backend == "nccl":
        # NCCL object collectives and eager communicator initialization must
        # select the correct GPU before initializing the process group.
        torch.cuda.set_device(local_rank)
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999, *, initialize=True):
        self.decay = decay
        self.shadow = {}
        if initialize:
            self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
