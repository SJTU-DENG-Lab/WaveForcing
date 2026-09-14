"""Explicit node-local FSDP shards and cross-node replicas.

Group construction belongs to the lifetime of WORLD, not to an individual
student/teacher/critic or stage. Every rank creates groups in the same order.
"""
from dataclasses import dataclass, field
import os
import socket

import torch.distributed as dist


_TOPOLOGY = None
_WORLD = None
_LAYOUT = "contiguous_nodes_v1"
_HYBRID = ("hybrid_full", "hybrid_zero2")


@dataclass(frozen=True)
class FSDPTopology:
    world_size: int
    gpus_per_node: int
    num_nodes: int
    sharding_strategy: str
    fsdp_shard_size: int
    fsdp_replica_size: int
    rank: int
    node_rank: int
    shard_ranks: tuple[int, ...]
    replica_ranks: tuple[int, ...]
    node_hostnames: tuple[str, ...]
    shard_group: object = field(repr=False, compare=False)
    replica_group: object = field(repr=False, compare=False)
    fsdp_group_layout: str = _LAYOUT

    @property
    def hybrid(self):
        return self.sharding_strategy in _HYBRID

    @property
    def export_replica(self):
        # FULL_STATE_DICT unshards only within a shard group. Every rank in
        # the selected group participates; only its rank zero retains data.
        return not self.hybrid or self.node_rank == 0

    def metadata(self, rank_specific=False):
        result = {name: getattr(self, name) for name in (
            "world_size", "gpus_per_node", "num_nodes", "sharding_strategy",
            "fsdp_shard_size", "fsdp_replica_size", "fsdp_group_layout",
        )}
        if rank_specific:
            result.update(
                fsdp_shard_rank=self.rank % self.fsdp_shard_size,
                fsdp_replica_rank=self.node_rank if self.hybrid else 0,
            )
        return result


def get_fsdp_topology():
    """Return topology for the current WORLD, never stale destroyed groups."""
    if not dist.is_initialized() or _WORLD is not dist.group.WORLD:
        return None
    return _TOPOLOGY


def reset_fsdp_topology():
    """Forget cached references after WORLD teardown; do not destroy groups.

    Existing FSDP modules must be discarded before resetting. Distributed
    teardown owns process-group destruction; stages normally reuse the cache.
    """
    global _TOPOLOGY, _WORLD
    _TOPOLOGY = _WORLD = None


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _local_record(config):
    rank, world = dist.get_rank(), dist.get_world_size()
    record = {"rank": rank, "world_size": world, "hostname": socket.gethostname()}
    try:
        expected_world = _positive_integer(config.world_size, "world_size")
        local = _positive_integer(getattr(config, "gpus_per_node", expected_world),
                                  "gpus_per_node")
        if expected_world % local:
            raise ValueError("world_size must be divisible by gpus_per_node")
        strategy = getattr(config, "sharding_strategy", "full")
        if strategy not in ("full", *_HYBRID):
            raise ValueError(f"unsupported FSDP topology strategy: {strategy}")
        hybrid = strategy in _HYBRID
        if hybrid and local < 2:
            # Torch downgrades a one-rank shard group to NO_SHARD and would
            # no longer reduce gradients across the replica process group.
            raise ValueError("hybrid sharding requires at least two ranks per node")
        nodes = expected_world // local
        shard, replica = (local, nodes) if hybrid else (expected_world, 1)
        layout = dict(world_size=expected_world, gpus_per_node=local,
                      num_nodes=nodes, sharding_strategy=strategy,
                      fsdp_shard_size=shard, fsdp_replica_size=replica,
                      fsdp_group_layout=_LAYOUT)
        for name, expected in layout.items():
            if getattr(config, name, expected) != expected:
                raise ValueError(f"{name} disagrees with the resolved FSDP topology")
        record.update(
            layout=layout,
            sp_size=_positive_integer(getattr(config, "sequence_parallel_size", 1),
                                      "sequence_parallel_size"),
            env_rank=int(os.environ.get("RANK", rank)),
            env_world=int(os.environ.get("WORLD_SIZE", world)),
            local_rank=int(os.environ.get("LOCAL_RANK", rank % local)),
            local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", local)),
        )
    except (AttributeError, TypeError, ValueError) as error:
        record["error"] = f"rank {rank}: {error}"
    return record


def _validate_records(records):
    """Validate the same gathered records on every rank before group creation."""
    errors = [record["error"] for record in records if "error" in record]
    if errors:
        raise ValueError("Invalid FSDP topology: " + "; ".join(errors))
    layout = records[0]["layout"]
    world, local = layout["world_size"], layout["gpus_per_node"]
    if len(records) != world:
        raise ValueError("configured world_size differs from initialized WORLD")
    sp_size = records[0]["sp_size"]
    if local % sp_size:
        raise ValueError("SP must divide gpus_per_node and stay within one node")
    for rank, record in enumerate(records):
        if record["layout"] != layout or record["sp_size"] != sp_size:
            raise ValueError("FSDP/SP configuration differs between WORLD ranks")
        if (record["rank"] != rank or record["env_rank"] != rank
                or record["world_size"] != world or record["env_world"] != world):
            raise ValueError("RANK/WORLD_SIZE disagree with initialized WORLD")
        if record["local_world_size"] != local or record["local_rank"] != rank % local:
            raise ValueError("LOCAL_RANK/LOCAL_WORLD_SIZE disagree with contiguous node layout")
    hosts = []
    for start in range(0, world, local):
        node_hosts = {record["hostname"] for record in records[start:start + local]}
        if len(node_hosts) != 1:
            raise ValueError("a configured node group spans multiple hostnames")
        hosts.append(next(iter(node_hosts)))
    if len(set(hosts)) != len(hosts):
        raise ValueError("different configured nodes report the same hostname")
    return layout, tuple(hosts)


def initialize_fsdp_topology(config):
    """Validate torchrun placement and cache explicit FSDP process groups.

    SP is checked on every call but is not part of the FSDP group identity.
    All WORLD ranks must call this once per stage, before constructing models.
    """
    global _TOPOLOGY, _WORLD
    if not dist.is_initialized():
        raise RuntimeError("initialize torch.distributed before FSDP topology")
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, _local_record(config))
    layout, hosts = _validate_records(records)
    current = get_fsdp_topology()
    if current is not None:
        if current.metadata() != layout or current.node_hostnames != hosts:
            raise RuntimeError("FSDP topology cannot change while WORLD is initialized")
        return current

    rank, world, local = dist.get_rank(), layout["world_size"], layout["gpus_per_node"]
    node = rank // local
    shard_group, replica_group = dist.group.WORLD, None
    shard_ranks, replica_ranks = tuple(range(world)), (rank,)
    if layout["sharding_strategy"] in _HYBRID:
        for start in range(0, world, local):
            ranks = tuple(range(start, start + local))
            group = dist.new_group(ranks=list(ranks))
            if rank in ranks:
                shard_group, shard_ranks = group, ranks
        for offset in range(local):
            ranks = tuple(range(offset, world, local))
            group = dist.new_group(ranks=list(ranks))
            if rank in ranks:
                replica_group, replica_ranks = group, ranks
    _TOPOLOGY = FSDPTopology(
        **layout, rank=rank, node_rank=node, shard_ranks=shard_ranks,
        replica_ranks=replica_ranks, node_hostnames=hosts,
        shard_group=shard_group, replica_group=replica_group,
    )
    _WORLD = dist.group.WORLD
    return _TOPOLOGY
