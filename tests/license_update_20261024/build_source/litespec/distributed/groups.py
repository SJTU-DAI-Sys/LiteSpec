import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist

_DEVICE_MESH = None
_TP_DEVICE_MESH = None
_TP_GROUP = None
_DP_DEVICE_MESH = None
_DP_GROUP = None
_DRAFT_DP_GROUP = None
_DRAFT_SP_GROUP = None
_SP_ULYSSES_GROUP = None
_SP_RING_GROUP = None
_DISAGG_TOPOLOGY = None
_BRIDGE_GROUP = None
_DISAGG_CONTROL_GROUP = None
_OWNED_DISAGG_GROUPS = []


@dataclass(frozen=True)
class DisaggregatedTopology:
    rank: int
    local_rank: int
    node_rank: int
    nnodes: int
    nproc_per_node: int
    target_ranks_per_node: int
    draft_ranks_per_node: int
    target_tp_size: int
    target_ep_size: int
    role: str
    target_replica_local_rank: Optional[int]
    target_tp_rank: Optional[int]
    target_tp_leader_global_rank: Optional[int]
    draft_local_rank: Optional[int]
    target_tp_group: Optional[dist.ProcessGroup]
    target_tp_cpu_group: Optional[dist.ProcessGroup]
    target_moe_ep_group: Optional[dist.ProcessGroup]
    target_moe_ep_cpu_group: Optional[dist.ProcessGroup]
    target_moe_tp_group: Optional[dist.ProcessGroup]
    target_moe_tp_cpu_group: Optional[dist.ProcessGroup]
    target_singleton_group: Optional[dist.ProcessGroup]
    target_singleton_cpu_group: Optional[dist.ProcessGroup]
    bridge_group: Optional[dist.ProcessGroup]
    bridge_cpu_group: Optional[dist.ProcessGroup]
    draft_group: Optional[dist.ProcessGroup]
    draft_cpu_group: Optional[dist.ProcessGroup]

    @property
    def target_replicas_per_node(self) -> int:
        return self.target_ranks_per_node // self.target_tp_size

    @property
    def is_target(self) -> bool:
        return self.role == "target"

    @property
    def is_draft(self) -> bool:
        return self.role == "draft"

    @property
    def is_target_leader(self) -> bool:
        return self.is_target and self.target_tp_rank == 0

    @property
    def draft_global_ranks(self) -> list[int]:
        return [
            node * self.nproc_per_node + self.target_ranks_per_node + local
            for node in range(self.nnodes)
            for local in range(self.draft_ranks_per_node)
        ]

    @property
    def node_target_leader_ranks(self) -> list[int]:
        base = self.node_rank * self.nproc_per_node
        return [
            base + replica * self.target_tp_size
            for replica in range(self.target_replicas_per_node)
        ]

    @property
    def node_draft_ranks(self) -> list[int]:
        base = self.node_rank * self.nproc_per_node + self.target_ranks_per_node
        return [base + local for local in range(self.draft_ranks_per_node)]


def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP


def get_dp_group():
    global _DP_GROUP
    return _DP_GROUP


def get_draft_dp_group():
    global _DRAFT_DP_GROUP
    return _DRAFT_DP_GROUP


def get_draft_sp_group():
    global _DRAFT_SP_GROUP
    return _DRAFT_SP_GROUP


def get_device_mesh():
    global _DEVICE_MESH
    return _DEVICE_MESH


def get_tp_device_mesh():
    global _TP_DEVICE_MESH
    return _TP_DEVICE_MESH


def get_dp_device_mesh():
    global _DP_DEVICE_MESH
    return _DP_DEVICE_MESH


def get_sp_ulysses_group():
    global _SP_ULYSSES_GROUP
    return _SP_ULYSSES_GROUP


def get_sp_ring_group():
    global _SP_RING_GROUP
    return _SP_RING_GROUP


def get_disaggregated_topology() -> Optional[DisaggregatedTopology]:
    return _DISAGG_TOPOLOGY


def get_bridge_group():
    return _BRIDGE_GROUP


def disaggregated_group_plan(
    *,
    nnodes: int,
    target_ranks_per_node: int,
    draft_ranks_per_node: int,
    target_tp_size: int,
    target_ep_size: int,
) -> dict[str, list[list[int]]]:
    """Return the canonical rank lists used by ``init_disaggregated``."""
    if (
        min(
            nnodes,
            target_ranks_per_node,
            draft_ranks_per_node,
            target_tp_size,
            target_ep_size,
        )
        <= 0
    ):
        raise ValueError("disaggregated topology sizes must be positive")
    if target_ranks_per_node % target_tp_size:
        raise ValueError("target_ranks_per_node must be divisible by target_tp_size")
    if target_tp_size % target_ep_size:
        raise ValueError("target_tp_size must be divisible by target_ep_size")
    per_node = target_ranks_per_node + draft_ranks_per_node
    replicas = target_ranks_per_node // target_tp_size
    plan = {key: [] for key in ("tp", "moe_ep", "moe_tp", "singleton", "bridge")}
    for node in range(nnodes):
        base = node * per_node
        for replica in range(replicas):
            replica_base = base + replica * target_tp_size
            tp = list(range(replica_base, replica_base + target_tp_size))
            plan["tp"].append(tp)
            moe_tp_size = target_tp_size // target_ep_size
            for lane in range(moe_tp_size):
                plan["moe_ep"].append(
                    [
                        replica_base + lane + ep * moe_tp_size
                        for ep in range(target_ep_size)
                    ]
                )
            for ep in range(target_ep_size):
                plan["moe_tp"].append(
                    list(
                        range(
                            replica_base + ep * moe_tp_size,
                            replica_base + (ep + 1) * moe_tp_size,
                        )
                    )
                )
            plan["singleton"].extend([[target_rank] for target_rank in tp])
        plan["bridge"].append(
            [base + replica * target_tp_size for replica in range(replicas)]
            + [
                base + target_ranks_per_node + local
                for local in range(draft_ranks_per_node)
            ]
        )
    plan["draft"] = [
        [
            node * per_node + target_ranks_per_node + local
            for node in range(nnodes)
            for local in range(draft_ranks_per_node)
        ]
    ]
    return plan


def _new_disagg_group(ranks: list[int], backend: str, current_rank: int):
    import os

    debug = os.environ.get("LITESPEC_GROUP_DEBUG") == "1"
    if debug:
        print(
            f"[rank {current_rank}] create {backend} group {ranks}",
            flush=True,
        )
    group = None
    if backend == "nccl":
        candidate = dist.new_group(ranks=ranks, backend=backend)
        if current_rank in ranks:
            group = candidate
    elif current_rank in ranks:
        group = dist.new_group(
            ranks=ranks,
            backend=backend,
            use_local_synchronization=True,
        )
        # Eagerly initialize NCCL communicators before ranks enter the global
        # control barrier; otherwise NCCL's lazy init can overlap rendezvous.
        dist.barrier(group=group)
    # Non-members may return from new_group before members finish rendezvous.
    # Serializing group creation on the default world prevents a faster rank
    # from entering a later Gloo/NCCL rendezvous with the same store sequence.
    dist.barrier(group=_DISAGG_CONTROL_GROUP)
    if debug:
        print(
            f"[rank {current_rank}] created {backend} group {ranks}",
            flush=True,
        )
    if group is not None:
        _OWNED_DISAGG_GROUPS.append(group)
        return group
    return None


def init_disaggregated(
    *,
    timeout: int,
    target_ranks_per_node: int,
    draft_ranks_per_node: int,
    target_tp_size: int,
    target_ep_size: int = 1,
) -> DisaggregatedTopology:
    """Create all role-specific communicators before target/draft divergence."""
    import os

    if (
        min(target_ranks_per_node, draft_ranks_per_node, target_tp_size, target_ep_size)
        <= 0
    ):
        raise ValueError("disaggregated topology sizes must be positive")
    if target_ranks_per_node % target_tp_size:
        raise ValueError("target_ranks_per_node must be divisible by target_tp_size")
    if target_tp_size % target_ep_size:
        raise ValueError("target_tp_size must be divisible by target_ep_size")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=timeout),
        device_id=torch.device("cuda", local_rank),
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()
    global _DISAGG_CONTROL_GROUP
    _OWNED_DISAGG_GROUPS.clear()
    _DISAGG_CONTROL_GROUP = dist.new_group(
        ranks=list(range(world_size)), backend="gloo"
    )
    _OWNED_DISAGG_GROUPS.append(_DISAGG_CONTROL_GROUP)
    nproc_per_node = target_ranks_per_node + draft_ranks_per_node
    if world_size % nproc_per_node:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by per-node ranks "
            f"{nproc_per_node}"
        )
    if local_rank >= nproc_per_node:
        raise ValueError(
            f"LOCAL_RANK={local_rank} is outside per-node world {nproc_per_node}"
        )
    nnodes = world_size // nproc_per_node
    node_rank = rank // nproc_per_node
    replicas = target_ranks_per_node // target_tp_size
    current = {
        "tp": None,
        "tp_cpu": None,
        "ep": None,
        "ep_cpu": None,
        "moe_tp": None,
        "moe_tp_cpu": None,
        "singleton": None,
        "singleton_cpu": None,
        "bridge": None,
        "bridge_cpu": None,
        "draft": None,
        "draft_cpu": None,
    }
    # Every global rank executes every call in this exact order. SGLang target
    # ranks later wrap these groups and therefore never call new_group alone.
    for node in range(nnodes):
        base = node * nproc_per_node
        for replica in range(replicas):
            replica_base = base + replica * target_tp_size
            tp_ranks = list(range(replica_base, replica_base + target_tp_size))
            for backend, key in (("nccl", "tp"), ("gloo", "tp_cpu")):
                group = _new_disagg_group(tp_ranks, backend, rank)
                if group is not None:
                    current[key] = group

            moe_tp_size = target_tp_size // target_ep_size
            for lane in range(moe_tp_size):
                ep_ranks = [
                    replica_base + lane + ep * moe_tp_size
                    for ep in range(target_ep_size)
                ]
                for backend, key in (("nccl", "ep"), ("gloo", "ep_cpu")):
                    group = _new_disagg_group(ep_ranks, backend, rank)
                    if group is not None:
                        current[key] = group
            for ep in range(target_ep_size):
                moe_tp_ranks = list(
                    range(
                        replica_base + ep * moe_tp_size,
                        replica_base + (ep + 1) * moe_tp_size,
                    )
                )
                for backend, key in (
                    ("nccl", "moe_tp"),
                    ("gloo", "moe_tp_cpu"),
                ):
                    group = _new_disagg_group(moe_tp_ranks, backend, rank)
                    if group is not None:
                        current[key] = group
            for target_rank in tp_ranks:
                singleton = [target_rank]
                for backend, key in (
                    ("nccl", "singleton"),
                    ("gloo", "singleton_cpu"),
                ):
                    group = _new_disagg_group(singleton, backend, rank)
                    if group is not None:
                        current[key] = group

    for node in range(nnodes):
        base = node * nproc_per_node
        bridge_ranks = [
            base + replica * target_tp_size for replica in range(replicas)
        ] + [base + target_ranks_per_node + i for i in range(draft_ranks_per_node)]
        group = _new_disagg_group(bridge_ranks, "nccl", rank)
        if group is not None:
            current["bridge"] = group

    draft_ranks = [
        node * nproc_per_node + target_ranks_per_node + local
        for node in range(nnodes)
        for local in range(draft_ranks_per_node)
    ]
    group = _new_disagg_group(draft_ranks, "nccl", rank)
    if group is not None:
        current["draft"] = group

    role = "target" if local_rank < target_ranks_per_node else "draft"
    target_replica = local_rank // target_tp_size if role == "target" else None
    target_tp_rank = local_rank % target_tp_size if role == "target" else None
    leader = (
        node_rank * nproc_per_node + int(target_replica) * target_tp_size
        if target_replica is not None
        else None
    )
    draft_local_rank = local_rank - target_ranks_per_node if role == "draft" else None
    topology = DisaggregatedTopology(
        rank=rank,
        local_rank=local_rank,
        node_rank=node_rank,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        target_ranks_per_node=target_ranks_per_node,
        draft_ranks_per_node=draft_ranks_per_node,
        target_tp_size=target_tp_size,
        target_ep_size=target_ep_size,
        role=role,
        target_replica_local_rank=target_replica,
        target_tp_rank=target_tp_rank,
        target_tp_leader_global_rank=leader,
        draft_local_rank=draft_local_rank,
        target_tp_group=current["tp"],
        target_tp_cpu_group=current["tp_cpu"],
        target_moe_ep_group=current["ep"],
        target_moe_ep_cpu_group=current["ep_cpu"],
        target_moe_tp_group=current["moe_tp"],
        target_moe_tp_cpu_group=current["moe_tp_cpu"],
        target_singleton_group=current["singleton"],
        target_singleton_cpu_group=current["singleton_cpu"],
        bridge_group=current["bridge"],
        bridge_cpu_group=current["bridge_cpu"],
        draft_group=current["draft"],
        draft_cpu_group=current["draft_cpu"],
    )
    global _TP_GROUP, _DP_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP
    global _SP_ULYSSES_GROUP, _SP_RING_GROUP, _DISAGG_TOPOLOGY, _BRIDGE_GROUP
    _TP_GROUP = current["tp"]
    _DP_GROUP = current["draft"]
    _DRAFT_DP_GROUP = current["draft"]
    _DRAFT_SP_GROUP = None
    _SP_ULYSSES_GROUP = None
    _SP_RING_GROUP = None
    _DISAGG_TOPOLOGY = topology
    _BRIDGE_GROUP = current["bridge"]
    print_with_rank(
        f"disaggregate role={role}, node={node_rank}, local_rank={local_rank}, "
        f"target_tp_rank={target_tp_rank}, draft_local_rank={draft_local_rank}"
    )
    dist.barrier(group=_DISAGG_CONTROL_GROUP)
    return topology


def init_distributed(
    timeout: int = 10, tp_size: int = 1, sp_ulysses_size: int = 1, sp_ring_size: int = 1
):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
        tp_size(int): The degree of tensor parallelism
    """
    from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    world_size = dist.get_world_size()
    dp_size = world_size // tp_size
    assert (
        world_size == tp_size * dp_size
    ), f"world size must be divisible by tp size, now {world_size=}, {(tp_size * dp_size)=} "

    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )

    assert (
        world_size % (sp_ulysses_size * sp_ring_size) == 0
    ), f"World size ({world_size}) cannot be evenly divided by total SP size ({sp_ulysses_size * sp_ring_size})"

    draft_dp_size = world_size // (sp_ulysses_size * sp_ring_size)
    draft_device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        (draft_dp_size, sp_ulysses_size * sp_ring_size),
        mesh_dim_names=("draft_dp", "sp"),
    )
    set_seq_parallel_pg(sp_ulysses_size, sp_ring_size, dist.get_rank(), world_size)

    print_with_rank(f"device mesh: {device_mesh}")
    tp_group = device_mesh.get_group("tp")
    dp_group = device_mesh.get_group("dp")

    sp_ulysses_group = PROCESS_GROUP.ULYSSES_PG
    sp_ring_group = PROCESS_GROUP.RING_PG
    # we need to create a 1D submesh
    tp_device_mesh = dist.DeviceMesh.from_group(tp_group, device_type="cuda")

    global _TP_GROUP, _DP_GROUP, _DEVICE_MESH, _TP_DEVICE_MESH, _DP_DEVICE_MESH, _SP_RING_GROUP, _SP_ULYSSES_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP
    _DEVICE_MESH = device_mesh
    _TP_GROUP = tp_group
    _TP_DEVICE_MESH = tp_device_mesh
    _SP_ULYSSES_GROUP = sp_ulysses_group
    _SP_RING_GROUP = sp_ring_group
    _DP_GROUP = dp_group
    _DRAFT_DP_GROUP = draft_device_mesh.get_group("draft_dp")
    _DRAFT_SP_GROUP = draft_device_mesh.get_group("sp")
    _DP_DEVICE_MESH = dist.DeviceMesh.from_group(dp_group, device_type="cuda")


def destroy_distributed():
    global _TP_GROUP, _DP_GROUP, _SP_ULYSSES_GROUP, _SP_RING_GROUP, _DRAFT_DP_GROUP
    if not dist.is_available() or not dist.is_initialized():
        return
    seen = set()
    for group in (
        *_OWNED_DISAGG_GROUPS,
        _TP_GROUP,
        _DP_GROUP,
        _SP_ULYSSES_GROUP,
        _SP_RING_GROUP,
        _DRAFT_DP_GROUP,
        _DRAFT_SP_GROUP,
        _BRIDGE_GROUP,
    ):
        if group is None or group in seen:
            continue
        seen.add(group)
        try:
            dist.destroy_process_group(group)
        except (AssertionError, RuntimeError, ValueError):
            pass
    if dist.is_initialized():
        dist.destroy_process_group()


def is_tp_rank_0():
    """Return True if current process is rank 0 in its TP group."""
    tp_group = get_tp_group()
    if tp_group is None:
        return True
    return dist.get_rank(group=tp_group) == 0


logger = logging.getLogger(__name__)


def print_with_rank(message) -> None:
    if dist.is_available() and dist.is_initialized():
        logger.info("rank %s: %s", dist.get_rank(), message)
    else:
        logger.info("non-distributed: %s", message)


def print_on_rank0(message) -> None:
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(message)
