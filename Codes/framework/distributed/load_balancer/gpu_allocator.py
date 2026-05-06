"""
GPUAllocator – dynamically selects the best GPU(s) for each job.

Inspired by ClusterOps' `list_available_gpus()` / `execute_on_gpu()`
and FastSpeed's heterogeneous GPU load balancing.
"""

import torch
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """How to choose GPUs."""
    BEST_FIT = "best_fit"          # pick GPU with most free memory
    ROUND_ROBIN = "round_robin"    # cycle through all GPUs
    PACK = "pack"                  # pack jobs onto as few GPUs as possible
    SPREAD = "spread"              # spread jobs across as many GPUs as possible
    TOPOLOGY_AWARE = "topology_aware"  # minimise inter‑GPU communication cost


@dataclass
class GPUInfo:
    """Snapshot of a single GPU's state."""
    index: int
    name: str
    total_memory_mb: float
    free_memory_mb: float
    allocated_mb: float
    utilization_pct: float
    temperature_c: float = 0.0
    node: str = "localhost"
    is_allocated: bool = False
    allocated_to: Optional[str] = None   # job_id


class GPUAllocator:
    """
    Singleton that discovers all GPUs across nodes and allocates them to jobs.

    Usage:
        allocator = GPUAllocator()
        gpus = allocator.allocate(num_gpus=2, strategy=AllocationStrategy.BEST_FIT,
                                  job_id="train-001")
        # ... run training ...
        allocator.release(job_id="train-001")
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self):
        if self._initialised:
            return
        self._gpus: Dict[str, List[GPUInfo]] = {}   # node → GPU list
        self._jobs: Dict[str, List[Tuple[str, int]]] = {}  # job_id → [(node, gpu_idx)]
        self._round_robin_counter = 0
        self.refresh()
        self._initialised = True
        logger.info("GPUAllocator initialised: %d GPU(s) across %d node(s)",
                    self.total_gpus, self.num_nodes)

    # ── Discovery ───────────────────────────────────────────────────
    def refresh(self) -> List[GPUInfo]:
        """Re‑discover all GPUs on the local node."""
        local = []
        if not torch.cuda.is_available():
            self._gpus["localhost"] = local
            return local
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            local.append(GPUInfo(
                index=i,
                name=props.name,
                total_memory_mb=total / (1024 ** 2),
                free_memory_mb=free / (1024 ** 2),
                allocated_mb=(total - free) / (1024 ** 2),
                utilization_pct=0.0,
                node="localhost",
            ))
        self._gpus["localhost"] = local
        return local

    def register_remote_gpus(self, node: str, gpu_infos: List[GPUInfo]) -> None:
        """Add GPUs from a remote node."""
        with self._lock:
            for g in gpu_infos:
                g.node = node
            self._gpus[node] = gpu_infos
            logger.info("Registered %d GPU(s) from node %s", len(gpu_infos), node)

    # ── Properties ──────────────────────────────────────────────────
    @property
    def num_nodes(self) -> int:
        return len(self._gpus)

    @property
    def total_gpus(self) -> int:
        return sum(len(v) for v in self._gpus.values())

    @property
    def available_gpus(self) -> int:
        with self._lock:
            return sum(1 for gpus in self._gpus.values()
                       for g in gpus if not g.is_allocated)

    def get_free_gpus(self, min_free_memory_mb: float = 0) -> List[GPUInfo]:
        """Return all free GPUs (optionally filtering by minimum free memory)."""
        with self._lock:
            result = []
            for node, gpus in self._gpus.items():
                for g in gpus:
                    if (not g.is_allocated and
                        g.free_memory_mb >= min_free_memory_mb):
                        result.append(g)
            return result

    # ── Allocation ──────────────────────────────────────────────────
    def allocate(
        self,
        num_gpus: int,
        strategy: AllocationStrategy = AllocationStrategy.BEST_FIT,
        job_id: str = None,
        min_free_memory_mb: float = 0,
        prefer_same_node: bool = True,
    ) -> List[GPUInfo]:
        """
        Allocate `num_gpus` GPUs using the given strategy.

        Returns list of allocated GPUInfo objects.
        """
        with self._lock:
            free = self.get_free_gpus(min_free_memory_mb)
            if len(free) < num_gpus:
                raise RuntimeError(
                    f"Need {num_gpus} GPU(s), only {len(free)} available "
                    f"(total: {self.total_gpus}, free: {len(free)})"
                )

            if strategy == AllocationStrategy.BEST_FIT:
                chosen = sorted(free, key=lambda g: g.free_memory_mb, reverse=True)[:num_gpus]
            elif strategy == AllocationStrategy.ROUND_ROBIN:
                start = self._round_robin_counter % len(free)
                indices = [(start + i) % len(free) for i in range(num_gpus)]
                chosen = [free[i] for i in indices]
                self._round_robin_counter += num_gpus
            elif strategy == AllocationStrategy.PACK:
                # Pack onto as few nodes as possible
                by_node = {}
                for g in free:
                    by_node.setdefault(g.node, []).append(g)
                chosen = []
                for node_gpus in sorted(by_node.values(), key=len, reverse=True):
                    for g in node_gpus:
                        if len(chosen) < num_gpus:
                            chosen.append(g)
                        else:
                            break
                    if len(chosen) >= num_gpus:
                        break
            elif strategy == AllocationStrategy.SPREAD:
                # One GPU per node, round‑robin across nodes
                nodes = list(set(g.node for g in free))
                chosen = []
                ri = 0
                while len(chosen) < num_gpus:
                    node = nodes[ri % len(nodes)]
                    node_gpus = [g for g in free if g.node == node and g not in chosen]
                    if node_gpus:
                        chosen.append(node_gpus[0])
                    ri += 1
            elif strategy == AllocationStrategy.TOPOLOGY_AWARE:
                # Prefer GPUs on the same node (lowest communication cost)
                chosen = self._topology_aware(free, num_gpus, prefer_same_node)
            else:
                chosen = free[:num_gpus]

            # Mark as allocated
            for g in chosen:
                g.is_allocated = True
                g.allocated_to = job_id
            if job_id:
                self._jobs[job_id] = [(g.node, g.index) for g in chosen]

            logger.info("Allocated %d GPU(s) to job %s via %s: %s",
                        num_gpus, job_id, strategy.value,
                        [(g.node, g.index) for g in chosen])
            return chosen

    def release(self, job_id: str) -> None:
        """Free all GPUs allocated to `job_id`."""
        with self._lock:
            if job_id not in self._jobs:
                return
            for node, gpu_idx in self._jobs.pop(job_id, []):
                gpus = self._gpus.get(node, [])
                if gpu_idx < len(gpus):
                    gpus[gpu_idx].is_allocated = False
                    gpus[gpu_idx].allocated_to = None
            logger.info("Released GPUs for job %s", job_id)

    def _topology_aware(
        self, free: List[GPUInfo], num: int, prefer_same_node: bool
    ) -> List[GPUInfo]:
        """Pick GPUs with best NVLink / PCI topology."""
        if prefer_same_node:
            # Group by node, pick the node with most free GPUs
            by_node = {}
            for g in free:
                by_node.setdefault(g.node, []).append(g)
            best_node = max(by_node, key=lambda n: len(by_node[n]))
            return by_node[best_node][:num]
        # Otherwise spread one per node in a round‑robin fashion
        return AllocationStrategy.SPREAD(None, free, num)

    def status(self) -> Dict:
        """Return current allocation summary."""
        with self._lock:
            return {
                "total_gpus": self.total_gpus,
                "free_gpus": sum(1 for gpus in self._gpus.values()
                                 for g in gpus if not g.is_allocated),
                "allocated_gpus": sum(1 for gpus in self._gpus.values()
                                      for g in gpus if g.is_allocated),
                "jobs": list(self._jobs.keys()),
                "nodes": {
                    node: {
                        "total": len(gpus),
                        "free": sum(1 for g in gpus if not g.is_allocated),
                    }
                    for node, gpus in self._gpus.items()
                },
            }