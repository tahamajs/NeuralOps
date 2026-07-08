"""GPU discovery, monitoring, and allocation."""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    BEST_FIT = "best_fit"
    ROUND_ROBIN = "round_robin"
    PACK = "pack"
    SPREAD = "spread"
    TOPOLOGY_AWARE = "topology_aware"


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_mb: float
    free_memory_mb: float
    allocated_mb: float
    utilization_pct: float
    node: str = "localhost"
    is_allocated: bool = False
    allocated_to: Optional[str] = None


class GPUAllocator:
    """Singleton GPU pool with multiple allocation strategies."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init = False
        return cls._instance

    def __init__(self):
        if self._init:
            return
        self._gpus: Dict[str, List[GPUInfo]] = {}
        self._jobs: Dict[str, List[Tuple[str, int]]] = {}
        self._rr_counter = 0
        self.refresh()
        self._init = True

    def refresh(self) -> List[GPUInfo]:
        local = []
        if not torch.cuda.is_available():
            self._gpus["localhost"] = local
            return local
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            local.append(
                GPUInfo(
                    index=i,
                    name=p.name,
                    total_memory_mb=total / (1024**2),
                    free_memory_mb=free / (1024**2),
                    allocated_mb=(total - free) / (1024**2),
                    utilization_pct=0.0,
                )
            )
        self._gpus["localhost"] = local
        return local

    def register_remote(self, node: str, gpus: List[GPUInfo]):
        with self._lock:
            for g in gpus:
                g.node = node
            self._gpus[node] = gpus

    @property
    def total_gpus(self):
        return sum(len(v) for v in self._gpus.values())

    @property
    def available_gpus(self):
        with self._lock:
            return sum(
                1 for gs in self._gpus.values() for g in gs if not g.is_allocated
            )

    def get_free(self, min_mem: float = 0) -> List[GPUInfo]:
        with self._lock:
            return [
                g
                for gs in self._gpus.values()
                for g in gs
                if not g.is_allocated and g.free_memory_mb >= min_mem
            ]

    def allocate(
        self,
        num: int,
        strategy=AllocationStrategy.BEST_FIT,
        job_id: str = None,
        min_mem: float = 0,
        prefer_same_node: bool = True,
    ) -> List[GPUInfo]:
        with self._lock:
            free = self.get_free(min_mem)
            if len(free) < num:
                raise RuntimeError(f"Need {num} GPU(s), only {len(free)} available")

            if strategy == AllocationStrategy.BEST_FIT:
                chosen = sorted(free, key=lambda g: g.free_memory_mb, reverse=True)[
                    :num
                ]
            elif strategy == AllocationStrategy.ROUND_ROBIN:
                start = self._rr_counter % len(free)
                chosen = [free[(start + i) % len(free)] for i in range(num)]
                self._rr_counter += num
            elif strategy == AllocationStrategy.PACK:
                by_node = {}
                _ = [by_node.setdefault(g.node, []).append(g) for g in free]
                chosen = []
                for ng in sorted(by_node.values(), key=len, reverse=True):
                    for g in ng:
                        if len(chosen) < num:
                            chosen.append(g)
                    if len(chosen) >= num:
                        break
            elif strategy == AllocationStrategy.SPREAD:
                nodes = list({g.node for g in free})
                chosen = []
                ri = 0
                while len(chosen) < num:
                    ng = [
                        g
                        for g in free
                        if g.node == nodes[ri % len(nodes)] and g not in chosen
                    ]
                    if ng:
                        chosen.append(ng[0])
                    ri += 1
            elif strategy == AllocationStrategy.TOPOLOGY_AWARE:
                if prefer_same_node:
                    by_node = {}
                    _ = [by_node.setdefault(g.node, []).append(g) for g in free]
                    best = max(by_node, key=lambda n: len(by_node[n]))
                    chosen = by_node[best][:num]
                else:
                    chosen = self.allocate(
                        num, AllocationStrategy.SPREAD, job_id, min_mem
                    )
            else:
                chosen = free[:num]

            for g in chosen:
                g.is_allocated = True
                g.allocated_to = job_id
            if job_id:
                self._jobs[job_id] = [(g.node, g.index) for g in chosen]
            logger.info("Allocated %d GPU(s) to %s via %s", num, job_id, strategy.value)
            return chosen

    def release(self, job_id: str):
        with self._lock:
            if job_id not in self._jobs:
                return
            for node, idx in self._jobs.pop(job_id):
                gpus = self._gpus.get(node, [])
                if idx < len(gpus):
                    gpus[idx].is_allocated = False
                    gpus[idx].allocated_to = None
            logger.info("Released GPUs for %s", job_id)

    def status(self) -> Dict:
        with self._lock:
            return {
                "total": self.total_gpus,
                "free": self.available_gpus,
                "jobs": list(self._jobs.keys()),
                "nodes": {
                    n: {
                        "total": len(gs),
                        "free": sum(1 for g in gs if not g.is_allocated),
                    }
                    for n, gs in self._gpus.items()
                },
            }
