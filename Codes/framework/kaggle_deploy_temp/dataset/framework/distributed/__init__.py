"""
Distributed Training Module — GPU allocation, multi‑node orchestration,
elastic training, and job scheduling.

Usage:
    from framework.distributed import (
        DistributedConfig, DistributedTrainer, GPUAllocator,
        ClusterManager, launch_distributed,
    )
"""

from .allocator import AllocationStrategy, GPUAllocator, GPUInfo
from .cluster import ClusterManager, NodeRole, NodeSpec
from .config import DistributedConfig
from .elastic import ElasticConfig, ElasticState, ElasticTrainer
from .launcher import launch_distributed, launch_elastic
from .rendezvous import InternetRendezvous
from .scheduler import DistributedJobScheduler, JobSpec, JobState, SchedulingPolicy
from .strategies import (
    STRATEGY_REGISTRY,
    BaseStrategy,
    DDPStrategy,
    DeepSpeedStrategy,
    FSDPStrategy,
)
from .trainer import DistributedTrainer

__all__ = [
    "DistributedConfig",
    "DistributedTrainer",
    "GPUAllocator",
    "GPUInfo",
    "AllocationStrategy",
    "STRATEGY_REGISTRY",
    "BaseStrategy",
    "DDPStrategy",
    "FSDPStrategy",
    "DeepSpeedStrategy",
    "launch_distributed",
    "launch_elastic",
    "ClusterManager",
    "NodeSpec",
    "NodeRole",
    "DistributedJobScheduler",
    "JobSpec",
    "SchedulingPolicy",
    "JobState",
    "ElasticTrainer",
    "ElasticConfig",
    "ElasticState",
    "InternetRendezvous",
]
