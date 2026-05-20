"""
Distributed Training Module — GPU allocation, multi‑node orchestration,
elastic training, and job scheduling.

Usage:
    from framework.distributed import (
        DistributedConfig, DistributedTrainer, GPUAllocator,
        ClusterManager, launch_distributed,
    )
"""

from .config import DistributedConfig
from .trainer import DistributedTrainer
from .allocator import GPUAllocator, GPUInfo, AllocationStrategy
from .strategies import STRATEGY_REGISTRY, BaseStrategy, DDPStrategy, FSDPStrategy, DeepSpeedStrategy
from .launcher import launch_distributed, launch_elastic
from .cluster import ClusterManager, NodeSpec, NodeRole
from .scheduler import DistributedJobScheduler, JobSpec, SchedulingPolicy, JobState
from .elastic import ElasticTrainer, ElasticConfig, ElasticState
from .rendezvous import InternetRendezvous

__all__ = [
    "DistributedConfig",
    "DistributedTrainer",
    "GPUAllocator", "GPUInfo", "AllocationStrategy",
    "STRATEGY_REGISTRY", "BaseStrategy", "DDPStrategy", "FSDPStrategy", "DeepSpeedStrategy",
    "launch_distributed", "launch_elastic",
    "ClusterManager", "NodeSpec", "NodeRole",
    "DistributedJobScheduler", "JobSpec", "SchedulingPolicy", "JobState",
    "ElasticTrainer", "ElasticConfig", "ElasticState",
    "InternetRendezvous",
]