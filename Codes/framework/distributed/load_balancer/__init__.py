"""
Complete load‑balancing layer for distributed training.

Inspired by:
  - Bagua Load Balanced Data Loader (ETH Zürich / Kuaishou)
  - FastSpeed heterogeneous GPU balancer
  - TorchElastic rendezvous & elastic training (Meta)
  - AdaptDL elastic scheduler (Petuum / CMU)
  - DeepSeek EPLB expert‑parallel load balancer
  - SkyPilot multi‑cloud GPU routing
  - ClusterOps dynamic GPU selection
  - TorchSubmit lightweight cluster management
"""

from .cluster_manager import ClusterManager, NodeRole, NodeSpec
from .data_balancer import BatchSizeBalancer, LoadBalancedDataLoader
from .elastic_trainer import ElasticConfig, ElasticState, ElasticTrainer
from .gpu_allocator import AllocationStrategy, GPUAllocator, GPUInfo
from .job_scheduler import DistributedJobScheduler, JobSpec, SchedulingPolicy

__all__ = [
    # GPU allocation
    "GPUAllocator",
    "GPUInfo",
    "AllocationStrategy",
    # Data balancing
    "LoadBalancedDataLoader",
    "BatchSizeBalancer",
    # Elastic training
    "ElasticTrainer",
    "ElasticConfig",
    "ElasticState",
    # Job scheduling
    "DistributedJobScheduler",
    "JobSpec",
    "SchedulingPolicy",
    # Cluster management
    "ClusterManager",
    "NodeSpec",
    "NodeRole",
]
