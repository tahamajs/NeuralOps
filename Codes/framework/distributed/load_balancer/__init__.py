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

from .gpu_allocator import GPUAllocator, GPUInfo, AllocationStrategy
from .data_balancer import LoadBalancedDataLoader, BatchSizeBalancer
from .elastic_trainer import ElasticTrainer, ElasticConfig, ElasticState
from .job_scheduler import DistributedJobScheduler, JobSpec, SchedulingPolicy
from .cluster_manager import ClusterManager, NodeSpec, NodeRole

__all__ = [
    # GPU allocation
    "GPUAllocator", "GPUInfo", "AllocationStrategy",
    # Data balancing
    "LoadBalancedDataLoader", "BatchSizeBalancer",
    # Elastic training
    "ElasticTrainer", "ElasticConfig", "ElasticState",
    # Job scheduling
    "DistributedJobScheduler", "JobSpec", "SchedulingPolicy",
    # Cluster management
    "ClusterManager", "NodeSpec", "NodeRole",
]