# Distributed Subsystem (`Codes/framework/distributed`)

The `distributed` subsystem provides multi-GPU, multi-node, and elastic training orchestration supporting Distributed Data Parallel (DDP), Fully Sharded Data Parallel (FSDP), and DeepSpeed ZeRO stages.

---

## 🏗 Architecture Overview

- **Strategy Abstraction (`strategies/`, `strategies.py`)**: Extensible `BaseStrategy` interface registering:
  - **DDP Strategy (`strategies/ddp.py`)**: Standard PyTorch DistributedDataParallel with gradient accumulation synchronization optimization (`model.no_sync()`) to eliminate redundant all-reduce communication.
  - **FSDP Strategy (`strategies/fsdp.py`)**: Zero-redundancy parameter, gradient, and optimizer state sharding across nodes for trillion-parameter scaling.
  - **DeepSpeed Strategy (`strategies/deepspeed.py`)**: ZeRO-Stage 1, 2, and 3 integration with CPU offloading and activation checkpointing.
- **Cluster & Hardware Management (`cluster.py`, `allocator.py`)**: Node discovery, GPU affinity assignment, NUMA node binding, and multi-node topology mapping.
- **Load Balancer & Elastic Scaling (`load_balancer/`, `elastic.py`)**: Dynamic micro-batch sizing, data pipeline balancing, and fault-tolerant worker rendezvous (`rendezvous.py`).
- **Distributed Launcher (`launcher.py`)**: Unified entry point for spawning distributed processes across Slurm, Torchrun, and standalone clusters.

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `trainer.py` | `DistributedTrainer` wrapper managing strategy lifecycle and distributed barriers. |
| `strategies/ddp.py` | DDP strategy with optimized non-blocking gradient accumulation. |
| `strategies/fsdp.py` | Fully Sharded Data Parallel strategy for large model scale. |
| `strategies/deepspeed.py` | DeepSpeed ZeRO-1/2/3 configuration and engine builder. |
| `load_balancer/gpu_allocator.py` | Multi-GPU device allocation and VRAM load balancing. |
| `launcher.py` | Programmatic process launcher for multi-GPU training jobs. |
| `rendezvous.py` | Fault-tolerant dynamic worker registration and health checks. |

---

## 🚀 Quick Start & Usage

### 1. Initializing Distributed Trainer with DDP
```python
from framework.distributed.trainer import DistributedTrainer
from framework.distributed.config import DistributedConfig

dist_config = DistributedConfig(strategy="ddp", backend="nccl", find_unused_parameters=False)
dist_trainer = DistributedTrainer(model=model, config=dist_config)
model = dist_trainer.setup_model()
```

### 2. Multi-GPU Launch via Torchrun
```bash
torchrun --nproc_per_node=4 --master_port=29500 train.py --distributed_strategy ddp
```
