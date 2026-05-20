
# `framework/distributed` — Distributed Training

The **distributed** module provides a complete, production‑grade system for training models across **multiple GPUs, multiple nodes, and even across the internet**.  
It is designed to be **zero‑code‑change** for existing training loops: wrap your training function and let the module handle the rest.

---

## Quick Start

```python
from framework.distributed import DistributedConfig, DistributedTrainer

# 1. Define config
cfg = DistributedConfig(strategy="ddp", num_gpus=4)

# 2. Wrap your training function
trainer = DistributedTrainer(cfg)
trainer.run(my_train_fn, model=model, dataloader=train_loader)
```

That’s it — your `my_train_fn` will now run on 4 GPUs using DistributedDataParallel, with GPU discovery, allocation, and cleanup handled automatically.

---

## Components

### 1. `DistributedConfig`
A single dataclass that controls **every** aspect of distributed training.  
Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `strategy` | `"ddp"` | Strategy: `"none"`, `"ddp"`, `"fsdp"`, `"deepspeed"` |
| `num_gpus` | `1` | GPUs to use (0 = all) |
| `num_nodes` | `1` | Number of nodes |
| `elastic` | `False` | Enable elastic training |
| `allocation_strategy` | `"best_fit"` | How GPUs are chosen (best_fit, round_robin, pack, spread, topology_aware) |
| `cluster_nodes` | `[]` | List of remote node specs |
| `remote_enabled` | `False` | Use remote workers over internet |

Full reference in `config.py`.

### 2. `GPUAllocator`
Singleton that discovers all GPUs across nodes and allocates them to jobs with five strategies:
- **best_fit** – pick GPUs with most free memory
- **round_robin** – cycle through all GPUs
- **pack** – pack jobs onto as few nodes as possible
- **spread** – spread across many nodes
- **topology_aware** – prefer GPUs on the same node

### 3. Distributed Strategies
Pluggable via `STRATEGY_REGISTRY`:
- **DDP** – PyTorch DistributedDataParallel
- **FSDP** – Fully Sharded Data Parallel
- **DeepSpeed** – ZeRO optimization (requires `deepspeed`)

### 4. `DistributedTrainer`
One‑call wrapper that:
1. Allocates GPUs via `GPUAllocator`
2. Sets environment variables (`CUDA_VISIBLE_DEVICES`, `LOCAL_RANK`)
3. Initialises the distributed strategy
4. Wraps the model
5. Runs your training function
6. Cleans up afterwards

### 5. Launcher
`launch_distributed()` uses `torchrun` (or `torchelastic`) to launch multi‑process training.  
Supports single‑node multi‑GPU, multi‑node, and elastic launch.

### 6. `ClusterManager`
SSH‑based cluster manager (no SLURM/Kubernetes needed):
- `add_node()` – register a remote machine
- `sync_code()` – rsync your code to all nodes
- `launch()` – start `torchrun` across the cluster

### 7. `ElasticTrainer`
Fault‑tolerant training with dynamic scaling:
- Workers can join and leave mid‑training
- Checkpoints are automatically saved and restored
- Uses TorchElastic rendezvous

### 8. `DistributedJobScheduler`
Priority queue for training jobs:
- Submit jobs with `submit(config, num_gpus, priority)`
- Jobs are dispatched as GPUs become free
- Supports FIFO, priority, fair‑share, shortest‑job scheduling

### 9. `InternetRendezvous`
Helps GPUs behind NAT find each other via a central coordinating server.

---

## Usage Scenarios

### Single‑Node Multi‑GPU
```python
cfg = DistributedConfig(strategy="ddp", num_gpus=4)
trainer = DistributedTrainer(cfg)
trainer.run(train_fn, model=model, dataloader=loader)
```

### Multi‑Node (via torchrun)
```python
# On master node:
cfg = DistributedConfig(strategy="fsdp", num_gpus=8, num_nodes=2,
                         master_addr="10.0.0.1", master_port=29500)
launch_distributed(cfg, train_fn)
```

### Cluster (SSH‑based)
```python
cluster = ClusterManager()
cluster.add_node("gpu1.example.com", role=NodeRole.MASTER)
cluster.add_node("gpu2.example.com")
cluster.sync_code(".", "~/workspace")
cluster.launch("python run.py --task sst2", nproc_per_node=4, nnodes=2)
```

### Elastic Training
```python
cfg = DistributedConfig(strategy="ddp", num_gpus=4, elastic=True,
                         elastic_min_nodes=1, elastic_max_nodes=8)
launch_elastic(cfg, train_fn)
```

### Job Scheduling
```python
scheduler = DistributedJobScheduler(SchedulingPolicy.PRIORITY)
scheduler.submit({"task": "sst2", "attn_type": "galerkin"}, num_gpus=2, priority=5)
scheduler.submit({"task": "mrpc", "attn_type": "nova"}, num_gpus=1, priority=3)
scheduler.run_loop(my_train_fn)   # blocks, dispatches jobs as GPUs free
```

---

## Dependencies

All heavy dependencies are optional:
- `torch` – required for DDP/FSDP
- `deepspeed` – only if you use `strategy="deepspeed"`
- `optuna` – only for hyperparameter sweeps
- `requests` – only for internet rendezvous

The module gracefully handles missing imports.

---

## Integration with the Framework

The distributed module is designed to work **directly** with the rest of the framework:
- `run.py` can be launched with `launch_distributed(config, main_fn)`
- The `DistributedTrainer` can wrap any model loaded via `framework.models`
- The `GPUAllocator` and `ElasticTrainer` are automatically used when `DistributedConfig.elastic=True`

This means you can take your existing single‑GPU experiment and scale it to hundreds of GPUs with a single config change.

---

The distributed module completes the framework’s journey from **single‑GPU research to internet‑scale training**, all within a clean, typed Python API.