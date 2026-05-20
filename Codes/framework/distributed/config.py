"""Unified configuration for all distributed training."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class DistributedConfig:
    """Complete distributed training configuration — one dataclass drives everything."""

    # ── Strategy ───────────────────────────────────────────
    strategy: str = "ddp"               # "none", "ddp", "fsdp", "deepspeed"
    num_gpus: int = 1                   # GPUs for this job (0 = all)
    gpu_ids: Optional[List[int]] = None # specific GPU indices

    # ── Multi‑node ─────────────────────────────────────────
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500

    # ── Elastic training ───────────────────────────────────
    elastic: bool = False
    elastic_min_nodes: int = 1
    elastic_max_nodes: int = 8
    elastic_rdzv_backend: str = "c10d"
    elastic_rdzv_endpoint: str = "localhost:29500"
    elastic_run_id: str = "elastic-run"
    elastic_max_restarts: int = 3
    elastic_checkpoint_dir: str = "./elastic_checkpoints"
    elastic_checkpoint_every_steps: int = 500

    # ── DeepSpeed specific ─────────────────────────────────
    deepspeed_config: Optional[Dict[str, Any]] = field(default_factory=dict)

    # ── GPU allocation ─────────────────────────────────────
    allocation_strategy: str = "best_fit"  # best_fit, round_robin, pack, spread, topology_aware
    min_free_memory_mb: float = 0
    prefer_same_node: bool = True
    auto_discover: bool = True

    # ── Cluster (SSH‑based) ────────────────────────────────
    cluster_nodes: List[Dict[str, Any]] = field(default_factory=list)
    cluster_ssh_user: str = "ubuntu"
    cluster_ssh_key: str = "~/.ssh/id_rsa"
    cluster_sync_dir: str = "~/workspace"

    # ── Scheduler ──────────────────────────────────────────
    scheduler_policy: str = "priority"   # fifo, priority, fair, sjf
    scheduler_max_concurrent: int = 4

    # ── Remote (internet) ──────────────────────────────────
    remote_enabled: bool = False
    remote_server_url: str = ""
    remote_auth_token: str = ""
    remote_num_gpus_needed: int = 1