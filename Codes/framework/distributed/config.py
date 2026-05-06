from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class DistributedConfig:
    """Complete distributed training configuration."""
    # Strategy: "ddp", "deepspeed", "fsdp", "none"
    strategy: str = "none"

    # GPU allocation (single machine)
    num_gpus: int = 1               # how many GPUs to use on this node
    gpu_ids: Optional[list] = None   # specific GPU indices (e.g. [0,2])

    # Multi‑node
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500

    # DeepSpeed specific (path to JSON config, or inline dict)
    deepspeed_config: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Automatic GPU discovery
    auto_discover: bool = True     # if True, use all available GPUs