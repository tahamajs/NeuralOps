"""Integration with MegaOrchestrator."""

import os
import torch
from pathlib import Path
from typing import Dict, Any

from .config import DistributedConfig
from .strategies import STRATEGY_REGISTRY
from .manager import GPUManager

def wrap_with_distributed(
    config_dict: Dict[str, Any],
    model: torch.nn.Module,
    job_id: str,
) -> torch.nn.Module:
    """
    Apply distributed training strategy to the model.
    Called by MegaOrchestrator after model loading.

    Returns the wrapped model (with _dist_strategy and _dist_cfg attached for cleanup).
    """
    cfg = DistributedConfig(**{
        k: v for k, v in config_dict.items()
        if k in DistributedConfig.__dataclass_fields__
    })

    if cfg.strategy == "none":
        return model

    # Allocate GPUs
    if cfg.auto_discover or cfg.num_gpus > 0:
        mgr = GPUManager()
        if not cfg.gpu_ids:
            cfg.gpu_ids = mgr.allocate(cfg.num_gpus, job_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.gpu_ids))

    # Strategy setup
    strategy_cls = STRATEGY_REGISTRY.resolve(cfg.strategy)
    strategy = strategy_cls(cfg)
    strategy.setup()
    wrapped = strategy.wrap_model(model, job_id=job_id)

    # Store for later cleanup
    model._dist_strategy = strategy
    model._dist_cfg = cfg
    return wrapped


def cleanup_distributed(model: torch.nn.Module, job_id: str) -> None:
    """Release GPUs and clean up process group."""
    cfg = getattr(model, '_dist_cfg', None)
    if cfg and cfg.gpu_ids:
        GPUManager().release(job_id)
    strategy = getattr(model, '_dist_strategy', None)
    if strategy:
        strategy.cleanup()