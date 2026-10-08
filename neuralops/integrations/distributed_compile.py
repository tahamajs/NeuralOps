"""DeepSpeed, FSDP, and torch.compile integrations."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

__all__ = [
    "get_deepspeed_config",
    "apply_fsdp_wrapping",
    "compile_attention_module",
]


def get_deepspeed_config(
    stage: int = 2,
    offload_optimizer: bool = False,
    fp16: bool = True,
) -> dict[str, Any]:
    """Generates standard DeepSpeed Zero configuration dictionary."""
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {
                "device": "cpu" if offload_optimizer else "none",
            },
        },
        "fp16": {"enabled": fp16},
    }


def apply_fsdp_wrapping(model: nn.Module, auto_wrap_policy: Any = None) -> nn.Module:
    """Wraps model in PyTorch FullyShardedDataParallel (FSDP)."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return FSDP(model, auto_wrap_policy=auto_wrap_policy)
    except Exception:
        return model


def compile_attention_module(
    module: nn.Module,
    mode: str = "default",
    dynamic: bool = True,
) -> nn.Module:
    """Applies torch.compile with dynamic shapes support to attention module."""
    if hasattr(torch, "compile"):
        try:
            return torch.compile(module, mode=mode, dynamic=dynamic)
        except Exception:
            return module
    return module
