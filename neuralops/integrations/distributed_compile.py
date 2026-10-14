"""DeepSpeed, FSDP, and torch.compile integrations."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

__all__ = [
    "get_deepspeed_config",
    "get_neuralops_fsdp_policy",
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


def get_neuralops_fsdp_policy(min_num_params: int = int(1e5)):
    """Generate PyTorch FSDP auto-wrap policy targeted at factorized SVD, PEFT, and attention blocks."""
    try:
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
        from neuralops.core.base import BaseAttention
        from neuralops.trainer.peft_recovery.svd_adapters import PiSSALinear, DoRALinear
        
        def custom_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
            if isinstance(module, (BaseAttention, PiSSALinear, DoRALinear)):
                return True
            return nonwrapped_numel >= min_num_params

        return custom_policy
    except Exception:
        return None


def apply_fsdp_wrapping(model: nn.Module, auto_wrap_policy: Any = None) -> nn.Module:
    """Wraps model in PyTorch FullyShardedDataParallel (FSDP) using neuralops-aware auto-wrap policy."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        policy = auto_wrap_policy or get_neuralops_fsdp_policy()
        return FSDP(model, auto_wrap_policy=policy)
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
