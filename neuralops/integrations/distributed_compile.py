#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Distributed compilation integration for DeepSpeed, FSDP, and ``torch.compile``.
"""

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
    """Generate a DeepSpeed ZeRO configuration for the requested precision mode."""
    if not isinstance(stage, int) or isinstance(stage, bool) or stage not in {0, 1, 2, 3}:
        raise ValueError(f"stage must be one of 0, 1, 2, or 3; got {stage}")
    if offload_optimizer and stage == 0:
        raise ValueError("optimizer offload requires ZeRO stage 1, 2, or 3")
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
    """Build an FSDP auto-wrap policy for attention and low-rank adapter modules."""
    if min_num_params <= 0:
        raise ValueError("min_num_params must be positive")
    try:
        from neuralops.core.base import BaseAttention
        from neuralops.trainer.peft_recovery.svd_adapters import PiSSALinear, DoRALinear
    except ImportError as exc:
        raise RuntimeError("NeuralOps FSDP policy dependencies could not be imported") from exc

    def custom_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        if isinstance(module, (BaseAttention, PiSSALinear, DoRALinear)):
            return True
        return nonwrapped_numel >= min_num_params

    return custom_policy


def apply_fsdp_wrapping(model: nn.Module, auto_wrap_policy: Any = None) -> nn.Module:
    """Wrap ``model`` with FSDP; distributed initialization errors are not suppressed."""
    if not torch.distributed.is_initialized():
        raise RuntimeError("FSDP wrapping requires initialized torch.distributed")
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    policy = auto_wrap_policy or get_neuralops_fsdp_policy()
    return FSDP(model, auto_wrap_policy=policy)


def compile_attention_module(
    module: nn.Module,
    mode: str = "default",
    dynamic: bool = True,
) -> nn.Module:
    """Compile an attention module; backend or graph errors propagate on invocation."""
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch version")
    return torch.compile(module, mode=mode, dynamic=dynamic)
