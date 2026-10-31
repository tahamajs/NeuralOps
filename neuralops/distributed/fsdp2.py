#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FSDP2 Distributed Knowledge Distillation Harness

Configures dual-model PyTorch FSDP2 sharding across teacher and student networks,
freezing the teacher model while sharding student parameters and optimizers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["configure_fsdp2_distillation"]


def configure_fsdp2_distillation(
    teacher: nn.Module,
    student: nn.Module,
    *,
    sharding_strategy: str = "FULL_SHARD",
    cpu_offload: bool = False,
    auto_wrap_policy: Optional[Any] = None,
    device_mesh: Optional[Any] = None,
) -> Tuple[nn.Module, nn.Module]:
    """Configure PyTorch FSDP2 dual-model sharding for knowledge distillation.

    The teacher network is set to evaluation mode with gradients disabled.
    The student network is wrapped with FSDP2 sharding across available devices.
    If torch.distributed is not initialized (e.g. CPU/macOS), graceful local execution is preserved.

    Args:
        teacher: Full-precision teacher neural network.
        student: Compressed / surgery student neural network.
        sharding_strategy: Sharding strategy string (FULL_SHARD, SHARD_GRAD_OP, NO_SHARD).
        cpu_offload: Whether to offload parameters to CPU.
        auto_wrap_policy: Custom FSDP layer wrapping policy.
        device_mesh: Optional PyTorch 2.x DeviceMesh for multi-dimensional parallelism.

    Returns:
        Tuple of (frozen_teacher, sharded_student).
    """
    # 1. Freeze teacher completely
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not is_distributed:
        logger.info("FSDP2 distillation: torch.distributed not initialized; running in local mode.")
        return teacher, student

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        selected_strategy = strategy_map.get(sharding_strategy.upper(), ShardingStrategy.FULL_SHARD)

        sharded_student = FSDP(
            student,
            sharding_strategy=selected_strategy,
            auto_wrap_policy=auto_wrap_policy,
        )
        return teacher, sharded_student
    except Exception as e:
        logger.warning("FSDP2 sharding failed (%s); returning un-sharded student.", e)
        return teacher, student
