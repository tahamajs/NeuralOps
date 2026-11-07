#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FSDP2-based Distributed Knowledge Distillation Engine.

Shards large student and frozen teacher models across multi-GPU clusters
using PyTorch fully_shard (FSDP2) for post-transformation recovery and micro-healing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.distributed.context import DistributedContext
from neuralops.distributed.errors import E_FSDP2_CONFIG_FAILED, DistributedError

logger = logging.getLogger(__name__)

try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
except (ImportError, AttributeError):
    fully_shard = None
    MixedPrecisionPolicy = None


class FSDP2Distillation:
    """
    Coordinates distributed knowledge distillation using PyTorch FSDP2 (fully_shard).
    """

    def __init__(
        self,
        context: Optional[DistributedContext] = None,
        mp_policy: Optional[Any] = None,
    ) -> None:
        self.context = context or DistributedContext()
        if mp_policy is not None:
            self.mp_policy = mp_policy
        elif MixedPrecisionPolicy is not None and torch.cuda.is_available():
            self.mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        else:
            self.mp_policy = None

    def configure(
        self,
        student: nn.Module,
        teacher: nn.Module,
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Configure student and teacher models for FSDP2 sharded execution.

        Teacher parameters are frozen (requires_grad=False), placed in eval mode,
        and sharded. Student layers are sharded per-block to enable overlapped
        all-gather and gradient all-reduce during distillation training.
        """
        if fully_shard is None:
            raise DistributedError(
                "FSDP2 fully_shard is unavailable. PyTorch 2.4+ is required.",
                code=E_FSDP2_CONFIG_FAILED,
            )

        # Freeze teacher parameters and set to eval
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()

        # Shard teacher
        teacher_sharded = fully_shard(teacher, mp_policy=self.mp_policy)

        # Shard student per transformer layer block
        from neuralops.executor.layer_parallel import LayerParallelExecutor

        student_layers = LayerParallelExecutor.find_layer_list(student)
        if student_layers is not None:
            for layer in student_layers:
                fully_shard(layer, mp_policy=self.mp_policy)

        # Shard student root
        student_sharded = fully_shard(student, mp_policy=self.mp_policy)

        return student_sharded, teacher_sharded

    def train_step(
        self,
        student: nn.Module,
        teacher: nn.Module,
        batch: Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, Any]],
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single distributed distillation training step.
        """
        device = self.context.get_device()

        if isinstance(batch, dict):
            device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            s_out = student(**device_batch)
            with torch.no_grad():
                t_out = teacher(**device_batch)
        elif isinstance(batch, (list, tuple)):
            device_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            s_out = student(*device_inputs)
            with torch.no_grad():
                t_out = teacher(*device_inputs)
        else:
            inp = batch.to(device)
            s_out = student(inp)
            with torch.no_grad():
                t_out = teacher(inp)

        s_logits = getattr(s_out, "logits", s_out)
        if isinstance(s_logits, tuple):
            s_logits = s_logits[0]

        t_logits = getattr(t_out, "logits", t_out)
        if isinstance(t_logits, tuple):
            t_logits = t_logits[0]

        if loss_fn is not None:
            loss = loss_fn(s_logits, t_logits)
        else:
            loss = F.mse_loss(s_logits.float(), t_logits.float())

        student_lr = 0.0
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if optimizer.param_groups:
                student_lr = float(optimizer.param_groups[0].get("lr", 0.0))

        vram_mb = 0
        if torch.cuda.is_available() and device.type == "cuda":
            vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

        return {
            "loss": float(loss.item()),
            "student_lr": float(student_lr),
            "vram_mb": int(vram_mb),
        }

    def heal_replaced_model_distributed(
        self,
        student: nn.Module,
        teacher: nn.Module,
        calibration_loader: Any,
        steps: int = 50,
        lr: float = 5e-5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute post-transformation healing with FSDP2 sharding if distributed,
        falling back gracefully to single-GPU heal_replaced_model otherwise.
        """
        if not self.context.is_distributed():
            logger.info("Distributed context is not active; falling back to single-device heal_replaced_model.")
            from neuralops.core._attention_surgery import heal_replaced_model

            return heal_replaced_model(
                teacher_model=teacher,
                student_model=student,
                calibration_inputs=calibration_loader,
                steps=steps,
                lr=lr,
                **kwargs,
            )

        student_sharded, teacher_sharded = self.configure(student, teacher)
        student_sharded.train()
        optimizer = torch.optim.AdamW(student_sharded.parameters(), lr=lr)

        if isinstance(calibration_loader, (list, tuple)):
            batches = list(calibration_loader)
        else:
            batches = [calibration_loader]

        initial_loss: Optional[float] = None
        current_loss: float = 0.0
        history: List[float] = []

        for step in range(steps):
            batch = batches[step % len(batches)]
            metrics = self.train_step(
                student=student_sharded,
                teacher=teacher_sharded,
                batch=batch,
                optimizer=optimizer,
            )
            cur_loss = metrics["loss"]
            if initial_loss is None:
                initial_loss = cur_loss
            current_loss = cur_loss
            history.append(cur_loss)

        student_sharded.eval()
        vram_mb = 0
        device = self.context.get_device()
        if torch.cuda.is_available() and device.type == "cuda":
            vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

        init_val = float(initial_loss if initial_loss is not None else 0.0)
        final_val = float(current_loss)

        return {
            "initial_loss": init_val,
            "final_loss": final_val,
            "loss_reduced": bool(final_val <= init_val),
            "steps": steps,
            "lr": lr,
            "vram_mb": vram_mb,
            "history": history,
            "student": student_sharded,
            "teacher": teacher_sharded,
        }


__all__ = ["FSDP2Distillation"]
