#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Adaptation Distill Operation Contract.

Specification Section 6.11: adaptation.distill@1.0.0
Connects to MultiLevelKDTrainer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.errors import OperationExecutionError
from neuralops.operations.registry import register_operation
from neuralops.trainer.distillation.multi_level_kd import distill

logger = logging.getLogger("NeuralOps.Operations.Adaptation.Distill")

CONTRACT = OperationContract(
    name="adaptation.distill",
    version="1.0.0",
    input_state="*",
    output_state="adapted",
    params_schema={
        "type": "object",
        "required": ["teacher_source", "steps"],
        "properties": {
            "teacher_source": {"type": "string"},
            "steps": {"type": "integer", "minimum": 10},
            "learning_rate": {"type": "number", "default": 5e-5},
            "pillars": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    },
    preconditions=["torch.cuda.is_available", "dataset_specified"],
    postconditions=["recovery_loss_converged"],
    declares_mutations=["*"],
    deterministic_given_seed=False,
    composable_with_self=True,
)


@register_operation(override=True)
class AdaptationDistillOperation(Operation):
    """Operation for post-surgery multi-objective knowledge distillation using MultiLevelKDTrainer."""

    def __init__(self) -> None:
        """Initialize AdaptationDistillOperation with canonical contract."""
        super().__init__(contract=CONTRACT)

    def apply(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply multi-level knowledge distillation from teacher to model.

        Args:
            model: Student neural network module to be distilled.
            config: Parameters including teacher/teacher_model/teacher_source,
                    train_loader/dataset/dataloader, steps, learning_rate/lr,
                    temperature, device, and pillar configuration.

        Returns:
            Tuple of (distilled_student_model, evidence_dict).

        Raises:
            OperationExecutionError: If distillation cannot be completed.
        """
        try:
            teacher = config.get("teacher", config.get("teacher_model", None))
            teacher_source = config.get("teacher_source", "")

            if teacher is None:
                if isinstance(teacher_source, nn.Module):
                    teacher = teacher_source
                else:
                    raise ValueError(
                        f"Teacher model instance must be provided via config['teacher'] or "
                        f"config['teacher_model']. Got teacher_source='{teacher_source}'"
                    )

            steps = int(config.get("steps", 10))
            lr = float(config.get("learning_rate", config.get("lr", 5e-5)))
            temperature = float(config.get("temperature", 2.0))
            train_loader = config.get("train_loader", config.get("dataset", config.get("dataloader", None)))
            pillars: Optional[List[str]] = config.get("pillars", None)
            device = config.get("device", None)

            if device is not None:
                dev = torch.device(device)
            else:
                student_params = list(model.parameters())
                dev = student_params[0].device if student_params else torch.device("cpu")

            kd_kwargs: Dict[str, Any] = {}
            if pillars:
                kd_kwargs["lambda_kd"] = 1.0 if "kd" in pillars or "logits" in pillars or "pillar1" in pillars else 0.0
                kd_kwargs["lambda_mse"] = (
                    0.5 if "mse" in pillars or "hidden" in pillars or "pillar2" in pillars else 0.0
                )
                kd_kwargs["lambda_attn"] = (
                    0.5 if "attn" in pillars or "attention" in pillars or "pillar3" in pillars else 0.0
                )
                kd_kwargs["lambda_rel"] = (
                    0.1 if "rel" in pillars or "minilm" in pillars or "pillar4" in pillars else 0.0
                )
                kd_kwargs["lambda_gram"] = 0.1 if "gram" in pillars or "pillar5" in pillars else 0.0
                kd_kwargs["lambda_cos"] = (
                    0.1 if "cos" in pillars or "cosine" in pillars or "pillar6" in pillars else 0.0
                )
                kd_kwargs["lambda_cka"] = 0.2 if "cka" in pillars else 0.0
                kd_kwargs["lambda_frob"] = 0.05 if "frob" in pillars else 0.0
                kd_kwargs["lambda_stab"] = 0.01 if "stab" in pillars else 0.0
            else:
                for key in (
                    "lambda_kd",
                    "lambda_mse",
                    "lambda_attn",
                    "lambda_rel",
                    "lambda_gram",
                    "lambda_cos",
                    "lambda_cka",
                    "lambda_frob",
                    "lambda_stab",
                ):
                    if key in config:
                        kd_kwargs[key] = float(config[key])

            if train_loader is None:
                logger.warning("No train_loader specified for adaptation.distill; constructing single batch fallback.")
                sample_dim = 16
                for p in teacher.parameters():
                    if p.ndim >= 2:
                        sample_dim = p.shape[-1]
                        break
                train_loader = [torch.randn(4, sample_dim, device=dev) for _ in range(steps)]

            results = distill(
                student=model,
                teacher=teacher,
                train_loader=train_loader,
                epochs=1,
                max_steps=steps,
                lr=lr,
                device=dev,
                temperature=temperature,
                **kd_kwargs,
            )

            evidence: Dict[str, Any] = {
                "operation": "adaptation.distill",
                "steps": steps,
                "learning_rate": lr,
                "temperature": temperature,
                "pillars": pillars,
                "distill_metrics": results,
            }
            logger.info("Successfully executed adaptation.distill: %s", results)
            return model, evidence

        except Exception as exc:
            if isinstance(exc, OperationExecutionError):
                raise
            raise OperationExecutionError(f"Failed to execute adaptation.distill: {exc}") from exc
