#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Hugging Face Trainer integration with optional LoRA and logit distillation."""

from __future__ import annotations
import copy
from typing import Any
import torch
import torch.nn.functional as F
from neuralops.trainer.distillation.multi_level_kd import LogitDistillationLoss
from neuralops.trainer.trainer import SVDAwareTrainer
from neuralops.utils.lora import apply_lora

class NeuralOpsTrainer(SVDAwareTrainer):
    """Standard Hugging Face Trainer with opt-in NeuralOps training features."""
    def __init__(self, *args: Any, kd_loss: bool = False, teacher_model: Any | None = None, lora: bool = False, lora_rank: int = 8, lora_alpha: float | None = None, lora_dropout: float = 0.0, kd_temperature: float = 2.0, kd_weight: float = 0.5, **kwargs: Any) -> None:
        model = kwargs.get('model')
        if model is None and args:
            model = args[0]
        self.kd_enabled = bool(kd_loss)
        self.kd_weight = float(kd_weight)
        self.kd_objective = LogitDistillationLoss(mode='forward_kl', temperature=kd_temperature)
        if lora and model is not None:
            model, _ = apply_lora(model, r=int(lora_rank), alpha=float(lora_alpha or 2 * lora_rank), dropout=float(lora_dropout))
            for name, parameter in model.named_parameters():
                if 'lora_' in name.lower():
                    parameter.requires_grad = True
            if 'model' in kwargs:
                kwargs['model'] = model
            elif args:
                args = (model, *args[1:])
        if self.kd_enabled:
            self.teacher_model = teacher_model or copy.deepcopy(model)
            self.teacher_model.eval()
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad = False
        else:
            self.teacher_model = None
        super().__init__(*args, **kwargs)

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, num_items_in_batch: int | None = None) -> Any:
        """Combine supervised cross entropy with optional teacher logit KL."""
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        supervised = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None
        if supervised is None and labels is not None:
            supervised = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        if supervised is None:
            supervised = logits.float().square().mean()
        loss = supervised
        if self.teacher_model is not None:
            model_device = next(model.parameters()).device
            if next(self.teacher_model.parameters()).device != model_device:
                self.teacher_model.to(model_device)
            teacher_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**teacher_inputs)
                teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs[0]
            loss = (1.0 - self.kd_weight) * supervised + self.kd_weight * self.kd_objective(logits, teacher_logits)
        return (loss, outputs) if return_outputs else loss
