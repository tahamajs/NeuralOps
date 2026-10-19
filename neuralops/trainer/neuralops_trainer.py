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
    def __init__(self, *args: Any, kd_loss: bool = False, teacher_model: Any | None = None, lora: bool = False, peft_method: str | None = None, lora_rank: int = 8, lora_r: int | None = None, lora_alpha: float | None = None, lora_dropout: float = 0.0, kd_temperature: float = 2.0, kd_weight: float = 0.5, **kwargs: Any) -> None:
        model = kwargs.get('model')
        if model is None and args:
            model = args[0]
        self.peft_method = (peft_method or ("lora" if lora else None))
        if self.peft_method is not None:
            self.peft_method = self.peft_method.lower()
            if self.peft_method not in {"lora", "pissa", "dora"}:
                raise ValueError("peft_method must be one of: lora, pissa, dora")
        lora_rank = int(lora_r or lora_rank)
        self.kd_enabled = bool(kd_loss or teacher_model is not None)
        self.kd_weight = float(kd_weight)
        self.kd_objective = LogitDistillationLoss(mode='forward_kl', temperature=kd_temperature)
        if self.peft_method and model is not None:
            for parameter in model.parameters():
                parameter.requires_grad = False
            if self.peft_method == "lora":
                model, _ = apply_lora(model, r=lora_rank, alpha=float(lora_alpha or 2 * lora_rank), dropout=float(lora_dropout))
            elif self.peft_method == "pissa":
                from neuralops.trainer.peft_recovery.svd_adapters import inject_pissa_adapters
                inject_pissa_adapters(model, rank=lora_rank)
            else:
                from neuralops.trainer.methods.lora import _apply_adapters
                _apply_adapters(model, lora_rank, float(lora_alpha or 2 * lora_rank), float(lora_dropout), "dora")
            for name, parameter in model.named_parameters():
                if any(token in name.lower() for token in ('lora_', 'magnitude', 'weight_residual')) or name.lower().endswith('.m'):
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

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """Save only PEFT/attention tensors when parameter-efficient training is active."""
        if self.peft_method:
            from neuralops.utils.hub_integration import save_attention_only

            target = output_dir or self.args.output_dir
            save_attention_only(self.model, target)
            return
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

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
