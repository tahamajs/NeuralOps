#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Model Surgery Calibration Utilities

Provides streaming activation and gradient-based calibration routines (P040).
Ensures backward passes are executed under active autograd (outside torch.no_grad blocks)
to prevent RuntimeError under PyTorch >= 2.1.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuralops.surgery.universal_calibrator import (
    UniversalActivationCalibrator,
    compress_model_universal,
    transplant_and_compress,
)

logger = logging.getLogger("NeuralOps.Surgery.Calibration")


class GradientCalibrator:
    """
    Gradient-aware activation and parameter calibrator (P040).

    Computes parameter sensitivity, Fisher information, and gradient energy.
    Guarantees backward passes execute under active autograd (outside torch.no_grad blocks),
    enabling robust calibration across PyTorch 2.1+ without RuntimeErrors.
    """

    def __init__(
        self,
        model: nn.Module,
        target_classes: Tuple[type, ...] = (nn.Linear,),
        exclude_keywords: Tuple[str, ...] = ("lm_head", "classifier", "score", "embed"),
        loss_fn: Optional[Callable[[Any], torch.Tensor]] = None,
        damping: float = 1e-4,
    ) -> None:
        self.model = model
        self.target_classes = target_classes
        self.exclude_keywords = exclude_keywords
        self.loss_fn = loss_fn
        if damping < 0:
            raise ValueError("damping must be non-negative")
        self.damping = damping

        self.grad_energy_dict: Dict[str, torch.Tensor] = {}
        self.sample_count_dict: Dict[str, int] = {}

    def _should_calibrate(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, self.target_classes):
            return False
        for ex in self.exclude_keywords:
            if ex in name.lower():
                return False
        return True

    def clear(self) -> None:
        self.grad_energy_dict.clear()
        self.sample_count_dict.clear()

    def calibrate(
        self,
        calibration_data: Union[torch.utils.data.DataLoader, List[Any], torch.Tensor],
        num_batches: Optional[int] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Runs gradient-based calibration.

        Forward and backward passes are explicitly executed with active autograd
        (outside torch.no_grad blocks or wrapped inside torch.enable_grad).
        """
        if num_batches is not None and num_batches < 0:
            raise ValueError("num_batches must be non-negative or None")

        self.clear()
        training_states = [(module, module.training) for module in self.model.modules()]
        grad_states = [(param, param.requires_grad) for param in self.model.parameters()]

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.eval()

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        if isinstance(calibration_data, torch.Tensor):
            batches = [calibration_data]
        else:
            batches = calibration_data

        try:
            batch_idx = 0
            for raw_batch in batches:
                if num_batches is not None and batch_idx >= num_batches:
                    break

                self.model.zero_grad(set_to_none=True)

                # P040: Autograd explicitly enabled to ensure backward() works
                # even if caller invoked calibrate() within a torch.no_grad() block.
                with torch.enable_grad():
                    if isinstance(raw_batch, dict):
                        batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in raw_batch.items()
                        }
                        out = self.model(**batch)
                    elif isinstance(raw_batch, (tuple, list)):
                        batch = [
                            v.to(device) if isinstance(v, torch.Tensor) else v
                            for v in raw_batch
                        ]
                        out = self.model(*batch)
                    elif isinstance(raw_batch, torch.Tensor):
                        batch = raw_batch.to(device)
                        out = self.model(batch)
                    else:
                        raise TypeError(
                            "Calibration batches must be tensors, tuples/lists, or mappings; "
                            f"received {type(raw_batch).__name__}"
                        )

                    if self.loss_fn is not None:
                        loss = self.loss_fn(out)
                    elif hasattr(out, "loss") and out.loss is not None:
                        loss = out.loss
                    elif isinstance(out, torch.Tensor):
                        loss = out.float().pow(2).mean()
                    elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                        loss = out[0].float().pow(2).mean()
                    elif isinstance(out, dict) and "logits" in out:
                        loss = out["logits"].float().pow(2).mean()
                    else:
                        raise ValueError(f"Cannot compute calibration loss from output of type {type(out).__name__}")

                    # Backward executed with autograd actively enabled
                    loss.backward()

                for name, module in self.model.named_modules():
                    if not self._should_calibrate(name, module):
                        continue

                    grad_energy = torch.zeros(1, device=device)
                    has_grad = False
                    for p in module.parameters():
                        if p.grad is not None:
                            grad_energy += p.grad.detach().float().pow(2).sum()
                            has_grad = True

                    if has_grad:
                        if name not in self.grad_energy_dict:
                            self.grad_energy_dict[name] = grad_energy
                            self.sample_count_dict[name] = 1
                        else:
                            self.grad_energy_dict[name] += grad_energy
                            self.sample_count_dict[name] += 1

                batch_idx += 1
        finally:
            self.model.zero_grad(set_to_none=True)
            for param, was_req in grad_states:
                param.requires_grad = was_req
            for module, was_training in training_states:
                module.training = was_training

        results: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, total_energy in self.grad_energy_dict.items():
            count = max(1, self.sample_count_dict.get(name, 1))
            mean_energy = total_energy / count
            results[name] = {
                "grad_energy": mean_energy,
                "importance": torch.sqrt(mean_energy + self.damping),
            }
        return results


def calibrate_gradient_importance(
    model: nn.Module,
    calibration_data: Union[torch.utils.data.DataLoader, List[Any], torch.Tensor],
    loss_fn: Optional[Callable[[Any], torch.Tensor]] = None,
    num_batches: Optional[int] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Convenience functional wrapper for gradient importance calibration."""
    calibrator = GradientCalibrator(model=model, loss_fn=loss_fn)
    return calibrator.calibrate(calibration_data, num_batches=num_batches)


__all__ = [
    "UniversalActivationCalibrator",
    "compress_model_universal",
    "transplant_and_compress",
    "GradientCalibrator",
    "calibrate_gradient_importance",
]
