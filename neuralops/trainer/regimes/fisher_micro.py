#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fisher-Weighted Curvature Micro-Steps (F-Micro) Regime.

Computes the empirical diagonal Fisher Information Matrix (FIM) on calibration
batches and applies natural gradient damping during the initial micro-steps post-surgery:
    Delta theta = -(F + lambda * I)^{-1} grad_theta L_{healing}
This protects foundational attention/backbone representations while allowing newly
transplanted or adapted weights to freely adapt in unconstrained directions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

logger = logging.getLogger("NeuralOps.Trainer.Regimes.FisherMicro")


def compute_empirical_diagonal_fisher(
    model: nn.Module,
    dataloader: Iterable[Any],
    num_samples: int = 64,
    loss_fn: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Computes diagonal empirical Fisher Information Matrix (FIM):
        F_diag = E_{x ~ D}[ (grad_theta L(x))^2 ]
    on calibration batches prior to micro-training.

    Args:
        model: Target model whose parameters are evaluated.
        dataloader: Calibration data stream.
        num_samples: Maximum number of samples to process.
        loss_fn: Optional surrogate loss callable. If None, uses mean squared output norm.
        device: Target execution device.

    Returns:
        Mapping of parameter name to diagonal empirical Fisher tensor matching parameter shape.
    """
    target_device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )
    was_training = model.training
    model.eval()

    fisher_accum: Dict[str, torch.Tensor] = {}
    for p_name, p in model.named_parameters():
        if p.requires_grad:
            fisher_accum[p_name] = torch.zeros_like(p.data, device=target_device)

    sample_count = 0

    for batch in dataloader:
        if sample_count >= num_samples:
            break

        model.zero_grad(set_to_none=True)

        if isinstance(batch, dict):
            batch_inputs = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(**batch_inputs)
        elif isinstance(batch, (tuple, list)):
            batch_inputs = [
                v.to(target_device) if isinstance(v, torch.Tensor) else v
                for v in batch
            ]
            outputs = model(*batch_inputs)
        elif isinstance(batch, torch.Tensor):
            outputs = model(batch.to(target_device))
        else:
            continue

        # Compute empirical surrogate loss for curvature (log-likelihood proxy)
        if loss_fn is not None:
            loss = loss_fn(outputs)
        else:
            if isinstance(outputs, torch.Tensor):
                loss = 0.5 * (outputs.float() ** 2).mean()
            elif hasattr(outputs, "logits"):
                loss = 0.5 * (outputs.logits.float() ** 2).mean()
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                loss = 0.5 * (outputs[0].float() ** 2).mean()
            else:
                loss = torch.tensor(0.0, device=target_device, requires_grad=True)

        loss.backward()

        with torch.no_grad():
            for p_name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_accum[p_name].add_(p.grad.data.detach() ** 2)

        batch_sz = 1
        if isinstance(batch, dict) and "input_ids" in batch:
            batch_sz = batch["input_ids"].size(0)
        elif isinstance(batch, torch.Tensor):
            batch_sz = batch.size(0)
        sample_count += batch_sz

    if was_training:
        model.train()

    norm_factor = max(1, sample_count)
    fisher_diag = {k: v / norm_factor for k, v in fisher_accum.items()}
    logger.info(
        "Computed empirical diagonal Fisher over %d samples across %d parameter tensors.",
        sample_count,
        len(fisher_diag),
    )
    return fisher_diag


class FMicroOptimizer(Optimizer):
    """
    Fisher-Weighted Curvature Optimizer (F-Micro).

    Scales parameter gradients with the inverse diagonal empirical Fisher Information:
        p.grad = p.grad / (F_diag + damping)
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        fisher_diag: Dict[str, torch.Tensor],
        lr: float = 1e-4,
        damping: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping parameter: {damping}")
        defaults = dict(lr=lr, damping=damping, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.fisher_diag = fisher_diag

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            damping = group["damping"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0.0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Match Fisher diagonal tensor
                f_diag = None
                if p in self.fisher_diag:
                    f_diag = self.fisher_diag[p]
                elif id(p) in self.fisher_diag:
                    f_diag = self.fisher_diag[id(p)]
                else:
                    for key, val in self.fisher_diag.items():
                        if val.shape == p.shape:
                            f_diag = val
                            break

                if f_diag is not None:
                    # Natural gradient damping: Delta theta = -(F + lambda I)^(-1) grad L
                    scale = 1.0 / (f_diag.to(p.device) + damping)
                    step_dir = grad * scale
                else:
                    step_dir = grad / damping

                p.data.add_(step_dir, alpha=-lr)

        return loss


class FisherMicroHealer:
    """
    Engine for computing empirical diagonal Fisher information and running F-Micro healing.
    """

    def __init__(
        self,
        model: nn.Module,
        damping: float = 1e-3,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model = model
        self.damping = damping
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.fisher_diag: Dict[str, torch.Tensor] = {}

    def compute_empirical_fisher(
        self,
        dataloader: Iterable[Any],
        num_samples: int = 64,
        loss_fn: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes diagonal empirical Fisher information matrix:
            F_diag = E_{x ~ D}[ (grad_theta L(x))^2 ]
        """
        self.fisher_diag = compute_empirical_diagonal_fisher(
            model=self.model,
            dataloader=dataloader,
            num_samples=num_samples,
            loss_fn=loss_fn,
            device=self.device,
        )
        return self.fisher_diag

    def create_optimizer(
        self,
        lr: float = 1e-4,
        damping: Optional[float] = None,
        weight_decay: float = 0.0,
    ) -> FMicroOptimizer:
        """Instantiates an FMicroOptimizer conditioned on the computed Fisher diagonal."""
        damp = damping if damping is not None else self.damping
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return FMicroOptimizer(
            trainable_params,
            fisher_diag=self.fisher_diag,
            lr=lr,
            damping=damp,
            weight_decay=weight_decay,
        )


__all__ = ["FMicroOptimizer", "FisherMicroHealer", "compute_empirical_diagonal_fisher"]
