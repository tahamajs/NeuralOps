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

from neuralops.core._attention_surgery import calibrate_attention_subspace, calibrate_surgically_replaced_model
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
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw_batch.items()}
                        out = self.model(**batch)
                    elif isinstance(raw_batch, (tuple, list)):
                        batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in raw_batch]
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


def ridge_regression_solve(
    X: torch.Tensor,
    Y: torch.Tensor,
    ridge_lambda: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-Form Ridge Regression Parameter Solver with Cross-Device Robustness.

    Solves:
        W = (X_c^T X_c + lambda * I)^{-1} X_c^T Y_c
        b = mu_Y - mu_X @ W

    Executes without autograd tracking in <1s.
    Monitors Gram condition number kappa = sigma_max / sigma_min; escalates
    Tikhonov damping lambda in [1e-5, 1.0] if kappa > 1e4.
    Seamlessly falls back to CPU SVD / pseudo-inverse on GPU solver failure.
    """
    with torch.no_grad():
        target_device = torch.device(device) if device is not None else X.device
        D_in = X.shape[-1]
        D_out = Y.shape[-1]

        X_flat = X.reshape(-1, D_in).float()
        Y_flat = Y.reshape(-1, D_out).float()

        if X_flat.device != target_device:
            X_flat = X_flat.to(target_device)
        if Y_flat.device != target_device:
            Y_flat = Y_flat.to(target_device)

        mu_X = X_flat.mean(dim=0, keepdim=True)
        mu_Y = Y_flat.mean(dim=0, keepdim=True)
        X_c = X_flat - mu_X
        Y_c = Y_flat - mu_Y

        XT_X_raw = torch.matmul(X_c.t(), X_c)
        current_lambda = float(ridge_lambda)

        # Dynamic Tikhonov damping escalation based on Gram matrix condition number kappa
        try:
            svals = torch.linalg.svdvals(XT_X_raw)
            sigma_max = svals[0].item()
            sigma_min = svals[-1].clamp(min=1e-12).item()
            cond = sigma_max / sigma_min
            if cond > 1e4:
                # Scale damping proportionally to log-condition excess, bounded in [1e-5, 1.0]
                escalated_lambda = max(current_lambda, min(1.0, current_lambda * (cond / 1e4)))
                escalated_lambda = max(1e-5, min(1.0, escalated_lambda))
                logger.warning(
                    "Gram matrix ill-conditioned (cond=%.2e > 1e4); escalating Ridge damping from %.1e to %.1e",
                    cond,
                    current_lambda,
                    escalated_lambda,
                )
                current_lambda = escalated_lambda
        except Exception as e:
            logger.debug("Condition number check failed: %s", e)

        reg = current_lambda * torch.eye(D_in, device=target_device, dtype=torch.float32)
        XT_X = XT_X_raw + reg
        XT_Y = torch.matmul(X_c.t(), Y_c)

        try:
            W = torch.linalg.solve(XT_X, XT_Y)
            b = (mu_Y - torch.matmul(mu_X, W)).squeeze(0)
        except Exception as err:
            logger.warning(
                "Ridge regression solve on device %s failed (%s); escalating damping and falling back to CPU SVD.",
                target_device,
                err,
            )
            try:
                # CPU SVD / pseudo-inverse fallback
                reg_cpu = max(current_lambda, 1e-2) * torch.eye(D_in, device="cpu", dtype=torch.float32)
                XT_X_cpu = XT_X_raw.cpu() + reg_cpu
                XT_Y_cpu = XT_Y.cpu()
                # Use SVD-based lstsq/pinv for maximum numerical stability on CPU
                W_cpu = torch.linalg.lstsq(XT_X_cpu, XT_Y_cpu, driver="gelsd").solution
                b_cpu = (mu_Y.cpu() - torch.matmul(mu_X.cpu(), W_cpu)).squeeze(0)
                W = W_cpu.to(target_device)
                b = b_cpu.to(target_device)
            except Exception as cpu_err:
                logger.error("CPU SVD fallback also failed: %s; using pinv", cpu_err)
                W_cpu = torch.linalg.pinv(XT_X_raw.cpu() + 1e-1 * torch.eye(D_in, device="cpu")) @ XT_Y.cpu()
                b_cpu = (mu_Y.cpu() - torch.matmul(mu_X.cpu(), W_cpu)).squeeze(0)
                W = W_cpu.to(target_device)
                b = b_cpu.to(target_device)

        return W, b


# Alias for pre-benchmark micro-training & surgical healing engine
compute_ridge_calibration = ridge_regression_solve


def align_output_projections(
    teacher: nn.Module,
    student: nn.Module,
    calibration_data: Union[torch.utils.data.DataLoader, List[Any], torch.Tensor],
    damping: float = 1e-4,
    device: Optional[Union[str, torch.device]] = None,
    max_batches: int = 4,
) -> None:
    """
    Executes closed-form Ridge projection calibration on output projection layers.

    Solves W_O* = (H_s^T H_s + lambda I)^(-1) H_s^T Y_t in <1s without autograd.
    Eliminates post-transplant representation shock by aligning student output projections
    directly to teacher activations.
    """
    logger.info("Executing closed-form Ridge alignment on student output projections...")
    teacher.eval()
    student.eval()

    if device is None:
        try:
            device = next(student.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    if isinstance(calibration_data, torch.Tensor):
        batches = [calibration_data]
    else:
        batches = calibration_data

    t_acts: List[torch.Tensor] = []
    s_acts: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in batches:
            if isinstance(batch, dict):
                b_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                t_out = teacher(**b_in)
                s_out = student(**b_in)
            elif isinstance(batch, (tuple, list)):
                b_in = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
                t_out = teacher(*b_in)
                s_out = student(*b_in)
            elif isinstance(batch, torch.Tensor):
                t_out = teacher(batch.to(device))
                s_out = student(batch.to(device))
            else:
                break

            def _extract_rep(out: Any) -> torch.Tensor:
                if hasattr(out, "logits") and out.logits is not None:
                    return out.logits
                if isinstance(out, torch.Tensor):
                    return out
                if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                    return out[0]
                if isinstance(out, dict) and "logits" in out:
                    return out["logits"]
                raise ValueError(f"Cannot extract representation from output of type {type(out).__name__}")

            t_rep = _extract_rep(t_out)
            s_rep = _extract_rep(s_out)

            t_acts.append(t_rep.detach())
            s_acts.append(s_rep.detach())
            if len(t_acts) >= max_batches:
                break

    if not t_acts or not s_acts:
        logger.warning("No activations captured for Ridge alignment.")
        return

    x = torch.cat(s_acts, dim=0)
    y = torch.cat(t_acts, dim=0)

    # Locate output projection layer in student
    target_proj: Optional[nn.Linear] = None
    for name, mod in student.named_modules():
        if any(sub in name for sub in ["out_proj", "o_proj", "dense", "classifier"]) and isinstance(mod, nn.Linear):
            target_proj = mod
            break

    if target_proj is not None and x.shape[-1] == target_proj.in_features:
        w_ridge, b_ridge = compute_ridge_calibration(
            x.reshape(-1, target_proj.in_features),
            y.reshape(-1, target_proj.out_features),
            ridge_lambda=damping,
            device=device,
        )
        with torch.no_grad():
            target_proj.weight.data.copy_(w_ridge.t())
            if target_proj.bias is not None and b_ridge is not None:
                target_proj.bias.data.copy_(b_ridge)
        logger.info(
            "Successfully updated output projection layer %s via Ridge calibration.", type(target_proj).__name__
        )
    else:
        logger.warning(
            "Could not match target projection layer for alignment (shape x: %s, target_proj: %s)",
            x.shape,
            target_proj,
        )


__all__ = [
    "UniversalActivationCalibrator",
    "compress_model_universal",
    "transplant_and_compress",
    "GradientCalibrator",
    "calibrate_gradient_importance",
    "calibrate_surgically_replaced_model",
    "calibrate_attention_subspace",
    "ridge_regression_solve",
    "compute_ridge_calibration",
    "align_output_projections",
]
