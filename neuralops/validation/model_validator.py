#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Model Architecture Validator

Validates model architecture construction, layer integrity, and forward pass execution.
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates PyTorch model forward and backward passes and parameter sanity."""

    def __init__(
        self,
        model: nn.Module,
        input_shape: Sequence[int],
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.model = model
        self.input_shape = tuple(input_shape)
        self.device = str(device)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """Run validation checks on model."""
        logger.info("Validating model on %s with input shape %s", self.device, self.input_shape)
        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            self.errors.append(f"Move to device failed: {e}")
            logger.error("Move to device failed: %s", e)
            return False

        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Total params: %.2fM, Trainable: %.2fM", n_params / 1e6, n_trainable / 1e6)

        dummy: Any = None
        try:
            self.model.eval()
            try:
                dummy = torch.randint(0, 1000, self.input_shape, device=self.device)
                with torch.no_grad():
                    out = self.model(dummy)
            except (RuntimeError, TypeError):
                dummy = torch.randn(self.input_shape, device=self.device)
                with torch.no_grad():
                    out = self.model(dummy)

            logits = getattr(out, "logits", out)
            if isinstance(logits, (tuple, list)) and len(logits) > 0:
                logits = logits[0]

            if isinstance(logits, torch.Tensor):
                if torch.isnan(logits).any():
                    self.errors.append("NaN in forward output!")
                if torch.isinf(logits).any():
                    self.errors.append("Inf in forward output!")
        except Exception as e:
            self.errors.append(f"Forward failed: {e}")
            return False

        try:
            self.model.train()
            loss = None
            if dummy is None:
                dummy = torch.randn(self.input_shape, device=self.device)

            # 1. Try forward with labels kwarg
            try:
                labels = torch.zeros(self.input_shape[0], dtype=torch.long, device=self.device)
                out = self.model(dummy, labels=labels)
                if hasattr(out, "loss") and out.loss is not None:
                    loss = out.loss
            except Exception:
                pass

            if loss is None:
                try:
                    labels = torch.zeros(
                        self.input_shape[0],
                        self.input_shape[1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    out = self.model(dummy, labels=labels)
                    if hasattr(out, "loss") and out.loss is not None:
                        loss = out.loss
                except Exception:
                    pass

            # 2. Try forward without labels kwarg
            if loss is None:
                out = self.model(dummy)
                if hasattr(out, "loss") and out.loss is not None:
                    loss = out.loss
                elif hasattr(out, "logits"):
                    loss = out.logits.sum()
                elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                    loss = out[0].sum()
                elif isinstance(out, torch.Tensor):
                    loss = out.sum()

            if loss is not None and isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                self.model.zero_grad()
            elif loss is not None and isinstance(loss, torch.Tensor) and not loss.requires_grad:
                if n_trainable > 0:
                    self.errors.append("Backward failed: loss does not require grad despite trainable parameters")
            else:
                self.errors.append("Backward failed: could not compute scalar loss from model output")
        except Exception as e:
            self.errors.append(f"Backward failed: {e}")

        n_nan_grads = sum(
            1 for p in self.model.parameters() if p.grad is not None and torch.isnan(p.grad).any()
        )
        if n_nan_grads > 0:
            self.errors.append(f"{n_nan_grads} params with NaN grads")

        logger.info("Validation complete. Errors: %d", len(self.errors))
        return len(self.errors) == 0


__all__ = ["ModelValidator"]
