#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
End-to-End Training Validator.

Executes end-to-end verification runs on synthetic datasets to ensure pipeline correctness.
Returns structured ValidationStatus without stdout printing (F-57, F-97).
"""

from __future__ import annotations

from typing import Any, Dict

import torch


class ValidationStatus(dict):
    """Structured validation status dictionary with property and bool access."""

    @property
    def status(self) -> str:
        return self.get("status", "PASSED")

    def __bool__(self) -> bool:
        return bool(self.get("passed", False))


class TrainingValidator:
    """End-to-end training and overfitting validator returning structured status."""

    @staticmethod
    def overfit_test(
        model: torch.nn.Module,
        batch: Dict[str, Any],
        n_samples: int = 8,
        steps: int = 50,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> ValidationStatus:
        small_batch = {
            k: v[:n_samples].to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        losses = []
        model.train()
        for _ in range(steps):
            out = model(**small_batch)
            loss = out.loss if hasattr(out, "loss") else out[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        initial_loss = losses[0] if losses else 0.0
        final_loss = losses[-1] if losses else 0.0
        passed = bool(final_loss < initial_loss * 0.5) if losses else False

        return ValidationStatus({
            "status": "passed" if passed else "failed",
            "passed": passed,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "reduction_ratio": (final_loss / initial_loss) if (initial_loss != 0) else 1.0,
            "steps": steps,
            "n_samples": n_samples,
        })


__all__ = ["TrainingValidator", "ValidationStatus"]
