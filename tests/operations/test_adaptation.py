#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for adaptation operations.

Tests for:
- adaptation.lora (LoRA adapter injection)
- adaptation.heal (DoRA directional adaptation and healing)
- adaptation.distill (Multi-level knowledge distillation)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.core.peft.lora import LoRALinear
from neuralops.operations.adaptation.distill import AdaptationDistillOperation
from neuralops.operations.adaptation.heal import AdaptationHealOperation
from neuralops.operations.adaptation.lora import AdaptationLoraOperation
from neuralops.operations.errors import OperationExecutionError
from neuralops.trainer.regimes.dora_healing import DoRAProjectionWrapper


class MultiLayerMLP(nn.Module):
    """Synthetic multi-layer MLP for unit tests."""

    def __init__(self, in_features: int = 16, hidden_features: int = 32, out_features: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.relu = nn.ReLU()
        self.q_proj = nn.Linear(hidden_features, hidden_features)
        self.v_proj = nn.Linear(hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.ln1(self.fc1(x)))
        attn_out = self.q_proj(h) + self.v_proj(h)
        return self.fc2(attn_out)


# =========================================================================
# Tests for adaptation.lora
# =========================================================================


def test_lora_operation_all_linear_layers() -> None:
    """Test LoRA injection into all linear layers when no targets specified."""
    model = MultiLayerMLP()
    op = AdaptationLoraOperation()
    assert op.name == "adaptation.lora"

    config = {"rank": 4, "alpha": 8.0}
    adapted_model, evidence = op.apply(model, config)

    assert evidence["operation"] == "adaptation.lora"
    assert evidence["rank"] == 4
    assert evidence["alpha"] == 8.0
    assert evidence["adapted_count"] == 4  # fc1, q_proj, v_proj, fc2
    assert isinstance(adapted_model.fc1, LoRALinear)
    assert isinstance(adapted_model.fc2, LoRALinear)
    assert isinstance(adapted_model.q_proj, LoRALinear)
    assert isinstance(adapted_model.v_proj, LoRALinear)

    # Check forward pass
    x = torch.randn(2, 16)
    out = adapted_model(x)
    assert out.shape == (2, 8)


def test_lora_operation_selective_targets() -> None:
    """Test LoRA injection selectively filtering target layers."""
    model = MultiLayerMLP()
    op = AdaptationLoraOperation()

    config = {"rank": 2, "alpha": 4.0, "targets": ["q_proj", "v_proj"]}
    adapted_model, evidence = op.apply(model, config)

    assert evidence["adapted_count"] == 2
    assert isinstance(adapted_model.q_proj, LoRALinear)
    assert isinstance(adapted_model.v_proj, LoRALinear)
    assert not isinstance(adapted_model.fc1, LoRALinear)
    assert not isinstance(adapted_model.fc2, LoRALinear)

    x = torch.randn(2, 16)
    out = adapted_model(x)
    assert out.shape == (2, 8)


def test_lora_operation_invalid_config() -> None:
    """Test LoRA validation error on invalid rank or alpha."""
    model = MultiLayerMLP()
    op = AdaptationLoraOperation()

    with pytest.raises(OperationExecutionError):
        op.apply(model, {"rank": 0, "alpha": 16.0})

    with pytest.raises(OperationExecutionError):
        op.apply(model, {"rank": 4, "alpha": 0.5})


# =========================================================================
# Tests for adaptation.heal
# =========================================================================


def test_heal_operation_dora_injection_and_collapse() -> None:
    """Test DoRA healing injection followed by weight collapse."""
    model = MultiLayerMLP()
    op = AdaptationHealOperation()
    assert op.name == "adaptation.heal"

    config = {
        "calibration_tokens": 1024,
        "mode": "full_quick",
        "rank": 4,
        "lora_alpha": 8.0,
        "targets": ["q_proj", "v_proj"],
        "collapse_after_heal": True,
    }

    healed_model, evidence = op.apply(model, config)
    assert evidence["operation"] == "adaptation.heal"
    assert evidence["injected_dora_adapters"] == 2
    assert evidence["collapsed"] is True
    # After collapse, modules must be standard nn.Linear layers
    assert isinstance(healed_model.q_proj, nn.Linear)
    assert not isinstance(healed_model.q_proj, DoRAProjectionWrapper)

    x = torch.randn(2, 16)
    out = healed_model(x)
    assert out.shape == (2, 8)


def test_heal_operation_without_collapse() -> None:
    """Test DoRA healing leaving adapters attached when collapse_after_heal=False."""
    model = MultiLayerMLP()
    op = AdaptationHealOperation()

    config = {
        "calibration_tokens": 1024,
        "mode": "full_quick",
        "rank": 4,
        "lora_alpha": 8.0,
        "targets": ["q_proj"],
        "collapse_after_heal": False,
    }

    healed_model, evidence = op.apply(model, config)
    assert evidence["collapsed"] is False
    assert isinstance(healed_model.q_proj, DoRAProjectionWrapper)
    assert isinstance(healed_model.v_proj, nn.Linear)

    x = torch.randn(2, 16)
    out = healed_model(x)
    assert out.shape == (2, 8)


def test_heal_operation_with_calibration_dataloader() -> None:
    """Test DoRA healing running calibration optimization steps."""
    model = MultiLayerMLP()
    op = AdaptationHealOperation()

    data = [torch.randn(4, 16) for _ in range(5)]
    config = {
        "calibration_tokens": 1024,
        "mode": "full_quick",
        "rank": 2,
        "lora_alpha": 4.0,
        "targets": ["q_proj", "v_proj"],
        "dataloader": data,
        "steps": 3,
        "lr": 1e-3,
        "collapse_after_heal": True,
    }

    healed_model, evidence = op.apply(model, config)
    assert evidence["steps_run"] == 3
    assert len(evidence["losses"]) == 3
    for loss in evidence["losses"]:
        assert isinstance(loss, float)


def test_heal_operation_modes() -> None:
    """Test different healing modes: norm_only and bias_only."""
    model_norm = MultiLayerMLP()
    op = AdaptationHealOperation()

    _, ev_norm = op.apply(
        model_norm,
        {
            "calibration_tokens": 1024,
            "mode": "norm_only",
            "collapse_after_heal": True,
        },
    )
    assert ev_norm["mode"] == "norm_only"

    model_bias = MultiLayerMLP()
    _, ev_bias = op.apply(
        model_bias,
        {
            "calibration_tokens": 1024,
            "mode": "bias_only",
            "collapse_after_heal": True,
        },
    )
    assert ev_bias["mode"] == "bias_only"


# =========================================================================
# Tests for adaptation.distill
# =========================================================================


def test_distill_operation_success() -> None:
    """Test multi-level knowledge distillation operation execution."""
    teacher = MultiLayerMLP()
    student = MultiLayerMLP()
    op = AdaptationDistillOperation()
    assert op.name == "adaptation.distill"

    batches = [torch.randn(4, 16) for _ in range(3)]
    config = {
        "teacher": teacher,
        "steps": 2,
        "learning_rate": 1e-3,
        "temperature": 2.0,
        "train_loader": batches,
        "pillars": ["logits", "mse"],
    }

    distilled_model, evidence = op.apply(student, config)
    assert evidence["operation"] == "adaptation.distill"
    assert evidence["steps"] == 2
    assert evidence["distill_metrics"]["status"] == "success"
    assert "final_loss" in evidence["distill_metrics"]

    x = torch.randn(2, 16)
    out = distilled_model(x)
    assert out.shape == (2, 8)


def test_distill_operation_teacher_fallback() -> None:
    """Test distill operation synthetic data fallback when no loader provided."""
    teacher = nn.Sequential(nn.Linear(16, 8))
    student = nn.Sequential(nn.Linear(16, 8))
    op = AdaptationDistillOperation()

    config = {
        "teacher": teacher,
        "steps": 2,
        "learning_rate": 1e-3,
    }

    distilled_model, evidence = op.apply(student, config)
    assert evidence["distill_metrics"]["status"] == "success"


def test_distill_operation_missing_teacher() -> None:
    """Test distill operation failure when teacher is omitted."""
    student = MultiLayerMLP()
    op = AdaptationDistillOperation()

    with pytest.raises(OperationExecutionError):
        op.apply(student, {"steps": 5})
