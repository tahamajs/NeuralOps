#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for secure export loading and fail-closed weight verification (F-90)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from neuralops.export.exporter import load_exported_model


class TinyTestModel(nn.Module):
    """Simple multi-layer module for weight loading tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def test_malicious_base_model_refused(tmp_path: Path) -> None:
    """A remote base_model without explicit consent raises ValueError and downloads nothing."""
    config_data = {
        "base_model": "untrusted-org/malicious-checkpoint",
        "format": "safetensors",
        "_neuralops_metadata": [],
    }
    with open(tmp_path / "neuralops_config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    # Put a valid dummy safetensors file so weight loading isn't the failure reason
    save_file({"dummy": torch.zeros(1)}, str(tmp_path / "model.safetensors"))

    with patch("transformers.AutoModel.from_pretrained") as mock_from_pretrained:
        with pytest.raises(ValueError, match="Refusing to download remote base model"):
            load_exported_model(tmp_path, allow_remote_base_model=False)

        # Assert no network call or from_pretrained occurred
        mock_from_pretrained.assert_not_called()


def test_weight_mismatch_fails(tmp_path: Path) -> None:
    """A weights file with missing keys must raise RuntimeError in fail-closed mode."""
    model = TinyTestModel()
    config_data = {
        "base_model": "local_toy",
        "format": "safetensors",
        "_neuralops_metadata": [],
    }
    with open(tmp_path / "neuralops_config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    # Save state dict with only fc1 keys, omitting fc2 entirely
    incomplete_state = {
        "fc1.weight": model.fc1.weight.detach(),
        "fc1.bias": model.fc1.bias.detach(),
    }
    save_file(incomplete_state, str(tmp_path / "model.safetensors"))

    with pytest.raises(RuntimeError, match=r"(State dict mismatch|Missing key)"):
        load_exported_model(tmp_path, base_model=TinyTestModel(), strict=True)


def test_safetensors_failure_fails(tmp_path: Path) -> None:
    """A corrupt safetensors file must raise RuntimeError rather than continuing silently."""
    config_data = {
        "base_model": "local_toy",
        "format": "safetensors",
        "_neuralops_metadata": [],
    }
    with open(tmp_path / "neuralops_config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    # Write invalid binary data as safetensors
    with open(tmp_path / "model.safetensors", "wb") as f:
        f.write(b"NOT_A_VALID_SAFETENSORS_HEADER_OR_FILE")

    with pytest.raises(RuntimeError, match="Failed to load weights from"):
        load_exported_model(tmp_path, base_model=TinyTestModel())
