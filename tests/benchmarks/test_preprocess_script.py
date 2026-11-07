#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for custom benchmark preprocessing scripts and contract enforcement (F-91)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from neuralops.benchmarks.core.data_loader import load_preprocess_script, make_dataloader


def test_preprocess_script_runs_or_flag_absent(tmp_path: Path) -> None:
    """A valid preprocessing script modifying batches is executed and alters dataloader output."""
    script_path = tmp_path / "valid_preprocess.py"
    script_code = """import torch

def preprocess(batch):
    if isinstance(batch, (list, tuple)):
        # Add a custom tensor flag or modify inputs
        return list(batch) + [torch.tensor([999.0])]
    elif isinstance(batch, dict):
        batch["custom_tag"] = "transformed_by_script"
        return batch
    return batch
"""
    script_path.write_text(script_code, encoding="utf-8")

    preprocess_fn = load_preprocess_script(script_path)
    assert callable(preprocess_fn)
    assert hasattr(preprocess_fn, "__preprocess_script_sha256__")
    assert hasattr(preprocess_fn, "__preprocess_script_path__")

    # Create dummy TensorDataset and dataloader
    tensors = torch.randn(8, 4)
    dataset = TensorDataset(tensors)
    loader = make_dataloader(dataset, batch_size=2, shuffle=False, preprocess_fn=preprocess_fn)

    batch = next(iter(loader))
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert torch.equal(batch[1], torch.tensor([999.0]))


def test_preprocess_script_missing_function_fails(tmp_path: Path) -> None:
    """A preprocessing script without a callable 'preprocess(batch)' fails with AttributeError."""
    script_path = tmp_path / "missing_fn.py"
    script_code = """
def invalid_name_function(batch):
    return batch
"""
    script_path.write_text(script_code, encoding="utf-8")

    with pytest.raises(AttributeError, match="does not expose a callable named 'preprocess"):
        load_preprocess_script(script_path)


def test_preprocess_script_nonexistent_fails() -> None:
    """Passing a nonexistent preprocessing path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Preprocessing script not found"):
        load_preprocess_script("non_existent_preprocessing_script_file.py")
