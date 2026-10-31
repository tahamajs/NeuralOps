#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for HuggingFace export and runtime compatibility verification (Sub-Agent E1, E2, E3).
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from neuralops.export.hf_compat import export_to_hf
from neuralops.export.runtime_compat import check_compatibility


class ToyLinearModel(nn.Module):
    def __init__(self, in_features: int = 16, out_features: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x=None, input_ids=None, **kwargs):
        inp = x if x is not None else input_ids
        return self.linear(inp)


def test_e1_export_to_hf_round_trip():
    """Verify export_to_hf saves model, reloads it, and verifies cosine similarity > 0.9999."""
    model = ToyLinearModel(in_features=16, out_features=4)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        res = export_to_hf(model, tmpdir)

        assert res.verified is True
        assert res.cosine_similarity is not None
        assert res.cosine_similarity > 0.9999
        assert (Path(tmpdir) / "config.json").exists()


def test_e2_check_compatibility_on_mac():
    """Verify check_compatibility accurately identifies HF as SUCCESS and vLLM as NOT_INSTALLED on macOS."""
    model = ToyLinearModel(in_features=16, out_features=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        res = export_to_hf(model, tmpdir)
        assert res.verified is True

        report = check_compatibility(tmpdir)
        summary = report.summary

        # HuggingFace config should be recognized
        assert summary.get("hf") == "SUCCESS"

        # On macOS development host without vLLM installed
        import importlib.util
        if importlib.util.find_spec("vllm") is None:
            assert summary.get("vllm") == "NOT_INSTALLED"
        elif not torch.cuda.is_available():
            assert summary.get("vllm") == "SKIPPED"
        else:
            assert summary.get("vllm") in ["SUCCESS", "FAILURE"]
