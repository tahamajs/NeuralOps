#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Test suite for Example 09: Multi-Stage Model Surgery Pipeline.
"""

from __future__ import annotations

import subprocess
import sys
import pytest
import torch

from examples._shared import TinyClassifier, tiny_inputs
from neuralops.surgery import ModelSurgery

pytestmark = pytest.mark.filterwarnings("ignore:.*experimental.*:UserWarning")


def test_09_runs_clean():
    """Verify example 09 executes cleanly via subprocess and exits 0."""
    proc = subprocess.run(
        [sys.executable, "examples/09_multi_stage_model_surgery.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 09 failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[PASS] Overall Example 09 Status" in proc.stdout
    assert "Multi-stage surgery pipeline executed and verified across all stages" in proc.stdout


def test_09_stages_reduce_parameters():
    """Verify multi-stage surgery preserves output shape and reduces parameter counts monotonically."""
    model = TinyClassifier().to("cpu")
    surgery = ModelSurgery(model)
    inputs = tiny_inputs()
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        out0 = surgery.model(**inputs)
        expected_shape = tuple((out0.logits if hasattr(out0, "logits") else out0).shape)

    # Stage 1: Attention replacement
    surgery.apply("attention.replace", target="cosformer")
    p1 = sum(p.numel() for p in surgery.model.parameters())
    with torch.no_grad():
        out1 = surgery.model(**inputs)
        logits1 = out1.logits if hasattr(out1, "logits") else out1
        assert tuple(logits1.shape) == expected_shape
        assert torch.isfinite(logits1).all().item()

    # Stage 2: MLP intermediate dimension reduction
    surgery.apply("architecture.reduce_mlp", ratio=0.8)
    p2 = sum(p.numel() for p in surgery.model.parameters())
    with torch.no_grad():
        out2 = surgery.model(**inputs)
        logits2 = out2.logits if hasattr(out2, "logits") else out2
        assert tuple(logits2.shape) == expected_shape
        assert torch.isfinite(logits2).all().item()

    # Stage 3: Low-rank SVD compression
    surgery.apply("weights.compress", method="svd", target="*ffn.lin*", target_ratio=0.7)
    p3 = sum(p.numel() for p in surgery.model.parameters())
    with torch.no_grad():
        out3 = surgery.model(**inputs)
        logits3 = out3.logits if hasattr(out3, "logits") else out3
        assert tuple(logits3.shape) == expected_shape
        assert torch.isfinite(logits3).all().item()

    # Assert monotonic reduction across stages 1 -> 2 -> 3
    assert p2 < p1, f"Stage 2 MLP reduction failed to reduce parameters: p2={p2} >= p1={p1}"
    assert p3 < p2, f"Stage 3 SVD compression failed to reduce parameters: p3={p3} >= p2={p2}"
