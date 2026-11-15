#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess and mathematical property tests for examples/19_spectral_and_hessian_diagnostics.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import torch
import torch.nn as nn

# Dynamic import helper for numeric filename
example_path = Path(__file__).resolve().parent.parent.parent / "examples" / "19_spectral_and_hessian_diagnostics.py"
spec = importlib.util.spec_from_file_location("example_19", example_path)
example_19 = importlib.util.module_from_spec(spec)
sys.modules["example_19"] = example_19
spec.loader.exec_module(example_19)

from examples._shared import TinyClassifier, tiny_inputs


def test_19_runs_clean():
    """Verify that example 19 executes cleanly via subprocess and exits 0."""
    proc = subprocess.run(
        [sys.executable, "examples/19_spectral_and_hessian_diagnostics.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 19 failed with stderr: {proc.stderr}"
    assert "[PASS] Overall Example 19 Status" in proc.stdout


def test_19_hvp_mathematical_property():
    """Verify HVP linearity: HVP(alpha * v) == alpha * HVP(v)."""
    torch.manual_seed(42)
    model = TinyClassifier(seed=42)
    model.train()
    inputs = tiny_inputs(seed=42)

    outputs = model(**inputs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    targets = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(logits, targets)

    target_param = model.transformer.classifier.weight
    v = torch.randint(0, 2, target_param.shape).float() * 2.0 - 1.0
    v = v / (v.norm() + 1e-12)

    compute_hvp = example_19.compute_hvp
    hvp_1 = compute_hvp(loss, target_param, v)

    alpha = 2.5
    hvp_alpha = compute_hvp(loss, target_param, alpha * v)

    assert torch.allclose(hvp_alpha, alpha * hvp_1, atol=1e-5), "HVP linearity property violated"
