#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess and mathematical unit tests for Example 29: DoRA Healing & Directional Adaptation.
"""

from __future__ import annotations

import subprocess
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.trainer.regimes.dora_healing import DoRAHealingHealer, DoRAProjectionWrapper


def test_29_runs_clean():
    """Verify example 29 executes cleanly end-to-end via subprocess."""
    proc = subprocess.run(
        [sys.executable, "examples/29_dora_healing_directional_adaptation.py"],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, f"Example 29 failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[PASS] Overall Example 29 Status" in proc.stdout


def test_29_dora_decomposition_and_gradient_flow():
    """Verify mathematical invariant of DoRA: W = m * (W_0 + BA) / ||W_0 + BA||_2."""
    torch.manual_seed(42)
    in_features, out_features = 16, 32
    rank = 4
    alpha = 8.0

    base_layer = nn.Linear(in_features, out_features, bias=True)
    wrapper = DoRAProjectionWrapper(base_layer, rank=rank, lora_alpha=alpha)

    # Initial state check: B is zero-initialized, so Delta W is 0
    # W_eff should match base weight normalized * magnitude
    # Since m = ||W_0||_2 initially, W_eff == W_0
    w_eff = wrapper.compute_effective_weight()
    assert torch.allclose(w_eff, base_layer.weight, atol=1e-6)

    # Verify forward computation matching
    x = torch.randn(4, in_features)
    out_wrapper = wrapper(x)
    out_base = base_layer(x)
    assert torch.allclose(out_wrapper, out_base, atol=1e-6)

    # Backward step: verify gradients populate to magnitude, lora_A, lora_B
    loss = out_wrapper.sum()
    loss.backward()

    assert wrapper.magnitude.grad is not None
    assert torch.isfinite(wrapper.magnitude.grad).all()
    assert wrapper.lora_A.grad is not None
    assert torch.isfinite(wrapper.lora_A.grad).all()
    assert wrapper.lora_B.grad is not None
    assert torch.isfinite(wrapper.lora_B.grad).all()

    # Verify merge and collapse
    merged = wrapper.merge_and_collapse()
    assert isinstance(merged, nn.Linear)
    with torch.no_grad():
        out_merged = merged(x)
    assert torch.allclose(out_merged, out_wrapper, atol=1e-6)
