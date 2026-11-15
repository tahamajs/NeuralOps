#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess and mathematical unit tests for Example 31: Fisher-Weighted Curvature Natural Gradient Damping.
"""

from __future__ import annotations

import subprocess
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.trainer.regimes.fisher_micro import (
    FisherMicroHealer,
    FMicroOptimizer,
    compute_empirical_diagonal_fisher,
)


def test_31_runs_clean():
    """Verify example 31 executes cleanly end-to-end via subprocess."""
    proc = subprocess.run(
        [sys.executable, "examples/31_fisher_damping_natural_gradient.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"Example 31 failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[PASS] Overall Example 31 Status" in proc.stdout


def test_31_fisher_computation_and_optimizer_step():
    """Verify positive semi-definiteness of empirical Fisher and finite update steps."""
    torch.manual_seed(42)

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 2)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    model = SimpleNet()
    batches = [torch.randn(4, 8) for _ in range(5)]

    fisher_diag = compute_empirical_diagonal_fisher(
        model=model,
        dataloader=batches,
        num_samples=20,
    )

    assert len(fisher_diag) == 4, f"Expected 4 parameter tensors, got {len(fisher_diag)}"
    for name, f_tensor in fisher_diag.items():
        assert (f_tensor >= 0.0).all(), f"Fisher diagonal elements must be non-negative for {name}"
        assert torch.isfinite(f_tensor).all(), f"Fisher diagonal elements must be finite for {name}"

    # Verify FMicroOptimizer step
    optimizer = FMicroOptimizer(
        model.parameters(),
        fisher_diag=fisher_diag,
        lr=0.01,
        damping=1e-2,
    )

    model.train()
    optimizer.zero_grad()
    out = model(batches[0])
    loss = out.sum()
    loss.backward()

    # Pre-step weights
    w_before = model.fc1.weight.clone()
    optimizer.step()
    w_after = model.fc1.weight.clone()

    assert not torch.equal(w_before, w_after), "Weights should update after optimizer step"
    assert torch.isfinite(w_after).all(), "Updated weights should be finite"
