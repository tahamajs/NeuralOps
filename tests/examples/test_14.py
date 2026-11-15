#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration test for Example 14 (Muon & Riemannian Stiefel SGD) and verification of Example 13 deletion.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from neuralops.trainer.optimizers.muon import zeropower_via_newtonschulz5
from neuralops.trainer.optimizers.stiefel import project_stiefel_gradient


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_13_is_removed() -> None:
    """Asserts examples/13_six_pillar_knowledge_distillation.py does not exist on disk."""
    example_13 = REPO_ROOT / "examples" / "13_six_pillar_knowledge_distillation.py"
    assert not example_13.exists(), f"Example 13 must be deleted, but found at {example_13}"


def test_14_runs_clean() -> None:
    """Asserts examples/14_advanced_optimizers_muon_stiefel.py executes cleanly with exit code 0."""
    example_14 = REPO_ROOT / "examples" / "14_advanced_optimizers_muon_stiefel.py"
    proc = subprocess.run(
        [sys.executable, str(example_14)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"Example 14 failed with returncode {proc.returncode}:\n{proc.stderr}"
    assert "[PASS] Overall Example 14 Status" in proc.stdout


def test_14_muon_orthogonality() -> None:
    """Verifies Newton-Schulz iteration reduces condition number toward 1.0."""
    torch.manual_seed(42)
    G = torch.randn(16, 16)
    s_orig = torch.linalg.svdvals(G)
    cond_orig = (s_orig[0] / s_orig[-1]).item()

    M = zeropower_via_newtonschulz5(G, steps=5)
    s_ns = torch.linalg.svdvals(M)
    cond_ns = (s_ns[0] / s_ns[-1]).item()

    assert cond_ns < 2.5, f"Condition number after NS {cond_ns:.2f} is not < 2.5 (was {cond_orig:.2f})"


def test_14_stiefel_tangent_projection() -> None:
    """Verifies Stiefel tangent projection satisfies skew-symmetry U^T grad_R + grad_R^T U = 0."""
    torch.manual_seed(42)
    n, p = 32, 16
    q_init, _ = torch.linalg.qr(torch.randn(n, p), mode="reduced")
    grad_e = torch.randn(n, p)
    grad_r = project_stiefel_gradient(q_init, grad_e)

    skew = q_init.T @ grad_r + grad_r.T @ q_init
    skew_norm = torch.linalg.norm(skew).item()
    assert skew_norm < 1e-4, f"Stiefel tangent projection skew norm {skew_norm} is not < 1e-4"
