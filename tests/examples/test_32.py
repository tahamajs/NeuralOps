#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for examples/32_simpo_orpo_preference_alignment.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM
from neuralops.trainer.rl_alignment.preference_and_onpolicy import ORPOLoss, SimPOLoss

EXAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "32_simpo_orpo_preference_alignment.py"


def test_32_runs_clean_cpu():
    """Verify Example 32 runs cleanly via subprocess with exit code 0."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--device", "cpu", "--steps", "3"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 32 failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[PASS] Overall Example 32 Status" in proc.stdout
    assert "SimPO Optimization Convergence" in proc.stdout
    assert "Preference Reward Margin Separation" in proc.stdout


def test_32_simpo_loss_mathematical_formulation():
    """Verify SimPO loss: L = -log sigmoid(beta/|yw| * log_pi(yw) - beta/|yl| * log_pi(yl) - gamma)."""
    torch.manual_seed(42)
    beta = 2.0
    gamma = 0.5
    criterion = SimPOLoss(beta=beta, gamma=gamma)

    # Synthetic log-probabilities: chosen has higher per-token logprob than rejected
    batch_size = 2
    seq_len = 8
    chosen_logp = torch.full((batch_size, seq_len), -0.5)
    rejected_logp = torch.full((batch_size, seq_len), -2.0)

    loss = criterion(chosen_logp, rejected_logp)

    # Manual expected calculation:
    # r_w = (-0.5 * 8 / 8) * 2.0 = -1.0
    # r_l = (-2.0 * 8 / 8) * 2.0 = -4.0
    # logits = r_w - r_l - gamma = -1.0 - (-4.0) - 0.5 = 2.5
    # expected_loss = -log sigmoid(2.5)
    expected_loss = -torch.log(torch.sigmoid(torch.tensor(2.5))).item()

    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-5), (
        f"SimPO loss mismatch: {loss.item()} vs {expected_loss}"
    )


def test_32_orpo_loss_positive_odds_penalty():
    """Verify ORPO loss returns finite positive loss penalizing rejected log probabilities."""
    criterion = ORPOLoss(lambda_orpo=0.2, sft_weight=1.0)
    sft_loss = torch.tensor(1.5)
    chosen_logp = torch.tensor([[-0.2, -0.3, -0.1]])
    rejected_logp = torch.tensor([[-1.5, -2.0, -1.8]])

    loss = criterion(sft_loss=sft_loss, chosen_logprobs=chosen_logp, rejected_logprobs=rejected_logp)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
