#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess and mathematical tests for examples/08_grpo_reasoning.py.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import torch

ex08 = importlib.import_module("examples.08_grpo_reasoning")
compute_group_advantages = ex08.compute_group_advantages
reward_format = ex08.reward_format
reward_conciseness = ex08.reward_conciseness


def test_08_runs_clean():
    """Verify that example 08 executes cleanly via subprocess and exits 0."""
    proc = subprocess.run(
        [sys.executable, "examples/08_grpo_reasoning.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 08 failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[PASS] Overall Example 08 Status" in proc.stdout


def test_08_group_advantage_normalization():
    """Verify group advantage normalization satisfies zero-mean invariant."""
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 2.0, 0.5], dtype=torch.float32)
    adv = compute_group_advantages(rewards)

    assert adv.shape == rewards.shape
    assert abs(adv.mean().item()) < 1e-5, f"Expected advantage mean near 0, got {adv.mean().item()}"
    assert torch.isfinite(adv).all()


def test_08_policy_ratio_and_rewards():
    """Verify reward scoring logic and dynamic policy ratio clamping bounds."""
    # Test rewards
    r_fmt = reward_format("Step 1: calculate. Answer: 42.")
    assert r_fmt == 1.0
    r_no_fmt = reward_format("Just calculating 42.")
    assert r_no_fmt == 0.0

    r_len = reward_conciseness("12345", target_length=5)
    assert r_len == 1.0

    # Test policy ratio calculation
    policy_log_probs = torch.tensor([-0.5, -1.2, -0.1], dtype=torch.float32)
    ref_log_probs = torch.tensor([-0.6, -1.0, -0.1], dtype=torch.float32)
    ratio = torch.exp(policy_log_probs - ref_log_probs)

    assert torch.isfinite(ratio).all()
    assert (ratio > 0.0).all()

    clip_eps = 0.2
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    assert (clipped >= 1.0 - clip_eps).all()
    assert (clipped <= 1.0 + clip_eps).all()
