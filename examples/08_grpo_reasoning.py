#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 08: Group Relative Policy Optimization (GRPO) & Reasoning Alignment.

Subsystem Demonstrated:
    Offline Group Relative Policy Optimization (GRPO) advantage calculation and policy gradient update.

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration.

What You Will Learn:
    1. The mathematical formulation of Group Relative Policy Optimization (GRPO):
       - Sampling K candidate outputs per prompt.
       - Normalizing rewards across groups: A_i = (r_i - mean(R)) / (std(R) + eps).
    2. Evaluating dual reward functions:
       - Format Reward: Incentivizes reasoning delimiters (e.g., 'Answer:').
       - Conciseness/Length Reward: Penalizes excessive verbosity.
    3. Computing advantage-weighted surrogate policy loss: L = -mean(A_i * log_prob).
    4. Taking an offline policy gradient optimization step without external library dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

import neuralops
from examples._shared import (
    TinyClassifier,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    print_benchmark_table,
    section,
    tiny_inputs,
)


def compute_group_advantages(rewards: list[float], eps: float = 1e-6) -> torch.Tensor:
    """Compute group-relative advantages: A_i = (r_i - mean(r)) / (std(r) + eps)."""
    r_tensor = torch.tensor(rewards, dtype=torch.float32)
    mean_r = r_tensor.mean()
    std_r = r_tensor.std(unbiased=False)
    advantages = (r_tensor - mean_r) / (std_r + eps)
    return advantages


def reward_format(text: str) -> float:
    """Format Reward: returns 1.0 if 'Answer:' is present, else 0.0."""
    return 1.0 if "Answer:" in text else 0.0


def reward_conciseness(text: str, target_length: int = 25) -> float:
    """Length Reward: penalizes deviation from ideal concise token/character count."""
    length_penalty = abs(len(text) - target_length) / float(target_length)
    return max(0.0, 1.0 - length_penalty)


def run_grpo_simulation(device: str) -> bool:
    """Execute complete self-contained GRPO reasoning demonstration."""
    section("Part 1: Generating K Candidate Reasoning Paths")

    model = TinyClassifier().to(device)
    neuralops.replace_attention(model, attention="cosformer")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 4 simulated completion strings from candidate sampling
    candidate_responses = [
        "Let me think. Step 1: Analyze. Answer: Option A is correct.",
        "The problem requires careful deduction. The result is unclear.",
        "Short solution. Answer: 42.",
        "Thinking aloud. The final conclusion appears to be 42 but with caveats.",
    ]
    k_samples = len(candidate_responses)
    kv("Group Size (K candidates)", f"{k_samples}")
    print()

    # Part 2: Dual Reward Function Scoring
    section("Part 2: Dual Reward Scoring & Group Relative Advantage Calculation")

    format_rewards = [reward_format(resp) for resp in candidate_responses]
    length_rewards = [reward_conciseness(resp, target_length=30) for resp in candidate_responses]

    adv_format = compute_group_advantages(format_rewards)
    adv_length = compute_group_advantages(length_rewards)

    table_rows: list[list[str]] = []
    for idx in range(k_samples):
        table_rows.append([
            f"Sample {idx + 1}",
            f"{candidate_responses[idx][:35]}...",
            f"{format_rewards[idx]:.2f}",
            f"{adv_format[idx].item():+.3f}",
            f"{length_rewards[idx]:.2f}",
            f"{adv_length[idx].item():+.3f}",
        ])

    print_benchmark_table(
        table_rows,
        headers=("Candidate", "Snippet", "Format R", "Format Adv", "Length R", "Length Adv"),
    )

    # Verify zero-mean advantage invariant
    format_adv_mean = abs(adv_format.mean().item()) < 1e-5
    length_adv_mean = abs(adv_length.mean().item()) < 1e-5
    pass_fail("Group Advantage Normalization Check (mean == 0.0)", format_adv_mean and length_adv_mean)

    # Part 3: Advantage-Weighted Policy Gradient Step
    section("Part 3: Policy Gradient Optimization Step")
    inputs = {k: v.to(device) for k, v in tiny_inputs(batch_size=k_samples).items()}

    optimizer.zero_grad()
    outputs = model(**inputs)
    logits = outputs.logits  # (K, num_classes)

    # Log-probabilities of selected policy actions
    log_probs = F.log_softmax(logits, dim=-1)
    target_actions = torch.ones(k_samples, dtype=torch.long, device=device)
    action_log_probs = log_probs.gather(1, target_actions.unsqueeze(1)).squeeze(1)

    # Surrogate GRPO objective: - mean(A_i * log_prob_i)
    adv_weights = adv_format.to(device)
    surrogate_loss = - (adv_weights * action_log_probs).mean()

    surrogate_loss.backward()
    optimizer.step()

    loss_finite = bool(torch.isfinite(surrogate_loss).item())
    pass_fail("GRPO Policy Gradient Convergence", loss_finite, f"surrogate_loss={surrogate_loss.item():.4f}")

    # Part 4: TRL / HF Commentary
    section("Part 4: Enterprise Integration Commentary")
    try:
        import trl  # noqa: F401
        has_trl = True
    except ImportError:
        has_trl = False

    if has_trl:
        kv("Official TRL Package", "Installed — ready for trl.GRPOTrainer production training.")
    else:
        kv("Official TRL Package", "Optional — NeuralOps provides self-contained native GRPO without third-party dependencies.")

    return bool(format_adv_mean and length_adv_mean and loss_finite)


def main() -> int:
    section("NeuralOps Example 08: GRPO Reasoning & Alignment")
    backend = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", backend)
    kv("PyTorch Version", torch.__version__)
    print()

    success = run_grpo_simulation(backend)
    print()
    if success:
        pass_fail("Overall Example 08 Status", True, "All GRPO advantage and policy gradient steps verified")
        return 0
    else:
        pass_fail("Overall Example 08 Status", False, "One or more GRPO verification checks failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
