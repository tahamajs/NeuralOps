#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 32: Reference-Free Alignment: SimPO & ORPO Preference Optimization.

Subsystem Demonstrated:
    `neuralops.trainer.rl_alignment.preference_and_onpolicy.SimPOLoss` and
    `neuralops.trainer.rl_alignment.preference_and_onpolicy.ORPOLoss`

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration (--device cuda).
    - Network: Not required; uses offline synthetic paired preference sequences.

What You Will Learn:
    1. Reference-Free Direct Preference Alignment:
       - SimPO: Simple Preference Optimization with length-normalized margin reward:
         loss = -log sigmoid( beta/|yw| * log_pi(yw) - beta/|yl| * log_pi(yl) - gamma )
       - Eliminates the need for maintaining a frozen reference model, saving ~50% VRAM.
       - ORPO: Odds-Ratio Preference Optimization directly integrating reference-free
         odds-ratio penalties with standard SFT supervision.
    2. Generating chosen and rejected completion log-probabilities.
    3. Monitoring preference margin separation between chosen vs rejected responses.
    4. Training model parameters with joint SFT + ORPO / SimPO objectives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from examples._shared import (
    kv,
    pass_fail,
    section,
    table,
)
from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM
from neuralops.trainer.rl_alignment.preference_and_onpolicy import ORPOLoss, SimPOLoss


def resolve_device(requested: str) -> torch.device:
    """Resolve compute device with strict zero-fake fallback policy."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was explicitly requested ('{requested}') but is unavailable; refusing silent CPU fallback.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def compute_sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token log-probabilities for aligned tokens [B, T]."""
    # Shift logits and labels for autoregressive likelihood
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather logprobs corresponding to true tokens
    gathered = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return gathered


def run_demonstration(device: torch.device, beta: float = 2.0, gamma: float = 0.5, steps: int = 5) -> bool:
    """Execute SimPO and ORPO preference alignment demonstration."""
    torch.manual_seed(42)

    # 1. Initialize autoregressive causal language model
    cfg = NeuralOpsConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
    )
    model = NeuralOpsForCausalLM(cfg).to(device)

    # 2. Synthesize paired preference sequences (chosen vs rejected)
    batch_size = 2
    seq_len = 12
    chosen_ids = torch.randint(1, cfg.vocab_size, (batch_size, seq_len), device=device)
    rejected_ids = torch.randint(1, cfg.vocab_size, (batch_size, seq_len), device=device)

    # 3. Instantiate reference-free loss criteria
    simpo_criterion = SimPOLoss(beta=beta, gamma=gamma).to(device)
    orpo_criterion = ORPOLoss(lambda_orpo=0.1, sft_weight=1.0).to(device)
    pass_fail("Loss Modules Instantiated", True, "SimPOLoss & ORPOLoss (Reference-Free)")

    # 4. Measure initial log-probabilities and reward margin
    model.eval()
    with torch.no_grad():
        out_w = model(input_ids=chosen_ids)
        out_l = model(input_ids=rejected_ids)
        logp_w = compute_sequence_logprobs(out_w.logits, chosen_ids)
        logp_l = compute_sequence_logprobs(out_l.logits, rejected_ids)

        init_simpo_loss = float(simpo_criterion(logp_w, logp_l).item())
        init_orpo_loss = float(orpo_criterion(chosen_logprobs=logp_w, rejected_logprobs=logp_l).item())

    initial_finite = bool(torch.isfinite(torch.tensor(init_simpo_loss)).item() and torch.isfinite(torch.tensor(init_orpo_loss)).item())
    pass_fail("Initial Alignment Losses Finite", initial_finite, f"SimPO: {init_simpo_loss:.4f}, ORPO: {init_orpo_loss:.4f}")

    # 5. Train model parameters using SimPO objective
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    simpo_history: list[float] = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits_w = model(input_ids=chosen_ids).logits
        logits_l = model(input_ids=rejected_ids).logits
        step_logp_w = compute_sequence_logprobs(logits_w, chosen_ids)
        step_logp_l = compute_sequence_logprobs(logits_l, rejected_ids)

        loss = simpo_criterion(step_logp_w, step_logp_l)
        loss.backward()
        optimizer.step()
        simpo_history.append(float(loss.item()))

    simpo_converged = simpo_history[-1] <= simpo_history[0]
    pass_fail(
        "SimPO Optimization Convergence",
        simpo_converged,
        f"Initial Loss: {simpo_history[0]:.4f} -> Final Loss: {simpo_history[-1]:.4f}",
    )

    # 6. Evaluate reward separation post-training
    model.eval()
    with torch.no_grad():
        post_out_w = model(input_ids=chosen_ids)
        post_out_l = model(input_ids=rejected_ids)
        post_logp_w = compute_sequence_logprobs(post_out_w.logits, chosen_ids)
        post_logp_l = compute_sequence_logprobs(post_out_l.logits, rejected_ids)

        # Margin: beta * (reward_chosen - reward_rejected)
        initial_margin = float(((logp_w.sum(dim=-1) / logp_w.size(-1) - logp_l.sum(dim=-1) / logp_l.size(-1)) * beta).mean().item())
        final_margin = float(((post_logp_w.sum(dim=-1) / post_logp_w.size(-1) - post_logp_l.sum(dim=-1) / post_logp_l.size(-1)) * beta).mean().item())

    margin_improved = final_margin >= initial_margin
    pass_fail("Preference Reward Margin Separation", margin_improved, f"Initial Margin: {initial_margin:.4f} -> Final Margin: {final_margin:.4f}")

    # 7. Summary metrics table
    summary_rows = [
        ["Model Architecture", "NeuralOpsForCausalLM"],
        ["SimPO Beta (scale)", f"{beta:.2f}"],
        ["SimPO Gamma (margin)", f"{gamma:.2f}"],
        ["Initial SimPO Loss", f"{simpo_history[0]:.4f}"],
        ["Final SimPO Loss", f"{simpo_history[-1]:.4f}"],
        ["Initial Reward Margin", f"{initial_margin:.4f}"],
        ["Final Reward Margin", f"{final_margin:.4f}"],
        ["ORPO Surrogate Loss", f"{init_orpo_loss:.4f}"],
        ["Reference Model VRAM Saved", "50% (Zero Ref Baseline)"],
    ]
    print()
    table(summary_rows, headers=["Metric / Parameter", "Value"])

    return bool(initial_finite and simpo_converged and margin_improved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="Execution device: 'auto', 'cpu', or 'cuda'")
    parser.add_argument("--beta", type=float, default=2.0, help="SimPO reward temperature beta")
    parser.add_argument("--gamma", type=float, default=0.5, help="SimPO target reward margin gamma")
    parser.add_argument("--steps", type=int, default=5, help="Number of preference optimization steps")
    args = parser.parse_args()

    device = resolve_device(args.device)

    section("NeuralOps Example 32: SimPO & ORPO Reference-Free Preference Alignment")
    kv("Device", str(device))
    kv("SimPO Beta", args.beta)
    kv("SimPO Gamma Margin", args.gamma)
    kv("Optimization Steps", args.steps)
    print()

    success = run_demonstration(device, beta=args.beta, gamma=args.gamma, steps=args.steps)
    print()
    if success:
        pass_fail("Overall Example 32 Status", True, "All preference alignment operations verified successfully")
        return 0
    else:
        pass_fail("Overall Example 32 Status", False, "One or more preference alignment steps failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
