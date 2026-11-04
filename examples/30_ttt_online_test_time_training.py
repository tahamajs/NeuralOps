#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 30: TTT-Online: Test-Time Training Adaptation per Benchmark Prompt.

Subsystem Demonstrated:
    `neuralops.trainer.regimes.ttt_online.TTTOnlineAdapter`

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration (--device cuda).
    - Network: Not required; uses offline synthetic sequences.

What You Will Learn:
    1. Online Test-Time Training (TTT) concept: adapting model states or projection weights
       on individual test prompts prior to autoregressive generation or classification.
    2. Self-supervised reconstruction objective:
       S_{test} = S_0 - eta * grad_S ||x_{prompt} - hat{x}_{prompt}||_2^2
    3. Performing prompt-level fast inner adaptation steps on autoregressive causal models.
    4. Safe state restoration / parameter rollback reversing prompt-specific weight mutations.
    5. Evaluating perplexity / loss reduction on evaluation prompts.
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

from examples._shared import (
    kv,
    pass_fail,
    section,
    table,
)
from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM
from neuralops.trainer.regimes.ttt_online import TTTOnlineAdapter


def resolve_device(requested: str) -> torch.device:
    """Resolve compute device with strict zero-fake fallback policy."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was explicitly requested ('{requested}') but is unavailable; refusing silent CPU fallback.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_demonstration(device: torch.device, inner_lr: float = 1e-3, inner_steps: int = 3) -> bool:
    """Execute TTT-Online adaptation demonstration."""
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
    model.eval()

    # 2. Construct synthetic evaluation prompt batch
    batch_size = 2
    seq_len = 8
    inputs = {"input_ids": torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)}

    with torch.no_grad():
        initial_out = model(**inputs)
        initial_logits = initial_out.logits.clone()

    # 3. Initialize TTT-Online Adapter targeting projection layers
    targets = ["o_proj", "v_proj", "q_proj", "k_proj"]
    ttt_adapter = TTTOnlineAdapter(
        model=model,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        target_substrings=targets,
    )

    target_params = ttt_adapter._get_target_parameters()
    eligible_count = len(target_params)
    pass_fail("Eligible TTT Target Parameters", eligible_count > 0, f"Found {eligible_count} target tensors")

    # 4. Cache baseline parameter snapshots for rollback verification
    backups = [p.data.clone() for p in target_params]

    # 5. Perform self-supervised online test-time adaptation on prompt batch
    adaptation_result = ttt_adapter.adapt_on_prompt(inputs)
    ttt_loss = adaptation_result.get("ttt_loss", 0.0)
    pass_fail("TTT Adaptation Step Executed", ttt_loss > 0.0, f"Average inner reconstruction loss: {ttt_loss:.6f}")

    # 6. Measure model output shift post-adaptation
    with torch.no_grad():
        adapted_out = model(**inputs)
        adapted_logits = adapted_out.logits

    shift_norm = (adapted_logits - initial_logits).norm().item()
    adapted_finite = bool(torch.isfinite(adapted_logits).all().item())
    pass_fail("Adapted Forward Finite Check", adapted_finite)
    pass_fail("Prompt Adaptation Weight Update Verified", shift_norm > 0.0, f"L2 Logit Shift: {shift_norm:.6f}")

    # 7. Reset parameters back to pre-prompt state
    ttt_adapter.reset_parameters(backups)
    with torch.no_grad():
        restored_out = model(**inputs)
        restored_logits = restored_out.logits

    restored_delta = (restored_logits - initial_logits).abs().max().item()
    is_restored = restored_delta < 1e-6
    pass_fail("Post-Prompt Parameter Restoration", is_restored, f"Max delta from initial: {restored_delta:.2e}")

    # 8. Summary metrics table
    summary_rows = [
        ["Model Architecture", "NeuralOpsForCausalLM"],
        ["Target Projections Count", str(eligible_count)],
        ["TTT Inner Steps", str(inner_steps)],
        ["Inner Learning Rate", f"{inner_lr:.1e}"],
        ["Inner Autoregressive Loss", f"{ttt_loss:.6f}"],
        ["Post-TTT Logit Shift Norm", f"{shift_norm:.6f}"],
        ["Rollback Parity Max Error", f"{restored_delta:.2e}"],
    ]
    print()
    table(summary_rows, headers=["Metric / Dimension", "Observed Value"])

    return bool(eligible_count > 0 and ttt_loss > 0.0 and adapted_finite and shift_norm > 0.0 and is_restored)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="Execution device: 'auto', 'cpu', or 'cuda'")
    parser.add_argument("--lr", type=float, default=1e-3, help="TTT inner adaptation learning rate")
    parser.add_argument("--steps", type=int, default=3, help="Number of TTT inner gradient steps")
    args = parser.parse_args()

    device = resolve_device(args.device)

    section("NeuralOps Example 30: TTT-Online Test-Time Training Adaptation")
    kv("Device", str(device))
    kv("Inner Learning Rate", args.lr)
    kv("Inner Steps", args.steps)
    print()

    success = run_demonstration(device, inner_lr=args.lr, inner_steps=args.steps)
    print()
    if success:
        pass_fail("Overall Example 30 Status", True, "All TTT adaptation and restoration steps passed")
        return 0
    else:
        pass_fail("Overall Example 30 Status", False, "One or more TTT adaptation steps failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
