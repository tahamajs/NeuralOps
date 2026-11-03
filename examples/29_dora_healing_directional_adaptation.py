#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 29: DoRA Healing & Directional Adaptation.

Subsystem Demonstrated:
    `neuralops.trainer.regimes.dora_healing.DoRAHealingHealer` and
    `neuralops.trainer.regimes.dora_healing.DoRAProjectionWrapper`

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration (--device cuda).
    - Network: Not required; uses synthetic offline tensors.

What You Will Learn:
    1. Weight-Decomposed Low-Rank Adaptation (DoRA) formulation:
       W = m * (W_0 + Delta W) / ||W_0 + Delta W||_F
    2. Decoupling directional representation updates from magnitude scaling.
    3. Injecting DoRA wrappers into post-surgery model projections.
    4. Executing directional micro-healing steps with loss convergence.
    5. Merging and collapsing DoRA adapters back into standard Linear layers for zero inference overhead.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from examples._shared import (
    TinyClassifier,
    baseline_output,
    kv,
    pass_fail,
    section,
    table,
    tiny_inputs,
)
from neuralops.trainer.regimes.dora_healing import DoRAHealingHealer, DoRAProjectionWrapper


def resolve_device(requested: str) -> torch.device:
    """Resolve compute device with strict zero-fake fallback policy."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was explicitly requested ('{requested}') but is unavailable; refusing silent CPU fallback.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_demonstration(device: torch.device, rank: int = 4, alpha: float = 8.0, steps: int = 5) -> bool:
    """Execute DoRA directional adaptation and folding demonstration."""
    torch.manual_seed(42)

    # 1. Initialize baseline model
    model = TinyClassifier(n_layers=2).to(device)
    inputs = {k: v.to(device) for k, v in tiny_inputs().items()}
    ref_logits = baseline_output(model, inputs)

    # 2. Inject DoRA adapters into linear projections
    target_keys = ["dense", "q_lin", "k_lin", "v_lin", "out_lin"]
    healer = DoRAHealingHealer(
        model,
        rank=rank,
        lora_alpha=alpha,
        target_substrings=target_keys,
    )
    injected_count = healer.inject_dora_adapters()
    pass_fail("DoRA Adapters Injected", injected_count > 0, f"Injected {injected_count} wrappers")

    # 3. Verify DoRA forward pass with active decomposition
    with torch.no_grad():
        out_dora = model(**inputs)
        logits_dora = out_dora.logits if hasattr(out_dora, "logits") else out_dora
    dora_finite = bool(torch.isfinite(logits_dora).all().item())
    pass_fail("DoRA Forward Pass Finite", dora_finite)

    # 4. Perform directional micro-healing steps
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-3,
        weight_decay=1e-4,
    )
    model.train()
    loss_history: list[float] = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(**inputs)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.mse_loss(logits, ref_logits)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    loss_decreased = loss_history[-1] <= loss_history[0]
    pass_fail(
        "Micro-Healing Convergence",
        loss_decreased,
        f"Initial Loss: {loss_history[0]:.6f} -> Final Loss: {loss_history[-1]:.6f}",
    )

    # 5. Collapse & Merge DoRA adapters back into standard nn.Linear layers
    model.eval()
    healer.collapse_adapters()
    remaining_dora = any(isinstance(m, DoRAProjectionWrapper) for m in model.modules())
    pass_fail("Adapters Collapsed to Base Linear", not remaining_dora)

    # 6. Verify collapsed model outputs
    with torch.no_grad():
        out_collapsed = model(**inputs)
        logits_collapsed = out_collapsed.logits if hasattr(out_collapsed, "logits") else out_collapsed

    collapsed_finite = bool(torch.isfinite(logits_collapsed).all().item())
    pass_fail("Collapsed Model Forward Finite", collapsed_finite)

    # 7. Print summary metrics table
    summary_rows = [
        ["Injected Adapters", str(injected_count)],
        ["DoRA Rank (r)", str(rank)],
        ["DoRA Alpha", str(alpha)],
        ["Healing Steps", str(steps)],
        ["Initial Step Loss", f"{loss_history[0]:.6f}"],
        ["Final Step Loss", f"{loss_history[-1]:.6f}"],
        ["Adapters Residual Count", "0 (Cleanly Collapsed)"],
    ]
    print()
    table(summary_rows, headers=["Metric / Parameter", "Value"])

    return bool(injected_count > 0 and dora_finite and loss_decreased and not remaining_dora and collapsed_finite)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="Execution device: 'auto', 'cpu', or 'cuda'")
    parser.add_argument("--rank", type=int, default=4, help="DoRA low-rank dimension")
    parser.add_argument("--alpha", type=float, default=8.0, help="DoRA scaling alpha")
    parser.add_argument("--steps", type=int, default=5, help="Number of directional healing steps")
    args = parser.parse_args()

    device = resolve_device(args.device)

    section("NeuralOps Example 29: DoRA Healing & Directional Adaptation")
    kv("Device", str(device))
    kv("DoRA Rank", args.rank)
    kv("DoRA Alpha", args.alpha)
    kv("Healing Steps", args.steps)
    print()

    success = run_demonstration(device, rank=args.rank, alpha=args.alpha, steps=args.steps)
    print()
    if success:
        pass_fail("Overall Example 29 Status", True, "All DoRA operations verified successfully")
        return 0
    else:
        pass_fail("Overall Example 29 Status", False, "One or more DoRA verification steps failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
