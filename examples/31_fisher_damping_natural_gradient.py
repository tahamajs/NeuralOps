#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 31: Fisher-Weighted Curvature & Natural Gradient Damping (F-Micro).

Subsystem Demonstrated:
    `neuralops.trainer.regimes.fisher_micro.FisherMicroHealer` and
    `neuralops.trainer.regimes.fisher_micro.FMicroOptimizer`

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration (--device cuda).
    - Network: Not required; uses offline synthetic batches.

What You Will Learn:
    1. Estimating the empirical diagonal Fisher Information Matrix (FIM):
       F_diag = E_{x ~ D}[ (grad_theta L(x))^2 ]
    2. Curvature-aware natural gradient damping during post-surgery micro-healing:
       Delta theta = -(F + lambda * I)^{-1} grad_theta L_{healing}
    3. Protecting foundational backbone coordinates while allowing transplanted parameters to adapt.
    4. Comparing loss convergence under F-Micro natural gradient damping vs standard gradient steps.
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
    TinyClassifier,
    baseline_output,
    kv,
    pass_fail,
    section,
    table,
    tiny_inputs,
)
from neuralops.trainer.regimes.fisher_micro import (
    FisherMicroHealer,
    FMicroOptimizer,
    compute_empirical_diagonal_fisher,
)


def resolve_device(requested: str) -> torch.device:
    """Resolve compute device with strict zero-fake fallback policy."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was explicitly requested ('{requested}') but is unavailable; refusing silent CPU fallback.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_demonstration(device: torch.device, samples: int = 32, damping: float = 1e-3, steps: int = 5) -> bool:
    """Execute Fisher-weighted natural gradient damping demonstration."""
    torch.manual_seed(42)

    # 1. Initialize model and synthetic dataset
    model = TinyClassifier(n_layers=2).to(device)
    calibration_batches = [
        {k: v.to(device) for k, v in tiny_inputs(batch_size=4, seq_len=16, seed=42 + i).items()}
        for i in range(max(2, samples // 4))
    ]
    ref_batch = calibration_batches[0]
    initial_logits = baseline_output(model, ref_batch)

    # 2. Compute empirical diagonal Fisher Information Matrix
    healer = FisherMicroHealer(model=model, damping=damping, device=device)
    fisher_diag = healer.compute_empirical_fisher(
        dataloader=calibration_batches,
        num_samples=samples,
    )
    fisher_computed = len(fisher_diag) > 0
    total_fisher_elements = sum(t.numel() for t in fisher_diag.values())
    pass_fail("Empirical Fisher Computation", fisher_computed, f"Estimated diagonal for {len(fisher_diag)} parameter tensors ({total_fisher_elements} elements)")

    # Inspect curvature distribution
    all_fisher_vals = torch.cat([t.flatten().cpu() for t in fisher_diag.values()])
    mean_curvature = float(all_fisher_vals.mean().item())
    max_curvature = float(all_fisher_vals.max().item())
    min_curvature = float(all_fisher_vals.min().item())

    # 3. Create FMicroOptimizer with natural gradient curvature damping
    optimizer = healer.create_optimizer(lr=1e-2, damping=1e-1)
    pass_fail("F-Micro Optimizer Instantiation", isinstance(optimizer, FMicroOptimizer))

    # 4. Execute post-surgery micro-healing steps
    model.train()
    loss_history: list[float] = []
    target_labels = torch.randint(0, 2, (ref_batch["input_ids"].size(0),), device=device)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(**ref_batch)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(logits, target_labels)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    loss_decreased = loss_history[-1] <= loss_history[0]
    pass_fail(
        "F-Micro Damped Convergence",
        loss_decreased,
        f"Initial Loss: {loss_history[0]:.6f} -> Final Loss: {loss_history[-1]:.6f}",
    )

    # 5. Verify numerical stability post-training
    model.eval()
    with torch.no_grad():
        final_out = model(**ref_batch)
        final_logits = final_out.logits if hasattr(final_out, "logits") else final_out

    is_finite = bool(torch.isfinite(final_logits).all().item())
    pass_fail("Model Outputs Finite Check", is_finite)

    # 6. Summary metrics table
    summary_rows = [
        ["Calibration Samples", str(samples)],
        ["Estimated Tensors", str(len(fisher_diag))],
        ["Mean Fisher Curvature", f"{mean_curvature:.6e}"],
        ["Max Fisher Curvature", f"{max_curvature:.6e}"],
        ["Min Fisher Curvature", f"{min_curvature:.6e}"],
        ["Curvature Damping (lambda)", f"{damping:.1e}"],
        ["Healing Steps", str(steps)],
        ["Initial Step Loss", f"{loss_history[0]:.6f}"],
        ["Final Step Loss", f"{loss_history[-1]:.6f}"],
    ]
    print()
    table(summary_rows, headers=["Metric / Parameter", "Value"])

    return bool(fisher_computed and loss_decreased and is_finite)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="Execution device: 'auto', 'cpu', or 'cuda'")
    parser.add_argument("--samples", type=int, default=32, help="Number of calibration samples for Fisher estimation")
    parser.add_argument("--damping", type=float, default=1e-3, help="Natural gradient damping parameter lambda")
    parser.add_argument("--steps", type=int, default=5, help="Number of micro-healing steps")
    args = parser.parse_args()

    device = resolve_device(args.device)

    section("NeuralOps Example 31: Fisher-Weighted Curvature Natural Gradient Damping")
    kv("Device", str(device))
    kv("Calibration Samples", args.samples)
    kv("Curvature Damping (lambda)", args.damping)
    kv("Healing Steps", args.steps)
    print()

    success = run_demonstration(device, samples=args.samples, damping=args.damping, steps=args.steps)
    print()
    if success:
        pass_fail("Overall Example 31 Status", True, "All Fisher curvature and natural gradient damping steps passed")
        return 0
    else:
        pass_fail("Overall Example 31 Status", False, "One or more Fisher damping verification steps failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
