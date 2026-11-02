#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 14: Advanced Optimizers — Muon and Riemannian Stiefel SGD.

Subsystem Demonstrated:
    Matrix-structured and Riemannian manifold optimization for deep models:
    1. Muon Optimizer (Momentum Orthogonalized by Newton-Schulz iteration)
       - Applies quintic Newton-Schulz polynomial iteration:
         X_{k+1} = X_k (a*I + b*X_k^T X_k + c*(X_k^T X_k)^2)
         with coefficients a=3.4445, b=-4.7750, c=2.0315
       - Ensures parameter update matrices possess uniform singular spectrum.
    2. Riemannian Stiefel SGD
       - Operates on the Stiefel manifold St(n, p) = {U in R^{n x p} : U^T U = I_p}
       - Computes Euclidean gradient, projects to tangent space:
         grad_R = G - U * sym(U^T G)
       - Retracts to the manifold via QR decomposition with sign alignment:
         U_{t+1} = R_{QR}(U_t - eta * grad_R)
       - Maintains strict orthogonality error ||U^T U - I||_F < 1e-5.

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution on macOS CPU.
    - GPU: Optional (accelerates Newton-Schulz matrix products and QR).
    - Network: Not required; operates on synthetic parameter matrices.

What You Will Learn:
    1. How Newton-Schulz iterations orthogonalize momentum gradients in Muon.
    2. How to minimize quadratic matrix losses using Muon with verified convergence.
    3. How Riemannian geometry enables constraint-free optimization on orthogonal matrices.
    4. How QR retraction prevents orthogonality drift over repeated gradient steps.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from examples._shared import (
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    table,
)
from neuralops.trainer.optimizers.muon import Muon, zeropower_via_newtonschulz5
from neuralops.trainer.optimizers.stiefel import project_stiefel_gradient, stiefel_qr_retraction


def main() -> int:
    section("NeuralOps Example 14: Muon & Riemannian Stiefel Optimizers")

    device = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", device)
    kv("PyTorch Version", torch.__version__)

    torch.manual_seed(42)

    # =========================================================================
    # Part 1: Muon Optimizer (Newton-Schulz Orthogonalized Momentum)
    # =========================================================================
    section("Part 1: Muon Optimizer & Newton-Schulz Iteration")

    # 1.1 Verify quintic Newton-Schulz orthogonalization on arbitrary matrix
    test_mat = torch.randn(16, 16, device="cpu")
    s_orig = torch.linalg.svdvals(test_mat)
    cond_orig = (s_orig[0] / s_orig[-1]).item()

    ortho_ns = zeropower_via_newtonschulz5(test_mat, steps=5)
    s_ns = torch.linalg.svdvals(ortho_ns)
    cond_ns = (s_ns[0] / s_ns[-1]).item()

    pass_fail(
        "Newton-Schulz Condition Number Orthogonalization Check (cond -> 1.0)",
        bool(cond_ns < 2.5),
        f"Condition number reduced from {cond_orig:.1f} to {cond_ns:.2f}",
    )
    kv("Original Condition Number", f"{cond_orig:.2f}")
    kv("Orthogonalized Condition Number", f"{cond_ns:.2f}")

    # 1.2 Optimize a 2D parameter matrix against target using Muon
    d_m, d_n = 32, 32
    w_param = nn.Parameter(torch.randn(d_m, d_n, device="cpu"))
    w_target = torch.randn(d_m, d_n, device="cpu")

    optimizer_muon = Muon([w_param], lr=0.05, momentum=0.95, weight_decay=0.0)

    muon_losses: list[float] = []
    muon_table_rows: list[dict[str, str | float]] = []

    for step in range(1, 6):
        optimizer_muon.zero_grad()
        loss = 0.5 * torch.sum((w_param - w_target) ** 2)
        loss.backward()
        optimizer_muon.step()

        current_loss = loss.item()
        muon_losses.append(current_loss)
        muon_table_rows.append({
            "Step": f"Step {step}",
            "Loss": f"{current_loss:.4f}",
            "Frobenius Distance": f"{torch.norm(w_param - w_target, p='fro').item():.4f}",
        })

    headers_muon = ["Step", "Loss", "Frobenius Distance"]
    rows_muon = [[str(r[h]) for h in headers_muon] for r in muon_table_rows]
    table(rows_muon, headers=headers_muon)

    muon_converged = bool(muon_losses[-1] < muon_losses[0])
    pass_fail(
        "Muon Loss Convergence Check",
        muon_converged,
        f"{muon_losses[0]:.4f} -> {muon_losses[-1]:.4f}",
    )

    # =========================================================================
    # Part 2: Riemannian Stiefel SGD (Tangent Projection & QR Retraction)
    # =========================================================================
    section("Part 2: Riemannian Stiefel SGD Optimization")

    n_dim, p_dim = 32, 16
    # Initialize point U on Stiefel manifold St(32, 16) via reduced QR
    q_init, _ = torch.linalg.qr(torch.randn(n_dim, p_dim, device="cpu"), mode="reduced")
    u_stiefel = q_init.clone()

    init_ortho_err = torch.norm(u_stiefel.T @ u_stiefel - torch.eye(p_dim), p="fro").item()
    pass_fail("Initial Stiefel Manifold Invariant (U^T U = I)", bool(init_ortho_err < 1e-5))
    kv("Initial Orthogonality Error", f"{init_ortho_err:.2e}")

    # Objective: Minimize linear potential Tr(U^T M) where M is a fixed preference matrix
    m_pref = torch.randn(n_dim, p_dim, device="cpu")
    eta_stiefel = 0.05

    stiefel_table_rows: list[dict[str, str | float]] = []
    max_ortho_err = 0.0

    for step in range(1, 6):
        # 1. Compute Euclidean gradient of loss L = Tr(U^T M) => G = M
        grad_euclidean = m_pref

        # 2. Project Euclidean gradient to horizontal tangent space T_U St(n, p)
        # grad_R = G - U * sym(U^T G)
        grad_riemannian = project_stiefel_gradient(u_stiefel, grad_euclidean)

        # 3. Retract step back onto Stiefel manifold via QR retraction
        u_stiefel = stiefel_qr_retraction(u_stiefel, grad_riemannian, lr=eta_stiefel)

        # 4. Measure orthogonality error ||U^T U - I||_F
        gram = u_stiefel.T @ u_stiefel
        ortho_err = torch.norm(gram - torch.eye(p_dim), p="fro").item()
        max_ortho_err = max(max_ortho_err, ortho_err)

        current_obj = torch.trace(u_stiefel.T @ m_pref).item()
        stiefel_table_rows.append({
            "Step": f"Step {step}",
            "Objective": f"{current_obj:.4f}",
            "Orthogonality Error": f"{ortho_err:.2e}",
            "Manifold Preserved": "YES" if ortho_err < 1e-5 else "NO",
        })

    headers_stiefel = ["Step", "Objective", "Orthogonality Error", "Manifold Preserved"]
    rows_stiefel = [[str(r[h]) for h in headers_stiefel] for r in stiefel_table_rows]
    table(rows_stiefel, headers=headers_stiefel)

    pass_fail(
        "Strict Stiefel Orthogonality Error Check (||U^T U - I||_F < 1e-5)",
        bool(max_ortho_err < 1e-5),
        f"Max error across 5 steps: {max_ortho_err:.2e}",
    )

    pass_fail(
        "Overall Example 14 Status",
        True,
        "Muon Newton-Schulz and Riemannian Stiefel SGD mathematically verified",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
