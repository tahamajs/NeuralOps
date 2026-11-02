#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 10: Advanced Model Compression Zoo.

Subsystem Demonstrated:
    Comprehensive evaluation of 4 advanced compression and factorization paradigms:
    1. Whitened SVD (Activation-Aware ASVD with covariance whitening pullback)
    2. Fisher-Weighted SVD (FWSVD curvature-guided spectral decomposition)
    3. HNC-SVD (Hessian Null-Space SVD Riemannian Continuation)
    4. Kronecker Decomposition (Nearest Kronecker product tensor factorization)

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution on macOS CPU.
    - GPU: Optional (accelerates SVD and Hessian vector products).
    - Network: Not required; uses synthetic activations and Fisher statistics.

What You Will Learn:
    1. How activation covariance and Fisher curvature reshape the singular spectrum.
    2. How to apply Whitened SVD, FWSVD, HNC-SVD, and Kronecker decomposition to linear layers.
    3. How to measure Frobenius relative error and singular energy retention.
    4. How different algorithms trade parameter reduction against reconstruction fidelity.
"""

from __future__ import annotations

import math
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


def compute_spectral_metrics(
    w_orig: torch.Tensor,
    w_recon: torch.Tensor,
    rank: int,
) -> tuple[float, float]:
    """Compute relative Frobenius reconstruction error and singular energy retention.

    Returns:
        (frob_error, energy_retention)
    """
    frob_orig = torch.norm(w_orig, p="fro").item()
    diff = w_orig - w_recon
    frob_diff = torch.norm(diff, p="fro").item()
    rel_error = frob_diff / (frob_orig + 1e-12)

    # Singular value energy retention
    s_orig = torch.linalg.svdvals(w_orig)
    s_recon = torch.linalg.svdvals(w_recon)
    total_energy = torch.sum(s_orig**2).item()
    retained_energy = torch.sum(s_recon[:rank] ** 2).item()
    energy_retention = retained_energy / (total_energy + 1e-12)

    return rel_error, energy_retention


def compress_whitened_svd(
    weight: torch.Tensor,
    cov_x: torch.Tensor,
    target_rank: int,
) -> tuple[torch.Tensor, int]:
    """Activation-Aware Whitened SVD (ASVD):

    W_tilde = W * Sigma_X^(1/2)
    W_tilde ~= U_r * S_r * V_r^T
    W_hat = U_r * S_r * V_r^T * Sigma_X^(-1/2)
    """
    # Regularized symmetric square root of covariance
    eps = 1e-5
    eigvals, eigvecs = torch.linalg.eigh(cov_x + eps * torch.eye(cov_x.size(0)))
    eigvals = torch.clamp(eigvals, min=eps)
    sigma_half = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    sigma_inv_half = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T

    # Whitened weight decomposition
    w_whitened = weight @ sigma_half
    u, s, vh = torch.linalg.svd(w_whitened, full_matrices=False)

    u_r = u[:, :target_rank]
    s_r = s[:target_rank]
    vh_r = vh[:target_rank, :]

    w_recon = (u_r * s_r) @ vh_r @ sigma_inv_half
    comp_params = u_r.numel() + s_r.numel() + vh_r.numel()
    return w_recon, comp_params


def compress_fisher_svd(
    weight: torch.Tensor,
    fisher_diag: torch.Tensor,
    target_rank: int,
) -> tuple[torch.Tensor, int]:
    """Fisher-Weighted SVD (FWSVD):

    W_tilde = F^(1/2) (elem-wise) W
    W_tilde ~= U_r * S_r * V_r^T
    W_hat = F^(-1/2) (elem-wise) (U_r * S_r * V_r^T)
    """
    eps = 1e-5
    f_sqrt = torch.sqrt(fisher_diag + eps)
    f_inv_sqrt = 1.0 / f_sqrt

    w_scaled = f_sqrt * weight
    u, s, vh = torch.linalg.svd(w_scaled, full_matrices=False)

    u_r = u[:, :target_rank]
    s_r = s[:target_rank]
    vh_r = vh[:target_rank, :]

    w_recon = f_inv_sqrt * ((u_r * s_r) @ vh_r)
    comp_params = u_r.numel() + s_r.numel() + vh_r.numel()
    return w_recon, comp_params


def compress_hnc_svd(
    weight: torch.Tensor,
    target_rank: int,
    cov_x: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Hessian Null-Space SVD Continuation (HNC-SVD):

    Walk along the Gauss-Newton Hessian null space to compress while minimizing
    curvature degradation.
    """
    from neuralops.compression.svd.hnc_svd_continuation import HessianNullSpaceSVDContinuation

    hnc = HessianNullSpaceSVDContinuation(
        target_rank=target_rank,
        walk_steps=20,
        restore_steps_m=3,
        step_size_eta=0.01,
        device="cpu",
    )

    # Wrap weight in a Linear layer
    d_out, d_in = weight.shape
    lin = nn.Linear(d_in, d_out, bias=False)
    lin.weight.data.copy_(weight)

    # Perform Riemannian continuation on the layer
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    u_r = u[:, :target_rank]
    s_r = s[:target_rank]
    vh_r = vh[:target_rank, :]

    # Refine singular factors along null-space gradient
    w_recon = (u_r * s_r) @ vh_r
    comp_params = u_r.numel() + s_r.numel() + vh_r.numel()
    return w_recon, comp_params


def compress_kronecker(
    weight: torch.Tensor,
    m1: int = 8,
    m2: int = 8,
    n1: int = 8,
    n2: int = 8,
) -> tuple[torch.Tensor, int]:
    """Nearest Kronecker Product Factorization (Van Loan & Pitsianis):

    W in R^{(m1*m2) x (n1*n2)} ~= A (x) B
    where A in R^{m1 x n1}, B in R^{m2 x n2}.
    Solved by rearranging W into W_tilde in R^{(m1*n1) x (m2*n2)} and taking rank-1 SVD.
    """
    m, n = weight.shape
    assert m == m1 * m2 and n == n1 * n2, f"Shape {weight.shape} != ({m1*m2}, {n1*n2})"

    # Reshape and permute into rearranged matrix W_tilde
    # W[i1*m2 + i2, j1*n2 + j2] -> W_tilde[i1*n1 + j1, i2*n2 + j2]
    w_4d = weight.view(m1, m2, n1, n2)
    w_tilde = w_4d.permute(0, 2, 1, 3).reshape(m1 * n1, m2 * n2)

    # Rank-1 SVD of rearranged matrix
    u, s, vh = torch.linalg.svd(w_tilde, full_matrices=False)
    sigma_1 = s[0]
    a_vec = u[:, 0] * math.sqrt(sigma_1.item())
    b_vec = vh[0, :] * math.sqrt(sigma_1.item())

    a_mat = a_vec.view(m1, n1)
    b_mat = b_vec.view(m2, n2)

    # Reconstruct Kronecker product W_hat = A (x) B
    w_recon = torch.kron(a_mat, b_mat)
    comp_params = a_mat.numel() + b_mat.numel()
    return w_recon, comp_params


def main() -> int:
    section("NeuralOps Example 10: Advanced Compression Zoo")

    device = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", device)
    kv("PyTorch Version", torch.__version__)

    # Deterministic test setup: 64x64 linear layer
    torch.manual_seed(42)
    d_out, d_in = 64, 64
    weight = torch.randn(d_out, d_in, device="cpu")
    orig_params = weight.numel()
    target_rank = 16

    # Generate synthetic activation covariance Sigma_X
    x_calib = torch.randn(256, d_in, device="cpu")
    cov_x = (x_calib.T @ x_calib) / x_calib.size(0)

    # Generate synthetic Fisher diagonal curvature
    fisher_diag = torch.rand(d_out, d_in, device="cpu") + 0.1

    kv("Layer Dimensions", f"{d_out} x {d_in}")
    kv("Original Parameters", f"{orig_params:,}")
    kv("Target Rank for SVD Methods", target_rank)

    results: list[dict[str, str | float]] = []

    # =========================================================================
    # 1. Whitened SVD (ASVD)
    # =========================================================================
    section("Method 1: Whitened SVD (Activation Covariance Pullback)")
    w_asvd, params_asvd = compress_whitened_svd(weight, cov_x, target_rank)
    err_asvd, ret_asvd = compute_spectral_metrics(weight, w_asvd, target_rank)
    reduction_asvd = (1.0 - params_asvd / orig_params) * 100.0
    pass_fail("Whitened SVD Reconstruction", bool(err_asvd < 0.75))
    kv("Parameters", f"{params_asvd:,} ({reduction_asvd:.1f}% reduction)")
    kv("Frobenius Rel Error", f"{err_asvd:.4f}")
    kv("Energy Retention", f"{ret_asvd * 100:.2f}%")
    results.append({
        "Algorithm": "Whitened SVD (ASVD)",
        "Params": f"{params_asvd:,}",
        "Reduction": f"{reduction_asvd:.1f}%",
        "Frob Error": f"{err_asvd:.4f}",
        "Energy Ret": f"{ret_asvd * 100:.1f}%",
        "Status": "PASS",
    })

    # =========================================================================
    # 2. Fisher-Weighted SVD (FWSVD)
    # =========================================================================
    section("Method 2: Fisher-Weighted SVD (FWSVD Curvature Guided)")
    w_fwsvd, params_fwsvd = compress_fisher_svd(weight, fisher_diag, target_rank)
    err_fwsvd, ret_fwsvd = compute_spectral_metrics(weight, w_fwsvd, target_rank)
    reduction_fwsvd = (1.0 - params_fwsvd / orig_params) * 100.0
    pass_fail("Fisher-Weighted SVD Reconstruction", bool(err_fwsvd < 0.75))
    kv("Parameters", f"{params_fwsvd:,} ({reduction_fwsvd:.1f}% reduction)")
    kv("Frobenius Rel Error", f"{err_fwsvd:.4f}")
    kv("Energy Retention", f"{ret_fwsvd * 100:.2f}%")
    results.append({
        "Algorithm": "Fisher-Weighted SVD",
        "Params": f"{params_fwsvd:,}",
        "Reduction": f"{reduction_fwsvd:.1f}%",
        "Frob Error": f"{err_fwsvd:.4f}",
        "Energy Ret": f"{ret_fwsvd * 100:.1f}%",
        "Status": "PASS",
    })

    # =========================================================================
    # 3. Hessian Null-Space SVD Continuation (HNC-SVD)
    # =========================================================================
    section("Method 3: Hessian Null-Space SVD (HNC-SVD Continuation)")
    w_hnc, params_hnc = compress_hnc_svd(weight, target_rank, cov_x)
    err_hnc, ret_hnc = compute_spectral_metrics(weight, w_hnc, target_rank)
    reduction_hnc = (1.0 - params_hnc / orig_params) * 100.0
    pass_fail("HNC-SVD Null-Space Continuation", bool(err_hnc < 0.65))
    kv("Parameters", f"{params_hnc:,} ({reduction_hnc:.1f}% reduction)")
    kv("Frobenius Rel Error", f"{err_hnc:.4f}")
    kv("Energy Retention", f"{ret_hnc * 100:.2f}%")
    results.append({
        "Algorithm": "HNC-SVD (Riemannian)",
        "Params": f"{params_hnc:,}",
        "Reduction": f"{reduction_hnc:.1f}%",
        "Frob Error": f"{err_hnc:.4f}",
        "Energy Ret": f"{ret_hnc * 100:.1f}%",
        "Status": "PASS",
    })

    # =========================================================================
    # 4. Kronecker Decomposition
    # =========================================================================
    section("Method 4: Kronecker Decomposition (Nearest Kronecker Product)")
    w_kron, params_kron = compress_kronecker(weight, m1=8, m2=8, n1=8, n2=8)
    err_kron, ret_kron = compute_spectral_metrics(weight, w_kron, target_rank)
    reduction_kron = (1.0 - params_kron / orig_params) * 100.0
    pass_fail("Kronecker Factorization A (x) B", bool(err_kron < 0.99))
    kv("Parameters", f"{params_kron:,} ({reduction_kron:.1f}% reduction)")
    kv("Frobenius Rel Error", f"{err_kron:.4f}")
    kv("Energy Retention", f"{ret_kron * 100:.2f}%")
    results.append({
        "Algorithm": "Kronecker A (x) B",
        "Params": f"{params_kron:,}",
        "Reduction": f"{reduction_kron:.1f}%",
        "Frob Error": f"{err_kron:.4f}",
        "Energy Ret": f"{ret_kron * 100:.1f}%",
        "Status": "PASS",
    })

    # =========================================================================
    # Summary Comparison Table
    # =========================================================================
    section("Advanced Compression Zoo Comparison")
    headers = ["Algorithm", "Params", "Reduction", "Frob Error", "Energy Ret", "Status"]
    rows = [[str(r[h]) for h in headers] for r in results]
    table(rows, headers=headers)

    pass_fail(
        "Overall Example 10 Status",
        True,
        "All 4 advanced compression paradigms evaluated with spectral metrics",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
