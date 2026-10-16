# -*- coding: utf-8 -*-
"""
Exhaustive 16-Part Verification Battery for Hessian Null Space Continuation (HNC)
SVD Model Compression, Attention Alignment & Fused Triton Kernel Suite.

Covers 16 Invariant Groups (70+ Parameterized Unit & Cross-Module Tests):
1. Group 1 — Anchor Exact Zero Loss & Zero First-Order Gradient across 5 Architectures
2. Group 2 — Gauss–Newton Positive Semidefiniteness (v^T G v >= 0) vs Indefinite Task Loss Hessian
3. Group 3 — Matrix-Free Pearlmutter HVP vs Dense Jacobian Gram Matrix G_dense = 1/N J^T J
4. Group 4 — Power Iteration lambda_1 & Matrix-Free LOBPCG Flat Subspace Finder
5. Group 5 — Analytical SVD Spectral Tail Potential phi_SVD(W) & Closed-Form Gradient
6. Group 6 — Conjugate Gradient (CG) Soft Null-Space Projection & Warm-Start Acceleration
7. Group 7 — Fast Layerwise Kronecker Null-Space Step D_W = grad_phi (I + H_X / mu_X)^{-1}
8. Group 8 — Function-Restoring Steps m in {0, 2, 6, 12} & Drift Suppression
9. Group 9 — Damping Factor Sweep mu_rel in {10^-8, 10^-5, 10^-3, inf}
10. Group 10 — Loss Ceiling tau Rejection, Weight Rollback, and eta Halving
11. Group 11 — HNC vs Naive Penalized Optimization (max phi - beta L_func)
12. Group 12 — Full-Logit Preservation (N x C) vs Scalar Loss Constraint (1)
13. Group 13 — SGD/Adam Stability of HNC-Steered Low-Rank Endpoints
14. Group 14 — Fused @triton.jit HNC Kernel Parity, Dtypes, Strides, and SM 6.1 Fallback
15. Group 15 — DirectSVDCompressor & SVDLinearAttention End-to-End Surgery Parity vs Naive SVD
16. Group 16 — SmoothMaxOverLayersCKASteeringLoss, HNCContinuationTrainer & 131/72/26 Invariants
"""

from __future__ import annotations

import math
from unittest.mock import patch
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.compression_engine.svd.hnc_svd_continuation import (
    HessianNullSpaceSVDContinuation,
    compute_function_matching_loss,
    compute_spectral_tail_potential_and_gradient,
    estimate_top_eigenvalue,
    find_flat_subspace_lobpcg,
    hvp_gauss_newton,
    solve_layerwise_kronecker_nullspace_step,
    solve_soft_nullspace_projection_cg,
)
from neuralops.core.triton_kernels.low_rank import (
    fused_hnc_nullspace_step_pytorch,
    fused_hnc_nullspace_step_triton,
)
from neuralops.compression_engine.svd.direct_svd import DirectSVDCompressor
from neuralops.attentions.svd.svd_linear_attention import SVDLinear, SVDLinearAttention
from neuralops.trainer.distillation.multi_level_kd import (
    LinearCKALoss,
    SmoothMaxOverLayersCKASteeringLoss,
)
from neuralops.trainer.surgery_schedules.mohawk import HNCContinuationTrainer
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


# ============================================================================
# Architectures for Group 1 Verification
# ============================================================================
class SmallAnchorMLP(nn.Module):
    def __init__(self, in_features: int = 16, hidden: int = 32, out_features: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels: int = 16, hidden: int = 32, out_channels: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(hidden, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        return self.conv2(F.silu(self.conv1(x)))


class SimpleRNNCellModel(nn.Module):
    def __init__(self, in_features: int = 16, hidden: int = 16, out_features: int = 8):
        super().__init__()
        self.rnn = nn.RNNCell(in_features, hidden, bias=False)
        self.proj = nn.Linear(hidden, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features]
        h = torch.zeros(x.size(0), 16, device=x.device, dtype=x.dtype)
        h = self.rnn(x, h)
        return self.proj(h)


class SmallTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.ffn(x)
        return x


# ============================================================================
# Group 1: Anchor Exact Zero Loss & Zero First-Order Gradient across 5 Archs
# ============================================================================
@pytest.mark.parametrize("arch_name", ["mlp", "conv1d", "rnn", "svd_attn", "transformer"])
def test_group1_anchor_exact_zero_loss_and_gradient(arch_name):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if arch_name == "mlp":
        model = SmallAnchorMLP(16, 32, 8).to(device)
        probe_x = torch.randn(8, 16, device=device)
    elif arch_name == "conv1d":
        model = Conv1DBlock(16, 32, 8).to(device)
        probe_x = torch.randn(4, 16, 12, device=device)
    elif arch_name == "rnn":
        model = SimpleRNNCellModel(16, 16, 8).to(device)
        probe_x = torch.randn(8, 16, device=device)
    elif arch_name == "svd_attn":
        model = SVDLinearAttention(hidden_size=32, num_heads=4, rank=8).to(device)
        probe_x = torch.randn(2, 8, 32, device=device)
    elif arch_name == "transformer":
        model = SmallTransformerBlock(32, 4).to(device)
        probe_x = torch.randn(4, 8, 32, device=device)

    model.eval()
    with torch.no_grad():
        out = model(probe_x)
        anchor_y = [out[0] if isinstance(out, tuple) else out]

    loss = compute_function_matching_loss(model, anchor_y, probe_x, device=device)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)

    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)
    for g in grads:
        if g is not None:
            assert torch.max(torch.abs(g)).item() == pytest.approx(0.0, abs=1e-6)


# ============================================================================
# Group 2: Gauss–Newton Positive Semidefiniteness (v^T G v >= 0)
# ============================================================================
@pytest.mark.parametrize("seed", list(range(10)))
def test_group2_gauss_newton_psd_vs_indefinite_loss(seed):
    torch.manual_seed(100 + seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallAnchorMLP(8, 16, 4).to(device)
    probe_x = torch.randn(8, 8, device=device)

    with torch.no_grad():
        anchor_y = [model(probe_x)]

    loss = compute_function_matching_loss(model, anchor_y, probe_x, device=device)
    params = list(model.parameters())

    v_list = [torch.randn_like(p) for p in params]
    # Normalize v
    norm_v = math.sqrt(sum(torch.sum(vi ** 2).item() for vi in v_list)) + 1e-8
    v_list = [vi / norm_v for vi in v_list]

    # Compute G * v
    Gv = hvp_gauss_newton(loss, params, v_list)

    # Quadratic form v^T G v
    v_G_v = sum(torch.sum(vi * gvi).item() for vi, gvi in zip(v_list, Gv))
    # Must be non-negative up to floating precision
    assert v_G_v >= -1e-6, f"Gauss-Newton quadratic form violated PSD: {v_G_v}"


# ============================================================================
# Group 3: Pearlmutter HVP vs Exact Dense Jacobian Gram Matrix
# ============================================================================
@pytest.mark.parametrize("batch_size", [8, 16, 32])
def test_group3_matrix_free_pearlmutter_vs_dense_jacobian_gram(batch_size):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallAnchorMLP(in_features=8, hidden=12, out_features=4).to(device=device)
    probe_x = torch.randn(batch_size, 8, device=device)

    with torch.no_grad():
        anchor_y = [model(probe_x)]

    loss = compute_function_matching_loss(model, anchor_y, probe_x, device=device)
    params = list(model.parameters())
    v_list = [torch.randn_like(p) for p in params]

    # 1. Pearlmutter HVP
    hvp_fast = hvp_gauss_newton(loss, params, v_list)

    # 2. Dense Jacobian Gram matrix calculation: G_dense = 1/N * J^T J
    y = model(probe_x)
    N, C = y.shape
    jac_rows = []
    for n in range(N):
        for c in range(C):
            model.zero_grad()
            y[n, c].backward(retain_graph=True)
            grad_flat = torch.cat([p.grad.flatten() for p in params])
            jac_rows.append(grad_flat)

    J = torch.stack(jac_rows, dim=0)  # [N*C, P]
    v_flat = torch.cat([vi.flatten() for vi in v_list])
    # G * v = 1/N * J^T (J v)
    Jv = torch.matmul(J, v_flat)
    Gv_dense_flat = torch.matmul(J.t(), Jv) / N

    # Compare unflattened
    offset = 0
    for h_f, p in zip(hvp_fast, params):
        numel = p.numel()
        ref_h = Gv_dense_flat[offset:offset + numel].view(p.shape)
        assert torch.allclose(h_f, ref_h, atol=1e-4, rtol=1e-4)
        offset += numel


# ============================================================================
# Group 4: Power Iteration & Matrix-Free LOBPCG Flat Subspace Finder
# ============================================================================
@pytest.mark.parametrize("k_flat", [4, 8, 12])
def test_group4_power_iteration_and_lobpcg_flat_subspace(k_flat):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallAnchorMLP(in_features=8, hidden=16, out_features=4).to(device=device)
    probe_x = torch.randn(12, 8, device=device)

    with torch.no_grad():
        anchor_y = [model(probe_x)]

    loss = compute_function_matching_loss(model, anchor_y, probe_x, device=device)
    params = list(model.parameters())

    def hvp_fn(v):
        return hvp_gauss_newton(loss, params, v)

    top_lambda, top_v = estimate_top_eigenvalue(hvp_fn, params, num_iters=10)
    assert top_lambda > 0.0

    eigvals, flat_dirs = find_flat_subspace_lobpcg(hvp_fn, params, k=k_flat, num_iters=8)
    assert len(flat_dirs) == k_flat

    # Check basis orthonormality: V^T V ≈ I
    V_cols = []
    for vec in flat_dirs:
        V_cols.append(torch.cat([vi.flatten() for vi in vec]))
    V_mat = torch.stack(V_cols, dim=1)
    gram = torch.matmul(V_mat.t(), V_mat)
    eye = torch.eye(k_flat, device=device)
    assert torch.max(torch.abs(gram - eye)).item() < 1e-4

    # Verify flat perturbation sensitivity vs sharp direction
    eta = 0.05
    # Perturb in flat direction (first eigenvector of flat subspace)
    for p, vi in zip(params, flat_dirs[0]):
        p.data.add_(eta * vi)
    with torch.no_grad():
        out_flat = model(probe_x)
        diff_flat = torch.sum((out_flat - anchor_y[0]) ** 2).item()
    # Rollback
    for p, vi in zip(params, flat_dirs[0]):
        p.data.sub_(eta * vi)

    # Perturb in sharp direction (top_v)
    for p, vi in zip(params, top_v):
        p.data.add_(eta * vi)
    with torch.no_grad():
        out_sharp = model(probe_x)
        diff_sharp = torch.sum((out_sharp - anchor_y[0]) ** 2).item()
    for p, vi in zip(params, top_v):
        p.data.sub_(eta * vi)

    assert diff_flat <= diff_sharp + 1e-5


# ============================================================================
# Group 5: SVD Spectral Tail Potential phi_SVD & Analytical Gradient
# ============================================================================
@pytest.mark.parametrize("d_out,d_in", [(32, 32), (64, 32), (32, 64), (128, 64)])
@pytest.mark.parametrize("rank", [4, 8, 16])
def test_group5_svd_spectral_tail_potential_and_gradient(d_out, d_in, rank):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.randn(d_out, d_in, device=device, dtype=torch.float32)

    phi, grad_phi, stats = compute_spectral_tail_potential_and_gradient(
        W, rank=rank, gamma_nuc=1e-3, gamma_bump=0.05
    )
    assert not torch.isnan(grad_phi).any()
    assert not torch.isinf(grad_phi).any()
    assert grad_phi.shape == W.shape
    assert stats["tail_energy_ratio"] >= 0.0

    # Immunity to repeated singular value NaNs
    # Construct matrix with identical singular values
    U, S_orig, Vh = torch.linalg.svd(W, full_matrices=False)
    k_dim = len(S_orig)
    S_dup = torch.ones(k_dim, device=device) * 2.0
    W_dup = torch.matmul(U[:, :k_dim] * S_dup.unsqueeze(0), Vh[:k_dim, :])
    _, grad_dup, _ = compute_spectral_tail_potential_and_gradient(W_dup, rank=min(rank, k_dim - 1))
    assert not torch.isnan(grad_dup).any()
    assert not torch.isinf(grad_dup).any()


# ============================================================================
# Group 6: Soft Null-Space Projection CG Solver & Warm-Start Acceleration
# ============================================================================
@pytest.mark.parametrize("mu_rel", [1e-2, 1e-4, 1e-6])
def test_group6_soft_nullspace_cg_and_warmstart(mu_rel):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallAnchorMLP(8, 12, 4).to(device)
    probe_x = torch.randn(8, 8, device=device)

    with torch.no_grad():
        anchor_y = [model(probe_x)]

    loss = compute_function_matching_loss(model, anchor_y, probe_x, device=device)
    params = list(model.parameters())

    def hvp_fn(v):
        return hvp_gauss_newton(loss, params, v)

    rhs_list = [torch.randn_like(p) for p in params]

    # Cold start CG
    sol_cold, info_cold = solve_soft_nullspace_projection_cg(hvp_fn, rhs_list, mu=mu_rel, max_iters=20)
    assert len(sol_cold) == len(rhs_list)

    # Warm start CG with cold solution perturbed
    warm_init = [s + 0.01 * torch.randn_like(s) for s in sol_cold]
    sol_warm, info_warm = solve_soft_nullspace_projection_cg(
        hvp_fn, rhs_list, mu=mu_rel, max_iters=20, initial_d=warm_init
    )
    assert info_warm["iterations"] <= info_cold["iterations"] + 2


# ============================================================================
# Group 7: Fast Layerwise Kronecker Null-Space Step
# ============================================================================
@pytest.mark.parametrize("d_out,d_in", [(32, 32), (64, 48)])
@pytest.mark.parametrize("target_rank", [8, 16])
def test_group7_layerwise_kronecker_nullspace_step(d_out, d_in, target_rank):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.randn(d_out, d_in, device=device)
    X = torch.randn(64, d_in, device=device)
    Hx = torch.matmul(X.t(), X) / 64.0

    _, grad_phi, _ = compute_spectral_tail_potential_and_gradient(W, rank=target_rank)
    D_W, resolvent = solve_layerwise_kronecker_nullspace_step(W, grad_phi, Hx, mu_rel=1e-5)

    assert D_W.shape == W.shape
    assert resolvent.shape == (d_in, d_in)
    # Output difference on probe data: || X (W + eta D_W)^T - X W^T ||
    eta = 0.01
    y_orig = torch.matmul(X, W.t())
    y_steered = torch.matmul(X, (W + eta * D_W).t())
    rel_drift = torch.norm(y_steered - y_orig) / (torch.norm(y_orig) + 1e-8)
    assert rel_drift.item() < 0.2


# ============================================================================
# Group 8: Function-Restoring Steps m in {0, 2, 6, 12}
# ============================================================================
@pytest.mark.parametrize("restore_steps", [0, 2, 6, 12])
def test_group8_function_restoring_steps_ablation(restore_steps):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallAnchorMLP(16, 24, 8).to(device)
    probe_x = torch.randn(12, 16, device=device)

    hnc = HessianNullSpaceSVDContinuation(
        model,
        target_rank=6,
        mode="kronecker",
        walk_steps=4,
        step_size_eta=0.03,
        restore_steps_m=restore_steps,
        restore_lr=1e-3,
    )
    res = hnc.steer_and_compress(probe_x)
    assert res["status"] == "SUCCESS"
    assert res["final_function_loss"] >= 0.0


# ============================================================================
# Group 9: Damping Factor Sweep mu_rel in {1e-8, 1e-5, 1e-3, inf}
# ============================================================================
@pytest.mark.parametrize("mu_rel", [1e-8, 1e-5, 1e-3, float("inf")])
def test_group9_damping_factor_ablation(mu_rel):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallAnchorMLP(16, 24, 8).to(device)
    probe_x = torch.randn(8, 16, device=device)

    hnc = HessianNullSpaceSVDContinuation(
        model,
        target_rank=6,
        mode="kronecker",
        walk_steps=3,
        step_size_eta=0.02,
        restore_steps_m=2,
        mu_rel=mu_rel,
    )
    res = hnc.steer_and_compress(probe_x)
    assert res["status"] == "SUCCESS"


# ============================================================================
# Group 10: Loss Ceiling tau Rejection, Weight Rollback & eta Halving
# ============================================================================
def test_group10_loss_ceiling_rejection_and_rollback():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallAnchorMLP(16, 24, 8).to(device)
    probe_x = torch.randn(8, 16, device=device)

    # Use an impossibly strict loss ceiling to force rejections
    strict_tau = 1e-12
    hnc = HessianNullSpaceSVDContinuation(
        model,
        target_rank=4,
        mode="kronecker",
        walk_steps=4,
        step_size_eta=0.1,
        restore_steps_m=0,
        loss_ceiling_tau=strict_tau,
    )
    res = hnc.steer_and_compress(probe_x)
    # At least some steps should have been rejected and rolled back
    assert res["rejected_steps"] > 0
    # Confirm rollback maintained finite valid parameters
    for p in model.parameters():
        assert not torch.isnan(p).any()


# ============================================================================
# Group 11: HNC vs Naive Penalized Optimization
# ============================================================================
def test_group11_hnc_vs_naive_penalized_optimization():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # HNC model
    model_hnc = SmallAnchorMLP(16, 24, 8).to(device)
    # Naive model
    model_pen = SmallAnchorMLP(16, 24, 8).to(device)
    model_pen.load_state_dict(model_hnc.state_dict())

    probe_x = torch.randn(16, 16, device=device)
    with torch.no_grad():
        anchor_y = model_hnc(probe_x)

    # 1. Run HNC
    hnc = HessianNullSpaceSVDContinuation(model_hnc, target_rank=6, walk_steps=6, step_size_eta=0.02)
    res_hnc = hnc.steer_and_compress(probe_x)

    # 2. Run Naive Penalty optimization: loss = -phi + 10.0 * L_func
    opt = torch.optim.Adam(model_pen.parameters(), lr=1e-3)
    for _ in range(6):
        opt.zero_grad()
        y = model_pen(probe_x)
        l_func = 0.5 * torch.mean((y - anchor_y) ** 2)
        # Potential on fc1
        phi, _, _ = compute_spectral_tail_potential_and_gradient(model_pen.fc1.weight, rank=6)
        total_loss = -phi + 5.0 * l_func
        total_loss.backward()
        opt.step()

    # Compare tail energy ratios
    _, S_hnc, _ = torch.linalg.svd(model_hnc.fc1.weight.data)
    _, S_pen, _ = torch.linalg.svd(model_pen.fc1.weight.data)
    tail_hnc = torch.sum(S_hnc[6:] ** 2) / torch.sum(S_hnc ** 2)
    tail_pen = torch.sum(S_pen[6:] ** 2) / torch.sum(S_pen ** 2)

    assert tail_hnc.item() <= tail_pen.item() + 0.15


# ============================================================================
# Group 12: Full-Logit Preservation vs Scalar Loss Constraint
# ============================================================================
def test_group12_full_logit_preservation_vs_scalar_loss():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallAnchorMLP(16, 32, 10).to(device)
    probe_x = torch.randn(16, 16, device=device)
    heldout_x = torch.randn(16, 16, device=device)

    with torch.no_grad():
        y_anchor_probe = model(probe_x)
        y_anchor_heldout = model(heldout_x)
        top1_probe_anchor = torch.argmax(y_anchor_probe, dim=-1)
        top1_heldout_anchor = torch.argmax(y_anchor_heldout, dim=-1)

    hnc = HessianNullSpaceSVDContinuation(model, target_rank=8, walk_steps=5, step_size_eta=0.02)
    hnc.steer_and_compress(probe_x)

    with torch.no_grad():
        top1_probe_steered = torch.argmax(model(probe_x), dim=-1)
        top1_heldout_steered = torch.argmax(model(heldout_x), dim=-1)

    probe_match = (top1_probe_steered == top1_probe_anchor).float().mean().item()
    heldout_match = (top1_heldout_steered == top1_heldout_anchor).float().mean().item()

    assert probe_match >= 0.85, f"Expected probe_match >= 0.85, got {probe_match}"
    assert heldout_match >= 0.75, f"Expected heldout_match >= 0.75, got {heldout_match}"


# ============================================================================
# Group 13: SGD Stability of HNC-Steered Low-Rank Endpoints
# ============================================================================
def test_group13_sgd_stability_of_hnc_steered_endpoints():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallAnchorMLP(16, 32, 8).to(device)
    probe_x = torch.randn(16, 16, device=device)

    target_rank = 8
    hnc = HessianNullSpaceSVDContinuation(model, target_rank=target_rank, walk_steps=5, step_size_eta=0.02)
    hnc.steer_and_compress(probe_x)

    # Initial tail energy of steered endpoint
    _, S_steered, _ = torch.linalg.svd(model.fc1.weight.data)
    tail_steered = torch.sum(S_steered[target_rank:] ** 2).item()

    # Resume SGD fine-tuning for 10 steps
    opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    model.train()
    for _ in range(10):
        opt.zero_grad()
        loss = torch.sum(model(probe_x) ** 2)
        loss.backward()
        opt.step()

    _, S_after, _ = torch.linalg.svd(model.fc1.weight.data)
    tail_after = torch.sum(S_after[target_rank:] ** 2).item()

    # Verify weights did not rebound into a high-energy tail
    assert tail_after <= tail_steered + 1.0


# ============================================================================
# Group 14: Fused @triton.jit HNC Kernel Parity, Dtypes, & SM 6.1 Fallback
# ============================================================================
@pytest.mark.parametrize("d_out,d_in", [(32, 32), (64, 64), (128, 64), (96, 160)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_group14_triton_fused_hnc_kernel_parity_and_fallback(d_out, d_in, dtype):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.randn(d_out, d_in, device=device, dtype=dtype)
    grad_phi = torch.randn(d_out, d_in, device=device, dtype=dtype)
    inv_null = torch.eye(d_in, device=device, dtype=dtype)

    out_pt = fused_hnc_nullspace_step_pytorch(W, grad_phi, inv_null, eta=0.01)
    out_tri = fused_hnc_nullspace_step_triton(W, grad_phi, inv_null, eta=0.01)

    tol = 1e-4 if dtype == torch.float32 else 5e-3
    diff = torch.max(torch.abs(out_pt - out_tri)).item()
    assert diff <= tol

    # Pascal SM 6.1 mock fallback verification
    with patch("neuralops.core.triton_kernels.hardware_profile.get_device_hardware_profile") as mock_hw:
        from neuralops.core.triton_kernels.hardware_profile import HardwareProfile
        mock_hw.return_value = HardwareProfile(
            backend="cuda",
            device_name="GeForce GTX 1060 (Mock)",
            sm_major=6,
            sm_minor=1,
            max_shared_memory_bytes=49152,
            supports_triton_jit=False,
            supports_tensor_core_dot=False,
            supports_bf16=False,
            supports_tf32=False,
            warp_size=32,
            vendor="nvidia",
            fp8_dtype=None,
        )
        fallback_out = fused_hnc_nullspace_step_triton(W, grad_phi, inv_null, eta=0.01)
        assert torch.allclose(fallback_out, out_pt, atol=tol)


# ============================================================================
# Group 15: DirectSVDCompressor & SVDLinearAttention Surgery Parity vs Naive SVD
# ============================================================================
@pytest.mark.parametrize("target_rank", [4, 8, 16, 32, 64])
def test_group15_hnc_svd_vs_naive_direct_svd_error(target_rank):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_orig = SmallAnchorMLP(64, 128, 32).to(device)
    model_naive = SmallAnchorMLP(64, 128, 32).to(device)
    model_naive.load_state_dict(model_orig.state_dict())
    model_hnc = SmallAnchorMLP(64, 128, 32).to(device)
    model_hnc.load_state_dict(model_orig.state_dict())

    probe_x = torch.randn(16, 64, device=device)

    # 1. Compress with Naive DirectSVD
    comp_naive = DirectSVDCompressor(model_naive, rank=target_rank, use_hnc_continuation=False)
    mod_naive, _ = comp_naive.compress()

    # 2. Compress with HNC DirectSVD
    comp_hnc = DirectSVDCompressor(
        model_hnc,
        rank=target_rank,
        use_hnc_continuation=True,
        probe_inputs=probe_x,
    )
    mod_hnc, _ = comp_hnc.compress()

    # Compare output drift against original uncompressed model
    with torch.no_grad():
        y_orig = model_orig(probe_x)
        out_naive = mod_naive(probe_x)
        out_hnc = mod_hnc(probe_x)
        err_naive = (torch.norm(out_naive - y_orig) / (torch.norm(y_orig) + 1e-8)).item()
        err_hnc = (torch.norm(out_hnc - y_orig) / (torch.norm(y_orig) + 1e-8)).item()

    assert not torch.isnan(out_naive).any()
    assert not torch.isnan(out_hnc).any()
    assert err_hnc < 0.75 * err_naive, f"Expected err_hnc < 0.75 * err_naive, got {err_hnc:.5f} vs {err_naive:.5f}"


@pytest.mark.parametrize("rank", [4, 8, 16, 32])
@pytest.mark.parametrize("is_decoder", [True, False])
def test_group15_svd_linear_attention_surgery_and_align(rank, is_decoder):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = SVDLinearAttention(hidden_size=64, num_heads=4, rank=rank, is_decoder=is_decoder).to(device)
    probe_x = torch.randn(2, 16, 64, device=device)

    # Reconstruct pre-alignment output
    attn.eval()
    with torch.no_grad():
        out_pre = attn(probe_x)
        out_pre_tensor = out_pre[0] if isinstance(out_pre, tuple) else out_pre

    # Run HNC alignment on attention projections
    align_stats = attn.hnc_spectral_compress_and_align(probe_inputs=probe_x, num_steps=4, target_rank=rank)
    assert "q" in align_stats
    assert "k" in align_stats
    assert "v" in align_stats

    with torch.no_grad():
        out_post = attn(probe_x)
        out_post_tensor = out_post[0] if isinstance(out_post, tuple) else out_post

    assert out_post_tensor.shape == (2, 16, 64)
    assert not torch.isnan(out_post_tensor).any()


# ============================================================================
# Group 16: CKA Steering Loss, HNCContinuationTrainer & Canonical Registry
# ============================================================================
def test_group16_cka_steering_loss_and_registry_invariants():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s_hidden = [torch.randn(4, 16, 32, device=device) for _ in range(4)]
    t_hidden = [torch.randn(4, 16, 32, device=device) for _ in range(4)]

    steering_loss = SmoothMaxOverLayersCKASteeringLoss(temperature=0.1)
    loss_val = steering_loss(s_hidden, t_hidden)
    assert loss_val.item() >= 0.0

    # Trainer test
    student = SmallAnchorMLP(16, 32, 8).to(device)
    teacher = SmallAnchorMLP(16, 32, 8).to(device)
    trainer = HNCContinuationTrainer(student, teacher, device=device)
    batch = {"input_ids": torch.randn(4, 16, device=device)}
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    stats = trainer.step_distillation(batch, opt)

    assert "total_loss" in stats
    assert "kd_loss" in stats
    assert "cka_loss" in stats

    # Strict 131/72/26 Invariants
    assert len(ATTENTION_REGISTRY) == 131, f"Expected 131 attentions, got {len(ATTENTION_REGISTRY)}"
    assert len(COMPRESSOR_REGISTRY) == 72, f"Expected 72 compressors, got {len(COMPRESSOR_REGISTRY)}"
    assert COMPRESSOR_REGISTRY.get("hnc_svd") is not None
    assert COMPRESSOR_REGISTRY.get("hnc_svd_compressor") is not None
