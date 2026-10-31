#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Mathematical Conditioning and Fisher Information Implementations:
1. FWSVDCompressor (empirical Fisher information, activation covariance, adaptive Tikhonov damping).
2. AdaSVDCompressor (eigenvalue spectrum regularization and ALS numerical divergence safeguarding).
3. SVDLLMCompressor (robust Cholesky factorization with eigenvalue clipping and SVD square root fallback).
"""

import math
import pytest
import torch
import torch.nn as nn

from neuralops.compression.low_rank.svd_llm_compressor import SVDLLMCompressor
from neuralops.compression.svd.adasvd_compressor import AdaSVDCompressor
from neuralops.compression.svd.fwsvd_compressor import FWSVDCompressor


class SimpleLinearNet(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=True)
        self.v_proj = nn.Linear(out_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_proj(self.q_proj(x))


def test_fwsvd_supplied_gradients_outer_product():
    """Verify FWSVD correctly ingests supplied gradients to construct empirical Fisher Information."""
    torch.manual_seed(42)
    in_f, out_f, r = 32, 32, 8
    lin = nn.Linear(in_f, out_f, bias=False)

    M = 16  # M < in_f (rank deficient)
    grads = torch.randn(M, in_f)
    expected_F = (grads.t() @ grads) / float(M)

    comp = FWSVDCompressor(nn.Sequential(lin), rank=r, target_attention_only=False)
    svd_layer, err, actual_r = comp.compress_linear(
        lin, rank=r, fisher_matrix=expected_F
    )

    assert svd_layer is not None
    assert actual_r == r
    assert math.isfinite(err)
    assert err < 1.0


def test_fwsvd_activation_covariance_with_tikhonov_damping():
    """Verify FWSVD computes damped full covariance when sample count M << d_in."""
    torch.manual_seed(42)
    in_f, out_f, r = 64, 64, 16
    model = SimpleLinearNet(in_f, out_f)

    # 8 samples for 64-dim hidden dimension (severe nullspace of dimension 56)
    calib_x = torch.randn(8, in_f)

    comp = FWSVDCompressor(
        model,
        rank=r,
        damping=1e-3,
        use_gradients=False,
        target_attention_only=True,
    )
    compressed_model, stats = comp.compress(calibration_data=calib_x)

    assert stats["compressed_layers"] > 0
    for name, F_mat in comp.fisher_matrices.items():
        assert F_mat.shape == (in_f, in_f)
        # Verify strictly positive eigenvalues from Tikhonov damping
        eigvals = torch.linalg.eigvalsh(F_mat)
        assert (eigvals > 0).all(), f"Eigenvalues of {name} covariance must be strictly positive"
        min_eig = eigvals.min().item()
        assert min_eig >= 1e-6, f"Minimum eigenvalue {min_eig} too small; Tikhonov damping failed"


def test_fwsvd_compress_linear_parity_with_unweighted():
    """Verify compress_linear produces valid low-rank factors with both 2D and 1D Fisher weighting."""
    torch.manual_seed(42)
    in_f, out_f, r = 48, 48, 12
    lin = nn.Linear(in_f, out_f)
    comp = FWSVDCompressor(nn.Sequential(lin), rank=r, target_attention_only=False)

    # 1D diagonal Fisher
    f_diag = torch.ones(in_f) * 2.0
    svd_diag, err_diag, _ = comp.compress_linear(lin, rank=r, fisher_diag=f_diag)
    assert svd_diag is not None
    assert math.isfinite(err_diag)

    # 2D full Fisher
    F_mat = torch.eye(in_f) * 2.0
    svd_mat, err_mat, _ = comp.compress_linear(lin, rank=r, fisher_matrix=F_mat)
    assert svd_mat is not None
    assert math.isfinite(err_mat)
    # When Fisher is proportional to identity, diagonal and 2D matrix weighting yield identical error
    assert abs(err_diag - err_mat) < 1e-4


def test_adasvd_spectrum_regularization():
    """Verify AdaSVD eigenvalue spectrum regularization bounds condition number."""
    torch.manual_seed(42)
    d = 64
    # Create rank-deficient covariance with rank 4
    U = torch.randn(d, 4)
    C_ill = U @ U.t()

    C_reg = AdaSVDCompressor._regularize_covariance_spectrum(C_ill, damping_factor=1e-4)
    assert C_reg.shape == (d, d)
    assert torch.allclose(C_reg, C_reg.t(), atol=1e-5)

    eigvals = torch.linalg.eigvalsh(C_reg)
    assert (eigvals > 0).all(), "Regularized covariance must be strictly positive-definite"
    cond_num = (eigvals.max() / eigvals.min()).item()
    assert cond_num <= 1e7, f"Condition number {cond_num} exceeds regularized bound"


def test_adasvd_als_divergence_safeguard_on_singular_covariance():
    """Verify ALS solver does not diverge or produce NaN/Inf on rank-deficient covariance."""
    torch.manual_seed(42)
    in_f, out_f, r = 64, 64, 16
    W = torch.randn(out_f, in_f)
    D_var = torch.ones(out_f)

    # Extremely ill-conditioned rank-2 covariance
    X_narrow = torch.randn(10, 2)
    P = torch.randn(2, in_f)
    X = X_narrow @ P
    C_singular = (X.t() @ X) / 10.0

    comp = AdaSVDCompressor(nn.Sequential(nn.Linear(in_f, out_f)), rank=r, target_attention_only=False)

    A_als, B_als, err_als, iters = comp._two_sided_als(
        W, r=r, C_X=C_singular, D_var=D_var, X_calib=X, num_iters=10
    )

    assert torch.isfinite(A_als).all(), "A factors contained NaN or Inf"
    assert torch.isfinite(B_als).all(), "B factors contained NaN or Inf"
    assert math.isfinite(err_als)
    assert err_als < 2.0


def test_svd_llm_cholesky_eigenvalue_clipping_and_svd_fallback():
    """Verify SVD-LLM robust Cholesky succeeds on non-positive definite / rank-deficient covariance."""
    torch.manual_seed(42)
    n = 32
    lin = nn.Linear(n, n)
    comp = SVDLLMCompressor(nn.Sequential(lin), rank_ratio=0.5, target_attention_only=False)

    # Rank-deficient covariance (rank 3)
    X_rank3 = torch.randn(64, 3) @ torch.randn(3, n)
    C_rank3 = (X_rank3.t() @ X_rank3) / 64.0

    # Cholesky with eigenvalue clipping
    S = comp._robust_cholesky(C_rank3, n)
    assert S.shape == (n, n)
    assert torch.isfinite(S).all()
    # S @ S.T must reconstruct a valid positive semi-definite matrix
    recon_C = S @ S.t()
    assert torch.isfinite(recon_C).all()
    assert (torch.linalg.eigvalsh(recon_C) >= -1e-6).all()

    # Intentionally corrupted covariance with negative eigenvalues
    C_corrupt = C_rank3.clone()
    C_corrupt[0, 0] = -10.0
    S_corrupt = comp._robust_cholesky(C_corrupt, n)
    assert S_corrupt.shape == (n, n)
    assert torch.isfinite(S_corrupt).all()
