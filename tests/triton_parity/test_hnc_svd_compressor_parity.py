#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test for HNC-SVD compressor Triton kernel against PyTorch reference.
"""

import pytest
import torch


@pytest.mark.gpu
def test_hnc_svd_compressor_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from neuralops.ops.triton.low_rank import hnc_svd_compress

    torch.manual_seed(42)
    W = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    X = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    cov = torch.matmul(X.t(), X) / 128.0

    # Reference implementation following HNC-SVD Cholesky whitening pullback
    with torch.no_grad():
        reg = 1e-3 * (torch.trace(cov) / 256.0).item() + 1e-6
        Sx = cov + reg * torch.eye(256, device="cuda", dtype=torch.float32)
        L = torch.linalg.cholesky(Sx)
        W_whitened = torch.matmul(W, L)
        U, S, Vh = torch.linalg.svd(W_whitened, full_matrices=False)
        U_r = U[:, :32] * torch.sqrt(S[:32]).unsqueeze(0)
        V_unwhitened = torch.linalg.solve_triangular(
            L.t(), (torch.sqrt(S[:32]).unsqueeze(1) * Vh[:32, :]).t(), upper=True
        ).t()
        ref_out = torch.matmul(U_r, V_unwhitened)

        tri_out = hnc_svd_compress(W, cov=cov, rank=32)

    ref_err = ((W - ref_out).norm() / W.norm()).item()
    tri_err = ((W - tri_out).norm() / W.norm()).item()

    assert abs(tri_err - ref_err) <= 1e-3, (
        f"hnc_svd Triton reconstruction error {tri_err:.6f} "
        f"differs from PyTorch {ref_err:.6f} by more than 1e-3"
    )

    assert not torch.isnan(tri_out).any(), "hnc_svd output contains NaN"
    assert not torch.isinf(tri_out).any(), "hnc_svd output contains Inf"
