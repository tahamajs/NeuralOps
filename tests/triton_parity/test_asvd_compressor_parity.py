#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test for ASVD compressor Triton kernel against PyTorch reference.
"""

import pytest
import torch


@pytest.mark.gpu
def test_asvd_compressor_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from neuralops.compression.svd.aa_svd_compressor import AASVDCompressor
    from neuralops.ops.triton.low_rank import asvd_compress

    torch.manual_seed(42)
    W = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    X = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    cov = torch.matmul(X.t(), X) / 128.0

    with torch.no_grad():
        linear = torch.nn.Linear(256, 256, bias=False, device="cuda")
        linear.weight.data.copy_(W)
        compressor = AASVDCompressor(linear, rank=32)
        svd_linear, _, _ = compressor.compress_linear(linear, rank=32, cov=cov)
        ref_out = torch.matmul(svd_linear.U, svd_linear.V)
        tri_out = asvd_compress(W, cov=cov, rank=32)

    ref_err = ((W - ref_out).norm() / W.norm()).item()
    tri_err = ((W - tri_out).norm() / W.norm()).item()

    assert abs(tri_err - ref_err) <= 1e-3, (
        f"asvd Triton reconstruction error {tri_err:.6f} "
        f"differs from PyTorch {ref_err:.6f} by more than 1e-3"
    )

    assert not torch.isnan(tri_out).any(), "asvd output contains NaN"
    assert not torch.isinf(tri_out).any(), "asvd output contains Inf"
