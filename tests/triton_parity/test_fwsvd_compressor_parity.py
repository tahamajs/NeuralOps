#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test for FWSVD compressor Triton kernel against PyTorch reference.
"""

import pytest
import torch


@pytest.mark.gpu
def test_fwsvd_compressor_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from neuralops.compression.svd.fwsvd_compressor import FWSVDCompressor
    from neuralops.ops.triton.low_rank import fwsvd_compress

    torch.manual_seed(42)
    W = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    fisher_diag = torch.abs(torch.randn(256, device="cuda", dtype=torch.float32)) + 0.1

    with torch.no_grad():
        linear = torch.nn.Linear(256, 256, bias=False, device="cuda")
        linear.weight.data.copy_(W)
        compressor = FWSVDCompressor(linear, rank=32)
        svd_linear, _, _ = compressor.compress_linear(linear, rank=32, fisher_diag=fisher_diag)
        ref_out = torch.matmul(svd_linear.U, svd_linear.V)
        tri_out = fwsvd_compress(W, fisher_diag=fisher_diag, rank=32)

    ref_err = ((W - ref_out).norm() / W.norm()).item()
    tri_err = ((W - tri_out).norm() / W.norm()).item()

    assert abs(tri_err - ref_err) <= 1e-3, (
        f"fwsvd Triton reconstruction error {tri_err:.6f} "
        f"differs from PyTorch {ref_err:.6f} by more than 1e-3"
    )

    assert not torch.isnan(tri_out).any(), "fwsvd output contains NaN"
    assert not torch.isinf(tri_out).any(), "fwsvd output contains Inf"
