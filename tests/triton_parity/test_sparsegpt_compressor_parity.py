#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test for SparseGPT pruning compressor Triton kernel against PyTorch reference.
"""

import pytest
import torch


@pytest.mark.gpu
def test_sparsegpt_compressor_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from neuralops.ops.triton.prune_sparsegpt import _sparsegpt_pytorch, sparsegpt_compress

    torch.manual_seed(42)
    W = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    diag_h_inv = torch.rand(256, device="cuda", dtype=torch.float32) + 0.1

    with torch.no_grad():
        ref_out = _sparsegpt_pytorch(W, sparsity=0.5, diag_h_inv=diag_h_inv)
        tri_out = sparsegpt_compress(W, sparsity=0.5, diag_h_inv=diag_h_inv)

    ref_mask = ref_out != 0
    tri_mask = tri_out != 0
    mask_match = (ref_mask == tri_mask).all().item()
    assert mask_match, "SparseGPT pruning mask does not match reference"

    ref_err = ((W - ref_out).norm() / W.norm()).item()
    tri_err = ((W - tri_out).norm() / W.norm()).item()

    assert abs(tri_err - ref_err) <= 1e-5, (
        f"SparseGPT Triton reconstruction error {tri_err:.6f} "
        f"differs from PyTorch {ref_err:.6f} by more than 1e-5"
    )

    assert not torch.isnan(tri_out).any(), "SparseGPT output contains NaN"
    assert not torch.isinf(tri_out).any(), "SparseGPT output contains Inf"
