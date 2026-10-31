#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test for ADMM 2:4 structured pruning compressor Triton kernel against PyTorch reference.
"""

import pytest
import torch


@pytest.mark.gpu
def test_admm_pruner_compressor_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from neuralops.ops.triton.prune_admm import _admm_pytorch, admm_pruner_compress

    torch.manual_seed(42)
    W = torch.randn(128, 256, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        ref_out = _admm_pytorch(W, dim=1)
        tri_out = admm_pruner_compress(W, dim=1)

    ref_mask = ref_out != 0
    tri_mask = tri_out != 0
    mask_match = (ref_mask == tri_mask).all().item()
    assert mask_match, "ADMM pruning mask does not match reference"

    # Verify strict 2:4 structured sparsity pattern
    nonzeros_per_block = (tri_out.reshape(-1, 4) != 0).sum(dim=1)
    assert (nonzeros_per_block == 2).all().item(), "ADMM does not satisfy strict 2:4 sparsity"

    ref_err = ((W - ref_out).norm() / W.norm()).item()
    tri_err = ((W - tri_out).norm() / W.norm()).item()

    assert abs(tri_err - ref_err) <= 1e-5, (
        f"ADMM Triton reconstruction error {tri_err:.6f} "
        f"differs from PyTorch {ref_err:.6f} by more than 1e-5"
    )

    assert not torch.isnan(tri_out).any(), "ADMM output contains NaN"
    assert not torch.isinf(tri_out).any(), "ADMM output contains Inf"
