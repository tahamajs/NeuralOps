#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test for Wanda pruning compressor Triton kernel against PyTorch reference.
"""

import pytest
import torch


@pytest.mark.gpu
def test_wanda_compressor_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from neuralops.ops.triton.prune_wanda import _wanda_pytorch, wanda_compress

    torch.manual_seed(42)
    W = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    act_norm = torch.rand(256, device="cuda", dtype=torch.float32) + 0.1

    with torch.no_grad():
        ref_out = _wanda_pytorch(W, sparsity=0.5, act_norm=act_norm)
        tri_out = wanda_compress(W, sparsity=0.5, act_norm=act_norm)

    ref_mask = ref_out != 0
    tri_mask = tri_out != 0
    mask_match = (ref_mask == tri_mask).all().item()
    assert mask_match, "Wanda pruning mask does not match reference"

    ref_err = ((W - ref_out).norm() / W.norm()).item()
    tri_err = ((W - tri_out).norm() / W.norm()).item()

    assert abs(tri_err - ref_err) <= 1e-5, (
        f"Wanda Triton reconstruction error {tri_err:.6f} "
        f"differs from PyTorch {ref_err:.6f} by more than 1e-5"
    )

    assert not torch.isnan(tri_out).any(), "Wanda output contains NaN"
    assert not torch.isinf(tri_out).any(), "Wanda output contains Inf"
