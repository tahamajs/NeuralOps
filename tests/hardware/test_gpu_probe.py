#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Hardware test requiring physical CUDA GPU device."""

import pytest
import torch


@pytest.mark.gpu
def test_gpu_hardware_availability():
    """Verify physical CUDA device is available and functional."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    device = torch.device("cuda:0")
    x = torch.randn(128, 128, device=device)
    y = x @ x.T
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()
