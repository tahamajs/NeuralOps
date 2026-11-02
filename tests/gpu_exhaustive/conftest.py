#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Shared fixtures and hardware verification guards for the GPU Exhaustive test suite.
"""

from __future__ import annotations

import os
import gc
import pytest
import torch


@pytest.fixture(autouse=True)
def require_gpu():
    """Ensure tests in this suite only run on genuine CUDA GPU accelerators."""
    if not torch.cuda.is_available():
        pytest.skip("GPU Exhaustive test suite requires CUDA hardware acceleration.")
    torch.cuda.empty_cache()
    gc.collect()
    yield
    torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture
def gpu_device() -> torch.device:
    """Return active CUDA device."""
    return torch.device("cuda:0")


@pytest.fixture(params=[torch.float32, torch.float16])
def fp_dtype(request) -> torch.dtype:
    """Parametrized floating point dtypes supported across CUDA hardware."""
    return request.param
