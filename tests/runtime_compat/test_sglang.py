#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration tests for SGLang serving runtime compatibility (deferred to GPU).
"""

import importlib.util
import pytest
import torch

from neuralops.export.runtime_compat import check_compatibility


@pytest.mark.gpu
@pytest.mark.sglang
def test_e3_sglang_runtime_compatibility():
    """Verify SGLang compatibility on GPU host (skips cleanly on CPU/macOS)."""
    if importlib.util.find_spec("sglang") is None:
        pytest.skip("SGLang library not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available for SGLang")

    report = check_compatibility(".", engines=["sglang"])
    assert report.results["sglang"].status in ["SUCCESS", "SKIPPED"]
