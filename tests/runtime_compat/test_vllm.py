#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration tests for vLLM serving runtime compatibility (deferred to GPU).
"""

import importlib.util
import pytest
import torch

from neuralops.export.runtime_compat import check_compatibility


@pytest.mark.gpu
@pytest.mark.vllm
def test_e3_vllm_runtime_compatibility():
    """Verify vLLM compatibility on GPU host (skips cleanly on CPU/macOS)."""
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM library not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available for vLLM")

    # This runs only on the remote GPU host
    report = check_compatibility(".", engines=["vllm"])
    assert report.results["vllm"].status in ["SUCCESS", "SKIPPED"]
