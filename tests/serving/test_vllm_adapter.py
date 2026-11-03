#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for vLLM Serving Adapter (Fail-Closed Contract).
"""

import pytest
from neuralops.core.errors import IntegrationNotImplementedError
import neuralops.integrations.vllm as vllm_integration
from neuralops.integrations.vllm import VLLMAdapter, HAS_VLLM


def test_vllm_adapter_fail_closed_or_minimal_generation():
    """Verify VLLMAdapter fails closed with IntegrationNotImplementedError when vllm is missing."""
    if not HAS_VLLM:
        with pytest.raises(IntegrationNotImplementedError) as exc_info:
            _ = VLLMAdapter()
        assert "vLLM is not installed" in str(exc_info.value)
    else:
        adapter = VLLMAdapter()
        assert adapter is not None
