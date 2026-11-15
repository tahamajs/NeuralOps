#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration and Unit Tests for Offline Headless Serving Mocks (vLLM & SGLang).

Verifies:
1. HeadlessVLLMWorker offline execution, paged state table allocation, and generation.
2. get_vllm_runner offline fail-closed vs headless mock dispatch.
3. register_neuralops_vllm_backend and register_neuralops_linear_method offline behavior.
4. NeuralOpsLinearMethod create_weights and apply execution.
5. HeadlessSGLangRunner offline execution, Radix prefix caching, and batch generation.
6. get_sglang_runner offline fail-closed vs headless mock dispatch.
7. register_neuralops_sglang_backend offline behavior with NEURALOPS_MOCK_SERVING.
8. End-to-end integration with PyTorch nn.Module in offline environment.
"""

from __future__ import annotations

import os
from unittest.mock import patch
import pytest
import torch
import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.integrations.vllm import (
    HeadlessVLLMWorker,
    get_vllm_runner,
    is_vllm_available,
    vllm_is_available,
)
from neuralops.integrations.vllm_plugin import (
    NeuralOpsLinearMethod,
    register_neuralops_linear_method,
    register_neuralops_vllm_backend,
)
from neuralops.integrations.sglang import (
    get_sglang_runner,
    sglang_is_available,
)
from neuralops.integrations.sglang_plugin import (
    HeadlessSGLangRunner,
    register_neuralops_sglang_backend,
)


class DummyModel(nn.Module):
    """Tiny mock model for offline execution testing."""

    def __init__(self, vocab_size: int = 128, hidden_size: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.config = type("Config", (), {"vocab_size": vocab_size, "hidden_size": hidden_size, "num_attention_heads": 2})()

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.embedding(input_ids)
        return self.linear(h)


def test_headless_vllm_worker_direct():
    """Verify HeadlessVLLMWorker initializes, generates tokens, and executes model requests."""
    worker = HeadlessVLLMWorker(block_size=16, num_blocks=32)
    assert worker._is_initialized is True

    # Test execute_model with synthetic dictionary
    req = {"input_ids": [10, 20, 30, 40]}
    res = worker.execute_model(req)
    assert res["status"] == "success"
    assert "tokens" in res
    assert res["cached_blocks"] >= 1

    # Test execute_model with tensor
    tensor_req = {"input_ids": torch.tensor([[10, 20, 30, 40]], dtype=torch.long)}
    res2 = worker.execute_model(tensor_req)
    assert res2["status"] == "success"
    assert len(res2["tokens"]) == 1

    # Test generate
    gen_res = worker.generate("test prompt", max_new_tokens=4)
    assert gen_res["status"] == "success"
    assert len(gen_res["generated_tokens"]) == 4
    assert len(gen_res["output_tokens"]) == len(gen_res["prompt_tokens"]) + 4

    worker.shutdown()
    assert worker._is_initialized is False


def test_headless_vllm_worker_with_torch_model():
    """Verify HeadlessVLLMWorker with real PyTorch model."""
    model = DummyModel()
    worker = HeadlessVLLMWorker(model_config=model.config, model=model)

    prompt = [5, 12, 19]
    res = worker.generate(prompt, max_new_tokens=3)
    assert res["status"] == "success"
    assert len(res["generated_tokens"]) == 3
    for tok in res["generated_tokens"]:
        assert isinstance(tok, int)
        assert 0 <= tok < 128


def test_get_vllm_runner_offline_dispatch():
    """Verify get_vllm_runner fail-closed without mock vs succeeding with headless_mock=True."""
    with patch("neuralops.integrations.vllm.HAS_VLLM", False):
        # 1. Fails closed without mock flag or env var
        with pytest.raises(IntegrationNotImplementedError) as exc_info:
            get_vllm_runner()
        assert "vLLM is not installed" in str(exc_info.value)

        # 2. Succeeds when headless_mock=True
        runner = get_vllm_runner(headless_mock=True)
        assert isinstance(runner, HeadlessVLLMWorker)

        # 3. Succeeds when NEURALOPS_MOCK_SERVING=1
        with patch.dict(os.environ, {"NEURALOPS_MOCK_SERVING": "1"}):
            runner2 = get_vllm_runner()
            assert isinstance(runner2, HeadlessVLLMWorker)


def test_vllm_plugin_registration_and_linear_method():
    """Verify register_neuralops_vllm_backend and NeuralOpsLinearMethod."""
    with patch("neuralops.integrations.vllm_plugin.HAS_VLLM", False):
        # Without mock: raises IntegrationNotImplementedError
        with pytest.raises(IntegrationNotImplementedError):
            register_neuralops_vllm_backend()

        # With mock: succeeds
        with patch.dict(os.environ, {"NEURALOPS_MOCK_SERVING": "1"}):
            assert register_neuralops_vllm_backend() is True
            assert register_neuralops_linear_method() is True

    # Test NeuralOpsLinearMethod create_weights & apply
    linear_method = NeuralOpsLinearMethod()
    dummy_layer = nn.Module()
    weights = linear_method.create_weights(
        layer=dummy_layer,
        input_size_per_partition=32,
        output_partition_sizes=[64],
        input_size=32,
        output_size=64,
        params_dtype=torch.float32,
    )
    assert "weight" in weights
    assert dummy_layer.weight.shape == (64, 32)

    # Initialize weights to non-zero for test
    dummy_layer.weight.data.normal_(0.0, 0.02)
    x = torch.randn(2, 4, 32)
    out = linear_method.apply(dummy_layer, x)
    assert out.shape == (2, 4, 64)
    assert torch.isfinite(out).all()

    # With bias
    bias = torch.randn(64)
    out_bias = linear_method.apply(dummy_layer, x, bias=bias)
    assert out_bias.shape == (2, 4, 64)


def test_headless_sglang_runner_direct():
    """Verify HeadlessSGLangRunner initializes, generates, and performs Radix caching."""
    runner = HeadlessSGLangRunner()
    assert runner._is_initialized is True

    # First generation (cold cache)
    prompt = [10, 20, 30]
    res1 = runner.generate(prompt, max_new_tokens=4)
    assert res1["status"] == "success"
    assert len(res1["generated_tokens"]) == 4

    # Second generation with prefix extension (Radix cache hit)
    res2 = runner.generate(res1["output_tokens"], max_new_tokens=2)
    assert res2["status"] == "success"
    assert res2["cached_tokens"] >= len(res1["output_tokens"])

    # Batch execution
    batch_reqs = [{"prompt": "hello"}, {"prompt": "world"}]
    batch_res = runner.execute_batch(batch_reqs)
    assert len(batch_res) == 2
    for r in batch_res:
        assert r["status"] == "success"

    runner.shutdown()
    assert runner._is_initialized is False


def test_headless_sglang_runner_with_torch_model():
    """Verify HeadlessSGLangRunner with PyTorch model."""
    model = DummyModel()
    runner = HeadlessSGLangRunner(model=model)

    res = runner.generate([1, 2, 3], max_new_tokens=3)
    assert res["status"] == "success"
    assert len(res["generated_tokens"]) == 3
    for tok in res["generated_tokens"]:
        assert isinstance(tok, int)
        assert 0 <= tok < 128


def test_get_sglang_runner_offline_dispatch():
    """Verify get_sglang_runner fail-closed without mock vs succeeding with headless_mock=True."""
    with patch("neuralops.integrations.sglang.sglang_is_available", return_value=False):
        # 1. Fails closed without mock flag or env var
        with pytest.raises(IntegrationNotImplementedError) as exc_info:
            get_sglang_runner()
        assert "SGLang is not installed" in str(exc_info.value)

        # 2. Succeeds when headless_mock=True
        runner = get_sglang_runner(headless_mock=True)
        assert isinstance(runner, HeadlessSGLangRunner)

        # 3. Succeeds when NEURALOPS_MOCK_SERVING=1
        with patch.dict(os.environ, {"NEURALOPS_MOCK_SERVING": "1"}):
            runner2 = get_sglang_runner()
            assert isinstance(runner2, HeadlessSGLangRunner)


def test_sglang_plugin_registration_mock():
    """Verify register_neuralops_sglang_backend offline behavior with NEURALOPS_MOCK_SERVING."""
    with patch("neuralops.integrations.sglang_plugin.sglang_is_available", return_value=False):
        # Without mock: raises IntegrationNotImplementedError
        with pytest.raises(IntegrationNotImplementedError):
            register_neuralops_sglang_backend()

        # With mock: succeeds
        with patch.dict(os.environ, {"NEURALOPS_MOCK_SERVING": "1"}):
            assert register_neuralops_sglang_backend() is True
