#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Shared pytest fixtures for neuralops unified test suite."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

if "numpy" in sys.modules:
    import numpy  # noqa: F401

import pytest
import torch

try:
    import triton
    _TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _TRITON_AVAILABLE = False


@pytest.fixture(autouse=True)
def _reset_torch_seed():
    """Reset random seed before each test to ensure determinism."""
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    try:
        import numpy as np
        np.random.seed(42)
    except ImportError:
        pass
    yield


@pytest.fixture
def device():
    """Return available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


@pytest.fixture(params=[torch.float32, torch.float16, torch.bfloat16])
def dtype(request):
    """Parametrized dtype fixture."""
    if request.param == torch.bfloat16:
        if not torch.cuda.is_available():
            pytest.skip("bfloat16 requires CUDA")
        if not torch.cuda.is_bf16_supported():
            pytest.skip("bfloat16 not supported on this GPU")
    return request.param


@pytest.fixture
def dummy_config():
    """Minimal config for testing attention modules."""
    class DummyConfig:
        hidden_size = 64
        num_attention_heads = 4
        num_key_value_heads = 4
        head_dim = 16
        dropout = 0.0
        is_decoder = False
        model_type = "bert"
    return DummyConfig()


@pytest.fixture
def dummy_hidden_states():
    """Random hidden states for forward passes."""
    return torch.randn(2, 16, 64)


@pytest.fixture
def dummy_qkv():
    """Random Q, K, V tensors."""
    B, H, N, D = 2, 4, 16, 16
    return (
        torch.randn(B, H, N, D),
        torch.randn(B, H, N, D),
        torch.randn(B, H, N, D),
    )


@pytest.fixture
def tiny_gpt2():
    """Lightweight GPT2 model for testing."""
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=64,
            vocab_size=256,
            n_positions=512,
        )
        return GPT2LMHeadModel(config)
    except Exception:
        class TinyGPT2Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {
                    "hidden_size": 64,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 2,
                    "vocab_size": 256,
                    "n_positions": 512,
                    "model_type": "gpt2",
                })()
                self.transformer = torch.nn.ModuleDict({
                    "wte": torch.nn.Embedding(256, 64),
                    "wpe": torch.nn.Embedding(512, 64),
                    "h": torch.nn.ModuleList([
                        torch.nn.Linear(64, 64) for _ in range(2)
                    ]),
                })
                self.lm_head = torch.nn.Linear(64, 256, bias=False)

            def forward(self, input_ids, **kwargs):
                x = self.transformer["wte"](input_ids)
                for layer in self.transformer["h"]:
                    x = layer(x)
                logits = self.lm_head(x)
                return type("Output", (), {"logits": logits, "loss": torch.tensor(0.0)})()

        return TinyGPT2Dummy()


@pytest.fixture
def repo_root():
    """Absolute path to repository root."""
    return _PROJECT_ROOT


@pytest.fixture
def tmp_results_dir(tmp_path):
    """Temporary results directory."""
    d = tmp_path / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def attention_registry():
    from neuralops.core.registry import ATTENTION_REGISTRY
    import neuralops.attentions  # noqa: F401
    return ATTENTION_REGISTRY


@pytest.fixture(scope="session")
def compressor_registry():
    from neuralops.core.registry import COMPRESSOR_REGISTRY
    import neuralops.compression_engine  # noqa: F401
    return COMPRESSOR_REGISTRY


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA",
)

requires_fp16_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="Requires CUDA with fp16/bf16 support",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA-capable GPU")
    config.addinivalue_line("markers", "triton: requires Triton compiler/kernels")
    config.addinivalue_line("markers", "slow: marks test as slow-running")
    config.addinivalue_line("markers", "integration: marks test as an integration test")
    config.addinivalue_line("markers", "smoke: smoke test subset")
    config.addinivalue_line("markers", "performance: performance benchmark tests")
    config.addinivalue_line("markers", "regression: regression tests")
    config.addinivalue_line("markers", "requires_cuda: skip if no CUDA")
    config.addinivalue_line("markers", "requires_fp16_cuda: skip if no fp16 CUDA")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    cuda_available = torch.cuda.is_available()
    skip_cuda = pytest.mark.skip(reason="CUDA device not available")
    skip_triton = pytest.mark.skip(reason="Triton compiler not available")

    for item in items:
        item_path = str(item.fspath).lower()
        item_name = item.name.lower()
        if "cuda" in item_path or "cuda" in item.keywords or "cuda" in item_name:
            item.add_marker(pytest.mark.gpu)
        if "triton" in item_path or "triton" in item.keywords or "triton" in item_name:
            item.add_marker(pytest.mark.triton)
            item.add_marker(pytest.mark.gpu)

        if not cuda_available:
            if "gpu" in item.keywords or any(item.iter_markers("gpu")):
                item.add_marker(skip_cuda)
            if "requires_cuda" in item.keywords or any(item.iter_markers("requires_cuda")):
                item.add_marker(skip_cuda)
            if "requires_fp16_cuda" in item.keywords or any(item.iter_markers("requires_fp16_cuda")):
                item.add_marker(skip_cuda)

        if not _TRITON_AVAILABLE:
            if "triton" in item.keywords or any(item.iter_markers("triton")):
                item.add_marker(skip_triton)
