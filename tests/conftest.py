# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Shared pytest fixtures for neuralops unified test suite."""
from __future__ import annotations

import os
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
    reason="Requires CUDA"
)

requires_fp16_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="Requires CUDA with fp16/bf16 support"
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: skip if no CUDA")
    config.addinivalue_line("markers", "requires_fp16_cuda: skip if no fp16 CUDA")
