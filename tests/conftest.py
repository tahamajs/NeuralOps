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
    sys.path.append(str(_TESTS_DIR))

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
    """Lightweight real GPT2 model for testing, requiring transformers."""
    pytest.importorskip("transformers", reason="transformers package required for tiny_gpt2 fixture")
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=256,
        n_positions=512,
    )
    return GPT2LMHeadModel(config)


@pytest.fixture
def tiny_gpt2_dummy():
    """Explicit hand-made mock GPT2 model for lightweight testing without transformers."""

    class TinyGPT2Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type(
                "Config",
                (),
                {
                    "hidden_size": 64,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 2,
                    "vocab_size": 256,
                    "n_positions": 512,
                    "model_type": "gpt2",
                },
            )()
            self.transformer = torch.nn.ModuleDict(
                {
                    "wte": torch.nn.Embedding(256, 64),
                    "wpe": torch.nn.Embedding(512, 64),
                    "h": torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(2)]),
                }
            )
            self.lm_head = torch.nn.Linear(64, 256, bias=False)

        def forward(self, input_ids, **kwargs):
            x = self.transformer["wte"](input_ids)
            for layer in self.transformer["h"]:
                x = layer(x)
            logits = self.lm_head(x)
            return type("Output", (), {"logits": logits, "loss": torch.tensor(0.0)})()

    return TinyGPT2Dummy()


@pytest.fixture
def tiny_bert():
    """Lightweight real or synthetic BERT model for testing."""
    try:
        from transformers import BertConfig, BertModel

        config = BertConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
            vocab_size=256,
            max_position_embeddings=512,
        )
        return BertModel(config)
    except Exception:
        class TinyBertDummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type(
                    "Config",
                    (),
                    {
                        "hidden_size": 64,
                        "num_attention_heads": 2,
                        "num_hidden_layers": 2,
                        "vocab_size": 256,
                        "max_position_embeddings": 512,
                        "model_type": "bert",
                    },
                )()
                self.embeddings = torch.nn.Embedding(256, 64)
                self.encoder = torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(2)])

            def forward(self, input_ids, **kwargs):
                x = self.embeddings(input_ids)
                for layer in self.encoder:
                    x = layer(x)
                return type("Output", (), {"last_hidden_state": x})()

        return TinyBertDummy()


@pytest.fixture
def tiny_llama():
    """Lightweight real or synthetic LLaMA model for testing."""
    try:
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=256,
            max_position_embeddings=512,
        )
        return LlamaForCausalLM(config)
    except Exception:
        class TinyLlamaDummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type(
                    "Config",
                    (),
                    {
                        "hidden_size": 64,
                        "intermediate_size": 128,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 2,
                        "vocab_size": 256,
                        "max_position_embeddings": 512,
                        "model_type": "llama",
                    },
                )()
                self.model = torch.nn.ModuleDict(
                    {
                        "embed_tokens": torch.nn.Embedding(256, 64),
                        "layers": torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(2)]),
                    }
                )
                self.lm_head = torch.nn.Linear(64, 256, bias=False)

            def forward(self, input_ids, **kwargs):
                x = self.model["embed_tokens"](input_ids)
                for layer in self.model["layers"]:
                    x = layer(x)
                logits = self.lm_head(x)
                return type("Output", (), {"logits": logits, "loss": torch.tensor(0.0)})()

        return TinyLlamaDummy()


@pytest.fixture
def mock_cuda_device(monkeypatch):
    """Mocks torch.cuda environment and device properties for testing without physical GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    class MockDeviceProp:
        name = "NVIDIA Mock Test GPU"
        total_memory = 16 * 1024 * 1024 * 1024
        major = 8
        minor = 0
        multi_processor_count = 108
        L2_cache_size = 4 * 1024 * 1024

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args, **kwargs: MockDeviceProp())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args, **kwargs: 2048 * 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args, **kwargs: 2048 * 1024)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    return torch.device("cuda:0")


@pytest.fixture
def mock_hf_api(monkeypatch):
    """Mocks Hugging Face API calls to prevent any remote network access during tests."""
    from unittest.mock import MagicMock

    mock_api = MagicMock()
    mock_api.upload_file.return_value = "https://huggingface.co/mock/repo/commit"
    mock_api.create_repo.return_value = "https://huggingface.co/mock/repo"
    mock_api.model_info.return_value = MagicMock(sha="mocksha1234567890abcdef1234567890abcdef")
    try:
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "HfApi", lambda *args, **kwargs: mock_api)
        monkeypatch.setattr(huggingface_hub, "create_repo", lambda *args, **kwargs: "https://huggingface.co/mock/repo")
        monkeypatch.setattr(huggingface_hub, "upload_file", lambda *args, **kwargs: "https://huggingface.co/mock/repo/commit")
    except ImportError:
        pass
    return mock_api


@pytest.fixture
def isolated_temp_ledger(tmp_path):
    """Provides an isolated ResultsLedger in a clean temporary directory."""
    from neuralops.analysis.results_ledger import ResultsLedger

    ledger_root = tmp_path / "test_ledger"
    ledger_root.mkdir(parents=True, exist_ok=True)
    return ResultsLedger(root_dir=ledger_root)


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
    import neuralops.attentions  # noqa: F401
    from neuralops.core.registry import ATTENTION_REGISTRY

    return ATTENTION_REGISTRY


@pytest.fixture(scope="session")
def compressor_registry():
    import neuralops.compression_engine  # noqa: F401
    from neuralops.core.registry import COMPRESSOR_REGISTRY

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
        # Don't classify static config/metadata verification tests as GPU tests
        is_static_config_test = any(
            cfg_mod in item_path
            for cfg_mod in (
                "test_version_sync",
                "test_repo_config",
                "test_dependency",
                "test_ops_triton_imports",
                "test_cuda_bindings",
                "test_cuda_catalog",
                "test_kernels_lazy",
                "test_config_single_source",
                "test_terminal_ui_single_source",
                "test_executor_pipeline_transform_canonical",
                "test_perplexity_single_gate",
                "test_cli_main",
                "test_standalone_cli",
                "test_results_ledger_single_owner",
                "test_monitor_single_server",
                "test_serving_canonical",
                "test_shard_placeholder",
                "test_cli_cuda_visible_devices",
                "test_changelog_sync",
                "test_notebook_catalog",
                "test_hardware_probe",
                "test_examples_smoke",
                "test_readme_truth",
                "test_logging_policy",
                "test_adapters_unified",
                "test_compare_cli",
                "test_import_boundaries",
                "test_config_validate_device",
                "test_dispatcher_leases",
                "test_flush_l2_dynamic",
                "test_memory_fragmentation",
                "test_cuda_ops_no_fake",
                "test_alibi_attention",
                "test_v2_migration",
                "test_triton_ops_all_kernels",
                "test_triton_sample",
                "test_timing_correct",
                "test_cli_cuda_visible_devices",
                "test_shared_fixtures",
            )
        ) or ("test_notebook_structure_and_syntax" in item_name or "test_notebook_smoke" in item_name)
        if ("cuda" in item_path or "cuda" in item.keywords or "cuda" in item_name) and not is_static_config_test:
            item.add_marker(pytest.mark.gpu)
        if ("triton" in item_path or "triton" in item.keywords or "triton" in item_name) and not is_static_config_test:
            item.add_marker(pytest.mark.triton)
            item.add_marker(pytest.mark.gpu)

        if not cuda_available and not is_static_config_test:
            if "gpu" in item.keywords or any(item.iter_markers("gpu")):
                item.add_marker(skip_cuda)
            if "requires_cuda" in item.keywords or any(item.iter_markers("requires_cuda")):
                item.add_marker(skip_cuda)
            if "requires_fp16_cuda" in item.keywords or any(item.iter_markers("requires_fp16_cuda")):
                item.add_marker(skip_cuda)

        if not _TRITON_AVAILABLE and not is_static_config_test:
            if "triton" in item.keywords or any(item.iter_markers("triton")):
                item.add_marker(skip_triton)
