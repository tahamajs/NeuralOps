#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Multi-Backend Serving and Model Export Verification Suite.
Validates safetensors serialization, HuggingFace Transformers GPU loading,
and vLLM/SGLang engine inference compliance.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from neuralops.export.exporter import export_model
from neuralops.integrations.hf_automodel import patch_model_attention


@pytest.mark.gpu
def test_export_safetensors_and_load_hf_gpu():
    """Verify exporting a patched model to safetensors and reloading on GPU."""
    device = torch.device("cuda:0")

    config = BertConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=1,
        vocab_size=100,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = BertModel(config).to(device=device)
    patch_model_attention(model, attention_type="cosformer")
    model.eval()

    # Run forward pass before export
    x = torch.randint(0, 100, (1, 16), device=device)
    with torch.no_grad():
        out_before = model(x).last_hidden_state

    with tempfile.TemporaryDirectory() as tmpdir:
        safetensors_path = Path(tmpdir) / "model.safetensors"
        export_model(model, safetensors_path, format="safetensors")

        assert safetensors_path.exists(), "Safetensors export file or package was not created on disk."
        safetensors_file = safetensors_path / "model.safetensors" if safetensors_path.is_dir() else safetensors_path
        assert safetensors_file.exists(), f"Safetensors binary weight file not found at {safetensors_file}"
        assert safetensors_file.stat().st_size > 0, "Safetensors binary file is empty."

        # Reload weights into a fresh model instance on GPU
        reloaded_model = BertModel(config).to(device=device)
        patch_model_attention(reloaded_model, attention_type="cosformer")

        try:
            from safetensors.torch import load_file
            try:
                state_dict = load_file(str(safetensors_file), device="cuda:0")
            except Exception:
                state_dict = load_file(str(safetensors_file), device="cpu")
                state_dict = {k: v.to(device) for k, v in state_dict.items()}
            reloaded_model.load_state_dict(state_dict)
        except Exception as exc:
            pytest.fail(f"Failed to reload exported safetensors on GPU: {exc}")

        reloaded_model.eval()
        with torch.no_grad():
            out_after = reloaded_model(x).last_hidden_state

        max_diff = (out_before - out_after).abs().max().item()
        assert max_diff <= 1e-4, f"Reloaded safetensors output mismatch: max_diff={max_diff:.6f}"


@pytest.mark.gpu
def test_vllm_serving_engine_inference():
    """Verify vLLM serving engine integration if installed, otherwise skip with explicit reason."""
    try:
        import vllm  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pytest.skip("vLLM serving engine is not installed in the current environment.")

    # If vLLM is installed, verify plugin integration hook
    from neuralops.integrations.vllm import register_neuralops_vllm_ops
    try:
        register_neuralops_vllm_ops()
    except Exception as exc:
        pytest.fail(f"vLLM operator registration failed: {exc}")


@pytest.mark.gpu
def test_sglang_serving_engine_inference():
    """Verify SGLang serving engine integration if installed, otherwise skip with explicit reason."""
    try:
        import sglang  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pytest.skip("SGLang serving engine is not installed in the current environment.")

    from neuralops.integrations.sglang import register_neuralops_sglang_ops
    try:
        register_neuralops_sglang_ops()
    except Exception as exc:
        pytest.fail(f"SGLang operator registration failed: {exc}")
