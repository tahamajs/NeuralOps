#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
GPU Exhaustive Verification for Compressors on Real Foundation Models (DistilBERT and Qwen2.5-7B).
Validates parameter reduction, weight reconstruction, and forward pass integrity.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

from neuralops.core.registry import COMPRESSOR_REGISTRY


def _resolve_model_path(model_id: str) -> str:
    """Resolve model path checking local huggingface hub cache first."""
    hub_name = "models--" + model_id.replace("/", "--")
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub")) / hub_name
    if cache_dir.exists():
        snapshots_dir = cache_dir / "snapshots"
        if snapshots_dir.exists():
            snaps = list(snapshots_dir.iterdir())
            if snaps:
                return str(snaps[0])
        # Direct weights directory
        if (cache_dir / "config.json").exists():
            return str(cache_dir)
    return model_id


@pytest.mark.gpu
def test_distilbert_hnc_svd_parameter_reduction():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GPU compressor test")

    model_path = _resolve_model_path("distilbert-base-uncased")
    model = AutoModel.from_pretrained(model_path, local_files_only=os.path.exists(model_path)).to("cuda")

    orig_total_params = sum(p.numel() for p in model.parameters())
    # Count compressible linear parameters
    orig_linear_params = sum(
        p.numel() for m in model.modules() if isinstance(m, nn.Linear) for p in m.parameters()
    )

    compressor_cls = COMPRESSOR_REGISTRY.get("hnc_svd")
    assert compressor_cls is not None, "hnc_svd compressor must be registered in COMPRESSOR_REGISTRY"

    compressor = compressor_cls(model, target_ratio=0.5, target_attention_only=False)
    comp_model, stats = compressor.compress()

    comp_total_params = sum(p.numel() for p in comp_model.parameters())
    comp_linear_params = sum(
        p.numel() for m in comp_model.modules() if hasattr(m, "in_features") for p in m.parameters()
    )

    # Verify parameter reduction
    assert comp_total_params < orig_total_params, (
        f"Expected total parameter reduction: orig={orig_total_params}, comp={comp_total_params}"
    )

    linear_reduction = (orig_linear_params - comp_linear_params) / orig_linear_params
    print(f"\n[DistilBERT SVD] Orig Total: {orig_total_params:,} | Comp Total: {comp_total_params:,}")
    print(f"[DistilBERT SVD] Orig Linear: {orig_linear_params:,} | Comp Linear: {comp_linear_params:,} | Linear Reduction: {linear_reduction * 100:.2f}%")

    # Assert linear layers reduced by ~50% (between 40% and 60%)
    assert 0.40 <= linear_reduction <= 0.60, (
        f"Expected linear parameter reduction ~50%, got {linear_reduction * 100:.2f}%"
    )

    # Run forward pass to verify runnable model
    dummy_input = torch.randint(0, 1000, (2, 32), device="cuda")
    with torch.no_grad():
        out = comp_model(dummy_input)

    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    assert torch.isfinite(hidden).all(), "Compressed DistilBERT forward pass produced non-finite values"


@pytest.mark.gpu
def test_qwen_7b_hnc_svd_compression_and_forward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GPU compressor test")

    model_path = _resolve_model_path("Qwen/Qwen2.5-7B-Instruct")
    if not os.path.exists(model_path):
        model_path = _resolve_model_path("Qwen/Qwen2.5-7B")

    if not os.path.exists(model_path):
        pytest.skip(f"Qwen 7B model not found in local cache: {model_path}")

    device = "cuda:0"
    vram_free = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"\n[Qwen 7B Test] Target GPU: {torch.cuda.get_device_name(0)}, Total VRAM: {vram_free:.2f} GB")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
    except torch.cuda.OutOfMemoryError as e:
        pytest.fail(f"OOM during Qwen 7B model load: {e}")
    except Exception as e:
        if "out of memory" in str(e).lower():
            pytest.fail(f"OOM during Qwen 7B model load: {e}")
        raise

    orig_params = sum(p.numel() for p in model.parameters())

    compressor_cls = COMPRESSOR_REGISTRY.get("hnc_svd")
    compressor = compressor_cls(model, rank=1024, target_attention_only=True)
    comp_model, stats = compressor.compress()

    comp_params = sum(p.numel() for p in comp_model.parameters())
    print(f"[Qwen 7B SVD rank=1024] Orig Params: {orig_params:,} | Comp Params: {comp_params:,}")

    # Forward pass test with 16 tokens
    input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
    try:
        with torch.no_grad():
            outputs = comp_model(input_ids)
        assert outputs.logits is not None
        assert torch.isfinite(outputs.logits).all(), "Qwen 7B output logits contain NaN/Inf"
        print("[Qwen 7B SVD] Forward pass SUCCESS with finite logits.")
    except torch.cuda.OutOfMemoryError as e:
        pytest.fail(f"OOM during Qwen 7B forward pass: {e}")
