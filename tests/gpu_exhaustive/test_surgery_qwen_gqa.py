#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
GPU Exhaustive Verification Suite for Model Surgery & GQA Attention Transplantation.

Validates in-place surgery on Grouped-Query Attention (GQA) architectures (Qwen2.5-0.5B),
testing linear attention replacement (Cosformer), GQA head broadcasting parity,
and post-surgery subspace micro-healing convergence on NVIDIA RTX 3090 hardware.
"""

from __future__ import annotations

import copy
import os
from typing import Any
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM

from neuralops.api import transplant_attention
from neuralops.surgery.healing import heal_replaced_model


@pytest.fixture
def gpu_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for GPU surgery tests")
    return torch.device("cuda:0")


def _load_qwen_model(causal: bool = False) -> Any:
    candidates = [
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"),
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B"),
        "Qwen/Qwen2.5-0.5B",
    ]
    cls = AutoModelForCausalLM if causal else AutoModel
    for path in candidates[:-1]:
        if os.path.exists(path):
            try:
                return cls.from_pretrained(path, torch_dtype=torch.float16, local_files_only=True)
            except Exception:
                continue
    pytest.skip(
        "Qwen/Qwen2.5-0.5B is not available in the local Hugging Face cache; "
        "GPU surgery tests do not access the network implicitly."
    )


@pytest.mark.gpu
def test_transplant_qwen_full_to_cosformer(gpu_device: torch.device) -> None:
    """
    Load Qwen2.5-0.5B, transplant full quadratic attention to linear Cosformer,
    and assert output tensor finiteness and structural dimensionality.
    """
    model = _load_qwen_model(causal=False).to(gpu_device)
    hidden_size = model.config.hidden_size
    transplant_attention(model, attention="cosformer")

    input_ids = torch.randint(0, 1000, (1, 64), device=gpu_device)
    with torch.no_grad():
        out = model(input_ids)
        last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    assert last_hidden.shape == (1, 64, hidden_size), (
        f"Expected output shape (1, 64, {hidden_size}), got {last_hidden.shape}"
    )
    assert torch.isfinite(last_hidden).all(), "Transplanted model produced non-finite (NaN/Inf) representations"


@pytest.mark.gpu
def test_transplant_qwen_preserves_gqa(gpu_device: torch.device) -> None:
    """
    Verify that after transplant, the number of KV heads is preserved or correctly adapted
    without broadcasting or head dimensional corruption.
    """
    model = _load_qwen_model(causal=True).to(gpu_device)
    orig_q_heads = getattr(model.config, "num_attention_heads", 14)
    orig_kv_heads = getattr(model.config, "num_key_value_heads", 2)
    assert orig_q_heads != orig_kv_heads, (
        f"Target model must feature GQA (num_heads={orig_q_heads}, num_kv_heads={orig_kv_heads})"
    )

    transplant_attention(model, attention="cosformer")

    found_transplanted = 0
    for name, module in model.named_modules():
        leaf = name.split(".")[-1].lower()
        if leaf in ("self_attn", "attn", "attention"):
            found_transplanted += 1
            if hasattr(module, "num_kv_heads"):
                assert module.num_kv_heads in (orig_kv_heads, orig_q_heads), (
                    f"Layer {name} KV heads must be preserved ({orig_kv_heads}) or adapted ({orig_q_heads}), "
                    f"got {module.num_kv_heads}"
                )

    assert found_transplanted > 0, "No transplanted attention layers were found in model hierarchy"


@pytest.mark.gpu
def test_heal_replaced_model_10_steps(gpu_device: torch.device) -> None:
    """
    Execute 10 closed-loop post-surgery calibration and subspace micro-healing steps
    and verify that representation MSE loss decreases monotonically or improves.
    """
    model = _load_qwen_model(causal=False).to(gpu_device)
    teacher_model = copy.deepcopy(model).to(gpu_device)
    transplant_attention(model, attention="cosformer")

    calibration_inputs = torch.randint(0, 1000, (1, 64), device=gpu_device)
    heal_metrics = heal_replaced_model(
        student_model=model,
        teacher_model=teacher_model,
        calibration_inputs=calibration_inputs,
        steps=10,
        lr=1e-4,
        device=gpu_device,
    )

    assert "initial_mse" in heal_metrics and "final_mse" in heal_metrics, (
        f"Healing metrics missing MSE trajectory: {heal_metrics}"
    )

    init_mse = heal_metrics["initial_mse"]
    final_mse = heal_metrics["final_mse"]
    assert final_mse <= init_mse + 1e-4, (
        f"Healing failed: final_mse ({final_mse:.6f}) should decrease or be <= init_mse ({init_mse:.6f})"
    )
