#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for real ecosystem integrations (patch_model_attention, vLLM, DeepSpeed, FSDP)."""
import torch
import torch.nn as nn
import pytest
from neuralops.integrations.hf_automodel import patch_model_attention
from neuralops.integrations.serving import VLLMAttentionAdapter
from neuralops.integrations.distributed_compile import get_deepspeed_config, apply_fsdp_wrapping


def test_patch_model_attention_real():
    class DummyTransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Linear(32, 32)
            self.mlp = nn.Linear(32, 32)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = DummyTransformerLayer()

    model = DummyModel()
    patched = patch_model_attention(model, attention_type="full_attention")
    assert not isinstance(patched.layer.self_attn, nn.Linear)


def test_vllm_and_deepspeed_fsdp_working(monkeypatch):
    adapter = VLLMAttentionAdapter(attention_type="paged_attention", head_dim=64, num_heads=8)
    meta = adapter.format_metadata([[0, 1], [2, 3]], [32, 64])
    assert meta["max_context_len"] == 64

    ds = get_deepspeed_config(stage=2, offload_optimizer=True, fp16=True)
    assert ds["zero_optimization"]["stage"] == 2
    assert ds["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    model = nn.Linear(16, 16)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized"):
        apply_fsdp_wrapping(model)
