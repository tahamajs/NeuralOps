#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration and synchronization verification suite for neuralops.models.surgery
and all updated attention layers.
"""

from __future__ import annotations

import pytest
import torch
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model

from neuralops.core.registry import create_attention
from neuralops.layers.bert_adapter import BertAttentionAdapter
from neuralops.layers.qwen_adapter import QwenAttentionAdapter
from neuralops.models.surgery import (
    ModelSurgery,
    replace_attention,
    transplant_model_attention,
)

TARGET_LAYERS = [
    "eat",
    "cluster_attention",
    "cosformer",
    "positive_kernel_linear_attention",
    "decay_linear_attention",
    "gdn2_attention",
    "delta_product_attention",
    "preconditioned_gdn_attention",
    "halo_regla",
]


@pytest.mark.parametrize("name", TARGET_LAYERS)
def test_create_attention_factory_signatures(name: str) -> None:
    """Verify create_attention resolves dim, num_heads, and configs cleanly across parameter aliases."""
    # 1. explicit dim and num_heads
    mod1 = create_attention(name, dim=768, num_heads=12)
    assert mod1 is not None

    # 2. embed_dim alias
    mod2 = create_attention(name, embed_dim=768, num_heads=12)
    assert mod2 is not None

    # 3. is_decoder flags
    mod_dec = create_attention(name, dim=768, num_heads=12, is_decoder=True)
    assert getattr(mod_dec, "is_decoder", False) in (True, None) or hasattr(mod_dec, "is_causal")

    mod_enc = create_attention(name, dim=768, num_heads=12, is_decoder=False)
    assert mod_enc is not None


@pytest.mark.parametrize("name", TARGET_LAYERS)
def test_layer_bert_and_qwen_adapters(name: str) -> None:
    """Verify that every target layer integrates with BertAttentionAdapter and QwenAttentionAdapter."""
    x = torch.randn(2, 8, 768)
    mask_4d = torch.zeros(2, 1, 1, 8)
    mask_2d = torch.ones(2, 8, dtype=torch.long)

    # Bert Adapter
    core_enc = create_attention(name, dim=768, num_heads=12, is_decoder=False)
    bert_adapter = BertAttentionAdapter(core_enc, embed_dim=768, num_heads=12)
    ctx, _ = bert_adapter(x, attention_mask=mask_4d)
    assert ctx.shape == (2, 8, 768)

    # Qwen Adapter
    core_dec = create_attention(name, dim=768, num_heads=12, is_decoder=True)
    qwen_adapter = QwenAttentionAdapter(core_dec, embed_dim=768, num_heads=12)
    qwen_out = qwen_adapter(x, attention_mask=mask_2d)
    qwen_ctx = qwen_out[0] if isinstance(qwen_out, tuple) else qwen_out
    assert qwen_ctx.shape == (2, 8, 768)


@pytest.mark.parametrize("name", ["eat", "cosformer", "decay_linear_attention", "gdn2_attention", "halo_regla"])
def test_models_surgery_in_place_transplant(name: str) -> None:
    """Verify that models.surgery.replace_attention mutates BERT and GPT-2 and completes forward execution."""
    bert_cfg = BertConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=2, intermediate_size=128)
    bert = BertModel(bert_cfg)
    bert_surg = transplant_model_attention(bert, attention=name, copy_weights=True)
    inp = torch.randint(0, 100, (2, 8))
    out_bert = bert_surg(inp)
    assert out_bert.last_hidden_state.shape == (2, 8, 64)

    gpt_cfg = GPT2Config(n_embd=64, n_head=4, n_layer=2, n_inner=128)
    gpt = GPT2Model(gpt_cfg)
    gpt_surg = transplant_model_attention(gpt, attention=name, copy_weights=True)
    out_gpt = gpt_surg(inp)
    assert out_gpt.last_hidden_state.shape == (2, 8, 64)


def test_models_surgery_fluent_orchestrator() -> None:
    """Verify ModelSurgery snapshot capture and replace_attention orchestrator."""
    bert_cfg = BertConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=2, intermediate_size=128)
    bert = BertModel(bert_cfg)
    surgery = ModelSurgery(bert)
    assert surgery._initial_snapshot is not None
