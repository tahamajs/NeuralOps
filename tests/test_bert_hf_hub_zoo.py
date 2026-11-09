#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and integration tests for the Pretrained BERT Attention Zoo pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import pytest

from neuralops.hub.bert_zoo import (
    generate_bert_model_card_markdown,
    list_pretrained_bert_models,
)
from neuralops.layers.bert_adapter import BertAttentionAdapter
from scripts.benchmarks.bert_zoo_train_wiki import (
    fast_gram_covariance_loss,
    InMemoryWikiTextDataset,
)


class MockCoreAttention(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        return self.proj(x)


class MockTupleAttention(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, **kwargs):
        return self.proj(x), torch.ones(x.size(0), x.size(1), x.size(1))


def test_bert_attention_adapter_shapes():
    embed_dim = 768
    batch_size = 2
    seq_len = 16

    core = MockCoreAttention(embed_dim)
    adapter = BertAttentionAdapter(core, embed_dim=embed_dim)

    x = torch.randn(batch_size, seq_len, embed_dim)
    context, attn_weights = adapter(x)

    assert context.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights is None


def test_bert_attention_adapter_tuple_output():
    embed_dim = 768
    batch_size = 2
    seq_len = 16

    core = MockTupleAttention(embed_dim)
    adapter = BertAttentionAdapter(core, embed_dim=embed_dim)

    x = torch.randn(batch_size, seq_len, embed_dim)
    context, attn_weights = adapter(x, output_attentions=True)

    assert context.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights is not None
    assert attn_weights.shape == (batch_size, seq_len, seq_len)


def test_fast_gram_covariance_loss():
    batch_size = 2
    seq_len = 8
    embed_dim = 64

    s = torch.randn(batch_size, seq_len, embed_dim)
    t = torch.randn(batch_size, seq_len, embed_dim)

    loss = fast_gram_covariance_loss(s, t)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_model_card_markdown_generation():
    metrics = {
        "teacher_ppl": 6.84,
        "teacher_acc": 68.42,
        "pre_kd_ppl": 18.25,
        "pre_kd_acc": 42.10,
        "post_kd_ppl": 7.95,
        "post_kd_acc": 66.80,
        "tokens_per_sec": 4820.5,
        "total_steps": 350,
        "peak_vram_mb": 14200,
    }
    card = generate_bert_model_card_markdown("gated_delta_net", metrics, "Hooshaai/bert-gated-delta-net-wiki")
    assert "---" in card
    assert "base_model: bert-base-uncased" in card
    assert "7.95" in card
    assert "66.80%" in card
    assert "gated_delta_net" in card


def test_in_memory_dataset_deterministic_masking():
    class DummyTokenizer:
        vocab_size = 1000
        mask_token_id = 103
        cls_token_id = 101
        sep_token_id = 102

    dummy_tokens = torch.randint(100, 900, (10, 32))
    tok = DummyTokenizer()

    ds = InMemoryWikiTextDataset(dummy_tokens, tok, is_eval=True, seed=42)
    item1 = ds[0]
    item2 = ds[0]

    # Deterministic evaluation masking check
    assert torch.equal(item1["input_ids"], item2["input_ids"])
    assert torch.equal(item1["labels"], item2["labels"])
    # Check that mask tokens are placed
    assert (item1["input_ids"] == tok.mask_token_id).any()
