#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end integration tests without real checkpoints."""

import pytest
import torch
from transformers import BertConfig, BertModel
from neuralops.api import replace_attention


@pytest.mark.integration
def test_integration_bert_replace_and_forward():
    """Verify end-to-end integration of loading BERT, replacing attention, and running forward pass."""
    config = BertConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=512,
    )
    model = BertModel(config)
    model = replace_attention(model, "cosformer")
    input_ids = torch.tensor([[1, 2, 3]])
    out = model(input_ids)
    assert out.last_hidden_state.shape == (1, 3, 64)
    assert torch.isfinite(out.last_hidden_state).all()
