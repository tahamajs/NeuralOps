#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Surgery demo: replace attention in BERT with linear/subquadratic attention."""

from __future__ import annotations

import torch
from transformers import BertConfig, BertModel
from neuralops.api import replace_attention
from neuralops.core.registry import ATTENTION_REGISTRY


def main() -> None:
    print("NeuralOps - Attention Surgery Demo")
    print(f"Total available attention mechanisms: {len(ATTENTION_REGISTRY)}")

    config = BertConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=256,
        vocab_size=1000,
    )
    model = BertModel(config)
    model.eval()

    target_attn = "cosformer"
    print(f"Performing surgery: replacing standard self-attention with '{target_attn}'...")
    modified_model = replace_attention(model, target_attn)

    # Validate output
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    with torch.no_grad():
        out = modified_model(input_ids=input_ids)

    print(f"Transplanted model forward pass successful!")
    print(f"Output hidden states shape: {out.last_hidden_state.shape}")
    assert out.last_hidden_state.shape == (batch_size, seq_len, 128)
    assert torch.isfinite(out.last_hidden_state).all()
    print("Surgery demo completed successfully.")


if __name__ == "__main__":
    main()
