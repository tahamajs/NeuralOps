#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Deterministic lightweight mock models for offline NeuralOps examples.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from examples._shared.offline_data import synthetic_tokens


class TinyClassifier(nn.Module):
    """
    Deterministic tiny transformer classifier for fast offline unit demonstrations.

    Provides authentic multi-head self-attention layers targeting `replace_attention()`
    and standard linear Feed-Forward Network (FFN/MLP) blocks targeting SVD/pruning/quantization.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        max_seq_len: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        dim: int = 32,
        hidden_dim: int = 64,
        num_labels: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.config = DistilBertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len,
            n_layers=n_layers,
            n_heads=n_heads,
            dim=dim,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
        )
        self.transformer = DistilBertForSequenceClassification(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> Any:
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def count_parameters(self) -> int:
        """Return the total number of parameters in the model."""
        return sum(param.numel() for param in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Return the number of trainable (requires_grad=True) parameters."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


def tiny_inputs(
    batch_size: int = 2,
    seq_len: int = 16,
    vocab_size: int = 128,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Return a small batch of synthetic tokens for model forward execution."""
    return synthetic_tokens(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size, seed=seed)


def baseline_output(
    model: nn.Module,
    inputs: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run the model once in evaluation mode and return output logits as a reference tensor."""
    if inputs is None:
        inputs = tiny_inputs()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits if hasattr(out, "logits") else out
    if was_training:
        model.train()
    return logits.clone()
