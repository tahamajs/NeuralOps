#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for neuralops.core.architecture introspection and leaf attention discovery."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.core.architecture import (
    detect_architecture,
    get_attention_modules,
    get_classifier_head,
    get_embedding_module,
    get_ffn_modules,
    get_transformer_layers,
)


class DummyAttention(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyMLP(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.up_proj = nn.Linear(embed_dim, embed_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.self_attn = DummyAttention(embed_dim)
        self.mlp = DummyMLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.self_attn(x))


class DummyTransformerModel(nn.Module):
    def __init__(self, num_layers: int = 4, embed_dim: int = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, embed_dim)
        self.layers = nn.ModuleList([DummyTransformerBlock(embed_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(embed_dim, 10)

        class Config:
            model_type = "llama"
            num_hidden_layers = num_layers

        self.config = Config()


def test_detect_architecture() -> None:
    """Verify detect_architecture detects supported canonical names."""
    model = DummyTransformerModel(4)
    assert detect_architecture(model) == "llama"

    model.config.model_type = "gpt2"
    assert detect_architecture(model) == "gpt2"

    model.config.model_type = "something_unknown"
    assert detect_architecture(model) == "unknown"


def test_get_attention_modules_count() -> None:
    """Verify get_attention_modules returns exactly all leaf attention sub-modules."""
    num_layers = 6
    model = DummyTransformerModel(num_layers)
    modules = get_attention_modules(model)
    assert len(modules) == num_layers
    for name, mod in modules:
        assert isinstance(mod, DummyAttention)
        assert name.endswith(".self_attn")


def test_get_ffn_modules() -> None:
    """Verify get_ffn_modules detects all feed-forward sub-modules."""
    num_layers = 4
    model = DummyTransformerModel(num_layers)
    ffn = get_ffn_modules(model)
    assert len(ffn) == num_layers
    for _, mod in ffn:
        assert isinstance(mod, DummyMLP)


def test_get_embedding_module() -> None:
    """Verify get_embedding_module retrieves input embeddings."""
    model = DummyTransformerModel(4)
    emb = get_embedding_module(model)
    assert isinstance(emb, nn.Embedding)


def test_get_classifier_head() -> None:
    """Verify get_classifier_head retrieves classifier head."""
    model = DummyTransformerModel(4)
    head = get_classifier_head(model)
    assert isinstance(head, nn.Linear)
