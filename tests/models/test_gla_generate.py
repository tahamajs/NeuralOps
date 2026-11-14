#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for GLAHFModel / GLAModel generation behavior."""

from __future__ import annotations

import pytest
import torch
from transformers import GenerationMixin

from neuralops.models.gla_hf_model import GLAConfig, GLAHFModel, GLAModel


def test_gla_model_does_not_inherit_generation_mixin() -> None:
    assert not issubclass(GLAModel, GenerationMixin)
    assert not issubclass(GLAHFModel, GenerationMixin)


def test_gla_model_generate_raises_not_implemented_error() -> None:
    config = GLAConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        gate_rank=4,
    )
    model = GLAHFModel(config)
    input_ids = torch.tensor([[1, 2, 3]])

    with pytest.raises(NotImplementedError, match="Autoregressive generation for GLAHFModel is not implemented yet."):
        model.generate(input_ids)
