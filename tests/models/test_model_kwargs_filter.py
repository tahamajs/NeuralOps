#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for filtering invalid kwargs passed to NeuralOpsModel decoder layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.models.configuration import NeuralOpsConfig
from neuralops.models.modeling import NeuralOpsModel


@ATTENTION_REGISTRY.register("dummy_attn")
class DummyAttn(nn.Module):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, None, None]:
        return x, None, None


class StrictDecoderLayer(nn.Module):
    """Decoder layer with a strict forward signature that rejects unknown kwargs."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: object = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, None, None]:
        # No **kwargs in signature! Any leaked kwargs will raise TypeError.
        return self.linear(hidden_states), None, None


def test_model_filters_leaked_kwargs_to_strict_layer() -> None:
    config = NeuralOpsConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        attention_type="dummy_attn",
    )
    model = NeuralOpsModel(config)
    # Replace default layer with our strict layer that does not accept arbitrary kwargs
    model.layers[0] = StrictDecoderLayer(hidden_size=32)

    input_ids = torch.tensor([[1, 2, 3]])

    # Passing unrecognized kwargs that HF generation or pipelines might pass
    # (e.g., cache_position, logits_to_keep, num_items_in_batch, invalid_arg)
    out = model(
        input_ids=input_ids,
        invalid_kwarg="should_be_filtered",
        cache_position=torch.arange(3),
        logits_to_keep=1,
    )
    assert out.last_hidden_state is not None
    assert out.last_hidden_state.shape == (1, 3, 32)
