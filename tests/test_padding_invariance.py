#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.linear.cosformer import CosformerAttention
from neuralops.core.mask_utils import prepare_mask_and_positions, mask_kv_states


def test_prepare_mask_and_positions_logic():
    hidden = torch.randn(2, 4, 16)
    mask = torch.tensor([
        [0, 0, 1, 1],
        [1, 1, 0, 0],
    ], dtype=torch.long)

    valid_mask, pos_ids, clean = prepare_mask_and_positions(hidden, attention_mask=mask)

    assert valid_mask[0].tolist() == [False, False, True, True]
    assert valid_mask[1].tolist() == [True, True, False, False]

    assert pos_ids[0, 2:].tolist() == [0, 1]
    assert pos_ids[1, :2].tolist() == [0, 1]


def test_cosformer_left_vs_right_padding_equivalence():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn = CosformerAttention(hidden_size=32, num_heads=2, is_causal=True).to(device).eval()

    tokens = torch.randn(1, 4, 32, device=device)

    pad = torch.zeros(1, 4, 32, device=device)
    left_x = torch.cat([pad, tokens], dim=1)
    left_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], device=device)

    right_x = torch.cat([tokens, pad], dim=1)
    right_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device)

    with torch.no_grad():
        left_out, _ = attn(left_x, attention_mask=left_mask)
        right_out, _ = attn(right_x, attention_mask=right_mask)

    valid_from_left = left_out[:, 4:, :]
    valid_from_right = right_out[:, :4, :]

    max_diff = (valid_from_left - valid_from_right).abs().max().item()
    assert max_diff < 1e-3, f"Left and Right padding diverged for valid tokens! Max diff: {max_diff}"
