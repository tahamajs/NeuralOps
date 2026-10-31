#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
End-to-end integration test applying multi-stage surgery recipe to a mock transformer model.

Note: This test uses a lightweight mock architecture for fast hermetic CI testing
without requiring downloading external weights from HuggingFace Hub.
"""

import torch
import torch.nn as nn

from neuralops.surgery.pipeline import SurgeryOp, SurgeryRecipe


class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        return x + attn_out + self.mlp(x)


class MockGPT2LM(nn.Module):
    def __init__(self, num_layers=3, dim=32, vocab_size=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([SimpleTransformerBlock(dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def generate(self, input_ids, max_new_tokens=5):
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token = torch.argmax(logits[:, -1:, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


def test_full_recipe_on_transformer():
    """Build transformer, validate recipe, apply surgeries, and test token generation."""
    model = MockGPT2LM()
    input_ids = torch.randint(0, 50, (1, 8))

    recipe = SurgeryRecipe([
        SurgeryOp(name="replace_attention", params={"attention": "chunk_gla"}),
        SurgeryOp(name="prune_layers", params={"drop_count": 1}),
    ])

    report = recipe.validate(model)
    assert report.is_safe

    model, apply_report = recipe.apply(model)
    assert apply_report.success
    assert "replace_attention" in apply_report.applied_ops
    assert "prune_layers" in apply_report.applied_ops
    assert len(model.layers) == 2

    # Generation check
    out = model.generate(input_ids, max_new_tokens=4)
    assert out.shape == (1, 12)
