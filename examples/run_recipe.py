#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Demonstration runner executing declarative surgery recipe on a mock transformer model.
"""

import torch.nn as nn

import neuralops
from neuralops.surgery.recipe import load_recipe


class ToyBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.c_fc = nn.Linear(dim, dim * 4)
        self.c_proj = nn.Linear(dim * 4, dim)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        return x + attn_out + self.c_proj(self.c_fc(x))


class MockGPT2Model(nn.Module):
    def __init__(self, num_layers=2, dim=32):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    recipe = load_recipe("examples/recipes/gpt2-fast.yaml")
    print(f"Loaded recipe: {recipe.name} (version {recipe.recipe_version})")
    print(f"Surgeries to execute: {[op.name for op in recipe.ops]}")

    # Instantiate mock model
    model = MockGPT2Model()

    # Apply recipe
    model, report = neuralops.surgery.apply_recipe(model, recipe)
    print(f"Applied successfully: {report.success}. VRAM reduction estimate: {report.vram_reduction_pct}%")


if __name__ == "__main__":
    main()
