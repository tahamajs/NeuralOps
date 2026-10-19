#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Register and instantiate a small user-defined attention module."""

import torch
from torch import nn

import neuralops


@neuralops.register_custom_attention("demo_identity_attention")
class DemoIdentityAttention(nn.Module):
    """Minimal attention-shaped module for extension demonstrations."""

    def __init__(self, embed_dim: int = 32, **_: object) -> None:
        super().__init__()
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.projection(x)


def main() -> None:
    module = neuralops.create_attention("demo_identity_attention", embed_dim=32, num_heads=2)
    output = module(torch.randn(2, 8, 32))
    assert torch.isfinite(output).all()
    print(f"Registered custom attention: output={tuple(output.shape)}")


if __name__ == "__main__":
    main()
