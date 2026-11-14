#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compression demo: apply SVD low-rank compression to GPT-2."""

from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from neuralops.api import compress
from neuralops.core.registry import COMPRESSOR_REGISTRY


def main() -> None:
    print("NeuralOps - Model Compression Demo")
    print(f"Total available compression methods: {len(COMPRESSOR_REGISTRY)}")

    # Create lightweight GPT-2 model
    config = GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=128,
        vocab_size=1000,
        n_positions=512,
    )
    model = GPT2LMHeadModel(config)
    model.eval()

    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {orig_params:,}")

    # Apply SVD compression with rank=16
    print("Applying SVD low-rank decomposition (target rank=16)...")
    compressed_model = compress(model, method="svd", rank=16)

    # Validate output
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    with torch.no_grad():
        out = compressed_model(input_ids=input_ids)

    print(f"Compressed model forward pass successful!")
    print(f"Output logits shape: {out.logits.shape}")
    assert out.logits.shape == (batch_size, seq_len, 1000)
    assert torch.isfinite(out.logits).all()
    print("Compression demo completed successfully.")


if __name__ == "__main__":
    main()
