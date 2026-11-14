#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Quickstart example: load BERT and run inference with NeuralOps."""

from __future__ import annotations

import torch
from transformers import BertConfig, BertForSequenceClassification
import neuralops


def main() -> None:
    print(f"NeuralOps v{neuralops.__version__} - Quickstart Demo")

    # Instantiate a lightweight BERT configuration
    config = BertConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=256,
        vocab_size=1000,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)
    model.eval()

    # Generate dummy input tokens
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    print(f"Running inference on batch size={batch_size}, seq_len={seq_len}...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    print(f"Inference complete! Logits shape: {logits.shape}")
    print(f"Class predictions: {predictions.tolist()}")
    print("Quickstart completed successfully.")


if __name__ == "__main__":
    main()
