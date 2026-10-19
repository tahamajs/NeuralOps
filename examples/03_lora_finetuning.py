#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Inject LoRA during attention surgery and run one training step."""

import torch

import neuralops
from _tiny import tiny_classifier, tiny_inputs


def main() -> None:
    model = neuralops.replace_attention(
        tiny_classifier(), "cosformer", copy_weights=False, inject_lora=True, lora_rank=2
    )
    inputs = tiny_inputs()
    labels = torch.randint(0, 2, (inputs["input_ids"].shape[0],))
    loss = model(**inputs, labels=labels).loss
    loss.backward()
    assert torch.isfinite(loss)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA step: loss={loss.item():.5f}; trainable_parameters={trainable:,}")


if __name__ == "__main__":
    main()
