#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Replace attention in a tiny Hugging Face model with the unified API."""

import torch

import neuralops
from _tiny import tiny_classifier, tiny_inputs


def main() -> None:
    inputs = tiny_inputs()
    for name in ("cosformer", "gla_attention"):
        model = neuralops.replace_attention(tiny_classifier(), name, copy_weights=False)
        with torch.no_grad():
            outputs = model(**inputs)
        assert torch.isfinite(outputs.logits).all()
        print(f"{name}: replaced attention; logits={tuple(outputs.logits.shape)}")


if __name__ == "__main__":
    main()
