#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Compress a tiny model and report the measured parameter footprint."""

import neuralops
from _tiny import tiny_classifier


def main() -> None:
    model = tiny_classifier()
    before = sum(parameter.numel() for parameter in model.parameters())
    compressed = neuralops.compress(model, method="svd_llm_v2", rank=32)
    after = sum(parameter.numel() for parameter in compressed.parameters())
    reduction = 100.0 * (before - after) / max(before, 1)
    print(f"svd_llm_v2: parameters={after:,}; reduction={reduction:.2f}%")


if __name__ == "__main__":
    main()
