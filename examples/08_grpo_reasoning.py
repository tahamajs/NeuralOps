#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Show the unified trainer entry point for reasoning/GRPO workflows."""

import neuralops
from _tiny import tiny_classifier


def main() -> None:
    model = neuralops.replace_attention(tiny_classifier(), "cosformer", copy_weights=False)
    print(f"Reasoning model prepared: {type(model).__name__}")
    try:
        import trl  # noqa: F401
    except ImportError:
        print("Optional TRL is not installed; call neuralops.train_grpo after installing trl.")
        return
    print("TRL is installed; use neuralops.train_grpo(model, reward_funcs, dataset, ...).")


if __name__ == "__main__":
    main()
