#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Run representative modules through the public benchmark API."""

import neuralops


def main() -> None:
    for name in ("cosformer", "mamba_block", "svd_linear_attention"):
        try:
            result = neuralops.benchmark(
                module=name,
                model="hf-internal-testing/tiny-random-distilbert",
                dataset="sst2",
                device="cpu",
                eval_only=True,
                timeout_s=120,
                max_eval_samples=16,
            )
            print(f"{name}: {result.status}")
        except Exception as exc:  # A missing optional dataset/model is reported, not hidden.
            print(f"{name}: unavailable in this environment ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
