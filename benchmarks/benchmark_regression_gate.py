#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Root entry point for benchmark regression comparison."""

from neuralops.benchmarks.regression_gate import main


if __name__ == "__main__":
    raise SystemExit(main())
