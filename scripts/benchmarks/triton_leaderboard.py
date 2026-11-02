#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Deprecated entry point. Use `python3 neuralops/benchmarks/benchmark_all_26_kernels.py` instead."""

from __future__ import annotations

import sys

print(
    "[NOTICE] This script was removed in October 2026. "
    "The unified benchmark runner is: python3 neuralops/benchmarks/benchmark_all_26_kernels.py",
    file=sys.stderr,
)
sys.exit(0)
