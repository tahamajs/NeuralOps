#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Root CLI entrypoint for benchmark_all_108_cuda_isolated.

Delegates directly to scripts.benchmarks.benchmark_all_108_cuda_isolated.
"""

from pathlib import Path
import runpy

_script = Path(__file__).resolve().parent.parent / "scripts" / "benchmarks" / "benchmark_all_108_cuda_isolated.py"

if __name__ == "__main__":
    runpy.run_path(str(_script), run_name="__main__")
