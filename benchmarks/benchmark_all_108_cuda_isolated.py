#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Taha Majlesi
"""Root CLI entrypoint for benchmark_all_108_cuda_isolated.

Delegates directly to scripts.benchmarks.benchmark_all_108_cuda_isolated.
"""

from pathlib import Path
import runpy

_script = Path(__file__).resolve().parent.parent / "scripts" / "benchmarks" / "benchmark_all_108_cuda_isolated.py"

if __name__ == "__main__":
    runpy.run_path(str(_script), run_name="__main__")
