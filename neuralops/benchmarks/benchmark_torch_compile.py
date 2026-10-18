#!/usr/bin/env python3
"""
Torch Compile Benchmark Alias

Canonical alias and execution runner for benchmark_torch_compile_131_attentions.

Author: Taha Majlesi
"""

import sys
from neuralops.benchmarks.benchmark_torch_compile_131_attentions import *
from neuralops.benchmarks.benchmark_torch_compile_131_attentions import main

if __name__ == "__main__":
    sys.exit(main())
