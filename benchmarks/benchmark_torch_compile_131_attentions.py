#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root CLI entrypoint for benchmark_torch_compile_131_attentions.

Delegates directly to canonical neuralops.benchmarks.benchmark_torch_compile_131_attentions.
"""

import sys
from neuralops.benchmarks.benchmark_torch_compile_131_attentions import *
from neuralops.benchmarks.benchmark_torch_compile_131_attentions import main

if __name__ == "__main__":
    sys.exit(main())
