#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root CLI entrypoint for benchmark_extrapolation_32k.

Delegates directly to canonical neuralops.benchmarks.benchmark_extrapolation_32k.
"""

import sys
from neuralops.benchmarks.benchmark_extrapolation_32k import *
from neuralops.benchmarks.benchmark_extrapolation_32k import main

if __name__ == "__main__":
    sys.exit(main())
