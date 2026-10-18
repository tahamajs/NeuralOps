#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root CLI entrypoint for benchmark_all_26_comprehensive.

Delegates directly to canonical neuralops.benchmarks.benchmark_all_26_comprehensive.
"""
import sys
from neuralops.benchmarks.benchmark_all_26_comprehensive import *
from neuralops.benchmarks.benchmark_all_26_comprehensive import main

if __name__ == "__main__":
    sys.exit(main())
