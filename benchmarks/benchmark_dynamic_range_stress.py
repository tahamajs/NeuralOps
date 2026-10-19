#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Taha Majlesi
"""Root CLI entrypoint for benchmark_dynamic_range_stress.

Delegates directly to canonical neuralops.benchmarks.benchmark_dynamic_range_stress.
"""

import sys
from neuralops.benchmarks.benchmark_dynamic_range_stress import main

if __name__ == "__main__":
    sys.exit(main())
