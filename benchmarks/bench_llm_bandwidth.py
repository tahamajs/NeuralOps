#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root CLI entrypoint for bench_llm_bandwidth.

Delegates directly to canonical neuralops.benchmarks.bench_llm_bandwidth.
"""

import sys
from neuralops.benchmarks.bench_llm_bandwidth import *
from neuralops.benchmarks.bench_llm_bandwidth import main

if __name__ == "__main__":
    sys.exit(main())
