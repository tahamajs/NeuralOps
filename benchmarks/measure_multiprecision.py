#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root CLI entrypoint for measure_multiprecision.

Delegates directly to canonical neuralops.benchmarks.measure_multiprecision.
"""
import sys
from neuralops.benchmarks.measure_multiprecision import *
from neuralops.benchmarks.measure_multiprecision import main

if __name__ == "__main__":
    sys.exit(main())
