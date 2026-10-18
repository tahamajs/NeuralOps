#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root CLI entrypoint for stress_test_seeds_100_200_300.

Delegates directly to canonical neuralops.benchmarks.stress_test_seeds_100_200_300.
"""

import sys
from neuralops.benchmarks.stress_test_seeds_100_200_300 import *
from neuralops.benchmarks.stress_test_seeds_100_200_300 import main

if __name__ == "__main__":
    sys.exit(main())
