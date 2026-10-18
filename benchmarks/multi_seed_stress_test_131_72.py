#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root CLI entrypoint for multi_seed_stress_test_131_72.

Delegates directly to canonical neuralops.benchmarks.multi_seed_stress_test_131_72.
"""

import sys
from neuralops.benchmarks.multi_seed_stress_test_131_72 import *
from neuralops.benchmarks.multi_seed_stress_test_131_72 import main

if __name__ == "__main__":
    sys.exit(main())
