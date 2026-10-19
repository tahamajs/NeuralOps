#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Taha Majlesi
"""Root CLI entrypoint for stress_test_seeds_404_777_999.

Delegates directly to canonical neuralops.benchmarks.stress_test_seeds_404_777_999.
"""

import sys
from neuralops.benchmarks.stress_test_seeds_404_777_999 import main

if __name__ == "__main__":
    sys.exit(main())
