#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Root CLI entrypoint for stress_test_seeds_404_777_999.

Delegates directly to canonical neuralops.benchmarks.stress_test_seeds_404_777_999.
"""

import sys
from neuralops.benchmarks.stress_test_seeds_404_777_999 import main

if __name__ == "__main__":
    sys.exit(main())
