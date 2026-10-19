#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Root CLI entrypoint for multi_seed_stress_test_131_72.

Delegates directly to canonical neuralops.benchmarks.multi_seed_stress_test_131_72.
"""

import sys
from neuralops.benchmarks.multi_seed_stress_test_131_72 import main

if __name__ == "__main__":
    sys.exit(main())
