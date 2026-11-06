#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Forwarder delegating to canonical neuralops.benchmarks.benchmark_all_26_comprehensive.

CLI Equivalent:
    neuralops benchmark --all-triton
    neuralops run --all-triton
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path when executed directly
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neuralops.benchmarks.benchmark_all_26_comprehensive import main

if __name__ == "__main__":
    sys.exit(main())
