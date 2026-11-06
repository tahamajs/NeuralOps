#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Root entry point for the canonical resource-scaling benchmark."""


import warnings
warnings.warn("Deprecated. Use neuralops benchmark instead.", DeprecationWarning, stacklevel=2)

import sys
from pathlib import Path

# Prevent benchmarks/logging.py from shadowing Python's standard logging module
# when this compatibility entry point is invoked as a file path.
_BENCHMARKS_DIR = str(Path(__file__).resolve().parent)
if _BENCHMARKS_DIR in sys.path:
    sys.path.remove(_BENCHMARKS_DIR)

from neuralops.benchmarks.resource_scaling import main


if __name__ == "__main__":
    raise SystemExit(main())
