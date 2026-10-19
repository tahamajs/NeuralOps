#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps CLI Entry Point

Unified command-line package execution interface routing directly to the benchmark and research CLI suite.
"""

from __future__ import annotations

import sys
from neuralops.benchmarks.cli import main

if __name__ == "__main__":
    sys.exit(main())
