#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Test Command

Runs smoke tests, kernel unit tests, and validation suites from the command line.
"""

import subprocess
import sys


def cmd_test():
    return subprocess.call(
        [sys.executable, "-m", "pytest", "tests/test_all_attention_modules.py",
         "tests/test_all_compressor_modules.py", "-q"]
    )
