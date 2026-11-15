#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess test for examples/02_model_compression.py.
"""

from __future__ import annotations

import subprocess
import sys


def test_example_02_execution():
    proc = subprocess.run(
        [sys.executable, "examples/02_model_compression.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 02 failed with stderr: {proc.stderr}"
    assert "[PASS] Overall Example 02 Status" in proc.stdout
