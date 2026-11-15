#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess test for examples/21_full_model_surgery.py.
"""

from __future__ import annotations

import subprocess
import sys


def test_example_21_execution():
    proc = subprocess.run(
        [sys.executable, "examples/21_full_model_surgery.py", "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 21 failed with stderr: {proc.stderr}\nstdout: {proc.stdout}"
    assert "[PASS] Overall Example 21 Status" in proc.stdout
    assert "[PASS] All Layers Attention Replacement" in proc.stdout
    assert "[PASS] Full-Model Forward Pass Integrity" in proc.stdout
    assert "[PASS] Trainable Gradients Propagation" in proc.stdout
