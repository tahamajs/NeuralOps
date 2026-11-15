#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess tests for examples/05_fla_style_benchmarking.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "05_fla_style_benchmarking.py"


def test_05_runs_clean():
    """Verify example 05 executes cleanly and exits with code 0."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"Example 05 failed with stderr:\n{result.stderr}"
    assert "[PASS] Overall Example 05 Status" in result.stdout


def test_05_no_fake_vram_on_cpu():
    """Verify that when run on CPU, output does not report nonzero CUDA VRAM."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, f"Example 05 failed on CPU with stderr:\n{result.stderr}"
    assert "0.0 MB" in result.stdout or "N/A" in result.stdout


def test_05_no_hardcoded_pass():
    """Verify example 05 does not hardcode pass_fail literal True."""
    content = EXAMPLE_PATH.read_text(encoding="utf-8")
    for line in content.splitlines():
        if 'pass_fail("Overall' in line:
            assert 'True,' not in line, f"Hardcoded pass detected: {line}"
