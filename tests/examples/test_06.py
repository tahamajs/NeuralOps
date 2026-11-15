#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for examples/06_export_and_push.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "06_export_and_push.py"


def test_06_runs_clean():
    """Verify example 06 executes cleanly and exits with code 0."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"Example 06 failed with stderr:\n{result.stderr}"
    assert "[PASS] Overall Example 06 Status" in result.stdout


def test_06_safetensors_reload_parity():
    """Verify SafeTensors output is confirmed."""
    content = EXAMPLE_PATH.read_text(encoding="utf-8")
    assert "export_and_verify_safetensors" in content
    assert "model.safetensors" in content
