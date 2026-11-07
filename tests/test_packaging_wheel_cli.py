#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for package packaging, wheel import hygiene, and sys.path isolation."""

from __future__ import annotations

import subprocess
import sys


def test_no_sys_path_manipulation_on_import() -> None:
    """Verify that importing neuralops does not mutate sys.path[0] and does not inject HF_HOME."""
    code = (
        "import sys, os\n"
        "orig_path0 = sys.path[0]\n"
        "had_hf_home = 'HF_HOME' in os.environ\n"
        "import neuralops\n"
        "assert sys.path[0] == orig_path0, f'sys.path[0] mutated from {orig_path0} to {sys.path[0]}'\n"
        "if not had_hf_home:\n"
        "    assert 'HF_HOME' not in os.environ, 'HF_HOME was unexpectedly injected by neuralops import'\n"
        "print('PASS')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Import test failed:\n{res.stdout}\n{res.stderr}"
    assert "PASS" in res.stdout


def test_neuralops_run_cli_smoke() -> None:
    """Verify that neuralops.main (registered as neuralops-run) provides a functional CLI help."""
    res = subprocess.run([sys.executable, "-m", "neuralops.main", "--help"], capture_output=True, text=True)
    assert res.returncode == 0, f"neuralops.main --help failed:\n{res.stdout}\n{res.stderr}"
    assert "Efficient Attention Framework" in res.stdout or "neuralops-run" in res.stdout
