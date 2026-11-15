#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for generate_reproduction_command and generate_reproduction_script.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
import pytest

from neuralops.benchmarks.results.reproduce import (
    generate_reproduction_command,
    generate_reproduction_script,
)
from neuralops.benchmarks.results.schema import BenchmarkResult


def test_generate_reproduction_command():
    """Verify generated command contains expected CLI flags."""
    res = BenchmarkResult(
        run_id="run_repro_123",
        module={"name": "test_attention"},
        model={"name": "distilbert-base-uncased"},
        dataset={"name": "sst2"},
        config={"rank": 32, "epochs": 2, "batch_size": 16, "max_length": 64, "seed": 99},
    )
    cmd = generate_reproduction_command(res)
    assert "python -m neuralops.benchmarks run" in cmd
    assert "--module test_attention" in cmd
    assert "--model distilbert-base-uncased" in cmd
    assert "--dataset sst2" in cmd
    assert "--rank 32" in cmd
    assert "--epochs 2" in cmd
    assert "--batch-size 16" in cmd
    assert "--seq-len 64" in cmd
    assert "--seed 99" in cmd
    assert "--output-dir results/reproduce_run_repro_123" in cmd


def test_generate_reproduction_script(tmp_path: Path):
    """Verify generated script is executable and contains environment sanity checks."""
    res = BenchmarkResult(
        run_id="run_repro_script",
        module={"name": "test_mod"},
        model={"name": "bert"},
        dataset={"name": "sst2"},
    )
    script_path = tmp_path / "reproduce.sh"
    out_path = generate_reproduction_script(res, script_path)
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "#!/usr/bin/env bash" in content
    assert "python -m neuralops.benchmarks run" in content

    # Verify executable bits
    mode = os.stat(out_path).st_mode
    assert bool(mode & stat.S_IXUSR)
