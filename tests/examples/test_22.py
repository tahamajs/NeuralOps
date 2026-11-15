#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess test for examples/22_surgery_baseline_report.py.
"""

from __future__ import annotations

import json
import subprocess
import sys


def test_example_22_execution(tmp_path):
    report_file = tmp_path / "baseline_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "examples/22_surgery_baseline_report.py",
            "--device",
            "cpu",
            "--output",
            str(report_file),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 22 failed with stderr: {proc.stderr}\nstdout: {proc.stdout}"
    assert "[PASS] Overall Example 22 Status" in proc.stdout
    assert "[PASS] Baseline Reference Evaluation" in proc.stdout
    assert "[PASS] Surgeried Model Evaluation" in proc.stdout
    assert "[PASS] Numerical Difference Bound" in proc.stdout
    assert "[PASS] Report Serialization" in proc.stdout

    # Verify generated report file
    assert report_file.exists()
    report_data = json.loads(report_file.read_text(encoding="utf-8"))
    assert report_data["status"] == "PASS"
    assert report_data["device"] == "cpu"
    assert report_data["attention"] == "cosformer"
    assert report_data["num_layers"] == 3
    assert report_data["baseline_params"] > 0
    assert report_data["surgeried_params"] > 0
    assert report_data["baseline_latency_ms"] > 0.0
    assert report_data["surgeried_latency_ms"] > 0.0
