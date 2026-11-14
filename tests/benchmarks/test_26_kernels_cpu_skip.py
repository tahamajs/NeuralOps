#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for 26-kernel comprehensive sweep graceful CPU / hardware skip."""

import json
from pathlib import Path
import pytest
import torch

from neuralops.benchmarks.benchmark_all_26_comprehensive import (
    CANONICAL_26_TRITON_KERNELS,
    SKIPPED,
    SKIPPED_UNSUPPORTED_HARDWARE,
    main,
    run_all_26_comprehensive,
)


def test_run_all_26_comprehensive_cpu_returns_skipped(monkeypatch, tmp_path: Path):
    """Verify run_all_26_comprehensive returns SKIPPED_UNSUPPORTED_HARDWARE / SKIPPED without raising RuntimeError."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    json_path = tmp_path / "test_skip.json"
    csv_path = tmp_path / "test_skip.csv"

    results = run_all_26_comprehensive(
        device_str="auto",
        output_csv=str(csv_path),
        output_json=str(json_path),
    )

    # Invariant 1: Equality checks for both skip statuses
    assert results == "SKIPPED_UNSUPPORTED_HARDWARE"
    assert results == "SKIPPED"
    assert results == SKIPPED_UNSUPPORTED_HARDWARE
    assert results == SKIPPED

    # Invariant 2: Result is a list covering all 26 canonical kernels
    assert isinstance(results, list)
    assert len(results) == len(CANONICAL_26_TRITON_KERNELS)

    # Invariant 3: Each kernel row has valid skipped status
    for row in results:
        assert row["status"] == "SKIPPED_UNSUPPORTED_HARDWARE"
        assert row["status"] == "SKIPPED"
        assert row["fallback_used"] is True
        assert "CUDA unavailable" in row["error"]

    # Invariant 4: Files were generated and reflect skip state
    assert json_path.exists()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(saved) == 26
    assert saved[0]["status"] == "SKIPPED_UNSUPPORTED_HARDWARE"


def test_run_all_26_comprehensive_explicit_cpu_device(monkeypatch):
    """Verify that passing device_str='cpu' triggers graceful skip rather than hard exception."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    results = run_all_26_comprehensive(device_str="cpu")
    assert results in ("SKIPPED_UNSUPPORTED_HARDWARE", "SKIPPED")
    assert len(results) == len(CANONICAL_26_TRITON_KERNELS)


def test_run_all_26_comprehensive_selected_kernels_skip(monkeypatch):
    """Verify selected_kernels subset is respected during hardware skip."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    subset = ["fused_rope", "sliding_window_attention"]
    results = run_all_26_comprehensive(device_str="auto", selected_kernels=subset)

    assert results == "SKIPPED_UNSUPPORTED_HARDWARE"
    assert len(results) == 2
    actual_kernels = [r["kernel"] for r in results]
    assert actual_kernels == subset


def test_main_cli_cpu_skip_exit_code_zero(monkeypatch, tmp_path: Path):
    """Verify CLI main entry point exits with code 0 on unsupported CPU host."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    out_json = str(tmp_path / "cli_out.json")
    out_csv = str(tmp_path / "cli_out.csv")

    exit_code = main(["--device", "auto", "--quick", "--output", out_csv, "--output-json", out_json])
    assert exit_code == 0
