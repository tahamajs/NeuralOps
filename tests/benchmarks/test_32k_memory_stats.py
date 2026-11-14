#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests verifying torch.cuda.memory_stats reporting in 32k extrapolation benchmark."""

import csv
import json
from pathlib import Path
import pytest
import torch

from neuralops.benchmarks.benchmark_extrapolation_32k import (
    collect_cuda_memory_stats,
    main,
    run_extrapolation_benchmark,
)


def test_collect_memory_stats_mocked(monkeypatch):
    """Verify collect_cuda_memory_stats extracts memory metrics and calculates fragmentation ratio."""
    mock_stats = {
        "reserved_bytes.all.current": 200 * 1024 * 1024,
        "allocated_bytes.all.current": 100 * 1024 * 1024,
        "active_bytes.all.current": 80 * 1024 * 1024,
    }

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda dev=None: mock_stats)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: 100 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev=None: 100 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev=None: 200 * 1024 * 1024)

    stats = collect_cuda_memory_stats("cuda:0")

    assert stats["peak_vram_mb"] == 100.0
    assert stats["reserved_vram_mb"] == 200.0
    assert stats["allocated_vram_mb"] == 100.0
    # fragmentation = (200 - 100) / 200 = 0.5
    assert stats["fragmentation_ratio"] == 0.5
    assert stats["memory_stats"]["reserved_bytes.all.current"] == 200 * 1024 * 1024


def test_collect_memory_stats_cpu():
    """Verify collect_cuda_memory_stats returns zeros on CPU without error."""
    stats = collect_cuda_memory_stats("cpu")
    assert stats["peak_vram_mb"] == 0.0
    assert stats["reserved_vram_mb"] == 0.0
    assert stats["allocated_vram_mb"] == 0.0
    assert stats["fragmentation_ratio"] == 0.0
    assert stats["memory_stats"] == {}


def test_32k_extrapolation_end_to_end_cpu(tmp_path: Path):
    """Verify run_extrapolation_benchmark runs end-to-end and writes CSV/JSON with fragmentation metrics."""
    csv_path = tmp_path / "extrapolation.csv"
    json_path = tmp_path / "extrapolation.json"

    results = run_extrapolation_benchmark(
        device_str="cpu",
        seq_lens=[64],
        warmup=1,
        repeat=1,
        output_csv=str(csv_path),
        output_json=str(json_path),
        output_tex=None,
    )

    assert len(results) > 0
    for row in results:
        assert "fragmentation_ratio" in row
        assert "reserved_vram_mb" in row
        assert "allocated_vram_mb" in row
        assert "memory_stats" in row
        assert row["fragmentation_ratio"] == 0.0

    # Verify JSON output
    assert json_path.exists()
    saved_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(saved_json) == len(results)
    assert "fragmentation_ratio" in saved_json[0]

    # Verify CSV output has fragmentation headers
    assert csv_path.exists()
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "fragmentation_ratio" in reader.fieldnames
        assert "reserved_vram_mb" in reader.fieldnames
        assert "allocated_vram_mb" in reader.fieldnames


def test_main_cli_execution(monkeypatch):
    """Verify main entrypoint executes cleanly without error."""
    monkeypatch.setattr(
        "neuralops.benchmarks.benchmark_extrapolation_32k.run_extrapolation_benchmark",
        lambda **kwargs: [],
    )
    exit_code = main(["--quick", "--device", "auto"])
    assert exit_code == 0
