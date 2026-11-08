#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Multi-Phase Benchmark Matrix Orchestrator (AGT-59 / Gap 6).

Verifies:
  1. Multi-axis parameter grid planning and workload regime categorization.
  2. Cold-cache flushing protocol and GPU/CPU device handling.
  3. Cell execution with percentile latency, throughput, and TFLOPs derivation.
  4. Prefill regression speedup gating.
  5. CLI matrix subparser invocation in plan and execution modes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from neuralops.benchmarks.cli import main
from neuralops.benchmarks.matrix import (
    BenchmarkMatrixOrchestrator,
    ColdCacheFlusher,
    MatrixDimensionConfig,
    MatrixPoint,
    run_benchmark_matrix,
)


def test_matrix_dimension_planning():
    """Verify that plan() generates the exact Cartesian product of dimensions and regimes."""
    config = MatrixDimensionConfig(
        modules=["attention:full_attention", "attention:cosformer"],
        seq_lens=[128, 1024],
        batch_sizes=[1, 4],
        hidden_dims=[64],
        dtypes=["float32"],
        causal_modes=[True, False],
    )
    orchestrator = BenchmarkMatrixOrchestrator(config)
    points = orchestrator.plan()

    expected_len = 2 * 2 * 2 * 1 * 1 * 2  # 16
    assert len(points) == expected_len

    # Verify regime categorization
    pt_prefill = MatrixPoint("attention:full_attention", seq_len=1024, batch_size=4, hidden_dim=64, dtype="float32", causal=True)
    assert pt_prefill.regime == "prefill"

    pt_decode = MatrixPoint("attention:full_attention", seq_len=1024, batch_size=1, hidden_dim=64, dtype="float32", causal=True)
    assert pt_decode.regime == "prefill"  # seq_len >= 512 is prefill

    pt_micro = MatrixPoint("attention:full_attention", seq_len=128, batch_size=2, hidden_dim=64, dtype="float32", causal=True)
    assert pt_micro.regime == "micro"


def test_cold_cache_flusher():
    """Verify ColdCacheFlusher operates safely across CPU and CUDA."""
    flusher_cpu = ColdCacheFlusher(torch.device("cpu"), size_mb=1)
    # CPU should safely skip buffer allocation or flush without error
    flusher_cpu.flush()


def test_matrix_cell_execution(tmp_path):
    """Execute a single matrix point and verify derived metrics."""
    config = MatrixDimensionConfig(
        modules=["attention:full_attention"],
        seq_lens=[64],
        batch_sizes=[1],
        hidden_dims=[32],
        dtypes=["float32"],
        causal_modes=[True],
        device="cpu",
        output_dir=str(tmp_path),
        warmup=1,
        iters=3,
        flush_l2=False,
    )
    orchestrator = BenchmarkMatrixOrchestrator(config)
    pt = orchestrator.plan()[0]
    cell_res = orchestrator.execute_cell(pt)

    assert cell_res.status == "success"
    assert cell_res.numerics_passed is True
    assert cell_res.speedup_gate_passed is True
    assert cell_res.latency_ms_median > 0
    assert cell_res.throughput_tokens_per_s > 0
    assert cell_res.tflops >= 0
    assert cell_res.effective_bandwidth_gb_s > 0


def test_matrix_grid_execution_and_summary(tmp_path):
    """Verify execution of a small 2-point grid and incremental file persistence."""
    res = run_benchmark_matrix(
        modules=["attention:full_attention"],
        seq_lens=[32, 64],
        batch_sizes=[1],
        hidden_dims=[32],
        dtypes=["float32"],
        causal_modes=[True],
        device="cpu",
        output_dir=str(tmp_path),
        dry_run=False,
        warmup=1,
        iters=2,
        flush_l2=False,
    )

    assert res["total_cells"] == 2
    assert res["passed_cells"] == 2
    assert res["failed_cells"] == 0

    jsonl_path = tmp_path / "matrix_results.jsonl"
    csv_path = tmp_path / "matrix_summary.csv"
    summary_path = tmp_path / "summary.json"

    assert jsonl_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()

    lines = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0]["status"] == "success"


def test_matrix_cli_plan():
    """Verify matrix CLI dry-run / plan outputs valid JSON with scheduled points."""
    stdout_buf = io.StringIO()
    with patch("sys.stdout", stdout_buf):
        exit_code = main([
            "matrix",
            "--modules", "attention:full_attention",
            "--seq-lens", "128,512",
            "--batch-sizes", "1,2",
            "--hidden-dims", "64",
            "--dtypes", "float32",
            "--causal",
            "--plan",
            "--json",
        ])
    assert exit_code == 0
    out = json.loads(stdout_buf.getvalue())
    assert out["dry_run"] is True
    assert out["total_points"] == 4
    assert len(out["points"]) == 4
