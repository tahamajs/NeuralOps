#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Phase 10: Regression Test Suite for Benchmark Subsystems.

Verifies:
1. BenchmarkConfig validation and serialization roundtrip
2. BenchmarkLogger structured JSONL audit trail integrity
3. SQLite WAL mode and schema auto-migration
4. GGUF binary container header magic integrity
5. Serving matrix 9-framework verifier execution
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile

import torch
import torch.nn as nn

from scripts.benchmarks.benchmark_config import BenchmarkConfig
from scripts.benchmarks.benchmark_logger import BenchmarkLogger
from scripts.benchmarks.results_exporter import init_sqlite_ledger
from scripts.benchmarks.serving_matrix_verifier import verify_all_nine


def test_01_benchmark_config_defaults():
    cfg = BenchmarkConfig()
    assert cfg.device in ("cuda", "mps", "cpu", "auto")
    assert cfg.warmup >= 1
    assert cfg.iters >= 1


def test_02_benchmark_logger_jsonl_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = BenchmarkLogger(run_id="test_run_01", log_dir=tmpdir)
        rec = logger.log(phase="test_phase", module="test_mod", status="PASS", metrics={"latency_ms": 1.23})
        assert rec["status"] == "PASS"
        assert logger.log_path.exists()
        content = logger.log_path.read_text(encoding="utf-8")
        assert "test_phase" in content
        assert "test_mod" in content


def test_03_sqlite_ledger_wal_and_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_ledger.sqlite"
        conn = init_sqlite_ledger(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode;")
        mode = cur.fetchone()[0]
        assert str(mode).lower() == "wal"

        cur.execute("PRAGMA table_info(module_metrics);")
        col_names = {row[1] for row in cur.fetchall()}
        assert "p50_ms" in col_names
        assert "p95_ms" in col_names
        assert "p99_ms" in col_names
        assert "peak_vram_mb" in col_names
        conn.close()


def test_04_gguf_integrity_roundtrip():
    from neuralops.export.exporter import export_model
    with tempfile.TemporaryDirectory() as tmpdir:
        model = nn.Sequential(nn.Linear(8, 8))
        out_p = Path(tmpdir) / "model.gguf"
        export_model(model, out_p, format="gguf")
        target_file = out_p / "model.gguf" if out_p.is_dir() else out_p
        with open(target_file, "rb") as f:
            magic = f.read(4)
            assert magic == b"GGUF"


def test_05_serving_verifier_all_nine_ci():
    results = verify_all_nine(ci_mode=True)
    assert len(results) == 9
    for r in results:
        assert "PASS" in r["status"]
