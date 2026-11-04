#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and Concurrency Tests for Hermetic Storage & Results Persistence.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
from pathlib import Path
import sqlite3

from neuralops.benchmarks.core.paths import StoragePathManager
from neuralops.results_saver import (
    BenchmarkSummaryGenerator,
    DualCommitPersistenceEngine,
    RollingSuiteAggregator,
    RunArtifactManager,
    SQLiteMaintenanceEngine,
    SQLiteMasterLedger,
    read_metrics_parquet,
    write_metrics_parquet,
)


def test_resolve_run_dir_uniqueness_and_layout(tmp_path: Path):
    """Verify that StoragePathManager resolves unique directories with proper layouts."""
    ctx1 = StoragePathManager.resolve_run_dir("Attention Benchmark", output_root=tmp_path)
    ctx2 = StoragePathManager.resolve_run_dir("Attention Benchmark", output_root=tmp_path)

    assert ctx1.run_id != ctx2.run_id
    assert ctx1.benchmark_name == "attention_benchmark"
    assert ctx1.run_dir.exists()
    assert ctx1.artifacts_dir.exists()
    assert ctx1.tensorboard_dir.exists()

    expected_dir = tmp_path / "attention_benchmark" / "runs" / ctx1.run_id
    assert ctx1.run_dir == expected_dir


def test_concurrent_resolve_run_dir_no_collisions(tmp_path: Path):
    """Verify thread-safe unique run ID generation and folder resolution."""
    def _create_run(i: int):
        return StoragePathManager.resolve_run_dir("concurrent_suite", output_root=tmp_path).run_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        run_ids = list(executor.map(_create_run, range(32)))

    assert len(set(run_ids)) == 32


def test_atomic_symlink_and_fallback(tmp_path: Path):
    """Test updating latest symlink and fallback pointer file."""
    ctx1 = StoragePathManager.resolve_run_dir("suite_alpha", run_id="run_001", output_root=tmp_path)
    StoragePathManager.update_latest_symlink(ctx1)

    suite_dir = tmp_path / "suite_alpha"
    latest_link = suite_dir / "latest"
    assert latest_link.exists() or (suite_dir / "latest.txt").exists()

    resolved = StoragePathManager.get_latest_run_dir("suite_alpha", output_root=tmp_path)
    assert resolved == ctx1.run_dir.resolve()

    # Update to newer run
    ctx2 = StoragePathManager.resolve_run_dir("suite_alpha", run_id="run_002", output_root=tmp_path)
    StoragePathManager.update_latest_symlink(ctx2)

    resolved2 = StoragePathManager.get_latest_run_dir("suite_alpha", output_root=tmp_path)
    assert resolved2 == ctx2.run_dir.resolve()


def test_artifact_manager_hashing_and_manifest(tmp_path: Path):
    """Verify RunArtifactManager stores files, computes SHA-256 hashes, and tracks manifests."""
    art_dir = tmp_path / "artifacts"
    test_content = b"Kernel PTX Assembly Dump Content 12345"

    saved_path = RunArtifactManager.save_artifact(
        artifacts_dir=art_dir,
        content_or_path=test_content,
        artifact_name="kernel.ptx",
        artifact_type="cuda_ptx",
    )

    assert saved_path.exists()
    assert saved_path.read_bytes() == test_content

    manifest_entries = RunArtifactManager.list_artifacts(art_dir)
    assert len(manifest_entries) == 1
    assert manifest_entries[0]["filename"] == "kernel.ptx"
    assert manifest_entries[0]["type"] == "cuda_ptx"
    assert manifest_entries[0]["size_bytes"] == len(test_content)
    assert RunArtifactManager.verify_integrity(art_dir) is True


def test_summary_generator_formats(tmp_path: Path):
    """Verify BenchmarkSummaryGenerator constructs valid JSON, CSV, and Markdown."""
    iterations = [
        {"step": 1, "mechanism_or_kernel": "m1", "latency_ms": 10.0, "vram_mb": 500.0, "status": "PASSED"},
        {"step": 2, "mechanism_or_kernel": "m1", "latency_ms": 20.0, "vram_mb": 600.0, "status": "PASSED"},
        {"step": 3, "mechanism_or_kernel": "m1", "latency_ms": 30.0, "vram_mb": 700.0, "status": "PASSED"},
    ]

    aggregates = BenchmarkSummaryGenerator.compute_aggregates(iterations)
    assert aggregates["total_iterations"] == 3
    assert aggregates["passed_iterations"] == 3
    assert aggregates["latency_mean_ms"] == 20.0
    assert aggregates["latency_p50_ms"] == 20.0
    assert aggregates["peak_vram_mb"] == 700.0

    json_file = tmp_path / "summary.json"
    csv_file = tmp_path / "summary.csv"
    md_file = tmp_path / "report.md"

    BenchmarkSummaryGenerator.generate_summary_json(
        run_id="run_123",
        benchmark_name="test_bench",
        aggregates=aggregates,
        iterations=iterations,
        target_path=json_file,
    )
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["run_id"] == "run_123"

    BenchmarkSummaryGenerator.generate_summary_csv(
        iterations=iterations,
        target_path=csv_file,
    )
    assert csv_file.exists()
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3

    md_content = BenchmarkSummaryGenerator.generate_markdown_report(
        run_id="run_123",
        benchmark_name="test_bench",
        aggregates=aggregates,
        iterations=iterations,
        target_path=md_file,
    )
    assert md_file.exists()
    assert "Benchmark Execution Report" in md_content


def test_rolling_suite_aggregator(tmp_path: Path):
    """Verify RollingSuiteAggregator appends multiple runs cleanly without corruption."""
    suite_dir = tmp_path / "suite_beta"

    agg1 = {"total_iterations": 10, "passed_iterations": 10, "latency_p50_ms": 1.5, "peak_vram_mb": 400.0}
    agg2 = {"total_iterations": 10, "passed_iterations": 9, "latency_p50_ms": 1.8, "peak_vram_mb": 420.0}

    RollingSuiteAggregator.append_run(suite_dir, "run_01", "suite_beta", agg1, duration_seconds=5.0)
    RollingSuiteAggregator.append_run(suite_dir, "run_02", "suite_beta", agg2, duration_seconds=5.2)

    master_csv = suite_dir / "summary.csv"
    assert master_csv.exists()

    with open(master_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["run_id"] == "run_01"
        assert rows[1]["run_id"] == "run_02"
        assert float(rows[0]["latency_p50_ms"]) == 1.5
        assert float(rows[1]["latency_p50_ms"]) == 1.8


def test_dual_commit_persistence_engine(tmp_path: Path):
    """Verify end-to-end dual commit to files and master SQLite ledger."""
    db_file = tmp_path / "test_ledger.sqlite"
    engine = DualCommitPersistenceEngine(ledger_path=db_file)

    ctx = StoragePathManager.resolve_run_dir("e2e_test", output_root=tmp_path)
    manifest = {
        "timestamp_start_utc": "2026-11-04T12:00:00Z",
        "timestamp_end_utc": "2026-11-04T12:01:00Z",
        "git_provenance": {"commit_sha": "abc12345", "branch": "main", "is_dirty": False},
        "device_name": "NVIDIA GeForce RTX 4090",
        "hostname": "workstation-node",
    }
    iterations = [
        {"step": 1, "mechanism_or_kernel": "flash_attention", "batch_size": 4, "seq_len": 2048, "dtype": "fp16", "latency_ms": 2.5, "vram_mb": 1200.0, "status": "PASSED"},
        {"step": 2, "mechanism_or_kernel": "flash_attention", "batch_size": 8, "seq_len": 2048, "dtype": "fp16", "latency_ms": 4.8, "vram_mb": 2200.0, "status": "PASSED"},
    ]

    success = engine.finalize_run(
        context=ctx,
        manifest=manifest,
        iterations=iterations,
        duration_seconds=60.0,
        enable_parquet=True,
    )

    assert success is True
    assert ctx.summary_json.exists()
    assert ctx.summary_csv.exists()
    assert ctx.report_md.exists()

    # Verify SQLite ledger records
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    run_row = cur.execute("SELECT run_id, benchmark_name, commit_sha, status FROM benchmark_runs WHERE run_id = ?", (ctx.run_id,)).fetchone()
    assert run_row is not None
    assert run_row[0] == ctx.run_id
    assert run_row[1] == "e2e_test"
    assert run_row[2] == "abc12345"

    metric_rows = cur.execute("SELECT step, mechanism_or_kernel, latency_p50_ms FROM benchmark_metrics WHERE run_id = ? ORDER BY step", (ctx.run_id,)).fetchall()
    assert len(metric_rows) == 2
    assert metric_rows[0][1] == "flash_attention"
    assert metric_rows[0][2] == 2.5
    conn.close()


def test_sqlite_maintenance_engine(tmp_path: Path):
    """Verify SQLiteMaintenanceEngine integrity checks and vacuum."""
    db_file = tmp_path / "maintenance.sqlite"
    with SQLiteMasterLedger(db_path=str(db_file)) as ledger:
        ledger.record_benchmark_run(
            run_id="m_run_1",
            benchmark_name="m_bench",
            commit_sha="sha99",
            hostname="host",
            device_name="device",
            run_dir="results/m_run_1",
            status="PASSED",
        )

    res_integrity = SQLiteMaintenanceEngine.check_integrity(db_file)
    assert res_integrity["status"] == "OK"

    res_vacuum = SQLiteMaintenanceEngine.vacuum_and_reindex(db_file)
    assert res_vacuum["status"] == "SUCCESS"


def test_parquet_writer_roundtrip(tmp_path: Path):
    """Verify columnar Parquet serialization and deserialization."""
    pq_path = tmp_path / "metrics.parquet"
    records = [
        {"step": 1, "kernel": "k1", "latency_ms": 1.23, "vram_mb": 450.0},
        {"step": 2, "kernel": "k2", "latency_ms": 2.34, "vram_mb": 550.0},
    ]

    out_path = write_metrics_parquet(records, pq_path)
    assert out_path is not None
    assert out_path.exists()

    loaded = read_metrics_parquet(pq_path)
    assert len(loaded) == 2
    assert loaded[0]["kernel"] == "k1"
    assert loaded[1]["latency_ms"] == 2.34
