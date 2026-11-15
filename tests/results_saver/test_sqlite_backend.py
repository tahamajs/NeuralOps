#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for SQLiteMasterLedger storage and SQL injection safety."""

from __future__ import annotations

import sqlite3
import pytest

from neuralops.results_saver.storage.sqlite_backend import (
    SQLiteMasterLedger,
    init_master_ledger,
)


def test_sqlite_master_ledger_context_manager(tmp_path):
    """Verify __enter__ and __exit__ correctly open and close the connection."""
    db_file = tmp_path / "test_ledger.sqlite"
    with SQLiteMasterLedger(str(db_file)) as ledger:
        assert ledger.conn is not None
        # Connection should be open and operational
        cursor = ledger.conn.cursor()
        cursor.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1

    # After exit, connection should be closed
    with pytest.raises(sqlite3.ProgrammingError):
        ledger.conn.cursor()


def test_sqlite_master_ledger_rollback_on_exception(tmp_path):
    """Verify that an exception inside the context manager causes rollback of uncommitted work."""
    db_file = tmp_path / "rollback_test.sqlite"
    # Create tables first
    with SQLiteMasterLedger(str(db_file)) as ledger:
        pass

    with pytest.raises(RuntimeError, match="simulated failure"):
        with SQLiteMasterLedger(str(db_file)) as ledger:
            cursor = ledger.conn.cursor()
            cursor.execute(
                "INSERT INTO hardware_telemetry (device_name, compute_capability, vram_total_gb, driver_version) VALUES (?, ?, ?, ?)",
                ("TestDev", "9.0", 16.0, "500"),
            )
            # Uncommitted transaction
            raise RuntimeError("simulated failure")

    # Verify uncommitted work was rolled back
    with SQLiteMasterLedger(str(db_file)) as ledger:
        cursor = ledger.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM hardware_telemetry")
        assert cursor.fetchone()[0] == 0


def test_record_run_and_result():
    """Verify hardware telemetry and result recording with SHA-256 provenance."""
    with SQLiteMasterLedger(":memory:") as ledger:
        run_id = ledger.record_run(
            device_name="A100-SXM4-80GB",
            compute_capability="8.0",
            vram_total_gb=80.0,
            driver_version="525.60",
        )
        assert isinstance(run_id, int)
        assert run_id > 0

        res_id, sha = ledger.record_result(
            run_id=run_id,
            architecture_name="llama",
            compression_method="svd",
            batch_size=8,
            seq_len=512,
            latency_ms=12.5,
            throughput_tokens_per_sec=1420.0,
            peak_vram_mb=4200.0,
            accuracy_pct=95.2,
        )
        assert isinstance(res_id, int)
        assert res_id > 0
        assert len(sha) == 64  # valid SHA-256 hex string


def test_record_and_get_benchmark_run():
    """Verify record_benchmark_run and retrieval via get_benchmark_run."""
    with SQLiteMasterLedger(":memory:") as ledger:
        ledger.record_benchmark_run(
            run_id="run-2026-001",
            benchmark_name="kernel_bench",
            commit_sha="abcdef12",
            hostname="test-host",
            device_name="cuda:0",
            run_dir="/tmp/run1",
            status="SUCCESS",
            suite="attention",
            timestamp_start_utc="2026-11-15T00:00:00Z",
            timestamp_end_utc="2026-11-15T00:05:00Z",
            duration_seconds=300.0,
            branch="main",
            is_dirty=0,
            tag="v2.4",
            total_cells=10,
            passed_cells=10,
            failed_cells=0,
        )

        run = ledger.get_benchmark_run("run-2026-001")
        assert run is not None
        assert run["run_id"] == "run-2026-001"
        assert run["benchmark_name"] == "kernel_bench"
        assert run["status"] == "SUCCESS"
        assert run["suite"] == "attention"
        assert run["commit_sha"] == "abcdef12"

        # Non-existent run
        assert ledger.get_benchmark_run("missing-run") is None


def test_record_and_get_benchmark_metrics():
    """Verify record_benchmark_metrics and retrieval via get_benchmark_metrics."""
    with SQLiteMasterLedger(":memory:") as ledger:
        # First create run
        ledger.record_benchmark_run(
            run_id="run-metrics-01",
            benchmark_name="bench",
            commit_sha="12345678",
            hostname="worker",
            device_name="gpu",
            run_dir="/tmp",
            status="SUCCESS",
        )

        metrics = [
            {
                "run_id": "run-metrics-01",
                "step": 1,
                "mechanism_or_kernel": "flash_attn_v2",
                "batch_size": 4,
                "seq_len": 1024,
                "dtype": "fp16",
                "latency_p50_ms": 2.1,
                "latency_p90_ms": 2.4,
                "latency_p99_ms": 2.8,
                "tflops": 120.5,
                "peak_vram_mb": 1024.0,
                "extra_metadata": {"head_dim": 64},
            },
            {
                "run_id": "run-metrics-01",
                "step": 2,
                "mechanism_or_kernel": "flash_attn_v2",
                "batch_size": 4,
                "seq_len": 2048,
                "dtype": "fp16",
                "latency_p50_ms": 4.5,
                "latency_p90_ms": 5.0,
                "latency_p99_ms": 5.5,
                "tflops": 125.0,
                "peak_vram_mb": 2048.0,
                "extra_metadata": "plain_string",
            },
        ]
        ledger.record_benchmark_metrics(metrics)

        retrieved = ledger.get_benchmark_metrics("run-metrics-01")
        assert len(retrieved) == 2
        assert retrieved[0]["mechanism_or_kernel"] == "flash_attn_v2"
        assert retrieved[0]["batch_size"] == 4
        assert retrieved[0]["latency_p50_ms"] == 2.1
        assert "head_dim" in retrieved[0]["extra_metadata"]


def test_list_benchmark_runs():
    """Verify listing runs with optional suite filter and limit."""
    with SQLiteMasterLedger(":memory:") as ledger:
        for i in range(5):
            ledger.record_benchmark_run(
                run_id=f"run-{i}",
                benchmark_name="bench",
                commit_sha="sha",
                hostname="h",
                device_name="d",
                run_dir="r",
                status="SUCCESS",
                suite="suite_a" if i % 2 == 0 else "suite_b",
                timestamp_start_utc=f"2026-11-15T0{i}:00:00Z",
            )

        all_runs = ledger.list_benchmark_runs(limit=10)
        assert len(all_runs) == 5

        suite_a_runs = ledger.list_benchmark_runs(suite="suite_a")
        assert len(suite_a_runs) == 3


def test_sql_injection_resistance():
    """Verify parametrized queries resist malicious SQL injection payloads."""
    with SQLiteMasterLedger(":memory:") as ledger:
        malicious_run_id = "run'; DROP TABLE benchmark_runs; --"
        malicious_kernel = "kernel'); DROP TABLE benchmark_metrics; --"

        # Insertion should safely escape
        ledger.record_benchmark_run(
            run_id=malicious_run_id,
            benchmark_name="injection_test",
            commit_sha="sha",
            hostname="h",
            device_name="d",
            run_dir="r",
            status="SUCCESS",
        )

        ledger.record_benchmark_metrics(
            [
                {
                    "run_id": malicious_run_id,
                    "step": 0,
                    "mechanism_or_kernel": malicious_kernel,
                    "batch_size": 1,
                    "seq_len": 128,
                    "dtype": "fp32",
                    "latency_p50_ms": 1.0,
                }
            ]
        )

        # Confirm tables still exist and malicious payloads were stored verbatim
        fetched_run = ledger.get_benchmark_run(malicious_run_id)
        assert fetched_run is not None
        assert fetched_run["run_id"] == malicious_run_id

        fetched_metrics = ledger.get_benchmark_metrics(malicious_run_id)
        assert len(fetched_metrics) == 1
        assert fetched_metrics[0]["mechanism_or_kernel"] == malicious_kernel


def test_init_master_ledger_convenience(tmp_path):
    """Verify init_master_ledger convenience initializer."""
    db_file = tmp_path / "conv_ledger.sqlite"
    ledger = init_master_ledger(str(db_file))
    try:
        assert isinstance(ledger, SQLiteMasterLedger)
        cursor = ledger.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        assert "benchmark_runs" in tables
        assert "benchmark_metrics" in tables
        assert "hardware_telemetry" in tables
        assert "master_benchmark_results" in tables
    finally:
        ledger.close()
