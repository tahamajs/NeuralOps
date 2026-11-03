#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for Unified Telemetry Persistence, SafeTensors IO, and Sweep Resumer.

Validates:
1. BenchmarkResult dataclass schema serialization/deserialization (.to_dict(), .to_json(), .from_dict(), .from_json())
2. MultiFormatSaver writing simultaneously across all 4 targets:
   - raw_results/<run_id>.json
   - .journal.jsonl (with atomic flush / fsync)
   - summary_matrix.csv
   - master_ledger.db (SQLite database)
3. BenchmarkSweepResumer journal parsing, skip detection, and pending matrix filtering
4. SafeTensors IO persistence and round-trip extraction
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import pytest

from neuralops.benchmarks.results import (
    BenchmarkResult,
    BenchmarkSweepResumer,
    MultiFormatSaver,
    dedup_results_ledger,
    load_surgerized_safetensors,
    save_surgerized_safetensors,
)


def _build_sample_result(
    run_id: str = "run_test_01",
    model: str = "gpt2",
    attn: str = "fms_scan",
    suite: str = "vllm_poisson",
    status: str = "success",
) -> BenchmarkResult:
    return BenchmarkResult(
        run_id=run_id,
        model_name=model,
        transplanted_attention=attn,
        pre_train_steps=150,
        benchmark_suite=suite,
        device_type="cuda:0",
        metrics={"ttft_p50_ms": 11.2, "throughput_req_sec": 52.4},
        metadata={"git_commit": "e860173817", "surgeon": "mohawk"},
        status=status,
    )


def test_benchmark_result_schema_methods():
    res = _build_sample_result()
    d = res.to_dict()
    assert d["run_id"] == "run_test_01"
    assert d["model_name"] == "gpt2"
    assert d["transplanted_attention"] == "fms_scan"
    assert d["pre_train_steps"] == 150
    assert d["benchmark_suite"] == "vllm_poisson"
    assert d["device_type"] == "cuda:0"
    assert d["metrics"]["ttft_p50_ms"] == 11.2
    assert d["metadata"]["surgeon"] == "mohawk"

    # Test to_json and from_json
    js_str = res.to_json()
    assert "run_test_01" in js_str
    reconstructed = BenchmarkResult.from_json(js_str)
    assert reconstructed.run_id == res.run_id
    assert reconstructed.model_name == res.model_name
    assert reconstructed.transplanted_attention == res.transplanted_attention
    assert reconstructed.pre_train_steps == res.pre_train_steps
    assert reconstructed.benchmark_suite == res.benchmark_suite
    assert reconstructed.device_type == res.device_type
    assert reconstructed.metrics == res.metrics


def test_multi_format_saver_four_targets():
    with tempfile.TemporaryDirectory() as tmp_dir:
        saver = MultiFormatSaver(output_dir=tmp_dir)
        res1 = _build_sample_result(run_id="run_alpha", status="success")
        res2 = _build_sample_result(run_id="run_beta", attn="gla_attention", status="failed")

        paths1 = saver.save(res1)
        paths2 = saver.save(res2)

        # 1. raw_results/<run_id>.json
        raw1 = paths1["raw_json"]
        raw2 = paths2["raw_json"]
        assert raw1.exists()
        assert raw2.exists()
        raw1_data = json.loads(raw1.read_text(encoding="utf-8"))
        assert raw1_data["run_id"] == "run_alpha"
        assert raw1_data["model_name"] == "gpt2"

        # 2. .journal.jsonl
        journal = Path(tmp_dir) / ".journal.jsonl"
        assert journal.exists()
        lines = journal.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])
        assert entry1["event"] == "CASE_SUCCESS"
        assert entry1["run_id"] == "run_alpha"
        assert entry2["event"] == "CASE_FAILED"
        assert entry2["run_id"] == "run_beta"

        # 3. summary_matrix.csv
        csv_file = Path(tmp_dir) / "summary_matrix.csv"
        assert csv_file.exists()
        csv_lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(csv_lines) == 3  # header + 2 rows
        assert "run_alpha" in csv_lines[1]
        assert "run_beta" in csv_lines[2]

        # 4. master_ledger.db (SQLite)
        db_file = Path(tmp_dir) / "master_ledger.db"
        assert db_file.exists()
        conn = sqlite3.connect(str(db_file))
        cur = conn.cursor()
        cur.execute("SELECT run_id, model_name, transplanted_attention, status FROM benchmark_runs ORDER BY run_id")
        rows = cur.fetchall()
        conn.close()
        assert len(rows) == 2
        assert rows[0] == ("run_alpha", "gpt2", "fms_scan", "success")
        assert rows[1] == ("run_beta", "gpt2", "gla_attention", "failed")


def test_benchmark_sweep_resumer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        saver = MultiFormatSaver(output_dir=tmp_dir)
        res1 = _build_sample_result(run_id="run_1", attn="fms_scan", status="success")
        res2 = _build_sample_result(run_id="run_2", attn="gla_attention", status="failed")
        saver.save(res1)
        saver.save(res2)

        resumer = BenchmarkSweepResumer(output_dir=tmp_dir)
        assert resumer.completed_count == 1

        # Check skip logic
        skip_done, reason_done = resumer.should_skip("gpt2", "fms_scan", "vllm_poisson")
        assert skip_done is True
        assert "Already completed" in reason_done

        # Failed run should not be skipped
        skip_failed, reason_failed = resumer.should_skip("gpt2", "gla_attention", "vllm_poisson")
        assert skip_failed is False

        # Unseen run should not be skipped
        skip_unseen, _ = resumer.should_skip("gpt2", "delta_net", "vllm_poisson")
        assert skip_unseen is False

        # Filter pending matrix
        sweep_matrix = [
            {"model_name": "gpt2", "transplanted_attention": "fms_scan", "benchmark_suite": "vllm_poisson"},
            {"model_name": "gpt2", "transplanted_attention": "gla_attention", "benchmark_suite": "vllm_poisson"},
            {"model_name": "gpt2", "transplanted_attention": "delta_net", "benchmark_suite": "vllm_poisson"},
        ]
        pending = resumer.filter_pending(sweep_matrix)
        assert len(pending) == 2
        assert pending[0]["transplanted_attention"] == "gla_attention"
        assert pending[1]["transplanted_attention"] == "delta_net"


def test_safetensors_io_roundtrip():
    import numpy as np

    fake_state = {
        "weight_a": np.ones((4, 4), dtype=np.float32),
        "weight_b": np.zeros((2, 2), dtype=np.float32),
    }
    meta = {
        "base_model": "gpt2",
        "transplanted_attention": "fms_scan",
        "retention_score": 0.985,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir) / "model.safetensors"
        saved_path = save_surgerized_safetensors(fake_state, target_path, metadata=meta)
        assert saved_path.exists()

        tensors, loaded_meta = load_surgerized_safetensors(saved_path)
        assert "weight_a" in tensors
        assert "weight_b" in tensors
        assert loaded_meta.get("base_model") == "gpt2"
        assert loaded_meta.get("transplanted_attention") == "fms_scan"
        assert float(loaded_meta.get("retention_score", 0)) == pytest.approx(0.985)


def test_dedup_results_ledger():
    with tempfile.TemporaryDirectory() as tmp_dir:
        saver = MultiFormatSaver(output_dir=tmp_dir)
        # Save two runs with identical composite key (gpt2, fms_scan, vllm_poisson)
        res_low = _build_sample_result(run_id="run_low", status="success")
        res_low.metrics = {"accuracy": 0.85}

        res_high = _build_sample_result(run_id="run_high", status="success")
        res_high.metrics = {"accuracy": 0.95}

        saver.save(res_low)
        saver.save(res_high)

        # Confirm 2 runs in sqlite
        db_path = Path(tmp_dir) / "master_ledger.db"
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM benchmark_runs")
        count_before = cur.fetchone()[0]
        conn.close()
        assert count_before == 2

        # Run deduplication
        deleted = dedup_results_ledger(db_path)
        assert deleted == 1

        # Confirm 1 run remaining and it is run_high
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT run_id FROM benchmark_runs")
        remaining = cur.fetchall()
        conn.close()
        assert len(remaining) == 1
        assert remaining[0][0] == "run_high"

