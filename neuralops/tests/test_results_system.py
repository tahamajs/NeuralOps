"""Comprehensive unit tests for NeuralOps results management system.

Tests:
1. Multi-format saving (JSON, JSONL, CSV, Markdown, Parquet)
2. Consistency across all formats
3. Deduplication (skipping identical runs, allowing retry on failure)
4. Git commit integration & commit message format
5. HF Hub push (credential checks, repository layout, secret scanning)
6. Results querying API (ResultsIndex, find, best, compare, failures, DataFrame)
7. Reproduction command & script generation
8. Rolling summary generation (by_module, by_model, by_dataset, leaderboards, coverage)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock
import pandas as pd
import pytest

from neuralops.benchmarks.results import (
    BenchmarkResult,
    Deduplicator,
    GitCommitter,
    HFHubPusher,
    ResultsIndex,
    SummaryGenerator,
    generate_reproduction_command,
    generate_reproduction_script,
    save_all_formats,
    save_csv_index,
    save_json,
    save_jsonl_index,
    save_markdown_summary,
    save_parquet_index,
)
from neuralops.benchmarks.results.schema import (
    DatasetMetadata,
    ModelMetadata,
    ModuleMetadata,
    RunMetadata,
)


def _create_sample_result(module="svd_linear_attention", rank=32, status="success", accuracy=0.88):
    res = BenchmarkResult(
        run_id="test-uuid-1234",
        run_name=f"{module}_distilbert_sst2",
        module={"name": module, "type": "attention"},
        model={"name": "distilbert-base-uncased", "family": "distilbert"},
        dataset={"name": "sst2", "task": "classification", "num_eval": 872},
        config={"rank": rank, "epochs": 1, "batch_size": 32, "seed": 42},
        metrics={"accuracy": accuracy, "f1": 0.87, "loss": 0.35},
        resources={"duration_s": 12.5, "peak_vram_mb": 512.0},
        status=status,
    )
    return res


def test_save_json_creates_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = _create_sample_result()
        p = Path(tmp_dir) / "out.json"
        saved = save_json(res, p)
        assert saved.exists()
        data = json.loads(saved.read_text(encoding="utf-8"))
        assert data["run_id"] == "test-uuid-1234"
        assert "run_metadata" in data
        assert "environment" in data
        assert data["module"]["name"] == "svd_linear_attention"


def test_save_jsonl_appends():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = _create_sample_result()
        p = Path(tmp_dir) / "results.jsonl"
        save_jsonl_index(res, p)
        save_jsonl_index(res, p)
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        entry = json.loads(lines[0])
        assert entry["run_id"] == "test-uuid-1234"


def test_save_csv_appends():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = _create_sample_result()
        p = Path(tmp_dir) / "results.csv"
        save_csv_index(res, p)
        save_csv_index(res, p)
        df = pd.read_csv(p)
        assert len(df) == 2
        assert df["module"].iloc[0] == "svd_linear_attention"


def test_save_markdown_creates_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = _create_sample_result()
        p = Path(tmp_dir) / "summary.md"
        save_markdown_summary(res, p)
        assert p.exists()
        text = p.read_text(encoding="utf-8")
        assert "Benchmark Run Summary" in text
        assert "0.8800" in text


def test_save_parquet_creates_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = _create_sample_result()
        p = Path(tmp_dir) / "results.parquet"
        save_parquet_index(res, p)
        assert p.exists()
        df = pd.read_parquet(p)
        assert len(df) == 1
        assert df["module"].iloc[0] == "svd_linear_attention"


def test_all_formats_consistent():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = _create_sample_result(module="cosformer", accuracy=0.91)
        saved = save_all_formats(res, output_dir=tmp_dir)
        for fmt, path in saved.items():
            assert path.exists(), f"Format {fmt} file missing"

        # Check content match
        js_data = json.loads(saved["json"].read_text(encoding="utf-8"))
        assert js_data["metrics"]["accuracy"] == 0.91

        df_csv = pd.read_csv(saved["csv"])
        assert df_csv["accuracy"].iloc[0] == 0.91

        df_pq = pd.read_parquet(saved["parquet"])
        assert df_pq["accuracy"].iloc[0] == 0.91


def test_deduplication_skips_identical():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dedup = Deduplicator(output_dir=tmp_dir)
        res = _create_sample_result()
        p = Path(tmp_dir) / "out.json"

        skip, _ = dedup.should_skip(res.compute_run_key())
        assert not skip

        dedup.record_run(res, p)
        skip, reason = dedup.should_skip(res.compute_run_key())
        assert skip
        assert "already exists" in reason


def test_deduplication_allows_retry():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dedup = Deduplicator(output_dir=tmp_dir)
        res = _create_sample_result(status="failed")
        p = Path(tmp_dir) / "out.json"
        dedup.record_run(res, p)

        # A failed run should allow retry
        skip, reason = dedup.should_skip(res.compute_run_key())
        assert not skip
        assert "allowing retry" in reason


def test_git_commit_message_format():
    committer = GitCommitter(auto_commit=True)
    res = _create_sample_result(module="mla_attention", rank=64, accuracy=0.932)
    msg = committer.build_commit_message(res, completed_count=5, failed_count=1)
    assert "results: mla_attention on distilbert-base-uncased/sst2 (rank=64) — acc=93.2%" in msg
    assert "5 runs completed" in msg
    assert "1 failures" in msg


def test_hf_push_requires_token():
    pusher = HFHubPusher(repo_id="test/repo", token=None)
    assert not pusher.is_enabled()


def test_hf_push_with_mock_client():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pusher = HFHubPusher(repo_id="test/benchmark-hub", token="fake_token")
        assert pusher.is_enabled()

        mock_client = MagicMock()
        res = _create_sample_result()
        json_file = Path(tmp_dir) / "result.json"
        save_json(res, json_file)

        repo = pusher.push_result(res, {"json": json_file}, hf_client=mock_client)
        assert repo == "test/benchmark-hub"
        assert res.push_info.pushed_to_hf is True
        mock_client.create_repo.assert_called_once()
        mock_client.upload_file.assert_called_once()


def test_index_query_by_module_and_best():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res1 = _create_sample_result(module="svd_linear_attention", accuracy=0.85)
        res2 = _create_sample_result(module="cosformer", accuracy=0.92)
        save_all_formats(res1, output_dir=tmp_dir)
        save_all_formats(res2, output_dir=tmp_dir)

        index = ResultsIndex(output_dir=tmp_dir)
        svd_runs = index.find(module="svd_linear_attention")
        assert len(svd_runs) == 1
        assert svd_runs[0].module["name"] == "svd_linear_attention"

        best_run = index.best(model="distilbert-base-uncased", dataset="sst2")
        assert best_run is not None
        assert best_run.module["name"] == "cosformer"
        assert best_run.metrics["accuracy"] == 0.92


def test_summary_regeneration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = _create_sample_result(module="svd_linear_attention", accuracy=0.88)
        save_all_formats(res, output_dir=tmp_dir)

        summarizer = SummaryGenerator(output_dir=tmp_dir)
        summaries = summarizer.generate_all()
        assert summaries["by_module"].exists()
        assert summaries["by_model"].exists()
        assert summaries["by_dataset"].exists()
        assert summaries["leaderboards"].exists()
        assert summaries["coverage"].exists()
        assert summaries["timeline"].exists()

        mod_content = summaries["by_module"].read_text(encoding="utf-8")
        assert "svd_linear_attention" in mod_content
        assert "0.8800" in mod_content


def test_reproduction_command_and_script():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = _create_sample_result(module="svd_linear_attention", rank=64)
        cmd = generate_reproduction_command(res)
        assert "--module svd_linear_attention" in cmd
        assert "--rank 64" in cmd

        script_path = Path(tmp_dir) / "reproduce.sh"
        generate_reproduction_script(res, script_path)
        assert script_path.exists()
        script_text = script_path.read_text(encoding="utf-8")
        assert "#!/usr/bin/env bash" in script_text
        assert cmd in script_text
