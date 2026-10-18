"""Unit Tests for ResultsSaver, Parquet Exporter & Isolated Storage."""

import os
import tempfile
from pathlib import Path

import pytest
from neuralops.results_saver.saver import (MASTER_CSV_COLUMNS, save_result,
                                           save_trajectories_parquet)


def test_master_csv_columns_integrity():
    assert "accuracy_pct" in MASTER_CSV_COLUMNS
    assert "f1_score" in MASTER_CSV_COLUMNS
    assert "mcc" in MASTER_CSV_COLUMNS
    assert "pearson" in MASTER_CSV_COLUMNS
    assert "fidelity_score" in MASTER_CSV_COLUMNS
    assert "throughput_tokens_sec" in MASTER_CSV_COLUMNS


def test_save_result_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = {
            "module_name": "test_svd",
            "model_family": "bert",
            "dataset": "sst2",
            "status": "PASS",
            "accuracy_pct": 88.5,
            "forward_latency_ms": 25.4,
        }
        csv_file = save_result(res, results_dir=tmpdir, csv_filename="test_results.csv")
        assert os.path.exists(csv_file)
        with open(csv_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "test_svd" in content
            assert "88.5" in content


def test_save_trajectories_parquet():
    with tempfile.TemporaryDirectory() as tmpdir:
        detailed_data = [
            {
                "run_id": "test_run_01",
                "module_name": "svd_linear_attention",
                "model_family": "bert",
                "dataset": "sst2",
                "trajectories": {
                    "loss": [0.69, 0.65, 0.60],
                    "grad_norm": [1.5, 1.2, 0.9],
                },
            }
        ]
        out_path = save_trajectories_parquet(detailed_data, output_dir=tmpdir)
        assert out_path is not None
        assert os.path.exists(out_path)
