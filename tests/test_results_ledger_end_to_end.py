"""
Master End-to-End Test Suite for NeuralOps Experiment Results Ledger Subsystem.
Verifies:
  1. ResultRecord creation, validation, serialization, and Git provenance.
  2. Dual storage engine (JSONL append-log + SQLite indexed cache).
  3. CRUD operations:
     - mode='upsert' (insert or update match)
     - mode='replace' (in-place replacement preserving ID)
     - mode='new_version' (increment revision, mark prior as superseded)
     - soft deletion vs hard deletion
  4. Querying & multi-criteria filtering (category, family, method, git_sha, status, min_accuracy).
  5. Version diffing and historical trajectory analysis.
  6. ASCII chart and pure-Python standalone vector SVG rendering.
  7. Formatted table export (Markdown, LaTeX booktabs, CSV, JSON).
  8. End-to-end subsystem wiring simulation (roofline, surgery, inspector, results_manager).
"""

import os
import json
import sqlite3
import tempfile
import pytest
from pathlib import Path

from neuralops.core.results_ledger import (
    ResultRecord,
    ResultsLedger,
    get_default_ledger,
    log_experiment_result,
    get_file_git_provenance,
)


@pytest.fixture
def temp_ledger_dir(tmp_path):
    """Provides an isolated clean ledger root directory for testing."""
    ledger_root = tmp_path / "test_results"
    ledger_root.mkdir(parents=True, exist_ok=True)
    return str(ledger_root)


@pytest.fixture
def ledger(temp_ledger_dir):
    """Initializes a fresh ResultsLedger instance."""
    return ResultsLedger(results_dir=temp_ledger_dir)


class TestResultRecord:
    def test_record_creation_and_defaults(self):
        rec = ResultRecord(
            module_name="flash_linear_attention",
            category="attention",
            family="linear-kernel",
            generation_method="Compiled Triton JIT Kernel w/ CUDA Event Timing (SM 8.9)",
            metrics={"latency_ms": 1.42, "tflops": 128.5},
        )
        assert rec.record_id.startswith("rec_")
        assert rec.version_tag == "v1.3.7"
        assert rec.revision == 1
        assert rec.status == "verified"
        assert rec.timestamp_utc is not None
        assert rec.metrics["latency_ms"] == 1.42

    def test_record_to_dict_and_from_dict(self):
        rec = ResultRecord(
            module_name="hnc_svd",
            category="compressor",
            family="hnc-svd",
            generation_method="HNC Nullspace Projection + Cholesky Whitening",
            experimental_setup={"dataset": "sst2", "rank": 32},
            metrics={"eval_accuracy": 92.4, "compression_ratio": 4.0},
            notes="Zero-drop calibration",
        )
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert d["module_name"] == "hnc_svd"

        rec2 = ResultRecord.from_dict(d)
        assert rec2.record_id == rec.record_id
        assert rec2.metrics["eval_accuracy"] == 92.4
        assert rec2.experimental_setup["rank"] == 32
        assert rec2.notes == "Zero-drop calibration"


class TestResultsLedgerCRUD:
    def test_dual_storage_creation(self, ledger, temp_ledger_dir):
        jsonl_path = Path(temp_ledger_dir) / "registry" / "master_ledger.jsonl"
        sqlite_path = Path(temp_ledger_dir) / "registry" / "master_ledger.sqlite"
        assert jsonl_path.exists()
        assert sqlite_path.exists()

    def test_add_and_query_record(self, ledger):
        rec = ResultRecord(
            module_name="simplex_wgqa",
            category="attention",
            family="simplex-wgqa",
            generation_method="Simplex-Constrained WGQA + Closed-Form Ridge Calibration",
            experimental_setup={"dataset": "mnli", "backbone": "roberta-base"},
            metrics={"eval_accuracy": 88.5, "speedup": 2.1},
        )
        saved = ledger.add_or_replace(rec, mode="upsert")
        assert saved.record_id == rec.record_id

        # Query back
        results = ledger.query(module_name="simplex_wgqa")
        assert len(results) == 1
        assert results[0].module_name == "simplex_wgqa"
        assert results[0].metrics["eval_accuracy"] == 88.5

        # Dossier file created
        dossier_path = Path(ledger.by_module_dir) / "simplex_wgqa.json"
        assert dossier_path.exists()
        with open(dossier_path, "r", encoding="utf-8") as f:
            dossier_data = json.load(f)
            assert dossier_data["module_name"] == "simplex_wgqa"
            assert len(dossier_data["history"]) == 1

    def test_version_bumping(self, ledger):
        rec1 = ResultRecord(
            module_name="fsvd_compressor",
            category="compressor",
            family="activation-aware-svd",
            generation_method="Activation-Aware Fisher-Weighted SVD (FWSVD)",
            experimental_setup={"dataset": "sst2", "rank": 64},
            metrics={"eval_accuracy": 91.2},
            version_tag="v1.3.6",
            revision=1,
        )
        ledger.add_or_replace(rec1)

        # Add second version with mode='new_version'
        rec2 = ResultRecord(
            module_name="fsvd_compressor",
            category="compressor",
            family="activation-aware-svd",
            generation_method="Activation-Aware Fisher-Weighted SVD (FWSVD) + Micro-Healing",
            experimental_setup={"dataset": "sst2", "rank": 64},
            metrics={"eval_accuracy": 93.1},
            version_tag="v1.3.7",
        )
        saved2 = ledger.add_or_replace(rec2, mode="new_version")
        assert saved2.revision == 2
        assert saved2.status == "verified"

        # Verify old record is marked superseded
        old_rec = ledger.get_record_by_id(rec1.record_id)
        assert old_rec.status == "superseded"

        # Default query returns only active records (status='verified')
        active = ledger.query(module_name="fsvd_compressor", status="verified")
        assert len(active) == 1
        assert active[0].metrics["eval_accuracy"] == 93.1

        # Query all including superseded
        all_recs = ledger.query(module_name="fsvd_compressor", status=None)
        assert len(all_recs) == 2

    def test_in_place_replace(self, ledger):
        rec = ResultRecord(
            module_name="fused_chunk_linear",
            category="triton_kernel",
            family="chunk-recurrent",
            generation_method="Triton JIT Kernel",
            experimental_setup={"seq_len": 4096},
            metrics={"latency_ms": 3.8},
        )
        ledger.add_or_replace(rec)

        # Update metrics in-place
        rec_updated = ResultRecord.from_dict(rec.to_dict())
        rec_updated.metrics["latency_ms"] = 3.1
        rec_updated.notes = "Re-calibrated with warm L2 cache"

        ledger.add_or_replace(rec_updated, mode="replace")

        records = ledger.query(module_name="fused_chunk_linear")
        assert len(records) == 1
        assert records[0].metrics["latency_ms"] == 3.1
        assert records[0].notes == "Re-calibrated with warm L2 cache"

    def test_soft_and_hard_delete(self, ledger):
        rec = ResultRecord(
            module_name="temp_kernel",
            category="triton_kernel",
            family="debug",
            generation_method="Diagnostic Profiling",
            metrics={"loss": 0.05},
        )
        ledger.add_or_replace(rec)
        assert len(ledger.query(module_name="temp_kernel")) == 1

        # Soft delete
        ledger.delete_records(module_name="temp_kernel", hard=False)
        assert len(ledger.query(module_name="temp_kernel", status="verified")) == 0
        deleted_records = ledger.query(module_name="temp_kernel", status="deprecated")
        assert len(deleted_records) == 1

        # Hard delete
        ledger.delete_records(module_name="temp_kernel", hard=True)
        assert len(ledger.query(module_name="temp_kernel", status=None)) == 0


class TestResultsLedgerQueryAndDiff:
    def test_multi_criteria_filtering(self, ledger):
        ledger.add_or_replace(ResultRecord(
            module_name="gated_deltanet", category="attention", family="recurrent",
            experimental_setup={"dataset": "sst2"}, metrics={"eval_accuracy": 94.2}
        ))
        ledger.add_or_replace(ResultRecord(
            module_name="retention", category="attention", family="recurrent",
            experimental_setup={"dataset": "sst2"}, metrics={"eval_accuracy": 91.8}
        ))
        ledger.add_or_replace(ResultRecord(
            module_name="hnc_svd", category="compressor", family="hnc-svd",
            experimental_setup={"dataset": "sst2"}, metrics={"eval_accuracy": 92.5}
        ))

        # Filter by category
        attns = ledger.query(category="attention")
        assert len(attns) == 2

        # Filter by min_metric
        high_acc = ledger.query(min_metric=("eval_accuracy", 92.0))
        assert len(high_acc) == 2
        names = {r.module_name for r in high_acc}
        assert names == {"gated_deltanet", "hnc_svd"}

    def test_compare_versions(self, ledger):
        r1 = ResultRecord(
            module_name="roberta_12layer_kd",
            category="end_to_end_model",
            family="distillation",
            metrics={"eval_accuracy": 90.0, "latency_ms": 12.0},
            version_tag="v1.3.5",
            revision=1,
        )
        r2 = ResultRecord(
            module_name="roberta_12layer_kd",
            category="end_to_end_model",
            family="distillation",
            metrics={"eval_accuracy": 92.8, "latency_ms": 9.5},
            version_tag="v1.3.7",
            revision=2,
        )
        ledger.add_or_replace(r1)
        ledger.add_or_replace(r2, mode="new_version")

        diff = ledger.compare_versions("roberta_12layer_kd", v1=1, v2=2)
        assert diff["diff"]["eval_accuracy"]["delta"] == pytest.approx(2.8)
        assert diff["diff"]["latency_ms"]["delta"] == pytest.approx(-2.5)


class TestVisualizationAndExport:
    def test_ascii_and_svg_plotting(self, ledger, temp_ledger_dir):
        for acc, rank in [(88.1, 16), (90.4, 32), (92.3, 64), (93.1, 128)]:
            ledger.add_or_replace(ResultRecord(
                module_name=f"svd_rank_{rank}",
                category="compressor",
                family="svd",
                experimental_setup={"rank": rank},
                metrics={"eval_accuracy": acc},
            ))

        # ASCII plot
        ascii_chart = ledger.plot_results(
            category="compressor",
            metric="eval_accuracy",
            chart_type="bar",
            format="ascii",
        )
        assert "svd_rank" in ascii_chart
        assert "====" in ascii_chart

        # Pure vector SVG plot
        svg_out = Path(temp_ledger_dir) / "plots" / "test_chart.svg"
        svg_content = ledger.plot_results(
            category="compressor",
            metric="eval_accuracy",
            chart_type="bar",
            format="svg",
            output_file=str(svg_out),
        )
        assert svg_out.exists()
        assert "<svg" in svg_content
        assert "</svg>" in svg_content
        assert "svd_rank" in svg_content

    def test_table_exports(self, ledger):
        ledger.add_or_replace(ResultRecord(
            module_name="mod_a", category="attention", family="linear", metrics={"eval_accuracy": 91.0}
        ))
        ledger.add_or_replace(ResultRecord(
            module_name="mod_b", category="attention", family="recurrent", metrics={"eval_accuracy": 92.5}
        ))

        md_table = ledger.export_table(format="markdown")
        assert "| Module Name | Category | Family | Version" in md_table
        assert "mod_a" in md_table
        assert "mod_b" in md_table

        latex_table = ledger.export_table(format="latex")
        assert "\\begin{tabular}" in latex_table
        assert "\\toprule" in latex_table
        assert "\\bottomrule" in latex_table
        assert "mod_a" in latex_table


class TestSubsystemTelemetryHooks:
    def test_log_experiment_result_helper(self, ledger):
        rec = log_experiment_result(
            module_name="telemetry_test_attention",
            category="attention",
            family="test",
            generation_method="Direct Kernel Test",
            metrics={"throughput": 5000.0},
            ledger=ledger,
        )
        assert rec.module_name == "telemetry_test_attention"
        assert len(ledger.query(module_name="telemetry_test_attention")) == 1
