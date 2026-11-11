#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for `scripts/benchmarks/export_dual_zoo_5formats.py` across all 6 Zoos."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_BENCHMARKS = REPO_ROOT / "scripts" / "benchmarks"
if str(SCRIPTS_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_BENCHMARKS))

import export_dual_zoo_5formats as exporter  # noqa: E402
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec) + "\n")
    return path


def test_dual_zoo_backward_compatibility(tmp_path: Path) -> None:
    """Existing 2-ledger CLI flags and output structure remain backward-compatible."""
    qwen_ledger = _write_jsonl(
        tmp_path / "qwen_attention_zoo" / "metrics.jsonl",
        [
            {
                "model": "Qwen/Qwen2.5-0.5B",
                "attn_key": "gated_delta_net",
                "dataset": "wikitext-2",
                "pattern": "all",
                "ppl_before": 48.2,
                "ppl_after": 19.5,
                "ppl_delta": -28.7,
                "train_secs": 120.5,
                "steps_s1": 100,
                "steps_s2": 200,
                "total_steps": 300,
                "peak_vram_mb": 4200,
                "timestamp": "2026-11-10T10:00:00Z",
                "status": "PASSED",
            }
        ],
    )
    bert_ledger = _write_jsonl(
        tmp_path / "bert_compressor_zoo" / "multirank_lora_ledger.jsonl",
        [
            {
                "compressor": "hnc_svd",
                "rank": 64,
                "ratio": 2.4,
                "orig_params": 109483778,
                "comp_params": 45618240,
                "zero_shot_acc": 72.1,
                "zero_shot_loss": 0.61,
                "delta_zero": -19.53,
                "healed_acc": 90.4,
                "healed_loss": 0.24,
                "delta_healed": -1.23,
                "lora_healed": True,
                "teacher_acc": 91.63,
                "duration_s": 44.2,
                "peak_vram_mb": 2048,
                "timestamp": "2026-11-10T10:05:00Z",
                "status": "PASSED",
            }
        ],
    )

    out_dir = tmp_path / "out_dual"
    rc = exporter.main(
        [
            "--qwen-ledger",
            str(qwen_ledger),
            "--bert-ledger",
            str(bert_ledger),
            "--output-dir",
            str(out_dir),
            "--no-require-parquet",
        ]
    )
    assert rc == exporter.EXIT_SUCCESS
    assert (out_dir / "results.jsonl").exists()
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "zoo_overview.json").exists()

    # Re-running without --overwrite must fail closed with usage error.
    rc_dup = exporter.main(
        [
            "--qwen-ledger",
            str(qwen_ledger),
            "--output-dir",
            str(out_dir),
            "--no-require-parquet",
        ]
    )
    assert rc_dup == exporter.EXIT_USAGE_ERROR

    # Re-running with --overwrite must succeed.
    rc_ow = exporter.main(
        [
            "--qwen-ledger",
            str(qwen_ledger),
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--no-require-parquet",
        ]
    )
    assert rc_ow == exporter.EXIT_SUCCESS
    lines = (out_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_all_six_zoos_export_and_overview(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """All 6 optional zoo ledgers aggregate into 5 canonical formats and zoo_overview.json."""
    qwen_ledger = _write_jsonl(
        tmp_path / "qwen_attention" / "metrics.jsonl",
        [
            {
                "model": "Qwen/Qwen2.5-0.5B",
                "attn_key": "gla_attention",
                "dataset": "wikitext-2",
                "ppl_before": 52.0,
                "ppl_after": 21.4,
                "train_secs": 95.0,
                "peak_vram_mb": 3900,
                "timestamp": "2026-11-11T01:00:00Z",
                "status": "PASSED",
            }
        ],
    )
    bert_comp_ledger = _write_jsonl(
        tmp_path / "bert_compressor" / "multirank_lora_ledger.jsonl",
        [
            {
                "compressor": "fisher_svd",
                "rank": 32,
                "ratio": 3.1,
                "orig_params": 109483778,
                "comp_params": 35317347,
                "zero_shot_acc": 68.0,
                "healed_acc": 89.2,
                "lora_healed": True,
                "duration_s": 38.5,
                "peak_vram_mb": 1800,
                "timestamp": "2026-11-11T01:05:00Z",
                "status": "PASSED",
            }
        ],
    )
    bert_attn_ledger = _write_jsonl(
        tmp_path / "bert_attention" / "metrics.jsonl",
        [
            {
                "attn_key": "adaptive_span_attention",
                "module": "adaptive_span_attention",
                "teacher_baseline_acc": 91.63,
                "pre_kd_acc": 51.2,
                "post_kd_acc": 88.9,
                "acc_delta": 37.7,
                "post_kd_loss": 0.29,
                "train_secs": 180.0,
                "steps_s1": 150,
                "steps_s2": 300,
                "total_steps": 450,
                "peak_vram_mb": 5120,
                "tokens_per_sec": 42000.0,
                "timestamp": "2026-11-11T01:10:00Z",
                "status": "PASSED",
            },
            {
                "attn_key": "oom_attention",
                "status": "SKIPPED_OOM",
                "error": "CUDA out of memory",
                "timestamp": "2026-11-11T01:12:00Z",
            },
        ],
    )
    qwen_comp_ledger = _write_jsonl(
        tmp_path / "qwen_compressor" / "metrics.jsonl",
        [
            {
                "model": "Qwen/Qwen2.5-0.5B",
                "compressor": "wanda_2_4",
                "rank": 16,
                "param_reduction": 1.65,
                "sparsity_pct": 50.0,
                "orig_params": 494032768,
                "comp_params": 299413798,
                "ppl_before": 40.5,
                "ppl_after": 18.7,
                "ppl_delta": -21.8,
                "train_secs": 140.2,
                "steps_s1": 80,
                "steps_s2": 160,
                "peak_vram_mb": 6400,
                "timestamp": "2026-11-11T01:15:00Z",
                "status": "PASSED",
            },
            {
                "model": "Qwen/Qwen2.5-0.5B",
                "compressor": "broken_comp",
                "status": "FAIL_TRAINING",
                "error": "Shape mismatch during surgery",
                "timestamp": "2026-11-11T01:16:00Z",
            },
        ],
    )
    vlm_ledger = _write_jsonl(
        tmp_path / "vlm_llava" / "metrics.jsonl",
        [
            {
                "model": "llava-hf/llava-interleave-qwen-0.5b-hf",
                "attn_key": "retnet_attention",
                "dataset": "derek-thomas/ScienceQA",
                "pattern": "middle_50",
                "total_layers": 24,
                "surgeried_layers": [6, 7, 8],
                "anchor_layers": [0, 1, 2],
                "teacher_acc": 74.5,
                "acc_before": 32.0,
                "acc_after": 69.8,
                "acc_delta": 37.8,
                "ppl_before": 19.4,
                "ppl_after": 7.8,
                "ppl_delta": -11.6,
                "train_secs": 210.0,
                "steps_s1": 90,
                "steps_s2": 180,
                "total_steps": 270,
                "peak_vram_mb": 11200,
                "timestamp": "2026-11-11T01:20:00Z",
                "status": "PASSED",
            }
        ],
    )
    tinyllama_ledger = _write_jsonl(
        tmp_path / "tinyllama" / "metrics.jsonl",
        [
            {
                "model": "TinyLlama/TinyLlama_v1.1",
                "attn_key": "mamba2_attention",
                "dataset": "wikitext-2",
                "pattern": "every_other",
                "total_layers": 22,
                "teacher_ppl": 9.8,
                "ppl_before": 35.1,
                "ppl_after": 12.6,
                "ppl_delta": -22.5,
                "train_secs": 240.0,
                "steps_s1": 120,
                "steps_s2": 240,
                "total_steps": 360,
                "peak_vram_mb": 14300,
                "timestamp": "2026-11-11T01:25:00Z",
                "status": "PASSED",
            }
        ],
    )

    out_dir = tmp_path / "all_6_zoos_export"
    rc = exporter.main(
        [
            "--qwen-ledger",
            str(qwen_ledger),
            "--bert-ledger",
            str(bert_comp_ledger),
            "--bert-attn-ledger",
            str(bert_attn_ledger),
            "--qwen-comp-ledger",
            str(qwen_comp_ledger),
            "--vlm-ledger",
            str(vlm_ledger),
            "--tinyllama-ledger",
            str(tinyllama_ledger),
            "--output-dir",
            str(out_dir),
            "--require-parquet",
            "--json",
        ]
    )
    assert rc == exporter.EXIT_SUCCESS

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["total_records"] == 8
    assert summary["qwen_records"] == 1
    assert summary["bert_records"] == 1
    assert summary["bert_attn_records"] == 2
    assert summary["qwen_comp_records"] == 2
    assert summary["vlm_records"] == 1
    assert summary["tinyllama_records"] == 1
    assert summary["verification_failures"] == 0

    # Verify JSONL index
    jsonl_path = out_dir / "results.jsonl"
    assert jsonl_path.exists()
    jsonl_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(jsonl_rows) == 8

    # Verify CSV index
    csv_path = out_dir / "results.csv"
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        csv_rows = list(csv.DictReader(fp))
    assert len(csv_rows) == 8
    modules_in_csv = {row["module"] for row in csv_rows}
    assert modules_in_csv == {
        "gla_attention",
        "fisher_svd",
        "adaptive_span_attention",
        "oom_attention",
        "wanda_2_4",
        "broken_comp",
        "retnet_attention",
        "mamba2_attention",
    }

    # Verify Parquet index
    assert (out_dir / "results.parquet").exists()

    # Verify individual canonical JSON & Markdown files
    assert len(summary["json_files"]) == 8
    assert len(summary["markdown_files"]) == 8
    for json_file in summary["json_files"]:
        jp = Path(json_file)
        assert jp.exists()
        res = BenchmarkResult.from_json(jp.read_text(encoding="utf-8"))
        assert res.run_id
    for md_file in summary["markdown_files"]:
        mp = Path(md_file)
        assert mp.exists()
        assert "Benchmark Run Summary:" in mp.read_text(encoding="utf-8")

    # Verify zoo_overview.json
    overview_path = out_dir / "zoo_overview.json"
    assert overview_path.exists()
    overview = json.loads(overview_path.read_text(encoding="utf-8"))
    assert overview["total_records"] == 8
    assert set(overview["zoos_included"]) == {
        "qwen_attention_zoo",
        "bert_compressor_zoo",
        "bert_attention_zoo",
        "qwen_compressor_zoo",
        "vlm_llava_scienceqa_zoo",
        "tinyllama_1.1b_zoo",
    }
    assert overview["zoos"]["bert_attention_zoo"]["passed_count"] == 1
    assert overview["zoos"]["bert_attention_zoo"]["skipped_count"] == 1
    assert overview["zoos"]["bert_attention_zoo"]["metrics_summary"]["accuracy"]["best_module"] == "adaptive_span_attention"
    assert overview["zoos"]["qwen_compressor_zoo"]["failed_count"] == 1
    assert overview["zoos"]["qwen_compressor_zoo"]["metrics_summary"]["compression_ratio"]["max"] == 1.65
    assert overview["zoos"]["vlm_llava_scienceqa_zoo"]["datasets"] == ["derek-thomas/ScienceQA"]
    assert overview["zoos"]["tinyllama_1.1b_zoo"]["metrics_summary"]["post_ppl"]["best_value"] == 12.6
    # Ensure no fabricated metrics appear for failed/skipped records
    oom_json = next(
        BenchmarkResult.from_json(Path(p).read_text(encoding="utf-8"))
        for p in summary["json_files"]
        if "oom_attention" in p
    )
    assert oom_json.status == BenchmarkStatus.SKIPPED.value
    assert oom_json.metrics == {}
    assert oom_json.error == "CUDA out of memory"


def test_fail_closed_on_missing_passed_metric_and_non_finite_values(tmp_path: Path) -> None:
    """PASSED records missing primary metrics or containing NaN/Inf fail closed."""
    vlm_ledger = _write_jsonl(
        tmp_path / "vlm_bad.jsonl",
        [
            {
                "attn_key": "missing_metrics_attn",
                "status": "PASSED",
                "timestamp": "2026-11-11T02:00:00Z",
            },
            {
                "attn_key": "nan_metric_attn",
                "acc_after": "NaN",
                "status": "PASSED",
                "timestamp": "2026-11-11T02:01:00Z",
            },
        ],
    )
    out_dir = tmp_path / "out_fail_closed"
    summary = exporter.export_dual_zoo_5formats(
        output_dir=out_dir,
        vlm_jsonl=vlm_ledger,
        require_parquet=False,
        strict=False,
    )
    assert summary["verification_failures"] == 2
    assert summary["status_counts"] == {
        BenchmarkStatus.FAILED.value: 1,
        BenchmarkStatus.FAIL_NUMERICS.value: 1,
    }

    # With --strict, CLI must exit with EXIT_NUMERICAL_FAILURE
    rc_strict = exporter.main(
        [
            "--vlm-ledger",
            str(vlm_ledger),
            "--output-dir",
            str(tmp_path / "out_strict"),
            "--no-require-parquet",
            "--strict",
        ]
    )
    assert rc_strict == exporter.EXIT_NUMERICAL_FAILURE


def test_no_ledgers_raises_usage_error(tmp_path: Path) -> None:
    """Invoking with no ledger flags fails closed with EXIT_USAGE_ERROR."""
    rc = exporter.main(["--output-dir", str(tmp_path / "empty_out")])
    assert rc == exporter.EXIT_USAGE_ERROR


def test_symmetric_six_zoos_and_canonical_5formats(tmp_path: Path) -> None:
    """Validate --bert-attn-ledger, --bert-comp-ledger, --qwen-attn-ledger, --qwen-comp-ledger,
    --vlm-ledger, and --tinyllama-ledger flags produce all 5 canonical formats."""
    qwen_ledger = _write_jsonl(
        tmp_path / "qwen_attn" / "metrics.jsonl",
        [
            {
                "model": "Qwen/Qwen2.5-0.5B",
                "attn_key": "qwen_attention_mod",
                "dataset": "wikitext-2",
                "ppl_before": 45.0,
                "ppl_after": 20.0,
                "train_secs": 100.0,
                "status": "PASSED",
            }
        ],
    )
    bert_comp_ledger = _write_jsonl(
        tmp_path / "bert_comp" / "multirank_lora_ledger.jsonl",
        [
            {
                "compressor": "hnc_svd",
                "rank": 32,
                "ratio": 2.5,
                "orig_params": 1000000,
                "comp_params": 400000,
                "zero_shot_acc": 70.0,
                "healed_acc": 90.0,
                "lora_healed": True,
                "duration_s": 50.0,
                "status": "PASSED",
            }
        ],
    )
    bert_attn_ledger = _write_jsonl(
        tmp_path / "bert_attn" / "metrics.jsonl",
        [
            {
                "attn_key": "primal_attention",
                "acc_before": 32.5,
                "acc_after": 64.38,
                "acc_delta": 31.88,
                "train_secs": 120.0,
                "status": "PASSED",
            }
        ],
    )
    qwen_comp_ledger = _write_jsonl(
        tmp_path / "qwen_comp" / "metrics.jsonl",
        [
            {
                "compressor": "magnitude_pruning",
                "rank": 16,
                "param_reduction": 1.5,
                "ppl_before": 40.0,
                "ppl_after": 22.0,
                "train_secs": 80.0,
                "status": "PASSED",
            }
        ],
    )
    vlm_ledger = _write_jsonl(
        tmp_path / "vlm_llava" / "metrics.jsonl",
        [
            {
                "attn_key": "padre_attention",
                "dataset": "derek-thomas/ScienceQA",
                "acc_before": 43.12,
                "acc_after": 65.62,
                "acc_delta": 22.5,
                "train_secs": 150.0,
                "status": "PASSED",
            }
        ],
    )
    tinyllama_ledger = _write_jsonl(
        tmp_path / "tinyllama" / "metrics.jsonl",
        [
            {
                "attn_key": "retnet_linear",
                "ppl_before": 30.0,
                "ppl_after": 15.0,
                "train_secs": 110.0,
                "status": "PASSED",
            }
        ],
    )

    out_dir = tmp_path / "canonical_5formats_test"
    rc = exporter.main(
        [
            "--qwen-attn-ledger",
            str(qwen_ledger),
            "--bert-comp-ledger",
            str(bert_comp_ledger),
            "--bert-attn-ledger",
            str(bert_attn_ledger),
            "--qwen-comp-ledger",
            str(qwen_comp_ledger),
            "--vlm-ledger",
            str(vlm_ledger),
            "--tinyllama-ledger",
            str(tinyllama_ledger),
            "--output-dir",
            str(out_dir),
            "--require-parquet",
        ]
    )
    assert rc == exporter.EXIT_SUCCESS

    # 1. results.json (complete structured export)
    results_json_path = out_dir / "results.json"
    assert results_json_path.exists()
    results_data = json.loads(results_json_path.read_text(encoding="utf-8"))
    assert isinstance(results_data, list)
    assert len(results_data) == 6
    for r in results_data:
        assert "run_id" in r
        assert "metrics" in r
        assert r["status"] == "success"

    # 2. results.jsonl (streaming ledger)
    results_jsonl_path = out_dir / "results.jsonl"
    assert results_jsonl_path.exists()
    jsonl_lines = [line for line in results_jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(jsonl_lines) == 6

    # 3. results.csv (tabular export)
    results_csv_path = out_dir / "results.csv"
    assert results_csv_path.exists()
    with results_csv_path.open("r", encoding="utf-8") as fp:
        csv_rows = list(csv.DictReader(fp))
    assert len(csv_rows) == 6
    assert {row["module"] for row in csv_rows} == {
        "qwen_attention_mod",
        "hnc_svd",
        "primal_attention",
        "magnitude_pruning",
        "padre_attention",
        "retnet_linear",
    }

    # 4. summary.md (publication-grade Markdown comparison card)
    summary_md_path = out_dir / "summary.md"
    assert summary_md_path.exists()
    md_text = summary_md_path.read_text(encoding="utf-8")
    assert "# NeuralOps Multi-Zoo Benchmark Summary Card" in md_text
    assert "## 1. Cross-Zoo Performance Matrix" in md_text
    assert "## 2. Top-Performing Modules by Zoo" in md_text
    assert "primal_attention" in md_text
    assert "padre_attention" in md_text

    # 5. results.parquet (compressed columnar storage via pyarrow)
    results_parquet_path = out_dir / "results.parquet"
    assert results_parquet_path.exists()
    assert results_parquet_path.stat().st_size > 0

    # Also check zoo_overview.json
    overview_path = out_dir / "zoo_overview.json"
    assert overview_path.exists()
    overview = json.loads(overview_path.read_text(encoding="utf-8"))
    assert overview["total_records"] == 6
    assert set(overview["zoos_included"]) == {
        "qwen_attention_zoo",
        "bert_compressor_zoo",
        "bert_attention_zoo",
        "qwen_compressor_zoo",
        "vlm_llava_scienceqa_zoo",
        "tinyllama_1.1b_zoo",
    }
