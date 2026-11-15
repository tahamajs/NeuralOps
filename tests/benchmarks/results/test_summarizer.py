#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for SummaryGenerator and robust record loading with warning logs.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pytest

from neuralops.benchmarks.results.saver import MultiFormatSaver
from neuralops.benchmarks.results.schema import BenchmarkResult
from neuralops.benchmarks.results.summarizer import SummaryGenerator


def test_summarizer_corrupt_record_logs_warning(tmp_path: Path, caplog):
    """Verify _load_all_records logs warnings on parse errors rather than swallowing silently."""
    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = out_dir / "results.jsonl"

    # Write one valid line and one corrupt line
    jsonl_file.write_text(
        '{"run_id": "r1", "status": "success", "module": "m1", "model": "m", "dataset": "d", "accuracy": 0.9}\n'
        "CORRUPTED_NON_JSON_LINE\n"
        '{"run_id": "r2", "status": "success", "module": "m2", "model": "m", "dataset": "d", "accuracy": 0.85}\n',
        encoding="utf-8",
    )

    gen = SummaryGenerator(out_dir)
    with caplog.at_level(logging.WARNING):
        records = gen._load_all_records()

    assert len(records) == 2
    assert any(
        "Could not parse JSON record" in record.message or "Could not read results.jsonl" in record.message
        for record in caplog.records
    )


def test_summarizer_generate_all(tmp_path: Path):
    """Verify generate_all writes out all summary markdown files."""
    out_dir = tmp_path / "results"
    saver = MultiFormatSaver(out_dir)

    for i, mod in enumerate(["flash_attn", "linear_attn"]):
        r = BenchmarkResult(
            run_id=f"run_sum_{i}",
            model_name="bert-base-uncased",
            transplanted_attention=mod,
            module={"name": mod, "type": "attention"},
            dataset={"name": "sst2"},
            benchmark_suite="sst2",
            status="success",
            metrics={"accuracy": 0.90 + (i * 0.05)},
        )
        saver.save(r)

    gen = SummaryGenerator(out_dir)
    outputs = gen.generate_all()

    assert "by_module" in outputs
    assert outputs["by_module"].exists()
    assert "by_model" in outputs
    assert outputs["by_model"].exists()
    assert "by_dataset" in outputs
    assert outputs["by_dataset"].exists()
    assert "leaderboards" in outputs
    assert outputs["leaderboards"].exists()
