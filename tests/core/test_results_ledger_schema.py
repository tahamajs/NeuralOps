#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for neuralops.core.results_ledger schema and record integrity."""

from __future__ import annotations

import dataclasses
import pytest

from neuralops.core.results_ledger import ResultRecord, ResultsLedger


def test_result_record_schema_fields() -> None:
    """Verify ResultRecord defines all required schema fields from specification."""
    field_names = {f.name for f in dataclasses.fields(ResultRecord)}
    expected = {
        "record_id",
        "module_name",
        "category",
        "family",
        "version_tag",
        "revision",
        "status",
        "timestamp_utc",
        "git_commit_sha",
        "generation_method",
        "experimental_setup",
        "metrics",
        "notes",
    }
    assert expected.issubset(field_names), f"Missing fields: {expected - field_names}"


def test_result_record_automatic_record_id_hashing() -> None:
    """Verify record_id is deterministically generated when empty."""
    rec1 = ResultRecord(
        record_id="",
        module_name="flash_attention",
        category="attention",
        family="fast_kernels",
        experimental_setup={"backbone": "bert-base", "seq_len": 512},
        generation_method={"algorithm": "fused_kernel"},
    )
    assert rec1.record_id
    assert len(rec1.record_id) == 16


def test_result_record_serialization_roundtrip() -> None:
    """Verify to_dict and from_dict roundtrip cleanly."""
    rec = ResultRecord(
        record_id="custom_hash_01",
        module_name="gla",
        category="attention",
        family="linear",
        version_tag="v2.4.0",
        revision=2,
        status="active",
        metrics={"latency_ms": 1.45, "tflops": 45.2},
        notes="Validated on synthetic benchmark",
    )
    data = rec.to_dict()
    assert isinstance(data, dict)
    assert data["record_id"] == "custom_hash_01"
    assert data["metrics"]["latency_ms"] == 1.45

    reconstructed = ResultRecord.from_dict(data)
    assert reconstructed.record_id == rec.record_id
    assert reconstructed.module_name == rec.module_name
    assert reconstructed.metrics == rec.metrics
    assert reconstructed.notes == rec.notes


def test_result_record_defensive_metrics_coercion() -> None:
    """Verify from_dict coerces string numbers and drops invalid metric values safely."""
    data = {
        "record_id": "test_coercion",
        "module_name": "mod_a",
        "category": "attention",
        "family": "dense",
        "metrics": {"valid_str": "12.5", "valid_int": 42, "invalid_str": "nan_or_bad", "none_val": None},
    }
    rec = ResultRecord.from_dict(data)
    assert rec.metrics["valid_str"] == 12.5
    assert rec.metrics["valid_int"] == 42.0
    assert rec.metrics["none_val"] == 0.0
    assert "invalid_str" not in rec.metrics


def test_results_ledger_instantiation(tmp_path) -> None:
    """Verify ResultsLedger initializes with designated root directory."""
    ledger = ResultsLedger(root_dir=tmp_path)
    assert ledger.root_dir == tmp_path
