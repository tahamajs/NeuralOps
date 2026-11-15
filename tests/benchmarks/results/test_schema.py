#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for canonical schema, BenchmarkStatus normalization, and backward-compatible forwarding.
"""

from __future__ import annotations

import warnings
import pytest

from neuralops.benchmarks.results.schema import (
    BenchmarkResult,
    BenchmarkStatus,
    normalize_legacy_result_dict,
)
from neuralops.benchmarks.schema import (
    BenchmarkResult as ForwardedResult,
    BenchmarkStatus as ForwardedStatus,
)


def test_schema_forwarding():
    """Verify neuralops.benchmarks.schema forwards canonical symbols."""
    assert ForwardedResult is BenchmarkResult
    assert ForwardedStatus is BenchmarkStatus


def test_status_normalize_valid():
    """Verify valid statuses map correctly."""
    assert BenchmarkStatus.normalize("pass") == BenchmarkStatus.SUCCESS
    assert BenchmarkStatus.normalize("PASS") == BenchmarkStatus.SUCCESS
    assert BenchmarkStatus.normalize("success") == BenchmarkStatus.SUCCESS
    assert BenchmarkStatus.normalize("fail") == BenchmarkStatus.FAILED
    assert BenchmarkStatus.normalize("timeout") == BenchmarkStatus.TIMEOUT
    assert BenchmarkStatus.normalize("skip") == BenchmarkStatus.SKIPPED
    assert BenchmarkStatus.normalize(BenchmarkStatus.SUCCESS) == BenchmarkStatus.SUCCESS


def test_status_normalize_unknown_warns():
    """Verify unknown status emits a UserWarning and defaults to FAILED."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        status = BenchmarkStatus.normalize("completely_unknown_status_xyz")
        assert status == BenchmarkStatus.FAILED
        assert len(w) == 1
        assert "Unknown benchmark status" in str(w[-1].message)


def test_status_normalize_empty_warns():
    """Verify empty/None status emits a warning and defaults to FAILED."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        status = BenchmarkStatus.normalize("")
        assert status == BenchmarkStatus.FAILED
        assert len(w) == 1
        assert "Unknown or empty benchmark status" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        status = BenchmarkStatus.normalize(None)
        assert status == BenchmarkStatus.FAILED
        assert len(w) == 1


def test_benchmark_result_roundtrip():
    """Verify BenchmarkResult serialization and deserialization."""
    res = BenchmarkResult(
        run_id="test_run_123",
        module={"name": "test_mod"},
        model={"name": "test_model"},
        dataset={"name": "test_ds"},
        status="success",
        metrics={"accuracy": 0.95, "latency_ms": 12.3},
    )
    json_str = res.to_json()
    res2 = BenchmarkResult.from_json(json_str)
    assert res2.run_id == res.run_id
    assert res2.status == BenchmarkStatus.SUCCESS.value
    assert res2.metrics["accuracy"] == 0.95
    assert res2.compute_checksum() == res.compute_checksum()


def test_normalize_legacy_result_dict():
    """Verify legacy dictionary mappings."""
    legacy = {
        "suite": "glue_sst2",
        "device": "cpu",
        "attention": "flash_attention",
        "model": "bert-base-uncased",
        "results": {"accuracy": 0.88},
    }
    res = normalize_legacy_result_dict(legacy)
    assert res.benchmark_suite == "glue_sst2"
    assert res.device_type == "cpu"
    assert res.transplanted_attention == "flash_attention"
    assert res.model_name == "bert-base-uncased"
    assert res.metrics.get("accuracy") == 0.88
