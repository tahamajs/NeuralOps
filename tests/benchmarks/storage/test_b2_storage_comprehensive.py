#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive unit and regression tests for Storage subsystem (W4-B2)."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from neuralops.benchmarks.storage.dispatcher import MultiGPUDispatcher, VRAMGovernor
from neuralops.benchmarks.storage.parquet_store import ParquetLedgerStore
from neuralops.benchmarks.storage.schema import CanonicalRecord


def test_parquet_store_deduplication_by_run_id(tmp_path: Path) -> None:
    """Verify ParquetLedgerStore deduplicates records with matching run_id."""
    store = ParquetLedgerStore(base_dir=tmp_path / "ledger", partition_by=["suite"])

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record1 = CanonicalRecord(
        run_id="run_100",
        timestamp=now_iso,
        suite="attention",
        module="linear_attention",
        model="distilbert",
        device="cpu",
        status="success",
        metrics={"latency_ms": 1.25},
    )
    record2 = CanonicalRecord(
        run_id="run_100",  # Duplicate run_id
        timestamp=now_iso,
        suite="attention",
        module="linear_attention",
        model="distilbert",
        device="cpu",
        status="success",
        metrics={"latency_ms": 1.30},
    )
    record3 = CanonicalRecord(
        run_id="run_200",
        timestamp=now_iso,
        suite="attention",
        module="cosformer",
        model="distilbert",
        device="cpu",
        status="success",
        metrics={"latency_ms": 1.45},
    )

    written = store.write_records([record1, record2, record3])
    # Duplicate run_100 should be filtered out
    assert len(written) == 2
    assert {r.run_id for r in written} == {"run_100", "run_200"}


def test_canonical_record_validation() -> None:
    """Verify CanonicalRecord validates status, run_id, and ISO 8601 timestamp."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Empty run_id must raise
    with pytest.raises(ValueError, match="run_id must be non-empty"):
        CanonicalRecord(
            run_id="",
            timestamp=now_iso,
            suite="test",
            module="test",
            model="test",
            device="cpu",
            status="success",
        )

    # Invalid timestamp must raise
    with pytest.raises(ValueError, match="timestamp must be ISO 8601"):
        CanonicalRecord(
            run_id="run_valid",
            timestamp="not_a_date",
            suite="test",
            module="test",
            model="test",
            device="cpu",
            status="success",
        )

    # Invalid status must raise
    with pytest.raises(ValueError, match="Invalid status"):
        CanonicalRecord(
            run_id="run_valid",
            timestamp=now_iso,
            suite="test",
            module="test",
            model="test",
            device="cpu",
            status="invalid_status_string",
        )

    # Valid record computes checksum automatically
    rec = CanonicalRecord(
        run_id="run_valid",
        timestamp=now_iso,
        suite="test",
        module="test",
        model="test",
        device="cpu",
        status="success",
        metrics={"acc": 0.95},
    )
    assert len(rec.checksum) == 64


def test_multi_gpu_dispatcher_lease_lifecycle() -> None:
    """Verify MultiGPUDispatcher acquires and releases device leases cleanly."""
    gov = VRAMGovernor(min_free_mb=0)
    dispatcher = MultiGPUDispatcher(device_ids=[0, 1], governor=gov)

    lease0 = dispatcher.acquire_device(task_id="task_a", preferred_device=0)
    assert lease0 is not None
    assert lease0.device_id == 0
    assert lease0.active

    # Release lease
    dispatcher.release_device(lease0)
    assert not lease0.active
