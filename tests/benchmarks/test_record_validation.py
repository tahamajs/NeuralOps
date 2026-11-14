import pytest
from neuralops.benchmarks.storage.schema import CanonicalRecord


def test_canonical_record_valid():
    rec = CanonicalRecord(
        run_id="run_123",
        timestamp="2026-11-14T22:00:00Z",
        suite="attention",
        module="flash_attn",
        model="bert-base-uncased",
        device="cuda",
        status="success",
        metrics={"latency_ms": 1.25},
    )
    assert rec.checksum != ""
    assert rec.run_id == "run_123"


def test_canonical_record_empty_run_id_raises():
    with pytest.raises(ValueError, match="run_id must be non-empty"):
        CanonicalRecord(
            run_id="",
            timestamp="2026-11-14T22:00:00Z",
            suite="attention",
            module="flash_attn",
            model="bert-base-uncased",
            device="cuda",
            status="success",
        )


def test_canonical_record_invalid_timestamp_raises():
    with pytest.raises(ValueError, match="timestamp must be ISO 8601"):
        CanonicalRecord(
            run_id="run_123",
            timestamp="invalid-timestamp",
            suite="attention",
            module="flash_attn",
            model="bert-base-uncased",
            device="cuda",
            status="success",
        )


def test_canonical_record_invalid_status_raises():
    with pytest.raises(ValueError, match="Invalid status"):
        CanonicalRecord(
            run_id="run_123",
            timestamp="2026-11-14T22:00:00Z",
            suite="attention",
            module="flash_attn",
            model="bert-base-uncased",
            device="cuda",
            status="unknown_status",
        )
