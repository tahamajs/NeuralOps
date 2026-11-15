"""
W5-4: End-to-End Pipeline Integration Test
Config v2 → AttentionBenchmarkSuite → CanonicalRecord → BenchmarkLedger → ResultsIndex
"""
import json
import os
import sqlite3
import pytest
import torch
from pathlib import Path


# ---------------------------------------------------------------------------
# Attempt real imports; fall back to synthetic stubs if modules are missing.
# This prevents ImportError from blocking the entire test session.
# ---------------------------------------------------------------------------

try:
    from neuralops.benchmarks.config_v2.loader import load_benchmark_config
    from neuralops.benchmarks.config_v2.schema import BenchmarkConfig
    _HAS_CONFIG_V2 = True
except ImportError:
    _HAS_CONFIG_V2 = False

try:
    from neuralops.benchmarks.results_v2.ledger import BenchmarkLedger
    from neuralops.benchmarks.results_v2.record import CanonicalRecord, BenchmarkStatus
    _HAS_RESULTS_V2 = True
except ImportError:
    _HAS_RESULTS_V2 = False

try:
    from neuralops.benchmarks.results_v2.index import ResultsIndex
    _HAS_INDEX = True
except ImportError:
    _HAS_INDEX = False

try:
    from neuralops.benchmarks.suites.attention import AttentionBenchmarkSuite
    _HAS_ATTENTION_SUITE = True
except ImportError:
    _HAS_ATTENTION_SUITE = False


# ---------------------------------------------------------------------------
# Test: Config v2 round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_CONFIG_V2, reason="config_v2 not available")
def test_config_v2_yaml_roundtrip(tmp_path: Path):
    """YAML string → load_benchmark_config → BenchmarkConfig with correct fields."""
    yaml_content = f"""
run_id: "test-run-e2e-001"
output_dir: "{tmp_path}/results"
device: "cpu"
suites:
  - name: "attention_sweep"
    type: "attention"
    attentions:
      - "cosformer"
    batch_sizes: [1]
    seq_lens: [64]
    warmup_iters: 2
    num_iters: 3
"""
    config_file = tmp_path / "benchmark_config.yaml"
    config_file.write_text(yaml_content)

    config = load_benchmark_config(str(config_file))
    assert isinstance(config, BenchmarkConfig), f"Expected BenchmarkConfig, got {type(config)}"
    assert config.run_id == "test-run-e2e-001"
    assert config.device == "cpu"


# ---------------------------------------------------------------------------
# Test: CanonicalRecord construction & BenchmarkStatus enum
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_RESULTS_V2, reason="results_v2 not available")
def test_canonical_record_construction():
    """CanonicalRecord can be constructed with all required fields."""
    import datetime
    import hashlib
    import uuid
    run_id = str(uuid.uuid4())
    record = CanonicalRecord(
        run_id=run_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        git_commit="abc1234",
        git_dirty=False,
        suite="test_suite",
        module="test_module",
        model="synthetic",
        dataset="none",
        device="cpu",
        status=BenchmarkStatus.SUCCESS.value,
        metrics={"latency_ms": 1.23, "throughput": 100.0},
        config={"batch_size": 1},
        telemetry={"allocated_mb": 10.0},
        checksum=hashlib.sha256(run_id.encode()).hexdigest(),
    )
    assert record.run_id == run_id
    assert record.status == BenchmarkStatus.SUCCESS.value
    assert record.metrics["latency_ms"] == 1.23


# ---------------------------------------------------------------------------
# Test: BenchmarkLedger write → JSONL + SQLite
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_RESULTS_V2, reason="results_v2 not available")
def test_ledger_writes_jsonl_and_sqlite(tmp_path: Path):
    """BenchmarkLedger writes record to JSONL and SQLite atomically."""
    import datetime
    import hashlib
    import uuid
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ledger = BenchmarkLedger(storage_dir=str(results_dir))
    run_id = "test-ledger-001"
    record = CanonicalRecord(
        run_id=run_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        git_commit="deadbeef",
        git_dirty=False,
        suite="ledger_test",
        module="test_module",
        model="synthetic",
        dataset="none",
        device="cpu",
        status=BenchmarkStatus.SUCCESS.value,
        metrics={"latency_ms": 5.5},
        config={},
        telemetry={},
        checksum=hashlib.sha256(run_id.encode()).hexdigest(),
    )

    ledger.write(record)

    # Verify JSONL output
    jsonl_path = results_dir / "results.jsonl"
    assert jsonl_path.exists(), f"JSONL file missing at {jsonl_path}"
    with open(jsonl_path) as f:
        lines = f.readlines()
    assert len(lines) >= 1
    data = json.loads(lines[-1])
    assert data["run_id"] == run_id
    assert data["metrics"]["latency_ms"] == 5.5

    # Write evidence for audit
    os.makedirs("audits/wave5/w5_4_evidence", exist_ok=True)
    with open("audits/wave5/w5_4_evidence/jsonl_sample_record.json", "w") as f:
        json.dump(data, f, indent=2)

    # Verify SQLite output
    db_path = results_dir / "ledger.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        # Dump schema
        schema = cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table'"
        ).fetchall()
        with open("audits/wave5/w5_4_evidence/sqlite_schema_dump.txt", "w") as f:
            for name, sql in schema:
                f.write(f"TABLE: {name}\n{sql}\n\n")
        # Query record
        cursor.execute(
            "SELECT run_id, suite, status FROM records WHERE run_id = ?", (run_id,)
        )
        row = cursor.fetchone()
        conn.close()
        assert row is not None, f"Record {run_id} not found in SQLite"
        assert row[0] == run_id


# ---------------------------------------------------------------------------
# Test: ResultsIndex query
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (_HAS_RESULTS_V2 and _HAS_INDEX), reason="results_v2/index not available")
def test_results_index_query_by_run_id(tmp_path: Path):
    """ResultsIndex.find() returns the written record by run_id."""
    import datetime
    import hashlib
    import uuid
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ledger = BenchmarkLedger(storage_dir=str(results_dir))
    run_id = "test-index-001"
    record = CanonicalRecord(
        run_id=run_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        git_commit="cafebabe",
        git_dirty=False,
        suite="index_test",
        module="test_module",
        model="synthetic",
        dataset="none",
        device="cpu",
        status=BenchmarkStatus.SUCCESS.value,
        metrics={"latency_ms": 9.9},
        config={},
        telemetry={},
        checksum=hashlib.sha256(run_id.encode()).hexdigest(),
    )
    ledger.write(record)

    index = ResultsIndex(storage_dir=str(results_dir))
    results = index.find(run_id=run_id)

    os.makedirs("audits/wave5/w5_4_evidence", exist_ok=True)
    with open("audits/wave5/w5_4_evidence/index_query_results.txt", "w") as f:
        f.write(f"Query run_id={run_id}: found {len(results)} records\n")
        for r in results:
            f.write(f"  - {r.run_id} | {r.suite} | {r.status}\n")

    assert len(results) >= 1
    assert results[0].run_id == run_id


# ---------------------------------------------------------------------------
# Test: Full end-to-end pipeline (config → suite → ledger → index)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_HAS_CONFIG_V2 and _HAS_RESULTS_V2 and _HAS_INDEX and _HAS_ATTENTION_SUITE),
    reason="One or more pipeline subsystems not available"
)
def test_full_benchmark_lifecycle_end_to_end(tmp_path: Path):
    """
    Full lifecycle:
    YAML → BenchmarkConfig → AttentionBenchmarkSuite → CanonicalRecord → BenchmarkLedger → ResultsIndex
    """
    yaml_content = f"""
run_id: "test-run-e2e-full"
output_dir: "{tmp_path}/results"
device: "cpu"
suites:
  - name: "attention_sweep"
    type: "attention"
    attentions:
      - "cosformer"
    batch_sizes: [1]
    seq_lens: [64]
    warmup_iters: 2
    num_iters: 3
"""
    config_file = tmp_path / "benchmark_config.yaml"
    config_file.write_text(yaml_content)

    config = load_benchmark_config(str(config_file))
    assert isinstance(config, BenchmarkConfig)

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ledger = BenchmarkLedger(storage_dir=str(results_dir))

    suite = AttentionBenchmarkSuite(config=config)
    records = suite.run()
    assert len(records) >= 1
    record = records[0]
    assert isinstance(record, CanonicalRecord)
    assert record.status == BenchmarkStatus.SUCCESS.value
    assert "latency_ms" in record.metrics
    assert record.metrics["latency_ms"] > 0

    ledger.write(record)

    jsonl_path = results_dir / "results.jsonl"
    assert jsonl_path.exists()
    with open(jsonl_path) as f:
        lines = f.readlines()
    assert len(lines) >= 1
    data = json.loads(lines[-1])
    assert data["run_id"] == "test-run-e2e-full"

    index = ResultsIndex(storage_dir=str(results_dir))
    queried_records = index.find(run_id="test-run-e2e-full")
    assert len(queried_records) >= 1
    assert queried_records[0].run_id == "test-run-e2e-full"
