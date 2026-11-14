import pytest
from neuralops.benchmarks.storage.parquet_store import ParquetLedgerStore
from neuralops.benchmarks.storage.schema import CanonicalRecord


def test_parquet_deduplicate_batch(tmp_path):
    pytest.importorskip("pyarrow", reason="pyarrow required for parquet test")
    store = ParquetLedgerStore(base_dir=tmp_path / "parquet")
    r1 = CanonicalRecord(
        run_id="run_1",
        timestamp="2026-11-14T22:00:00Z",
        suite="attention",
        module="flash",
        model="gpt2",
        device="cpu",
        status="success",
        metrics={"ms": 10.0},
    )
    r2 = CanonicalRecord(
        run_id="run_1",  # duplicate
        timestamp="2026-11-14T22:00:00Z",
        suite="attention",
        module="flash",
        model="gpt2",
        device="cpu",
        status="success",
        metrics={"ms": 10.0},
    )
    r3 = CanonicalRecord(
        run_id="run_2",
        timestamp="2026-11-14T22:00:00Z",
        suite="attention",
        module="flash",
        model="gpt2",
        device="cpu",
        status="success",
        metrics={"ms": 12.0},
    )

    written = store.write_records([r1, r2, r3])
    assert len(written) == 2
    assert [r.run_id for r in written] == ["run_1", "run_2"]
