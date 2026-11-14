from neuralops.benchmarks.results.dedup import Deduplicator


def test_dedup_save_index_flock(tmp_path):
    dedup = Deduplicator(output_dir=tmp_path / "results")
    dedup._index["key_1"] = {"status": "success", "timestamp": "2026-11-14T22:00:00Z"}
    dedup.save_index()

    # Re-read
    dedup2 = Deduplicator(output_dir=tmp_path / "results")
    assert "key_1" in dedup2._index
    assert dedup2._index["key_1"]["status"] == "success"
