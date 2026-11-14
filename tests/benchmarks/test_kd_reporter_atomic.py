from pathlib import Path
from neuralops.benchmarks.kd.reporter import KDReporter


def test_kd_reporter_atomic_writes(tmp_path):
    target = tmp_path / "test.json"
    KDReporter._atomic_write_json(target, {"hello": "world"})
    assert target.exists()

    txt_target = tmp_path / "test.txt"
    KDReporter._atomic_write_text(txt_target, "sample text")
    assert txt_target.exists()
    assert txt_target.read_text(encoding="utf-8") == "sample text"
