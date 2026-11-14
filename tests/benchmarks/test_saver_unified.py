import warnings
import pytest
from neuralops.benchmarks.results.saver import MultiFormatSaver, save_all_formats
from neuralops.benchmarks.results.schema import BenchmarkResult


def test_save_all_formats_deprecated_warning(tmp_path):
    res = BenchmarkResult(
        run_id="run_dep_test",
        timestamp="2026-11-14T22:00:00Z",
        benchmark_suite="attention",
        module={"name": "flash"},
        model={"name": "bert"},
        status="success",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = save_all_formats(res, output_dir=tmp_path / "results")
        assert any("save_all_formats is deprecated" in str(item.message) for item in w)
        assert "raw_json" in out
        assert out["raw_json"].exists()


def test_index_unknown_metric_raises(tmp_path):
    from neuralops.benchmarks.results.index import ResultsIndex

    saver = MultiFormatSaver(tmp_path / "results")
    res = BenchmarkResult(
        run_id="run_m1",
        timestamp="2026-11-14T22:00:00Z",
        benchmark_suite="attention",
        module={"name": "flash"},
        model={"name": "bert"},
        status="success",
        metrics={"latency_ms": 1.2},
    )
    saver.save(res)
    idx = ResultsIndex(tmp_path / "results")
    with pytest.raises(ValueError, match="not found in any results"):
        idx.best(metric="unknown_custom_metric")
