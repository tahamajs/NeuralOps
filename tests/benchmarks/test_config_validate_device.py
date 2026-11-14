import warnings
import pytest


def test_invalid_cuda_index_raises(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    from neuralops.benchmarks.config import BenchmarkConfig

    with pytest.raises(RuntimeError, match="out of range"):
        BenchmarkConfig(device="cuda:5").validate_device()


def test_cpu_device_passes():
    from neuralops.benchmarks.config import BenchmarkConfig

    BenchmarkConfig(device="cpu").validate_device()  # no raise


def test_unknown_status_warns():
    from neuralops.benchmarks.results.schema import BenchmarkStatus

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        status = BenchmarkStatus.normalize("weird_status")
        assert status == BenchmarkStatus.FAILED
        assert len(w) == 1
        assert "weird_status" in str(w[0].message)


def test_benchmark_result_none_handling():
    from neuralops.benchmarks.results.schema import BenchmarkResult

    res = BenchmarkResult(
        run_metadata=None,
        environment=None,
        hardware_usage=None,
        metrics=None,
        module=None,
        model=None,
    )
    assert res.run_metadata is not None
    assert res.environment is not None
    assert res.metrics == {}
    assert res.status == "success"
