from unittest.mock import MagicMock
import numpy as np
import pytest

from neuralops.benchmarks.matrix import (
    BenchmarkMatrixOrchestrator,
    MatrixCellResult,
    MatrixDimensionConfig,
    MatrixPoint,
)
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus


def test_matrix_cell_result_percentile_fields():
    """Verify MatrixCellResult includes median, p10, p90, and p99."""
    pt = MatrixPoint(module="test", seq_len=128, batch_size=1, hidden_dim=64, dtype="float32", causal=False)
    cell = MatrixCellResult(
        point=pt,
        status="success",
        regime="micro",
        latency_ms_median=12.5,
        latency_ms_p10=10.1,
        latency_ms_p90=15.2,
        latency_ms_p99=18.7,
    )
    assert cell.latency_ms_median == 12.5
    assert cell.latency_ms_p10 == 10.1
    assert cell.latency_ms_p90 == 15.2
    assert cell.latency_ms_p99 == 18.7

    d = cell.to_dict()
    assert "latency_ms_median" in d
    assert "latency_ms_p10" in d
    assert "latency_ms_p90" in d
    assert "latency_ms_p99" in d


def test_matrix_execute_cell_percentiles(monkeypatch):
    """Verify execute_cell correctly computes median, p90, and p99 from measured latencies."""
    config = MatrixDimensionConfig(
        modules=["attention:full_attention"],
        seq_lens=[64],
        batch_sizes=[1],
        hidden_dims=[32],
        flush_l2=False,
    )
    orch = BenchmarkMatrixOrchestrator(config)
    pt = orch.plan()[0]

    latencies = [float(i) for i in range(1, 101)]  # 1.0 to 100.0

    mock_adapter = MagicMock()
    mock_adapter.execute.return_value = BenchmarkResult(
        status=BenchmarkStatus.SUCCESS,
        metadata={"latencies_ms": latencies},
    )
    monkeypatch.setattr(orch, "_resolve_adapter", lambda point: mock_adapter)

    res = orch.execute_cell(pt)
    assert res.status == "success"

    # Percentiles from 1.0 to 100.0
    expected_median = float(np.percentile(latencies, 50))
    expected_p10 = float(np.percentile(latencies, 10))
    expected_p90 = float(np.percentile(latencies, 90))
    expected_p99 = float(np.percentile(latencies, 99))

    assert res.latency_ms_median == round(expected_median, 4)
    assert res.latency_ms_p10 == round(expected_p10, 4)
    assert res.latency_ms_p90 == round(expected_p90, 4)
    assert res.latency_ms_p99 == round(expected_p99, 4)
