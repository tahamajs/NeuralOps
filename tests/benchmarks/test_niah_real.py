from unittest.mock import MagicMock
from neuralops.benchmarks.suites.niah import NeedleInAHaystackBenchmark


def test_niah_functional_execution():
    """Verify NIAH benchmark executes functionally without raising NotImplementedError."""
    mock_config = MagicMock()
    mock_config.hardware.device = "cpu"
    mock_config.custom_parameters = {"limit": 1}
    mock_tracker = MagicMock()
    mock_context = MagicMock()
    mock_context.run_id = "test-niah-run"

    bm = NeedleInAHaystackBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.setup()
    res = bm.run_iteration(step=0, ctx_len=32, depth=0.5)
    assert isinstance(res, dict)
    assert "accuracy" in res
    assert "latency_ms" in res

