from unittest.mock import MagicMock
from neuralops.benchmarks.suites.triton_ops import TritonOpsBenchmark


def test_triton_ops_benchmark_setup():
    mock_config = MagicMock()
    mock_config.hardware.device = "cpu"
    mock_config.custom_parameters = {}
    mock_tracker = MagicMock()
    mock_context = MagicMock()

    bm = TritonOpsBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.setup()
    assert hasattr(bm, "target_kernels")
    assert isinstance(bm.target_kernels, list)
    assert len(bm.target_kernels) > 0
