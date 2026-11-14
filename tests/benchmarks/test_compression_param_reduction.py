from unittest.mock import MagicMock
from neuralops.benchmarks.suites.compression import CompressionBenchmark


def test_compression_benchmark_setup():
    mock_config = MagicMock()
    mock_config.hardware.device = "cpu"
    mock_config.custom_parameters = {}
    mock_tracker = MagicMock()
    mock_context = MagicMock()

    bm = CompressionBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.setup()
    assert hasattr(bm, "compressors")
    assert isinstance(bm.compressors, list)
