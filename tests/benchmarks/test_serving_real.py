from unittest.mock import MagicMock
import pytest
from neuralops.benchmarks.suites.e2e_serving import ServingBenchmark


def test_serving_raises_not_implemented():
    mock_config = MagicMock()
    mock_config.hardware.device = "cpu"
    mock_tracker = MagicMock()
    mock_context = MagicMock()

    bm = ServingBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.setup()
    with pytest.raises(NotImplementedError, match="e2e_serving requires real serving"):
        bm.run_iteration(step=0, batch_size=1, prompt_len=128, gen_len=32)
