from unittest.mock import MagicMock
import pytest
from neuralops.benchmarks.suites.niah import NeedleInAHaystackBenchmark


def test_niah_raises_not_implemented():
    mock_config = MagicMock()
    mock_config.hardware.device = "cpu"
    mock_tracker = MagicMock()
    mock_context = MagicMock()

    bm = NeedleInAHaystackBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.setup()
    with pytest.raises(NotImplementedError, match="NIAH requires real model"):
        bm.run_iteration(step=0, ctx_len=1024, depth=0.5)
