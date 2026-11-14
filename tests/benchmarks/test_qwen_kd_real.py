from unittest.mock import MagicMock
import pytest
from neuralops.benchmarks.suites.qwen_kd import QwenKDBenchmark


def test_qwen_kd_raises_not_implemented():
    mock_config = MagicMock()
    mock_config.hardware.device = "cpu"
    mock_tracker = MagicMock()
    mock_context = MagicMock()

    bm = QwenKDBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.setup()
    with pytest.raises(NotImplementedError, match="qwen_kd requires real model"):
        bm.run_iteration(step=0, stage="distillation_step")
