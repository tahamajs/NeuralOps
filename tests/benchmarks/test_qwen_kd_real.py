from unittest.mock import MagicMock
from neuralops.benchmarks.suites.qwen_kd import QwenKDBenchmark


def test_qwen_kd_functional_execution():
    """Verify Qwen KD benchmark executes functionally without raising NotImplementedError."""
    mock_config = MagicMock()
    mock_config.hardware.device = "cpu"
    mock_config.custom_parameters = {"limit": 1}
    mock_tracker = MagicMock()
    mock_context = MagicMock()
    mock_context.run_id = "test-qwen-kd-run"

    bm = QwenKDBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.setup()
    res = bm.run_iteration(step=0, stage="distillation_step")
    assert isinstance(res, dict)
    assert "kd_loss" in res
    assert "latency_ms" in res

