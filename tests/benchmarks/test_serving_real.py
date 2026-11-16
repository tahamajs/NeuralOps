from unittest.mock import MagicMock
from neuralops.benchmarks.suites.e2e_serving import ServingBenchmark


def test_serving_functional_execution():
    """Verify Serving benchmark executes functionally without raising NotImplementedError."""
    mock_config = MagicMock()
    mock_config.hardware.device = "cpu"
    mock_config.custom_parameters = {"limit": 1}
    mock_tracker = MagicMock()
    mock_context = MagicMock()
    mock_context.run_id = "test-serving-run"

    bm = ServingBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.setup()
    res = bm.run_iteration(step=0, batch_size=1, prompt_len=16, gen_len=4)
    assert isinstance(res, dict)
    assert "ttft_ms" in res
    assert "throughput_tokens_per_sec" in res

