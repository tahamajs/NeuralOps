from unittest.mock import MagicMock
from neuralops.benchmarks.suites.attention import AttentionBenchmark


def test_attention_suite_init_failure():
    mock_config = MagicMock()
    mock_config.hardware.device = "cpu"
    mock_config.attention_sweep = None
    mock_config.custom_parameters = {}
    mock_tracker = MagicMock()
    mock_context = MagicMock()

    bm = AttentionBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.setup()

    # Pass a mechanism that fails to initialize
    out = bm.run_iteration(
        step=0,
        mechanism="definitely_non_existent_attention_mechanism_123",
        batch_size=1,
        seq_len=16,
    )
    assert out["latency_ms"] == 0.0
    assert len(bm.results_table) == 1
    assert "FAILED_INIT" in bm.results_table[0]["status"]
