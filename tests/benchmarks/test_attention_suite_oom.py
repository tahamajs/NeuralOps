from unittest.mock import MagicMock
import torch
from neuralops.benchmarks.suites.attention import AttentionBenchmark


def test_attention_suite_oom(monkeypatch):
    mock_config = MagicMock()
    mock_config.hardware.device = "cpu"
    mock_config.attention_sweep = None
    mock_config.custom_parameters = {}
    mock_tracker = MagicMock()
    mock_context = MagicMock()

    bm = AttentionBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.setup()

    # Mock measure_latency to raise OOM
    def mock_measure(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(bm, "measure_latency", mock_measure)
    monkeypatch.setattr("neuralops.benchmarks.suites.attention.create_attention", lambda *a, **k: torch.nn.Identity())

    out = bm.run_iteration(
        step=0,
        mechanism="full",
        batch_size=1,
        seq_len=16,
    )
    assert out["latency_ms"] == 0.0
    assert len(bm.results_table) == 1
    assert bm.results_table[0]["status"] == "FAILED_OOM"
