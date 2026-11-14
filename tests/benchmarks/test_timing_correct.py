from unittest.mock import MagicMock
import torch
from neuralops.benchmarks.core.base import AbstractBaseBenchmark, ExecutionSummary


class ConcreteBenchmark(AbstractBaseBenchmark):
    def setup(self):
        pass

    def run(self):
        return self.generate_summary()

    def run_iteration(self, iteration: int):
        pass

    def generate_summary(self):
        return ExecutionSummary(run_id="test", benchmark_name="concrete")

    def teardown(self):
        pass


def test_measure_latency_cuda_events(monkeypatch):
    mock_config = MagicMock()
    mock_config.hardware.device = "cuda:0"
    mock_tracker = MagicMock()
    mock_context = MagicMock()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda dev: None)

    event_records = []

    class MockEvent:
        def __init__(self, enable_timing=True):
            pass

        def record(self):
            event_records.append("record")

        def synchronize(self):
            pass

        def elapsed_time(self, end_event):
            return 1.25

    monkeypatch.setattr(torch.cuda, "Event", MockEvent)

    bm = ConcreteBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.device = torch.device("cuda:0")

    mean_ms, latencies = bm.measure_latency(lambda: None, warmup=1, repeat=2)
    assert mean_ms == 1.25
    assert len(latencies) == 2
    assert len(event_records) >= 4  # 2 start records + 2 end records
