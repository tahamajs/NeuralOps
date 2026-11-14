from unittest.mock import MagicMock
import torch
from neuralops.benchmarks.core.base import AbstractBaseBenchmark, ExecutionSummary


class DummyBenchmark(AbstractBaseBenchmark):
    def setup(self):
        pass

    def run(self):
        return self.generate_summary()

    def run_iteration(self, iteration: int):
        pass

    def generate_summary(self):
        return ExecutionSummary(run_id="test", benchmark_name="dummy")

    def teardown(self):
        pass


def test_flush_l2_dynamic(monkeypatch):
    mock_config = MagicMock()
    mock_config.hardware.device = "cuda:0"
    mock_tracker = MagicMock()
    mock_context = MagicMock()

    # Mock cuda
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    mock_props = MagicMock()
    mock_props.L2_cache_size = 32 * 1024 * 1024  # 32MB
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: mock_props)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda dev: None)

    allocated_tensors = []

    def mock_empty(num_elements, dtype, device):
        t = MagicMock()
        allocated_tensors.append(num_elements)
        return t

    monkeypatch.setattr(torch, "empty", mock_empty)

    bm = DummyBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.device = torch.device("cuda:0")

    bm.flush_l2_cache()
    # 32MB + 16MB = 48MB -> 48 * 1024 * 1024 // 4 = 12582912 elements
    assert len(allocated_tensors) == 1
    assert allocated_tensors[0] == (48 * 1024 * 1024) // 4
