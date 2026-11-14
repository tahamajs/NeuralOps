from unittest.mock import MagicMock
import torch
from neuralops.benchmarks.core.evaluation import get_memory_fragmentation


def test_memory_fragmentation_cpu():
    stats = get_memory_fragmentation("cpu")
    assert stats["reserved_mb"] == 0.0
    assert stats["allocated_mb"] == 0.0
    assert stats["fragmentation_ratio"] == 0.0


def test_memory_fragmentation_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda dev: {
            "reserved_bytes.all.current": 200 * 1024 * 1024,
            "allocated_bytes.all.current": 100 * 1024 * 1024,
        },
    )
    stats = get_memory_fragmentation("cuda:0")
    assert stats["reserved_mb"] == 200.0
    assert stats["allocated_mb"] == 100.0
    assert stats["fragmentation_ratio"] == 0.5
