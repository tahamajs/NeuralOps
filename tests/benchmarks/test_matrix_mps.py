from unittest.mock import MagicMock
import pytest
import torch

from neuralops.benchmarks.matrix import ColdCacheFlusher


def test_cold_cache_flusher_mps_physical_if_available():
    """Verify ColdCacheFlusher creates an MPS buffer when MPS is available on host."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        flusher = ColdCacheFlusher(torch.device("mps"), size_mb=1)
        assert flusher.device.type == "mps"
        assert flusher._buffer is not None
        assert flusher._buffer.device.type == "mps"
        flusher.flush()


def test_cold_cache_flusher_mps_mocked(monkeypatch):
    """Verify ColdCacheFlusher allocates buffer for device.type == 'mps'."""
    allocated = []

    def mock_empty(num_elements, dtype, device):
        allocated.append((num_elements, dtype, str(device)))
        return MagicMock()

    monkeypatch.setattr(torch, "empty", mock_empty)

    flusher = ColdCacheFlusher(torch.device("mps"), size_mb=32)
    assert flusher.size_mb == 32
    # 32MB -> 32 * 1024 * 1024 // 4 = 8388608 elements
    assert len(allocated) == 1
    assert allocated[0][0] == (32 * 1024 * 1024) // 4
    assert allocated[0][2] == "mps"


def test_cold_cache_flusher_dynamic_l2(monkeypatch):
    """Verify dynamic L2 cache querying via torch.cuda.get_device_properties(device).L2_cache_size."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    mock_props = MagicMock()
    mock_props.L2_cache_size = 48 * 1024 * 1024  # 48MB
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: mock_props)

    allocated = []

    def mock_empty(num_elements, dtype, device):
        allocated.append((num_elements, dtype, str(device)))
        return MagicMock()

    monkeypatch.setattr(torch, "empty", mock_empty)

    # When size_mb is None, dynamic L2 should be queried: 48MB + 16MB = 64MB
    flusher = ColdCacheFlusher(torch.device("cuda:0"))
    assert flusher.size_mb == 64
    assert len(allocated) == 1
    assert allocated[0][0] == (64 * 1024 * 1024) // 4
