#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for ColdCacheFlusher across MPS, CUDA, and CPU devices."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from neuralops.benchmarks.matrix import ColdCacheFlusher


def test_cold_cache_flusher_cpu():
    """Verify CPU devices safely skip buffer allocation and flush is a no-op."""
    flusher = ColdCacheFlusher(torch.device("cpu"), size_mb=1)
    assert flusher._buffer is None
    # flush() must execute cleanly without error
    flusher.flush()


def test_cold_cache_flusher_string_device():
    """Verify ColdCacheFlusher accepts device specified as string."""
    flusher = ColdCacheFlusher("cpu", size_mb=1)
    assert isinstance(flusher.device, torch.device)
    assert flusher.device.type == "cpu"
    assert flusher._buffer is None
    flusher.flush()


def test_cold_cache_flusher_mps_hardware():
    """Verify MPS buffer allocation and flush on Apple Silicon if available."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available on this platform")

    flusher = ColdCacheFlusher(torch.device("mps"), size_mb=1)
    assert flusher._buffer is not None
    assert flusher._buffer.device.type == "mps"
    assert flusher._buffer.shape[0] == (1 * 1024 * 1024) // 4

    # Test flush zeroes buffer and synchronizes
    flusher.flush()
    assert (flusher._buffer == 0).all().item()


def test_cold_cache_flusher_mps_mocked():
    """Verify MPS buffer allocation and synchronization when mocked."""
    mock_tensor = MagicMock()
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.empty", return_value=mock_tensor) as mock_empty,
        patch("torch.mps.synchronize") as mock_sync,
    ):
        device = torch.device("mps")
        flusher = ColdCacheFlusher(device, size_mb=2)

        expected_elements = (2 * 1024 * 1024) // 4
        mock_empty.assert_called_once_with(expected_elements, dtype=torch.int32, device=device)
        assert flusher._buffer is mock_tensor

        # Verify flush calls zero_ and synchronize
        flusher.flush()
        mock_tensor.zero_.assert_called_once()
        assert mock_sync.call_count >= 1


def test_cold_cache_flusher_mock_dynamic_l2():
    """Verify CUDA dynamically uses L2 cache size from device properties."""
    mock_tensor = MagicMock()
    mock_props = MagicMock()
    l2_size_bytes = 48 * 1024 * 1024  # 48 MB
    mock_props.L2_cache_size = l2_size_bytes

    device = torch.device("cuda:0")
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_properties", return_value=mock_props),
        patch("torch.empty", return_value=mock_tensor) as mock_empty,
        patch("torch.cuda.synchronize") as mock_sync,
    ):
        flusher = ColdCacheFlusher(device, size_mb=256)

        # Expected elements from dynamic L2 cache size (48MB // 4)
        expected_elements = l2_size_bytes // 4
        mock_empty.assert_called_once_with(expected_elements, dtype=torch.int32, device=device)
        assert flusher._buffer is mock_tensor

        flusher.flush()
        mock_tensor.zero_.assert_called_once()
        mock_sync.assert_called_once_with(device)


def test_cold_cache_flusher_mock_fallback_l2():
    """Verify CUDA falls back to size_mb if L2_cache_size is 0 or unavailable."""
    mock_tensor = MagicMock()
    mock_props = MagicMock()
    mock_props.L2_cache_size = 0

    device = torch.device("cuda:0")
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_properties", return_value=mock_props),
        patch("torch.empty", return_value=mock_tensor) as mock_empty,
        patch("torch.cuda.synchronize"),
    ):
        flusher = ColdCacheFlusher(device, size_mb=32)

        # Fallback to 32MB
        expected_elements = (32 * 1024 * 1024) // 4
        mock_empty.assert_called_once_with(expected_elements, dtype=torch.int32, device=device)
