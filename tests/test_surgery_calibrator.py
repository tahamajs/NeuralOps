"""Unit tests for Phase 6 Universal Streaming Activation Calibrator."""

import torch
import torch.nn as nn
import pytest

from neuralops.surgery.universal_calibrator import (
    UniversalActivationCalibrator,
    compress_model_universal,
    transplant_and_compress,
)


class MockMLP(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_streaming_activation_calibrator():
    model = MockMLP(d=32)
    calibrator = UniversalActivationCalibrator(model)

    # Provide calibration batches
    batches = [torch.randn(4, 32) for _ in range(3)]
    results = calibrator.calibrate(batches)

    assert "fc1" in results
    assert "fc2" in results
    assert "hessian" in results["fc1"]
    assert "channel_scale" in results["fc1"]
    assert results["fc1"]["hessian"].shape == (32, 32)
    assert results["fc1"]["channel_scale"].shape == (32,)


def test_compress_model_universal():
    model = MockMLP(d=32)
    batches = [torch.randn(4, 32) for _ in range(2)]
    compressed_model = compress_model_universal(
        model, method="svd", rank=16, calibration_data=batches
    )
    assert compressed_model is not None
    # Model remains runnable
    out = compressed_model(torch.randn(2, 32))
    assert out.shape == (2, 32)
