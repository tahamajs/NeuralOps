"""Tests for production SmoothQuant."""
import torch
import torch.nn as nn
from neuralops.compression_engine.quantization.smoothquant import smoothquant_layer, smoothquant_model


def test_smoothquant_layer_real():
    torch.manual_seed(42)
    layer = nn.Linear(32, 32, bias=False)
    x = torch.randn(16, 32)
    # Add activation outlier
    x[:, 5] *= 10.0

    orig_out = layer(x)
    mod, s = smoothquant_layer(layer, alpha=0.5, calib_data=x)

    # Verified invariant: mod(x / s) == orig_out
    scaled_x = x / s.unsqueeze(0)
    new_out = mod(scaled_x)

    assert torch.allclose(orig_out, new_out, atol=1e-4)


def test_smoothquant_model_real():
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
    calib = [torch.randn(4, 16) for _ in range(2)]
    scales = smoothquant_model(model, calib, alpha=0.5)
    assert len(scales) == 2
