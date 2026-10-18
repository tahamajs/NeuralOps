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

    # Verified invariant: mod applies s_inv internally so mod(x) == orig_out
    new_out = mod(x)
    assert torch.allclose(orig_out, new_out, atol=1e-4)

    # And manual scaled linear: (x / s) @ (W * s)^T == x @ W^T == orig_out
    import torch.nn.functional as F
    scaled_x = x / s.unsqueeze(0)
    manual_out = F.linear(scaled_x, mod.weight)
    assert torch.allclose(orig_out, manual_out, atol=1e-4)


def test_smoothquant_model_real():
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
    calib = [torch.randn(4, 16) for _ in range(2)]
    scales = smoothquant_model(model, calib, alpha=0.5)
    assert len(scales) == 2
