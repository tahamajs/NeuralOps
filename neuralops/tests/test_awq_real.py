"""Tests for production AWQ quantizer."""
import torch
import torch.nn as nn
from neuralops.compression_engine.quantization.awq import AWQQuantizer, quantize_model_awq


def test_awq_numerical_preservation():
    torch.manual_seed(42)
    layer = nn.Linear(64, 64, bias=False)
    orig_w = layer.weight.data.clone()
    awq = AWQQuantizer(layer, bits=4, alpha=0.5)

    x = torch.randn(32, 64)
    # Introduce significant activation outlier channel
    x[:, 10] *= 20.0
    awq.observe(x)

    w_q = awq.quantize()
    assert not torch.equal(layer.weight.data, orig_w)

    with torch.no_grad():
        out_orig = torch.matmul(x, orig_w.t())
        out_q = layer(x)
        rel_err = torch.norm(out_orig - out_q) / torch.norm(out_orig)
        assert rel_err < 0.5


def test_quantize_model_awq():
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 16))
    calib = [torch.randn(8, 32) for _ in range(2)]
    quantize_model_awq(model, calib, bits=4)
    out = model(torch.randn(2, 32))
    assert out.shape == (2, 16)
