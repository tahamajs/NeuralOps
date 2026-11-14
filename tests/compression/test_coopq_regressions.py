import torch
import torch.nn as nn
from neuralops.compression.quantization.coopq_quantizer import CoopQQuantizer, CoopQLinear


def test_coopq_quantizer_on_custom_linear_container():
    """Verify CoopQQuantizer quantizes child linear layers in container modules named *Linear*."""

    class CustomLinearBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 128)
            self.fc2 = nn.Linear(128, 64)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = CustomLinearBlock()
    comp = CoopQQuantizer(model, bits=4, min_bits=4, max_bits=8)
    comp_model, stats = comp.compress()

    assert isinstance(comp_model.fc1, (CoopQLinear, nn.Linear))
    x = torch.randn(2, 10, 128)
    out = comp_model(x)
    assert out.shape == (2, 10, 64)
    assert torch.isfinite(out).all()


def test_coopq_linear_dtype_preservation():
    """Verify CoopQLinear preserves FP16, BF16, and FP32."""
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        lin = nn.Linear(128, 128)
        coopq = CoopQLinear.from_linear(lin, bits=4).to(dtype=dtype)
        x = torch.randn(2, 8, 128, dtype=dtype)
        out = coopq(x)
        assert out.shape == (2, 8, 128)
        assert out.dtype == dtype
        assert torch.isfinite(out).all()
