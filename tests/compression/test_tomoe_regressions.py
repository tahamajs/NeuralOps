import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.tomoe_compressor import ToMoELinear, ToMoECompressor


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tomoe_linear_dtype_preservation(dtype: torch.dtype):
    """Regression test for FP16/BF16 forward pass in ToMoELinear."""
    lin = nn.Linear(768, 768)
    tomoe = ToMoELinear(lin, num_experts=4, top_k=2, rank=16).to(dtype=dtype)
    x = torch.randn(2, 64, 768, dtype=dtype)
    out = tomoe(x)
    assert out.shape == (2, 64, 768)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


def test_tomoe_compressor_fp16_model():
    """Verify ToMoECompressor compressed model supports FP16 inference."""

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(768, 768)

        def forward(self, x):
            return self.fc(x)

    model = ToyModel()
    comp = ToMoECompressor(model, num_experts=4, top_k=2, rank=16)
    comp_model, stats = comp.compress()
    comp_model = comp_model.half()

    x = torch.randn(2, 64, 768, dtype=torch.float16)
    out = comp_model(x)
    assert out.shape == (2, 64, 768)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
