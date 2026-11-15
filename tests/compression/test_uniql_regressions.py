import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.uniql_compressor import UniQLCompressor, UniQLinear


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_uniql_linear_dtype_preservation(dtype: torch.dtype):
    """Regression test for FP16/BF16 forward pass in UniQLinear."""
    U_q = torch.randint(-127, 127, (64, 16), dtype=torch.int8)
    S_q = torch.randint(-127, 127, (16,), dtype=torch.int8)
    Vt_q = torch.randint(-127, 127, (16, 64), dtype=torch.int8)
    scale_u = torch.randn(64) * 0.01
    scale_s = torch.randn(1).squeeze() * 0.01
    scale_v = torch.randn(16) * 0.01
    bias = torch.randn(64) * 0.01

    layer = UniQLinear(
        in_features=64,
        out_features=64,
        rank=16,
        U_q=U_q,
        S_q=S_q,
        Vt_q=Vt_q,
        scale_u=scale_u,
        scale_s=scale_s,
        scale_v=scale_v,
        bias=bias,
    ).to(dtype=dtype)

    x = torch.randn(2, 8, 64, dtype=dtype)
    out = layer(x)
    assert out.shape == (2, 8, 64)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


def test_uniql_compressor_fp16_model():
    """Verify UniQLCompressor compressed model supports FP16 inference."""

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 64)

        def forward(self, x):
            return self.fc(x)

    model = ToyModel()
    comp = UniQLCompressor(model, rank=16)
    comp_model, stats = comp.compress()
    comp_model = comp_model.half()

    x = torch.randn(2, 8, 64, dtype=torch.float16)
    out = comp_model(x)
    assert out.shape == (2, 8, 64)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
