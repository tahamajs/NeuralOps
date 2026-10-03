"""Subagent 5: All Registered Compressors Verification Suite."""
import pytest
import torch
import torch.nn as nn
from framework.core.registry import COMPRESSOR_REGISTRY, is_helper_module


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 32)
    def forward(self, x):
        return self.fc2(self.fc1(x))

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)

@pytest.mark.parametrize("comp_name", [
    "a3", "aasvd", "adaptive_ranks", "direct_svd", "kronecker", "replaceme"
])
def test_compressor_compression_and_forward(comp_name):
    factory = COMPRESSOR_REGISTRY.get(comp_name)
    if factory is None:
        pytest.skip(f"{comp_name} not registered")

    model = DummyModel()

    try:
        res = factory(model, rank=8)
    except TypeError:
        res = factory(model)

    if hasattr(res, "compress"):
        compressed_model, stats = res.compress()
    elif isinstance(res, tuple):
        compressed_model, stats = res
    elif isinstance(res, nn.Module):
        compressed_model, stats = res, {}
    else:
        pytest.fail(f"Unexpected compressor return type: {type(res)}")

    assert isinstance(compressed_model, nn.Module)
    x = torch.randn(2, 32)
    out = compressed_model(x)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()
