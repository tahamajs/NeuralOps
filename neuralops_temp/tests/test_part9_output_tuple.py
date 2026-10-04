"""Part 9 Verification: Output Tuple Standardization in UniversalAttentionWrapper."""
import pytest
import torch
import torch.nn as nn
from neuralops.utils.replace_attention import UniversalAttentionWrapper


def test_wrapper_standardizes_tensor_to_tuple():
    class RawTensorAttention(nn.Module):
        def forward(self, x):
            return x
    raw = RawTensorAttention()
    wrapper = UniversalAttentionWrapper(raw)
    x = torch.randn(2, 4, 16)
    out = wrapper(x)
    assert isinstance(out, tuple)
    assert len(out) >= 2
    assert torch.equal(out[0], x)
    assert out[1] is None
