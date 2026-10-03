"""Part 2 Verification: Shape-Based Weight Layout Detection (RCA-2)."""
import pytest
import torch
import torch.nn as nn
from framework.core.attention_utils import (_copy_projection_weights,
                                            _slice_fused_qkv_weights,
                                            compute_reconstruction_error)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)

def test_conv1d_transposed_slicing():
    # [in_dim, 3*out_dim] -> [64, 192]
    w = torch.randn(64, 192)
    b = torch.randn(192)
    sliced = _slice_fused_qkv_weights(w, b, num_heads=4)
    q_w, q_b = sliced["q"]
    assert q_w.shape == (64, 64)
    assert q_b.shape == (64,)
    assert torch.allclose(q_w, w[:, :64].t(), atol=1e-6)

def test_linear_fused_slicing():
    # [3*out_dim, in_dim] -> [192, 64]
    w = torch.randn(192, 64)
    b = torch.randn(192)
    sliced = _slice_fused_qkv_weights(w, b, num_heads=4)
    q_w, q_b = sliced["q"]
    assert q_w.shape == (64, 64)
    assert torch.allclose(q_w, w[:64, :], atol=1e-6)

def test_reconstruction_error_zero():
    target = nn.Linear(64, 64)
    src_w = torch.randn(64, 64)
    _copy_projection_weights(target, src_w, None)
    err = compute_reconstruction_error(src_w, target)
    assert err < 1e-5
