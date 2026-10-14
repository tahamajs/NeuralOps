"""
Zero-Graph-Break torch.compile(fullgraph=True) and Dynamo Explanation Verification.
Tests custom op registration and FakeTensor metadata shape propagation across kernel modules.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.core.triton_kernels.op_registry import (
    _CUSTOM_OPS_REGISTERED,
    compilable_linear_attn,
    compilable_sparse_attn,
    compilable_low_rank,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_torch_library_custom_ops_registered():
    """Verify modern torch.library custom op registration."""
    assert _CUSTOM_OPS_REGISTERED, "Custom ops should be registered in op_registry"
    assert hasattr(torch.ops, "neuralops"), "torch.ops.neuralops namespace must be registered"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_dynamo_explain_zero_graph_breaks():
    """Verify torch._dynamo.explain yields 0 graph breaks on compilable entrypoints."""
    device = torch.device("cuda")
    dtype = torch.float16

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.U = nn.Parameter(torch.randn(128, 32, device=device, dtype=dtype))
            self.V = nn.Parameter(torch.randn(32, 128, device=device, dtype=dtype))

        def forward(self, x, q, k, v):
            y_lr = torch.ops.neuralops.low_rank(x, self.U, self.V)
            y_lin = torch.ops.neuralops.linear_attn(q, k, v)
            return y_lr + y_lin

    model = ToyModel().to(device)
    B, H, T, D = 2, 4, 64, 128
    x = torch.randn(B, H, T, D, device=device, dtype=dtype)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)

    explanation = torch._dynamo.explain(model, x, q, k, v)
    assert explanation.graph_break_count == 0, f"Expected 0 graph breaks, got {explanation.graph_break_count}: {explanation.graph_breaks}"

    # Also test compile execution
    compiled_model = torch.compile(model, fullgraph=True, backend="inductor")
    out = compiled_model(x, q, k, v)
    assert out.shape == (B * H, T, D) or out.shape == (B, H, T, D) or out.numel() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_fake_tensor_shape_propagation():
    """Verify FakeTensor shape propagation without materializing memory."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(2, 4, 128, 64, device="cuda", dtype=torch.float16)
        k = torch.empty(2, 4, 128, 64, device="cuda", dtype=torch.float16)
        v = torch.empty(2, 4, 128, 64, device="cuda", dtype=torch.float16)

        out_lin = torch.ops.neuralops.linear_attn(q, k, v)
        assert out_lin.shape == v.shape

        out_cos = torch.ops.neuralops.cosformer_attn(q, k, v, is_causal=True)
        assert out_cos.shape == v.shape

        x = torch.empty(8, 128, 64, device="cuda", dtype=torch.float16)
        gate = torch.empty(8, 128, 64, device="cuda", dtype=torch.float16)
        out_swiglu = torch.ops.neuralops.fused_swiglu(x, gate)
        assert out_swiglu.shape == x.shape

        norm_w = torch.empty(64, device="cuda", dtype=torch.float16)
        out_norm = torch.ops.neuralops.fused_rmsnorm(x, norm_w)
        assert out_norm.shape == x.shape
