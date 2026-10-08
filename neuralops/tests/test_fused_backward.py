"""Tests for fused attention backward."""
import torch
import torch.nn.functional as F
from neuralops.core.autograd.fused_backward import fused_attention_backward


def test_fused_attention_backward_accuracy():
    torch.manual_seed(42)
    B, H, N, D = 2, 2, 8, 16
    q = torch.randn(B, H, N, D, requires_grad=True)
    k = torch.randn(B, H, N, D, requires_grad=True)
    v = torch.randn(B, H, N, D, requires_grad=True)
    grad_out = torch.randn(B, H, N, D)

    # Autograd reference
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = F.softmax(scores, dim=-1)
    out = torch.matmul(p, v)

    # Backward using autograd
    out.backward(grad_out)
    ref_gq, ref_gk, ref_gv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    # Backward using fused reference
    with torch.no_grad():
        gq, gk, gv = fused_attention_backward(grad_out, q.detach(), k.detach(), v.detach(), out.detach())

    assert torch.allclose(ref_gq, gq, atol=1e-5)
    assert torch.allclose(ref_gk, gk, atol=1e-5)
    assert torch.allclose(ref_gv, gv, atol=1e-5)
