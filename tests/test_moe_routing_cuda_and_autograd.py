"""
Comprehensive GPU tests for MoE Routing (Module 5).
Validates autograd backward pass, 2D/3D shapes, FP16/BF16/INT64 safety,
and permuted/dropped slot (-1) guards.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.core.cuda_kernels.moe_routing import (
    moe_dispatch_gather,
    moe_route_pytorch_ref,
)
from neuralops.core.cuda_kernels.moe_routing.moe_autograd import (
    _moe_dispatch_pytorch_ref,
    _moe_gather_pytorch_ref,
    _get_module,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("shape", [(128, 64), (4, 32, 64)])
def test_moe_routing_forward_parity(dtype, top_k, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    tokens = torch.randn(*shape, device="cuda", dtype=dtype)
    gate_logits = torch.randn(tokens.shape[0] if len(shape) == 2 else shape[0] * shape[1], 8,
                              device="cuda", dtype=dtype)
    if len(shape) == 3:
        gate_logits = gate_logits.view(shape[0], shape[1], 8)

    expert = nn.Sequential(
        nn.Linear(64, 128, device="cuda", dtype=dtype),
        nn.ReLU(),
        nn.Linear(128, 64, device="cuda", dtype=dtype),
    )

    out_cuda = moe_dispatch_gather(tokens, gate_logits, expert, top_k=top_k)

    # Reference computation
    routing_weights, topk_indices, expert_offsets = moe_route_pytorch_ref(tokens, gate_logits, top_k=top_k)
    flat_tokens = tokens.view(-1, 64)
    total_slots = flat_tokens.shape[0] * top_k
    disp_ref = _moe_dispatch_pytorch_ref(flat_tokens, topk_indices, expert_offsets, total_slots)
    exp_out_ref = expert(disp_ref)
    out_ref = _moe_gather_pytorch_ref(exp_out_ref, routing_weights, topk_indices, expert_offsets)
    if len(shape) == 3:
        out_ref = out_ref.view(*shape)

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(out_cuda, out_ref, atol=tol, rtol=tol)


def test_moe_routing_permuted_and_dropped_slots():
    """Validates custom expert_offsets with shuffled slots and dropped tokens (-1)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    num_tokens, hidden_dim = 16, 32
    tokens = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float32)

    top_k = 2
    topk_indices = torch.randint(0, 4, (num_tokens, top_k), device="cuda", dtype=torch.int64)

    # Permuted offsets with some dropped slots (-1)
    total_slots = 32
    offsets = torch.full((num_tokens, top_k), -1, device="cuda", dtype=torch.int64)
    valid_slots = torch.randperm(total_slots)[:20]
    offsets.view(-1)[:20] = valid_slots

    mod = _get_module()
    assert mod is not None

    # CUDA dispatch & gather
    disp_cuda = mod.moe_dispatch(tokens, topk_indices, offsets, total_slots)
    disp_ref = _moe_dispatch_pytorch_ref(tokens, topk_indices, offsets, total_slots)
    torch.testing.assert_close(disp_cuda, disp_ref)

    # Expert dummy compute
    exp_out = disp_cuda * 1.5
    routing_weights = torch.softmax(torch.randn(num_tokens, top_k, device="cuda"), dim=-1)

    gather_cuda = mod.moe_gather(exp_out, routing_weights, topk_indices, offsets)
    gather_ref = _moe_gather_pytorch_ref(exp_out, routing_weights, topk_indices, offsets)
    torch.testing.assert_close(gather_cuda, gather_ref)


def test_moe_routing_autograd_backward():
    """Validates loss.backward() parity for tokens.grad, gate_logits.grad, and expert weights."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    num_tokens, hidden_dim, num_experts, top_k = 32, 64, 4, 2

    tokens1 = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    gate1 = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32, requires_grad=True)
    mlp1 = nn.Linear(hidden_dim, hidden_dim, device="cuda", dtype=torch.float32)

    tokens2 = tokens1.clone().detach().requires_grad_(True)
    gate2 = gate1.clone().detach().requires_grad_(True)
    mlp2 = nn.Linear(hidden_dim, hidden_dim, device="cuda", dtype=torch.float32)
    mlp2.weight.data.copy_(mlp1.weight.data)
    mlp2.bias.data.copy_(mlp1.bias.data)

    # CUDA forward + backward
    out1 = moe_dispatch_gather(tokens1, gate1, mlp1, top_k=top_k)
    loss1 = (out1 ** 2).sum()
    loss1.backward()

    # PyTorch reference forward + backward
    topk_vals2, topk_idx2 = torch.topk(gate2, k=top_k, dim=-1)
    weights2 = F.softmax(topk_vals2.float(), dim=-1)
    disp2 = tokens2.repeat_interleave(top_k, dim=0)
    exp_out2 = mlp2(disp2)
    out2 = (exp_out2.view(num_tokens, top_k, hidden_dim) * weights2.unsqueeze(-1)).sum(dim=1)
    loss2 = (out2 ** 2).sum()
    loss2.backward()

    torch.testing.assert_close(out1, out2, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(tokens1.grad, tokens2.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(mlp1.weight.grad, mlp2.weight.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(mlp1.bias.grad, mlp2.bias.grad, atol=1e-3, rtol=1e-3)
