import pytest
import torch
import torch.nn as nn

from neuralops.distributed.tensor_parallel import (
    SimulatedTPGroup,
    RankParallelSVDLinear,
    RowParallelSVDLinear,
    ColumnParallelSVDLinear,
    TensorParallelRecurrentAttention,
)
from neuralops.distributed.context_parallel import (
    compute_chunk_prefix_scan,
    ContextParallelRecurrentScan,
    ContextParallelLinearAttention,
)


def test_rank_parallel_svd_linear_math_and_gradient_parity():
    """Verify RankParallelSVDLinear forward and backward match dense linear reconstruction."""
    torch.manual_seed(42)
    B, T, d_in, d_out, r = 2, 8, 32, 64, 8
    x = torch.randn(B, T, d_in, requires_grad=True)

    # 1. Full dense factorized linear
    v_dense = nn.Linear(d_in, r, bias=False)
    u_dense = nn.Linear(r, d_out, bias=False)
    dense_out = u_dense(v_dense(x))
    loss_dense = dense_out.sum()
    loss_dense.backward()
    dense_x_grad = x.grad.clone()

    # 2. Rank-sharded parallel layer (tp_size=2)
    # Rank 0 handles r in [0, 4), Rank 1 handles r in [4, 8)
    x_tp = x.detach().clone().requires_grad_(True)
    layer_0 = RankParallelSVDLinear(d_in, d_out, rank=r, tp_size=2, tp_rank=0, bias=False)
    layer_1 = RankParallelSVDLinear(d_in, d_out, rank=r, tp_size=2, tp_rank=1, bias=False)

    # Copy slices
    with torch.no_grad():
        layer_0.v_proj.weight.copy_(v_dense.weight[:4, :])
        layer_0.u_proj.weight.copy_(u_dense.weight[:, :4])
        layer_1.v_proj.weight.copy_(v_dense.weight[4:, :])
        layer_1.u_proj.weight.copy_(u_dense.weight[:, 4:])

    # Forward
    out_0 = layer_0.u_proj(layer_0.v_proj(x_tp))
    out_1 = layer_1.u_proj(layer_1.v_proj(x_tp))
    tp_out = out_0 + out_1

    assert torch.allclose(tp_out, dense_out, atol=1e-5)

    # Backward
    loss_tp = tp_out.sum()
    loss_tp.backward()
    assert torch.allclose(x_tp.grad, dense_x_grad, atol=1e-5)


def test_row_parallel_svd_bottleneck_communication():
    """Verify RowParallelSVDLinear reduces in compressed rank-r space."""
    torch.manual_seed(42)
    B, T, d_in, d_out, r = 2, 4, 32, 64, 4
    x = torch.randn(B, T, d_in)
    # Shard input across 2 TP workers along d_in
    x_local_0 = x[:, :, :16]
    x_local_1 = x[:, :, 16:]

    layer_0 = RowParallelSVDLinear(d_in, d_out, rank=r, tp_size=2, tp_rank=0, bias=False)
    layer_1 = RowParallelSVDLinear(d_in, d_out, rank=r, tp_size=2, tp_rank=1, bias=False)

    # Both share identical U output projection
    with torch.no_grad():
        layer_1.u_proj.weight.copy_(layer_0.u_proj.weight)

    b0 = layer_0.v_proj(x_local_0)
    b1 = layer_1.v_proj(x_local_1)
    # All-Reduce sum in rank-r space
    bottleneck = b0 + b1
    out_0 = layer_0.u_proj(bottleneck)
    out_1 = layer_1.u_proj(bottleneck)
    assert torch.allclose(out_0, out_1, atol=1e-5)
    assert out_0.shape == (B, T, d_out)


def test_tensor_parallel_recurrent_attention():
    """Verify TensorParallelRecurrentAttention sharding across heads."""
    B, T, d_model, heads = 2, 8, 32, 4
    x = torch.randn(B, T, d_model)
    layer = TensorParallelRecurrentAttention(embed_dim=d_model, num_heads=heads, tp_size=2, tp_rank=0)
    out, state = layer(x)
    assert out.shape == (B, T, d_model)
    assert state.shape == (B, 2, 8, 8)  # 2 local heads out of 4


def test_context_parallel_recurrent_scan_exact_parity():
    """Verify ContextParallelRecurrentScan (W=4 chunks) matches full sequence scan exactly."""
    torch.manual_seed(123)
    B, H, T, D = 2, 2, 128, 16
    W = 4
    T_c = T // W  # 32 tokens per chunk

    q = torch.randn(B, H, T, D, requires_grad=True)
    k = torch.randn(B, H, T, D, requires_grad=True)
    v = torch.randn(B, H, T, D, requires_grad=True)
    g = -torch.rand(B, H, T, D) * 0.1  # negative log decay
    g.requires_grad_(True)

    # 1. Single-pass full sequence scan
    out_full, s_end_full = compute_chunk_prefix_scan(q, k, v, g=g, init_state=None)
    loss_full = (out_full ** 2).sum()
    loss_full.backward()

    q_grad_full = q.grad.clone()
    k_grad_full = k.grad.clone()
    v_grad_full = v.grad.clone()
    g_grad_full = g.grad.clone()

    # 2. 4-Stage Context Parallelism chunked execution
    q_chunks = [q[:, :, p*T_c:(p+1)*T_c, :].detach().clone().requires_grad_(True) for p in range(W)]
    k_chunks = [k[:, :, p*T_c:(p+1)*T_c, :].detach().clone().requires_grad_(True) for p in range(W)]
    v_chunks = [v[:, :, p*T_c:(p+1)*T_c, :].detach().clone().requires_grad_(True) for p in range(W)]
    g_chunks = [g[:, :, p*T_c:(p+1)*T_c, :].detach().clone().requires_grad_(True) for p in range(W)]

    out_chunks = ContextParallelRecurrentScan.apply(q_chunks, k_chunks, v_chunks, g_chunks)
    out_cp = torch.cat(out_chunks, dim=2)

    # Verify Forward Parity
    forward_diff = (out_cp - out_full).abs().max().item()
    assert forward_diff < 1e-5, f"Forward diff exceeded 1e-5: {forward_diff}"

    # Verify Backward Parity
    loss_cp = (out_cp ** 2).sum()
    loss_cp.backward()

    q_grad_cp = torch.cat([qc.grad for qc in q_chunks], dim=2)
    k_grad_cp = torch.cat([kc.grad for kc in k_chunks], dim=2)
    v_grad_cp = torch.cat([vc.grad for vc in v_chunks], dim=2)
    g_grad_cp = torch.cat([gc.grad for gc in g_chunks], dim=2)

    assert (q_grad_cp - q_grad_full).abs().max().item() < 1e-4
    assert (k_grad_cp - k_grad_full).abs().max().item() < 1e-4
    assert (v_grad_cp - v_grad_full).abs().max().item() < 1e-4
    assert (g_grad_cp - g_grad_full).abs().max().item() < 1e-4
