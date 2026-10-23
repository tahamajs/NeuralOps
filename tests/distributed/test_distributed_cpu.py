#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CPU-safe tests for NeuralOps distributed modules (tensor parallel and context parallel).

Verifies single-process execution, mathematical parity, context parallel chunking,
and informative exception handling when collectives are invoked without initialized process groups.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist

import neuralops.distributed as dist_pkg
from neuralops.distributed.tensor_parallel import (
    SimulatedTPGroup,
    ColumnParallelLinear,
    RowParallelLinear,
    RankParallelSVDLinear,
    RowParallelSVDLinear,
    ColumnParallelSVDLinear,
    TensorParallelRecurrentAttention,
    shard_model_tensor_parallel,
    copy_to_model_parallel_region,
    reduce_from_model_parallel_region,
)
from neuralops.distributed.context_parallel import (
    compute_chunk_prefix_scan,
    ContextParallelRecurrentScan,
    ContextParallelLinearAttention,
    ring_attention_chunk_exchange,
    RingLinearAttention,
)


def test_import_distributed_package():
    """Verify neuralops.distributed package exports all expected components cleanly."""
    expected_exports = [
        "SimulatedTPGroup",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "RankParallelSVDLinear",
        "RowParallelSVDLinear",
        "ColumnParallelSVDLinear",
        "TensorParallelRecurrentAttention",
        "shard_model_tensor_parallel",
        "copy_to_model_parallel_region",
        "reduce_from_model_parallel_region",
        "compute_chunk_prefix_scan",
        "ContextParallelRecurrentScan",
        "ContextParallelLinearAttention",
        "ring_attention_chunk_exchange",
        "RingLinearAttention",
    ]
    for name in expected_exports:
        assert hasattr(dist_pkg, name), f"Expected export '{name}' missing from neuralops.distributed"
        assert name in dist_pkg.__all__, f"'{name}' not included in dist_pkg.__all__"


def test_simulated_tp_group():
    """Test SimulatedTPGroup creation and metadata reporting."""
    group = SimulatedTPGroup(world_size=4)
    assert group.get_world_size() == 4

    with pytest.raises(ValueError, match="world_size must be positive"):
        SimulatedTPGroup(world_size=0)

    with pytest.raises(ValueError, match="world_size must be positive"):
        SimulatedTPGroup(world_size=-2)

    # SimulatedTPGroup must be rejected in collective operations
    x = torch.randn(2, 4)
    with pytest.raises(TypeError, match="SimulatedTPGroup cannot be used for collective operations"):
        copy_to_model_parallel_region(x, group=group)

    with pytest.raises(TypeError, match="SimulatedTPGroup cannot be used for collective operations"):
        reduce_from_model_parallel_region(x, group=group)

    layer = RowParallelLinear(8, 8, tp_size=2, tp_rank=0, process_group=group)
    with pytest.raises(TypeError, match="SimulatedTPGroup is metadata-only"):
        layer(torch.randn(2, 4))


def test_column_parallel_linear_cpu_single_process():
    """Test ColumnParallelLinear forward, backward, and parity on CPU with tp_size=1."""
    torch.manual_seed(42)
    B, T, in_features, out_features = 2, 4, 16, 32
    x = torch.randn(B, T, in_features, requires_grad=True)

    # With bias
    col_layer = ColumnParallelLinear(in_features, out_features, tp_size=1, tp_rank=0, bias=True)
    out = col_layer(x)
    assert out.shape == (B, T, out_features)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert col_layer.weight.grad is not None
    assert col_layer.bias.grad is not None

    # Parity with standard nn.Linear
    dense_linear = nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        col_layer.weight.copy_(dense_linear.weight)
        col_layer.bias.copy_(dense_linear.bias)

    x_test = torch.randn(B, T, in_features)
    assert torch.allclose(col_layer(x_test), dense_linear(x_test), atol=1e-5)

    # Without bias
    col_layer_nobias = ColumnParallelLinear(
        in_features, out_features, tp_size=1, tp_rank=0, bias=False
    )
    assert col_layer_nobias.bias is None
    out_nobias = col_layer_nobias(x_test)
    assert out_nobias.shape == (B, T, out_features)


def test_row_parallel_linear_cpu_single_process():
    """Test RowParallelLinear forward, backward, and parity on CPU with tp_size=1."""
    torch.manual_seed(42)
    B, T, in_features, out_features = 2, 4, 16, 32
    x = torch.randn(B, T, in_features, requires_grad=True)

    # Input is parallel (default)
    row_layer = RowParallelLinear(
        in_features, out_features, tp_size=1, tp_rank=0, bias=True, input_is_parallel=True
    )
    out = row_layer(x)
    assert out.shape == (B, T, out_features)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert row_layer.weight.grad is not None
    assert row_layer.bias.grad is not None

    # Parity with standard nn.Linear
    dense_linear = nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        row_layer.weight.copy_(dense_linear.weight)
        row_layer.bias.copy_(dense_linear.bias)

    x_test = torch.randn(B, T, in_features)
    assert torch.allclose(row_layer(x_test), dense_linear(x_test), atol=1e-5)

    # input_is_parallel = False
    row_layer_full = RowParallelLinear(
        in_features, out_features, tp_size=1, tp_rank=0, bias=True, input_is_parallel=False
    )
    with torch.no_grad():
        row_layer_full.weight.copy_(dense_linear.weight)
        row_layer_full.bias.copy_(dense_linear.bias)
    assert torch.allclose(row_layer_full(x_test), dense_linear(x_test), atol=1e-5)


def test_svd_linear_variants_cpu_single_process():
    """Test RankParallelSVDLinear, RowParallelSVDLinear, ColumnParallelSVDLinear on CPU with tp_size=1."""
    torch.manual_seed(42)
    B, T, in_features, out_features, rank = 2, 4, 16, 32, 4
    x = torch.randn(B, T, in_features, requires_grad=True)

    # RankParallelSVDLinear
    rank_svd = RankParallelSVDLinear(in_features, out_features, rank=rank, tp_size=1, tp_rank=0)
    out_rank = rank_svd(x)
    assert out_rank.shape == (B, T, out_features)
    out_rank.sum().backward()
    assert x.grad is not None
    x.grad.zero_()

    # RowParallelSVDLinear
    row_svd = RowParallelSVDLinear(in_features, out_features, rank=rank, tp_size=1, tp_rank=0)
    out_row = row_svd(x)
    assert out_row.shape == (B, T, out_features)
    out_row.sum().backward()
    assert x.grad is not None
    x.grad.zero_()

    # ColumnParallelSVDLinear
    col_svd = ColumnParallelSVDLinear(in_features, out_features, rank=rank, tp_size=1, tp_rank=0)
    out_col = col_svd(x)
    assert out_col.shape == (B, T, out_features)
    out_col.sum().backward()
    assert x.grad is not None


def test_tensor_parallel_recurrent_attention_cpu_single_process():
    """Test TensorParallelRecurrentAttention on CPU with tp_size=1, state retention, and gradient flow."""
    torch.manual_seed(42)
    B, T, embed_dim, num_heads = 2, 4, 16, 4
    x = torch.randn(B, T, embed_dim, requires_grad=True)

    tp_attn = TensorParallelRecurrentAttention(
        embed_dim=embed_dim, num_heads=num_heads, tp_size=1, tp_rank=0
    )
    out, next_state = tp_attn(x, state=None)
    assert out.shape == (B, T, embed_dim)
    assert next_state.shape == (B, num_heads, embed_dim // num_heads, embed_dim // num_heads)

    # Recurrent continuation with next_state
    x2 = torch.randn(B, T, embed_dim, requires_grad=True)
    out2, final_state = tp_attn(x2, state=next_state)
    assert out2.shape == (B, T, embed_dim)

    loss = out2.sum() + final_state.sum()
    loss.backward()
    assert x2.grad is not None


def test_context_parallel_utilities_cpu():
    """Test compute_chunk_prefix_scan and ContextParallelRecurrentScan chunk splitting/concatenation on CPU."""
    torch.manual_seed(42)
    B, H, T, D = 2, 2, 16, 8
    W = 4
    T_c = T // W

    q = torch.randn(B, H, T, D, requires_grad=True)
    k = torch.randn(B, H, T, D, requires_grad=True)
    v = torch.randn(B, H, T, D, requires_grad=True)
    g = -torch.rand(B, H, T, D) * 0.1
    g.requires_grad_(True)

    # Reference scan over full sequence
    out_ref, end_state_ref = compute_chunk_prefix_scan(q, k, v, g=g)
    assert out_ref.shape == (B, H, T, D)
    assert end_state_ref.shape == (B, H, D, D)

    # Split into W context chunks
    q_chunks = [q[:, :, p * T_c : (p + 1) * T_c, :].detach().clone().requires_grad_(True) for p in range(W)]
    k_chunks = [k[:, :, p * T_c : (p + 1) * T_c, :].detach().clone().requires_grad_(True) for p in range(W)]
    v_chunks = [v[:, :, p * T_c : (p + 1) * T_c, :].detach().clone().requires_grad_(True) for p in range(W)]
    g_chunks = [g[:, :, p * T_c : (p + 1) * T_c, :].detach().clone().requires_grad_(True) for p in range(W)]

    # Run ContextParallelRecurrentScan
    out_chunks = ContextParallelRecurrentScan.apply(q_chunks, k_chunks, v_chunks, g_chunks)
    assert len(out_chunks) == W
    out_cp = torch.cat(out_chunks, dim=2)

    # Parity between chunked scan and full scan
    assert torch.allclose(out_cp, out_ref, atol=1e-5)

    # ContextParallelLinearAttention module wrapper
    cp_module = ContextParallelLinearAttention(embed_dim=16, num_heads=2, cp_size=W)
    out_chunks_mod = cp_module.forward_chunks(q_chunks, k_chunks, v_chunks)
    assert len(out_chunks_mod) == W

    # Ring attention world_size=1
    k_chunk = torch.randn(B, H, T_c, D)
    v_chunk = torch.randn(B, H, T_c, D)
    k_ret, v_ret = ring_attention_chunk_exchange(k_chunk, v_chunk, rank=0, world_size=1)
    assert torch.equal(k_ret, k_chunk)
    assert torch.equal(v_ret, v_chunk)

    # RingLinearAttention module on CPU with ring_size=1
    ring_attn = RingLinearAttention(embed_dim=16, num_heads=2, ring_size=1, rank=0)
    x_local = torch.randn(B, T_c, 16)
    out_ring = ring_attn(x_local)
    assert out_ring.shape == (B, T_c, 16)


def test_runtime_error_when_multi_rank_collective_uninitialized(monkeypatch):
    """Verify RuntimeError is raised when multi-rank collectives are invoked without initialized process group."""
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    # RowParallelLinear with tp_size > 1
    row_layer = RowParallelLinear(16, 8, tp_size=2, tp_rank=0)
    with pytest.raises(RuntimeError, match="require.*torch.distributed to be initialized"):
        row_layer(torch.randn(2, 4, 8))

    # RowParallelSVDLinear with tp_size > 1
    row_svd = RowParallelSVDLinear(16, 8, rank=4, tp_size=2, tp_rank=0)
    with pytest.raises(RuntimeError, match="require.*torch.distributed to be initialized"):
        row_svd(torch.randn(2, 4, 8))

    # RankParallelSVDLinear with tp_size > 1
    rank_svd = RankParallelSVDLinear(16, 8, rank=4, tp_size=2, tp_rank=0)
    with pytest.raises(RuntimeError, match="require.*torch.distributed to be initialized"):
        rank_svd(torch.randn(2, 4, 16))

    # TensorParallelRecurrentAttention with tp_size > 1
    tp_attn = TensorParallelRecurrentAttention(embed_dim=16, num_heads=4, tp_size=2, tp_rank=0)
    with pytest.raises(RuntimeError, match="require.*torch.distributed to be initialized"):
        tp_attn(torch.randn(2, 4, 16))

    # ColumnParallelLinear with gather_output=True and tp_size > 1
    col_gather = ColumnParallelLinear(16, 8, tp_size=2, tp_rank=0, gather_output=True)
    with pytest.raises(RuntimeError, match="require.*torch.distributed to be initialized"):
        col_gather(torch.randn(2, 4, 16))

    # copy_to_model_parallel_region backward with enabled=True
    x = torch.randn(2, 4, requires_grad=True)
    out_copy = copy_to_model_parallel_region(x, enabled=True)
    with pytest.raises(RuntimeError, match="requires initialized torch.distributed"):
        out_copy.sum().backward()

    # reduce_from_model_parallel_region forward with enabled=True
    with pytest.raises(RuntimeError, match="requires initialized torch.distributed"):
        reduce_from_model_parallel_region(torch.randn(2, 4), enabled=True)

    # ring_attention_chunk_exchange with world_size > 1
    k_chunk = torch.randn(2, 2, 4, 8)
    v_chunk = torch.randn(2, 2, 4, 8)
    with pytest.raises(RuntimeError, match="requires initialized torch.distributed"):
        ring_attention_chunk_exchange(k_chunk, v_chunk, rank=0, world_size=2)

    # RingLinearAttention with ring_size > 1
    ring_attn = RingLinearAttention(embed_dim=16, num_heads=2, ring_size=2, rank=0)
    with pytest.raises(RuntimeError, match="requires initialized torch.distributed"):
        ring_attn(torch.randn(2, 4, 16))


def test_validation_and_not_implemented():
    """Verify input validation and NotImplementedError for unsupported automatic model sharding."""
    # shard_model_tensor_parallel
    with pytest.raises(NotImplementedError, match="Automatic model sharding is not implemented"):
        shard_model_tensor_parallel(nn.Linear(16, 16), tp_size=2, tp_rank=0)

    # Dimension divisibility validations
    with pytest.raises(ValueError, match="must be divisible by tp_size"):
        ColumnParallelLinear(16, 15, tp_size=2, tp_rank=0)

    with pytest.raises(ValueError, match="must be divisible by tp_size"):
        RowParallelLinear(15, 16, tp_size=2, tp_rank=0)

    with pytest.raises(ValueError, match="must be divisible by tp_size"):
        RankParallelSVDLinear(16, 16, rank=5, tp_size=2, tp_rank=0)

    # Rank bounds validations
    with pytest.raises(ValueError, match="tp_rank must be in"):
        ColumnParallelLinear(16, 16, tp_size=2, tp_rank=2)

    with pytest.raises(ValueError, match="tp_rank must be in"):
        ColumnParallelLinear(16, 16, tp_size=2, tp_rank=-1)

    with pytest.raises(TypeError, match="tp_size must be an integer"):
        ColumnParallelLinear(16, 16, tp_size=1.5, tp_rank=0)  # type: ignore[arg-type]
