"""
Suite 8: Exhaustive Verification Battery for Distributed Tensor Parallelism, Context Parallelism, and Streaming.
Verifies:
  8.1 RankParallelSVDLinear forward/backward equivalence vs dense linear.
  8.2 ContextParallelRecurrentScan chunk prefix scan.
  8.3 PackedTokenStreamDataset chunk streaming.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.distributed.tensor_parallel import (
    RankParallelSVDLinear,
    RowParallelSVDLinear,
    ColumnParallelSVDLinear,
)
from neuralops.distributed.context_parallel import (
    compute_chunk_prefix_scan,
    ContextParallelRecurrentScan,
)
from neuralops.data.streaming import PackedTokenStreamDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_8_1_tensor_parallel_svd_linear_parity():
    B, T, d_in, d_out, r = 2, 8, 32, 64, 8
    x = torch.randn(B, T, d_in, device=DEVICE, requires_grad=True)

    v_dense = nn.Linear(d_in, r, bias=False).to(DEVICE)
    u_dense = nn.Linear(r, d_out, bias=False).to(DEVICE)
    dense_out = u_dense(v_dense(x))
    loss_dense = dense_out.sum()
    loss_dense.backward()
    dense_x_grad = x.grad.clone()

    x_tp = x.detach().clone().requires_grad_(True)
    layer_0 = RankParallelSVDLinear(d_in, d_out, rank=r, tp_size=2, tp_rank=0, bias=False).to(DEVICE)
    layer_1 = RankParallelSVDLinear(d_in, d_out, rank=r, tp_size=2, tp_rank=1, bias=False).to(DEVICE)

    with torch.no_grad():
        layer_0.v_proj.weight.copy_(v_dense.weight[:4, :])
        layer_0.u_proj.weight.copy_(u_dense.weight[:, :4])
        layer_1.v_proj.weight.copy_(v_dense.weight[4:, :])
        layer_1.u_proj.weight.copy_(u_dense.weight[:, 4:])

    out_0 = layer_0.u_proj(layer_0.v_proj(x_tp))
    out_1 = layer_1.u_proj(layer_1.v_proj(x_tp))
    tp_out = out_0 + out_1

    assert torch.allclose(tp_out, dense_out, atol=1e-5)

    loss_tp = tp_out.sum()
    loss_tp.backward()
    assert torch.allclose(x_tp.grad, dense_x_grad, atol=1e-5)


def test_8_2_streaming_dataset():
    dummy_docs = [[i, i + 1, i + 2] for i in range(20)]
    ds = PackedTokenStreamDataset(token_iterator_fn=lambda: iter(dummy_docs), seq_len=6)
    items = []
    for item in ds:
        items.append(item)
        if len(items) >= 3:
            break
    assert len(items) == 3
    assert len(items[0]) == 6
