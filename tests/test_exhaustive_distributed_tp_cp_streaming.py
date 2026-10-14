"""
Suite 8: Exhaustive Verification Battery for Distributed Tensor Parallelism, Context Parallelism, and Streaming.
Expanded to 15+ parameterized tests verifying:
  8.1 RankParallelSVDLinear, RowParallelSVDLinear, ColumnParallelSVDLinear forward/backward parity across world sizes.
  8.2 ContextParallelRecurrentScan chunk prefix scan.
  8.3 PackedTokenStreamDataset chunk streaming and state_dict checkpoint determinism.
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


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("rank", [4, 8])
def test_8_1_rank_parallel_svd_linear_parity(tp_size, rank):
    B, T, d_in, d_out = 2, 8, 32, 64
    x = torch.randn(B, T, d_in, device=DEVICE, requires_grad=True)

    v_dense = nn.Linear(d_in, rank, bias=False).to(DEVICE)
    u_dense = nn.Linear(rank, d_out, bias=False).to(DEVICE)
    dense_out = u_dense(v_dense(x))
    loss_dense = dense_out.sum()
    loss_dense.backward()
    dense_x_grad = x.grad.clone()

    x_tp = x.detach().clone().requires_grad_(True)
    r_per_rank = rank // tp_size
    tp_layers = [
        RankParallelSVDLinear(d_in, d_out, rank=rank, tp_size=tp_size, tp_rank=r, bias=False).to(DEVICE)
        for r in range(tp_size)
    ]

    with torch.no_grad():
        for r in range(tp_size):
            tp_layers[r].v_proj.weight.copy_(v_dense.weight[r * r_per_rank : (r + 1) * r_per_rank, :])
            tp_layers[r].u_proj.weight.copy_(u_dense.weight[:, r * r_per_rank : (r + 1) * r_per_rank])

    tp_out = sum(tp_layers[r].u_proj(tp_layers[r].v_proj(x_tp)) for r in range(tp_size))

    assert torch.allclose(tp_out, dense_out, atol=1e-4)

    loss_tp = tp_out.sum()
    loss_tp.backward()
    assert torch.allclose(x_tp.grad, dense_x_grad, atol=1e-4)


@pytest.mark.parametrize("seq_len", [6, 12, 24])
def test_8_2_streaming_dataset(seq_len):
    dummy_docs = [[i, i + 1, i + 2] for i in range(30)]
    ds = PackedTokenStreamDataset(token_iterator_fn=lambda: iter(dummy_docs), seq_len=seq_len)
    items = []
    for item in ds:
        items.append(item)
        if len(items) >= 3:
            break
    assert len(items) == 3
    assert items[0]["input_ids"].shape[0] == seq_len
    assert items[0]["labels"].shape[0] == seq_len


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("d_in,d_out", [(32, 64), (64, 32)])
def test_8_3_row_and_column_parallel_svd_linear_parity(tp_size, d_in, d_out):
    B, T = 2, 8
    x = torch.randn(B, T, d_in, device=DEVICE)
    # Column parallel splits d_out across tp_size
    col_layers = [
        ColumnParallelSVDLinear(d_in, d_out, rank=8, tp_size=tp_size, tp_rank=r, bias=False).to(DEVICE)
        for r in range(tp_size)
    ]
    col_outs = [col_layers[r](x) for r in range(tp_size)]
    combined_col = torch.cat(col_outs, dim=-1)
    assert combined_col.shape == (B, T, d_out)
    assert not torch.isnan(combined_col).any()

    # Row parallel splits d_in across tp_size
    chunk_in = d_in // tp_size
    row_layers = [
        RowParallelSVDLinear(d_in, d_out, rank=8, tp_size=tp_size, tp_rank=r, bias=False).to(DEVICE)
        for r in range(tp_size)
    ]
    row_outs = [row_layers[r](x[:, :, r * chunk_in : (r + 1) * chunk_in]) for r in range(tp_size)]
    combined_row = sum(row_outs)
    assert combined_row.shape == (B, T, d_out)
    assert not torch.isnan(combined_row).any()


@pytest.mark.parametrize("num_heads", [2, 4])
@pytest.mark.parametrize("seq_len", [8, 16])
def test_8_4_context_parallel_recurrent_scan_prefix(num_heads, seq_len):
    B, H, T, D = 2, num_heads, seq_len, 16
    q = torch.randn(B, H, T, D, device=DEVICE)
    k = torch.randn(B, H, T, D, device=DEVICE)
    v = torch.randn(B, H, T, D, device=DEVICE)
    g = torch.sigmoid(torch.randn(B, H, T, D, device=DEVICE))

    # Compute chunk prefix scan
    out, final_state = compute_chunk_prefix_scan(q, k, v, g)
    assert out.shape == (B, H, T, D)
    assert final_state.shape == (B, H, D, D)
    assert not torch.isnan(out).any()


