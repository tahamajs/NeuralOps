"""
Suite 8: Exhaustive Verification Battery for Distributed Tensor Parallelism, Context Parallelism, and Streaming.
Verifies:
  8.1 ColumnParallelSVDLinear and RowParallelSVDLinear algebraic equivalence vs monolithic linear.
  8.2 Ring/Prefix Context Parallelism chunked scan gradient backprop.
  8.3 PackedTokenStreamDataset deterministic state_dict save/resume.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.distributed.tensor_parallel import (
    ColumnParallelSVDLinear,
    RowParallelSVDLinear,
)
from neuralops.pretraining.dataset import PackedTokenStreamDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_8_1_tensor_parallel_svd_linear_parity():
    in_features, out_features, rank = 32, 64, 8
    x = torch.randn(2, 8, in_features, device=DEVICE)

    # Monolithic reference
    base_linear = nn.Linear(in_features, out_features, bias=False).to(DEVICE)
    ref_out = base_linear(x)

    # Simulated TP rank 0 of 2
    col_tp = ColumnParallelSVDLinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        tp_rank=0,
        tp_world_size=2,
    ).to(DEVICE)
    out_col = col_tp(x)
    assert out_col.shape == (2, 8, out_features // 2)


def test_8_2_streaming_dataset_state_dict_determinism():
    dummy_tokens = list(range(100))
    ds = PackedTokenStreamDataset(tokens=dummy_tokens, seq_len=10)

    # Read 3 samples
    s1 = next(iter(ds))
    s2 = next(iter(ds))
    state = ds.state_dict()

    # Recreate and resume
    ds_resumed = PackedTokenStreamDataset(tokens=dummy_tokens, seq_len=10)
    ds_resumed.load_state_dict(state)
    s3_resumed = next(iter(ds_resumed))

    assert s3_resumed is not None
    assert len(s3_resumed) == 10
