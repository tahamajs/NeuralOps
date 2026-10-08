"""Unit tests for block-sparse attention."""
import torch
import pytest
from neuralops.core.triton_kernels.block_sparse import butterfly_mask, block_sparse_attention


def test_butterfly_mask_shape_and_diag():
    mask = butterfly_mask(128, block_size=32)
    assert mask.shape == (128, 128)
    # Diagonal must be True
    assert mask.diag().all()


def test_block_sparse_attention_forward():
    q = torch.randn(2, 4, 64, 32)
    k = torch.randn(2, 4, 64, 32)
    v = torch.randn(2, 4, 64, 32)
    out = block_sparse_attention(q, k, v, block_size=16)
    assert out.shape == (2, 4, 64, 32)
    assert not torch.isnan(out).any()
