"""Tests for Paged Attention kernel."""
import torch
from neuralops.core.triton_kernels.paged_attention import paged_attention_kernel


def test_paged_attention_kernel_matches():
    torch.manual_seed(42)
    B, H, D = 2, 4, 32
    block_size = 8
    num_blocks = 10

    q = torch.randn(B, H, D)
    k_cache = torch.randn(num_blocks, block_size, H, D)
    v_cache = torch.randn(num_blocks, block_size, H, D)

    # 2 sequences, lengths 12 and 15
    seq_lens = torch.tensor([12, 15], dtype=torch.long)
    # block tables: seq0 uses [0, 2], seq1 uses [1, 3]
    block_tables = torch.tensor([[0, 2, 0, 0], [1, 3, 0, 0]], dtype=torch.long)

    out = paged_attention_kernel(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)
    assert out.shape == (B, H, D)
    assert not torch.isnan(out).any()
