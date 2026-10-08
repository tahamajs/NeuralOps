"""Unit tests for variable-length attention."""
import torch
import pytest
from neuralops.core.triton_kernels.varlen_attention import varlen_attention_pytorch


def test_varlen_attention_shapes():
    # 2 sequences: len 10 and len 20
    cu_seqlens = torch.tensor([0, 10, 30], dtype=torch.int32)
    H, D = 4, 32
    total_tokens = 30
    
    q = torch.randn(total_tokens, H, D)
    k = torch.randn(total_tokens, H, D)
    v = torch.randn(total_tokens, H, D)
    
    out = varlen_attention_pytorch(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=20,
        max_seqlen_k=20,
        causal=True,
    )
    assert out.shape == (total_tokens, H, D)
