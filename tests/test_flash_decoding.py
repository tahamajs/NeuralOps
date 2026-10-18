"""Tests for flash decoding attention."""
import torch
import torch.nn.functional as F
from neuralops.core.triton_kernels.flash_decoding import flash_decoding_attention


def test_flash_decoding_accuracy():
    torch.manual_seed(42)
    B, H, N_q, N_kv, D = 2, 4, 1, 256, 32
    q = torch.randn(B, H, N_q, D)
    k = torch.randn(B, H, N_kv, D)
    v = torch.randn(B, H, N_kv, D)

    # Standard attention reference
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = F.softmax(scores, dim=-1)
    ref_out = torch.matmul(weights, v)

    # Flash decoding with chunk_size 64
    fd_out = flash_decoding_attention(q, k, v, chunk_size=64)

    assert torch.allclose(ref_out, fd_out, atol=1e-5)
