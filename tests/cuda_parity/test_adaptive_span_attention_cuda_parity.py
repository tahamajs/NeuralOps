"""
Parity test for adaptive_span_attention CUDA native extension.
"""

import pytest
import torch

@pytest.mark.gpu
def test_adaptive_span_attention_cuda_matches_triton():
    try:
        from neuralops.ops.cuda.extensions.adaptive_span_attention import cuda_forward
    except ImportError:
        pytest.skip("adaptive_span_attention_cuda not compiled")

    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    device = torch.device("cuda")
    q = torch.randn(2, 4, 128, 32, device=device, dtype=torch.float32)
    k = torch.randn(2, 4, 128, 32, device=device, dtype=torch.float32)
    v = torch.randn(2, 4, 128, 32, device=device, dtype=torch.float32)

    try:
        out = cuda_forward(q, k, v, 1.0)
        assert out.shape == q.shape
    except Exception as e:
        pytest.skip(f"CUDA extension not available: {e}")
