import pytest
import torch
import torch.nn as nn
from neuralops.attentions.linear.cosformer import CosformerAttention
from neuralops.attentions.linear.galerkin_attention import GalerkinAttention
from neuralops.utils.amp_safety import safe_linear_attn_denominator, safe_l2_normalize, safe_svd, safe_cholesky


def test_safe_amp_math_numerical_stability():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # 1. Denominator safety under near-zero and extreme values
    near_zero = torch.tensor([0.0, 1e-9, -1e-8, 1e-5], device=device)
    safe_denom = safe_linear_attn_denominator(near_zero, eps=1e-5, floor_val=1e-4)
    assert not torch.isnan(safe_denom).any()
    assert not torch.isinf(safe_denom).any()
    assert (safe_denom.abs() >= 1e-4).all()

    # 2. SVD with AMP enabled
    with torch.amp.autocast(device_type=device_type, enabled=True):
        mat = torch.randn(32, 32, device=device, dtype=torch.float16 if device_type == "cuda" else torch.float32)
        U, S, Vh = safe_svd(mat)
        assert not torch.isnan(U).any()
        assert not torch.isnan(S).any()
        assert not torch.isnan(Vh).any()

    # 3. Cholesky with AMP enabled and ill-conditioned matrix
    with torch.amp.autocast(device_type=device_type, enabled=True):
        # Semi-definite matrix
        A = mat @ mat.T
        L = safe_cholesky(A)
        assert not torch.isnan(L).any()


def test_cosformer_long_seq_extreme_fp16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"

    attn = CosformerAttention(
        hidden_size=64,
        num_heads=4,
        is_causal=True,
    ).to(device)
    attn.eval()

    # Stress test: T = 2048, scaled by 5.0 in FP16 to challenge dynamic range
    B, N, D = 1, 2048, 64
    x = (torch.randn(B, N, D, device=device) * 5.0)

    if device_type == "cuda":
        x = x.half()
        attn = attn.half()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out, _ = attn(x)
    else:
        out, _ = attn(x)

    assert not torch.isnan(out).any(), "Cosformer produced NaNs on long sequence extreme stress test!"
    assert not torch.isinf(out).any(), "Cosformer produced Infs on long sequence extreme stress test!"
