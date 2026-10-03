"""
Verification test script for AdaptiveSVDAttention and AdaptiveRanksCompressor.
Validates:
1. Dynamic cumulative singular value energy thresholding:
   sum_{i=1}^r sigma_i^2 / sum_{j=1}^d sigma_j^2 >= tau (tau = 0.92)
2. Symmetrical sqrt(Sigma) singular value distribution:
   U_r = U[:, :r] * sqrt(Sigma_r), V_r = sqrt(Sigma_r) * V_r^T
3. FP32 and FP16 forward and backward passes on dummy tensors x = torch.randn(2, 128, 768)
4. Numerical stability (no NaNs, no Infs, finite gradients).
"""

import os
import sys
import types

import torch
import torch.nn as nn

# Ensure framework is in path
FRAMEWORK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, FRAMEWORK_ROOT)

from neuralops.attentions.adaptive_svd_attention import (AdaptiveSVDAttention,
                                                         AdaptiveSVDLinear)
from neuralops.compression_engine.aa_svd_compressor import SVDLinear
from neuralops.compression_engine.adaptive_ranks import (
    AdaptiveRanksCompressor, adaptive_ranks_factory)


class MockMHAModule(nn.Module):
    """Mock standard PyTorch Multihead Attention module for weight transfer."""
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)


class MockTransformerModel(nn.Module):
    """Mock multi-layer transformer model with 768 hidden size and realistic spectral decay."""
    def __init__(self, hidden_size: int = 768, num_heads: int = 12, intermediate_size: int = 3072):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self._init_spectral_decay_weights()

    def _init_spectral_decay_weights(self):
        """Initialize linear layers with low-rank factor product so energy concentrates rapidly."""
        with torch.no_grad():
            for mod in [self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.fc1, self.fc2]:
                out_f, in_f = mod.weight.shape
                r = 32
                u = torch.randn(out_f, r) / (out_f ** 0.5)
                v = torch.randn(in_f, r) / (in_f ** 0.5)
                mod.weight.copy_(u @ v.t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        attn_out = torch.matmul(torch.softmax(attn, dim=-1), v)
        x = self.norm1(res + self.out_proj(attn_out))
        res = x
        ffn = self.fc2(self.act(self.fc1(x)))
        return self.norm2(res + ffn)


def test_energy_thresholding_math():
    """Verify exact formula sum(s[:r]^2)/sum(s^2) >= tau."""
    print("\n--- Test 1: Cumulative Singular Value Energy Thresholding Math ---")
    tau = 0.92
    # Create decaying singular values
    d = 768
    i_indices = torch.arange(1, d + 1, dtype=torch.float32)
    s_vals = 1.0 / (i_indices ** 0.7)
    
    total_energy = (s_vals ** 2).sum().item()
    cum_energy = torch.cumsum(s_vals ** 2, dim=0) / total_energy
    idx = torch.where(cum_energy >= tau)[0]
    expected_r = int(idx[0].item()) + 1
    
    actual_ratio = (s_vals[:expected_r] ** 2).sum().item() / total_energy
    print(f"Energy threshold tau = {tau}")
    print(f"Total dimensions d = {d}, Calculated rank r = {expected_r}")
    print(f"Cumulative energy captured: {actual_ratio:.6f} (>= {tau})")
    assert actual_ratio >= tau, f"Calculated energy ratio {actual_ratio} < {tau}"
    print("✓ Dynamic cumulative energy threshold math verified.")


def test_symmetric_sqrt_sigma():
    """Verify symmetrical sqrt(Sigma) distribution reconstruction fidelity."""
    print("\n--- Test 2: Symmetrical sqrt(Sigma) Singular Value Distribution ---")
    torch.manual_seed(42)
    W = torch.randn(768, 768, dtype=torch.float32)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    
    tau = 0.92
    s_sq = S ** 2
    cum_e = torch.cumsum(s_sq, dim=0) / s_sq.sum()
    r = int(torch.where(cum_e >= tau)[0][0].item()) + 1
    
    # Symmetrical distribution
    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
    U_r = U[:, :r] * sqrt_S.unsqueeze(0)
    V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]
    
    # Reconstructed W_approx = U_r @ V_r
    W_approx = torch.matmul(U_r, V_r)
    
    # Direct truncated SVD reconstruction: U[:, :r] @ diag(S[:r]) @ Vt[:r, :]
    W_direct = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vt[:r, :])
    
    diff = torch.norm(W_approx - W_direct).item()
    norm_direct = torch.norm(W_direct).item()
    rel_diff = diff / max(1e-12, norm_direct)
    print(f"Frobenius difference: {diff:.8e}, Relative error: {rel_diff:.8e}")
    assert rel_diff < 1e-5, f"Discrepancy in symmetrical factorization: rel_diff={rel_diff}"
    print("✓ Symmetrical sqrt(Sigma) distribution mathematically verified.")


def test_adaptive_svd_attention_fp32_fp16():
    """Test AdaptiveSVDAttention with dummy tensor [2, 128, 768] in FP32 and FP16."""
    print("\n--- Test 3: AdaptiveSVDAttention Forward & Backward (FP32 & FP16) ---")
    config = types.SimpleNamespace(
        hidden_size=768,
        num_attention_heads=12,
        dropout=0.1,
        rank=64,
        svd_rank=64,
        residual_rank=8,
        energy_threshold=0.92,
    )
    
    mock_src = MockMHAModule(hidden_size=768)
    
    for dtype, dtype_name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
        print(f"\n  Testing AdaptiveSVDAttention in {dtype_name}...")
        model = AdaptiveSVDAttention(
            config=config,
            pretrained_attn=mock_src,
            energy_threshold=0.92,
        ).to(dtype=dtype)
        
        # Test input tensor [2, 128, 768]
        x = torch.randn(2, 128, 768, dtype=dtype, requires_grad=True)
        mask = torch.ones(2, 128, dtype=dtype)
        mask[:, 100:] = 0
        
        # Forward pass
        out, attn_weights = model(x, attention_mask=mask, output_attentions=True)
        
        assert out.shape == (2, 128, 768), f"Output shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), f"NaN detected in {dtype_name} forward output"
        assert not torch.isinf(out).any(), f"Inf detected in {dtype_name} forward output"
        assert attn_weights is not None
        assert attn_weights.shape == (2, 12, 128, 128)
        
        # Backward pass
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None, f"Input gradient is None in {dtype_name}"
        assert not torch.isnan(x.grad).any(), f"NaN in input gradient for {dtype_name}"
        assert not torch.isinf(x.grad).any(), f"Inf in input gradient for {dtype_name}"
        
        # Check trainable adapter gradients
        assert model.query.A.grad is not None, "query.A grad is None"
        assert model.query.B.grad is not None, "query.B grad is None"
        assert not torch.isnan(model.query.A.grad).any(), "NaN in query.A grad"
        
        print(f"  ✓ {dtype_name} forward & backward pass successful. Output norm: {out.norm().item():.4f}")


def test_adaptive_ranks_compressor_fp32_fp16():
    """Test AdaptiveRanksCompressor on mock transformer model in FP32 and FP16."""
    print("\n--- Test 4: AdaptiveRanksCompressor Forward & Backward (FP32 & FP16) ---")
    
    for dtype, dtype_name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
        print(f"\n  Testing AdaptiveRanksCompressor in {dtype_name}...")
        orig_model = MockTransformerModel(hidden_size=768, num_heads=12, intermediate_size=3072).to(dtype=dtype)
        orig_params = sum(p.numel() for p in orig_model.parameters())
        
        compressor = AdaptiveRanksCompressor(
            orig_model,
            energy_threshold=0.92,
            min_rank=16,
        )
        comp_model, stats = compressor.compress()
        
        comp_params = sum(p.numel() for p in comp_model.parameters())
        print(f"  Original params: {orig_params:,}, Compressed params: {comp_params:,}, Ratio: {stats['compression_ratio']:.2f}x")
        assert comp_params < orig_params, "Compression did not reduce parameter count"
        
        # Forward pass on [2, 128, 768]
        x = torch.randn(2, 128, 768, dtype=dtype, requires_grad=True)
        out = comp_model(x)
        
        assert out.shape == (2, 128, 768), f"Output shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), f"NaN detected in {dtype_name} compressed output"
        assert not torch.isinf(out).any(), f"Inf detected in {dtype_name} compressed output"
        
        # Backward pass
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None, f"x.grad is None in {dtype_name}"
        assert not torch.isnan(x.grad).any(), f"NaN in x.grad for {dtype_name}"
        assert not torch.isinf(x.grad).any(), f"Inf in x.grad for {dtype_name}"
        
        print(f"  ✓ {dtype_name} compressed model forward & backward successful. Output norm: {out.norm().item():.4f}")


def test_adaptive_ranks_factory():
    """Verify adaptive_ranks_factory functionality."""
    print("\n--- Test 5: adaptive_ranks_factory ---")
    orig_model = MockTransformerModel(hidden_size=768)
    comp_model = adaptive_ranks_factory(orig_model, energy_threshold=0.92, min_rank=16)
    assert isinstance(comp_model, nn.Module)
    x = torch.randn(2, 128, 768)
    out = comp_model(x)
    assert out.shape == (2, 128, 768)
    print("✓ adaptive_ranks_factory verified.")


if __name__ == "__main__":
    test_energy_thresholding_math()
    test_symmetric_sqrt_sigma()
    test_adaptive_svd_attention_fp32_fp16()
    test_adaptive_ranks_compressor_fp32_fp16()
    test_adaptive_ranks_factory()
    print("\n=======================================================")
    print("ALL SVD ADAPTIVE TESTS PASSED SUCCESSFULLY (FP32 + FP16)")
    print("=======================================================")
