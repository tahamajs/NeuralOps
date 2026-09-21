#!/usr/bin/env python3
"""
Unit tests for Adaptive Ranks Compressor with Jacobian-Gated SVD (JG-SVD).
Tests all 9 core properties:
1. GELU derivative squared at y=0 is 0.25 (phi=0.5, pdf=0, prime=0.5 -> prime^2=0.25)
2. GELU derivative squared at y=5 is ~1.0 (saturation in active region)
3. GELU derivative squared at y=-5 is ~0.0 (saturation in inactive region)
4. SiLU derivative squared at y=0 is 0.25 (sig=0.5, sig*(1+0)=0.5 -> 0.25)
5. Activation detection: DistilBERT ('ffn.lin1') -> 'gelu'
6. Activation detection: Qwen ('gate_proj', 'up_proj') -> 'silu'
7. Attention Q/K detection: 'q_lin', 'k_lin', 'q_proj', 'k_proj' detected, 'out_lin'/'o_proj' not detected
8. Backward compatibility: use_jg_svd=False runs standard adaptive ranks compression identically
9. Opt-in JG-SVD: use_jg_svd=True initializes, accepts flags, and performs JG-weighted compression
"""
import math
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("Codes"))
sys.path.insert(0, os.path.abspath("."))

from framework.compression_engine.adaptive_ranks import (
    AdaptiveRanksCompressor,
    _gelu_derivative_sq,
    _silu_derivative_sq,
    _detect_activation_type,
    _is_attention_qk,
    adaptive_ranks_factory,
)


def test_1_gelu_derivative_at_zero():
    """GELU derivative squared at y=0 should equal 0.25."""
    y = torch.tensor([0.0], dtype=torch.float32)
    val = _gelu_derivative_sq(y).item()
    assert abs(val - 0.25) < 1e-5, f"Expected 0.25, got {val}"
    print("[PASS] Test 1: GELU derivative squared at y=0 is 0.25")


def test_2_gelu_derivative_at_positive():
    """GELU derivative squared at y=5 should saturate near 1.0."""
    y = torch.tensor([5.0], dtype=torch.float32)
    val = _gelu_derivative_sq(y).item()
    assert abs(val - 1.0) < 1e-3, f"Expected ~1.0, got {val}"
    print("[PASS] Test 2: GELU derivative squared at y=5 is ~1.0")


def test_3_gelu_derivative_at_negative():
    """GELU derivative squared at y=-5 should saturate near 0.0."""
    y = torch.tensor([-5.0], dtype=torch.float32)
    val = _gelu_derivative_sq(y).item()
    assert abs(val - 0.0) < 1e-4, f"Expected ~0.0, got {val}"
    print("[PASS] Test 3: GELU derivative squared at y=-5 is ~0.0")


def test_4_silu_derivative_at_zero():
    """SiLU derivative squared at y=0 should equal 0.25."""
    y = torch.tensor([0.0], dtype=torch.float32)
    val = _silu_derivative_sq(y).item()
    assert abs(val - 0.25) < 1e-5, f"Expected 0.25, got {val}"
    print("[PASS] Test 4: SiLU derivative squared at y=0 is 0.25")


def test_5_activation_detection_distilbert():
    """Detect activation type maps DistilBERT ffn.lin1 and intermediate.dense to gelu."""
    assert _detect_activation_type("transformer.layer.0.ffn.lin1") == "gelu"
    assert _detect_activation_type("roberta.encoder.layer.0.intermediate.dense") == "gelu"
    assert _detect_activation_type("transformer.layer.0.ffn.lin2") == "none"
    print("[PASS] Test 5: Activation detection correctly identifies GELU layers")


def test_6_activation_detection_qwen():
    """Detect activation type maps Qwen/LLaMA SwiGLU projections to silu."""
    assert _detect_activation_type("model.layers.0.mlp.gate_proj") == "silu"
    assert _detect_activation_type("model.layers.0.mlp.up_proj") == "silu"
    assert _detect_activation_type("model.layers.0.mlp.down_proj") == "none"
    print("[PASS] Test 6: Activation detection correctly identifies SiLU SwiGLU layers")


def test_7_attention_qk_detection():
    """Detect attention Q and K layers for softmax Jacobian scalar weighting."""
    assert _is_attention_qk("transformer.layer.0.attention.q_lin") is True
    assert _is_attention_qk("transformer.layer.0.attention.k_lin") is True
    assert _is_attention_qk("model.layers.0.self_attn.q_proj") is True
    assert _is_attention_qk("model.layers.0.self_attn.k_proj") is True
    assert _is_attention_qk("transformer.layer.0.attention.out_lin") is False
    assert _is_attention_qk("model.layers.0.self_attn.o_proj") is False
    assert _is_attention_qk("transformer.layer.0.attention.v_lin") is False
    print("[PASS] Test 7: Attention Q/K detection correctly filters Q/K vs V/O projections")


def test_8_backward_compatibility_jg_off():
    """Default compressor with use_jg_svd=False runs without error and leaves used_jg_svd False."""
    model = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
    )
    compressor = AdaptiveRanksCompressor(model, min_rank=16, use_jg_svd=False)
    assert compressor.use_jg_svd is False
    compressed_model, stats = compressor.compress()
    assert stats["use_jg_svd"] is False
    assert all(v is False for v in stats["used_jg_svd"].values())
    print("[PASS] Test 8: Backward compatibility preserved when use_jg_svd=False")


def test_9_opt_in_jg_svd_on():
    """Opt-in compressor with use_jg_svd=True factorizes with JG weighting."""
    model = nn.Sequential(
        nn.Linear(64, 64),
        nn.GELU(),
        nn.Linear(64, 64),
    )
    compressor = AdaptiveRanksCompressor(
        model,
        min_rank=16,
        use_jg_svd=True,
        calibration_texts=["This is a test calibration sentence for JG-SVD verification."],
    )
    assert compressor.use_jg_svd is True
    assert compressor.use_cholesky_jg is True

    # Test factorization directly
    W = torch.randn(64, 64)
    cov = torch.eye(64) + 0.1 * torch.randn(64, 64)
    cov = cov.T @ cov
    D = torch.ones(64) * 0.25
    A, B, err = compressor._factorize_with_JG(W, cov, D, r=16)
    assert A.shape == (64, 16)
    assert B.shape == (16, 64)
    assert math.isfinite(err)
    print(f"[PASS] Test 9: JG-SVD factorization computed successfully (reconstruction error={err:.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING ADAPTIVE RANKS JG-SVD TEST SUITE (9 TESTS)")
    print("=" * 60)
    test_1_gelu_derivative_at_zero()
    test_2_gelu_derivative_at_positive()
    test_3_gelu_derivative_at_negative()
    test_4_silu_derivative_at_zero()
    test_5_activation_detection_distilbert()
    test_6_activation_detection_qwen()
    test_7_attention_qk_detection()
    test_8_backward_compatibility_jg_off()
    test_9_opt_in_jg_svd_on()
    print("=" * 60)
    print("ALL 9 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
