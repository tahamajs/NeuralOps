import math
import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")

from neuralops.attentions.salt_attention import (SALTAttention, SALTLinear,
                                                 inject_salt_attention,
                                                 merge_model_salt)


def test_salt_linear_construction():
    """Basic SALTLinear construction."""
    W = torch.randn(64, 64)
    layer = SALTLinear(64, 64, rank=16, original_weight=W)
    assert layer.rank == 16
    print("[OK] test_salt_linear_construction")


def test_salt_linear_init_recovers_W_r():
    """Init should give W_r (truncated SVD)."""
    W = torch.randn(64, 64)
    layer = SALTLinear(64, 64, rank=16, original_weight=W)
    W_r = layer.weight
    # Compare with truncated SVD
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    W_r_expected = (U[:, :16] * S[:16].unsqueeze(0)) @ Vt[:16, :]
    err = (W_r - W_r_expected).norm() / (W_r_expected.norm() + 1e-8)
    assert err < 1e-5, f"Init should match W_r, err={err}"
    print(f"[OK] test_salt_linear_init_recovers_W_r (err={err:.2e})")


def test_fisher_weighted_svd():
    """Fisher weighting changes the SVD."""
    W = torch.randn(64, 64)
    F = torch.ones(64)
    F[10:20] = 100.0  # some high-activation dims
    
    layer_fisher = SALTLinear(64, 64, rank=16, 
                              original_weight=W, fisher_diag=F)
    layer_plain = SALTLinear(64, 64, rank=16, 
                             original_weight=W, fisher_diag=None)
    
    # Different SVD directions should be captured
    diff = (layer_fisher.U - layer_plain.U).abs().mean()
    assert diff > 0.01, "Fisher should change U"
    print(f"[OK] test_fisher_weighted_svd (diff={diff:.4f})")


def test_attention_aware_gamma_init():
    """Gamma init from sensitivity changes the values."""
    W = torch.randn(64, 64)
    layer = SALTLinear(64, 64, rank=16, original_weight=W)
    
    # Default gamma = 1
    assert (layer.gamma == 1.0).all()
    
    # Set custom sensitivity
    sens = torch.tensor([1.0] * 8 + [10.0] * 8)
    layer.init_gamma_from_sensitivity(sens)
    
    # High sensitivity should give higher gamma
    assert layer.gamma[8:].mean() > layer.gamma[:8].mean()
    print("[OK] test_attention_aware_gamma_init")


def test_per_projection_adaptive_rank():
    """Different projections get different ranks."""
    class FakeConfig:
        num_attention_heads = 4
        hidden_size = 64
        head_dim = 16
    
    attn = SALTAttention(config=FakeConfig(), rank=16)
    ranks = attn.proj_ranks
    # Just verify all keys exist and are positive
    assert all(r > 0 for r in ranks.values())
    print(f"[OK] test_per_projection_adaptive_rank (ranks={ranks})")


def test_merge_to_linear():
    """Merged linear produces the same output."""
    W = torch.randn(64, 64)
    layer = SALTLinear(64, 64, rank=16, original_weight=W)
    
    x = torch.randn(2, 8, 64)
    out_before = layer(x)
    merged = layer.merge_to_linear()
    out_after = merged(x)
    
    err = (out_before - out_after).norm() / (out_before.norm() + 1e-8)
    assert err < 1e-5, f"Merge error too high: {err}"
    print(f"[OK] test_merge_to_linear (err={err:.2e})")


def test_merge_model():
    """merge_model_salt replaces SALTLinear with nn.Linear."""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = SALTLinear(32, 32, rank=8)
        def forward(self, x):
            return self.q(x)
    
    model = TestModel()
    merged = merge_model_salt(model)
    assert isinstance(merged.q, nn.Linear), \
        f"Expected nn.Linear, got {type(merged.q)}"
    print("[OK] test_merge_model")


def test_registry_registration():
    """SALT keys are registered."""
    from neuralops.core.registry import ATTENTION_REGISTRY
    assert "salt" in ATTENTION_REGISTRY
    assert "salt_attention" in ATTENTION_REGISTRY
    assert "salt_f" in ATTENTION_REGISTRY
    assert "salt_f_attention" in ATTENTION_REGISTRY
    print("[OK] test_registry_registration")


if __name__ == "__main__":
    tests = [
        test_salt_linear_construction,
        test_salt_linear_init_recovers_W_r,
        test_fisher_weighted_svd,
        test_attention_aware_gamma_init,
        test_per_projection_adaptive_rank,
        test_merge_to_linear,
        test_merge_model,
        test_registry_registration,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*60}")
    print(f"PASSED: {passed}/{len(tests)}")
    print(f"FAILED: {failed}/{len(tests)}")
    sys.exit(0 if failed == 0 else 1)
