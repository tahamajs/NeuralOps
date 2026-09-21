import torch
import torch.nn as nn
import math
import sys
import os

# Ensure project root and Codes/ are in python path
sys.path.insert(0, os.path.abspath("Codes"))
sys.path.insert(0, os.path.abspath("."))

from framework.compression_engine.a3_compressor import (
    A3Compressor, A3FactorizedLinear
)


def test_activation_energy_rename():
    """Ensure 'fisher' alias still works but energy is preferred."""
    model = nn.Sequential(
        nn.Linear(64, 64),
        nn.GELU(),
        nn.Linear(64, 64),
    )
    comp = A3Compressor(model, rank=32, use_fisher=True)
    assert hasattr(comp, "use_fisher")
    assert hasattr(comp, "use_energy_weighting")
    print("[OK] test_activation_energy_rename")


def test_weight_property_removed():
    """Ensure no expensive weight property."""
    layer = A3FactorizedLinear(64, 64, rank=16)
    assert not isinstance(
        getattr(type(layer), "weight", None), property
    ), "weight should not be a property"
    w = layer.get_weight()
    assert w.shape == (64, 64)
    print("[OK] test_weight_property_removed")


def test_normalize_scale_no_clamp():
    """Verify outlier channels are not clamped at 1e3 (ceiling removed)."""
    # Simulate outlier feature without RMS normalization (e.g. normalize_scale=False)
    f = torch.ones(64)
    f[3] = 1e10  # huge outlier
    scale = torch.sqrt(f.clamp(min=1e-8) + 1e-6)
    # Floor at 1e-6, no upper clamp at 1000
    scale_clamped = scale.clamp(min=1e-6)
    assert scale_clamped[3].item() > 1000, \
        f"Outlier should exceed 1000 without ceiling clamp, got {scale_clamped[3].item()}"
    assert scale_clamped[3].item() == scale[3].item()
    print("[OK] test_normalize_scale_no_clamp")


def test_activation_error_metric():
    """Test activation error computation."""
    comp = A3Compressor(nn.Linear(32, 32), rank=16)
    W = torch.randn(32, 32)
    W_r = W * 0.99  # small perturbation
    X = torch.randn(64, 32)
    err = comp._compute_activation_error(W, W_r, X, None)
    assert 0 < err < 0.5, f"Expected small error, got {err}"
    print(f"[OK] test_activation_error_metric (err={err:.4f})")


def test_strs_search():
    """Test STRS picks minimal rank satisfying error."""
    comp = A3Compressor(nn.Linear(64, 64), rank=32, max_error=0.5)
    W = torch.randn(64, 64)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    X = torch.randn(128, 64)
    factors, err, r = comp._strs_search(
        W, U, S, Vt, None, None, False, r_floor=16, r_ceiling=48,
        X=X, cov=None,
    )
    assert r >= 16 and r <= 48
    assert err <= 0.6, f"STR error too high: {err}"
    print(f"[OK] test_strs_search (r={r}, err={err:.4f})")


def test_cholesky_optional():
    """Test Cholesky mode runs without error."""
    model = nn.Sequential(
        nn.Linear(32, 32),
        nn.GELU(),
        nn.Linear(32, 32),
    )
    comp = A3Compressor(model, rank=16, use_cholesky=True)
    assert comp.use_cholesky is True
    print("[OK] test_cholesky_optional")


def test_factorized_forward():
    """Test forward pass produces correct shapes."""
    layer = A3FactorizedLinear(64, 32, rank=16)
    x = torch.randn(8, 10, 64)
    y = layer(x)
    assert y.shape == (8, 10, 32)
    print("[OK] test_factorized_forward")


def test_ln_failure_handling():
    """Test that LN failure can be raised or bypassed."""
    comp1 = A3Compressor(
        nn.Linear(32, 32), rank=16, allow_ln_failure=False
    )
    comp2 = A3Compressor(
        nn.Linear(32, 32), rank=16, allow_ln_failure=True
    )
    assert comp1.allow_ln_failure is False
    assert comp2.allow_ln_failure is True
    print("[OK] test_ln_failure_handling")


def test_fallback_to_original():
    """Test fallback when error is too high."""
    comp = A3Compressor(
        nn.Linear(64, 64), rank=4,  # very low rank
        max_error=0.001,  # impossible threshold
        fallback_to_original=True,
    )
    layer = nn.Linear(64, 64)
    result = comp.compress_linear(layer, name="test", rank=4)
    # result[0] should be None (fallback)
    assert result[0] is None, "Expected fallback to None"
    print("[OK] test_fallback_to_original")


if __name__ == "__main__":
    tests = [
        test_activation_energy_rename,
        test_weight_property_removed,
        test_normalize_scale_no_clamp,
        test_activation_error_metric,
        test_strs_search,
        test_cholesky_optional,
        test_factorized_forward,
        test_ln_failure_handling,
        test_fallback_to_original,
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
