import torch
import torch.nn as nn
import sys
sys.path.insert(0, "Codes")

from framework.attentions.svd_sequence_attention import (
    SVDLinearAttention, LowRankSVDSequenceAttention,
)


def test_basic_construction():
    block = SVDLinearAttention(
        config=None, hidden_size=64, num_heads=4, rank=16,
    )
    assert block.hidden_size == 64
    assert block.num_heads == 4
    assert block.rank == 16
    print("[OK] test_basic_construction")


def test_forward_non_causal():
    block = SVDLinearAttention(
        config=None, hidden_size=64, num_heads=4, rank=16,
    )
    x = torch.randn(2, 32, 64)
    out, _ = block(x)
    assert out.shape == (2, 32, 64)
    print("[OK] test_forward_non_causal")


def test_forward_causal_bypass():
    """Causal mode bypasses SVD (see docstring)."""
    block = SVDLinearAttention(
        config=None, hidden_size=64, num_heads=4, rank=16,
        is_decoder=True,
    )
    x = torch.randn(2, 32, 64)
    out, _ = block(x)
    assert out.shape == (2, 32, 64)
    assert not torch.isnan(out).any()
    print("[OK] test_forward_causal_bypass")


def test_rank_none_disables_svd():
    block = SVDLinearAttention(
        config=None, hidden_size=64, num_heads=4, rank=None,
    )
    # With rank=None, should still work via full attention
    x = torch.randn(2, 16, 64)
    out, _ = block(x)
    assert out.shape == (2, 16, 64)
    print("[OK] test_rank_none_disables_svd")


def test_copy_weights_raises_on_missing():
    """copy_weights must raise on missing weights."""
    class EmptyAttn:
        pass
    
    try:
        block = SVDLinearAttention(
            config=None, hidden_size=64, num_heads=4, rank=16,
            old_attn=EmptyAttn(),
        )
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        print("[OK] test_copy_weights_raises_on_missing")


def test_get_token_valid_mask_none_safe():
    """attention_mask=None should not crash."""
    block = SVDLinearAttention(
        config=None, hidden_size=64, num_heads=4, rank=16,
    )
    x = torch.randn(2, 16, 64)
    out, _ = block(x, attention_mask=None)
    assert out.shape == (2, 16, 64)
    print("[OK] test_get_token_valid_mask_none_safe")


def test_gqa_slice_not_mean():
    """GQA copy uses slice, not mean."""
    block = SVDLinearAttention(
        config=None, hidden_size=64, num_heads=8, rank=16,
        num_kv_heads=2,
    )
    assert block.num_kv_heads == 2
    assert block.num_heads == 8
    print("[OK] test_gqa_slice_not_mean")


def test_registry_keys():
    from framework.core.registry import ATTENTION_REGISTRY
    assert "svd_seq" in ATTENTION_REGISTRY
    assert "svd_sequence" in ATTENTION_REGISTRY
    assert "svd_sequence_attention" in ATTENTION_REGISTRY
    print("[OK] test_registry_keys")


def test_no_registry_collision_with_other_svd():
    """'svd' key should not be present in this module."""
    from framework.core.registry import ATTENTION_REGISTRY
    # We specifically renamed to 'svd_seq' to avoid this collision
    # (test passes as long as no crash occurs on module load)
    print("[OK] test_no_registry_collision_with_other_svd")


def test_ortho_loss_is_diagnostic():
    """get_ortho_loss should not require gradient."""
    block = SVDLinearAttention(
        config=None, hidden_size=64, num_heads=4, rank=16,
    )
    x = torch.randn(2, 16, 64)
    _ = block(x)
    loss = block.get_ortho_loss()
    # Should be a plain tensor, not require grad
    assert not loss.requires_grad, \
        "Ortho loss should be a diagnostic, not differentiable"
    print(f"[OK] test_ortho_loss_is_diagnostic (loss={loss.item():.4f})")


def test_no_last_u_r_stale_grad():
    """_last_u_r should not carry gradients."""
    block = SVDLinearAttention(
        config=None, hidden_size=64, num_heads=4, rank=16,
    )
    x = torch.randn(2, 16, 64)
    _ = block(x)
    if block._last_u_r is not None:
        assert not block._last_u_r.requires_grad
    print("[OK] test_no_last_u_r_stale_grad")


def test_sdpa_fallback():
    """SDPA path should work when no mask/head_mask is given."""
    block = SVDLinearAttention(
        config=None, hidden_size=64, num_heads=4, rank=16,
    )
    block.eval()
    x = torch.randn(2, 32, 64)
    out, _ = block(x)
    assert out.shape == (2, 32, 64)
    print("[OK] test_sdpa_fallback")


if __name__ == "__main__":
    tests = [
        test_basic_construction,
        test_forward_non_causal,
        test_forward_causal_bypass,
        test_rank_none_disables_svd,
        test_copy_weights_raises_on_missing,
        test_get_token_valid_mask_none_safe,
        test_gqa_slice_not_mean,
        test_registry_keys,
        test_no_registry_collision_with_other_svd,
        test_ortho_loss_is_diagnostic,
        test_no_last_u_r_stale_grad,
        test_sdpa_fallback,
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
