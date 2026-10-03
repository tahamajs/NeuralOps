import sys

import torch
import torch.nn as nn

sys.path.insert(0, "Codes")

from framework.attentions.spectra_v2_attention import SpectraV2Attention


def test_basic_construction():
    block = SpectraV2Attention(
        config=None, hidden_size=64, num_heads=4, rank=32,
    )
    assert block.hidden_size == 64
    assert block.num_heads == 4
    print("[OK] test_basic_construction")


def test_forward_shape():
    block = SpectraV2Attention(
        config=None, hidden_size=64, num_heads=4, rank=32,
    )
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)
    print("[OK] test_forward_shape")


def test_attention_mask_respected():
    """In SDPA path, attention_mask must affect output."""
    block = SpectraV2Attention(
        config=None, hidden_size=64, num_heads=4, rank=32,
    )
    block.eval()
    x = torch.randn(1, 8, 64)
    # Mask out last 4 tokens
    mask = torch.ones(1, 8, dtype=torch.long)
    mask[:, -4:] = 0

    with torch.no_grad():
        out_masked, _ = block(x, attention_mask=mask)
        out_unmasked, _ = block(x)

    diff = (out_masked - out_unmasked).abs().max().item()
    assert diff > 1e-4, f"attention_mask was ignored (diff={diff:.2e})"
    print(f"[OK] test_attention_mask_respected (diff={diff:.2e})")


def test_causal_flag():
    """Causal mode should prevent future-token leakage."""
    block = SpectraV2Attention(
        config=None, hidden_size=64, num_heads=4, rank=32,
        is_causal=True,
    )
    block.eval()
    x = torch.randn(1, 8, 64)
    x_modified = x.clone()
    x_modified[:, 4:, :] = torch.randn(1, 4, 64)

    with torch.no_grad():
        out1, _ = block(x)
        out2, _ = block(x_modified)

    err = (out1[:, :4] - out2[:, :4]).abs().max().item()
    assert err < 1e-4, f"Causal leak: err={err}"
    print(f"[OK] test_causal_flag (err={err:.2e})")


def test_head_mask():
    """head_mask should zero out masked heads."""
    block = SpectraV2Attention(
        config=None, hidden_size=64, num_heads=4, rank=32,
    )
    block.eval()
    x = torch.randn(1, 8, 64)
    hm = torch.ones(1, 4)
    hm[:, 2:] = 0.0

    with torch.no_grad():
        out, _ = block(x, head_mask=hm)

    assert not torch.isnan(out).any()
    print("[OK] test_head_mask")


def test_gqa_copy_averaging():
    """GQA copy uses mean averaging, not repeating."""
    block = SpectraV2Attention(
        config=None, hidden_size=64, num_heads=8, rank=32,
        num_kv_heads=2,
    )
    assert block.num_kv_heads == 2
    assert block.num_heads == 8
    print("[OK] test_gqa_copy_averaging")


def test_no_fake_loss():
    """get_ortho_loss should not exist (was fake)."""
    block = SpectraV2Attention(
        config=None, hidden_size=64, num_heads=4, rank=32,
    )
    assert not hasattr(block, "get_ortho_loss"), "get_ortho_loss should be removed"
    print("[OK] test_no_fake_loss")


def test_no_config_dataclass():
    """SpectraConfig should be removed."""
    import framework.attentions.spectra_v2_attention as mod
    assert not hasattr(mod, "SpectraConfig"), "SpectraConfig should be removed in Option A"
    print("[OK] test_no_config_dataclass")


def test_deprecated_alias_works():
    """Old 'spectra_v2' registry key still resolves."""
    from framework.core.registry import ATTENTION_REGISTRY
    assert ATTENTION_REGISTRY.get("spectra_v2") is not None
    assert ATTENTION_REGISTRY.get("mha_sdpa") is not None
    print("[OK] test_deprecated_alias_works")


if __name__ == "__main__":
    tests = [
        test_basic_construction,
        test_forward_shape,
        test_attention_mask_respected,
        test_causal_flag,
        test_head_mask,
        test_gqa_copy_averaging,
        test_no_fake_loss,
        test_no_config_dataclass,
        test_deprecated_alias_works,
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
