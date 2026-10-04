import math
import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")

from neuralops.attentions.hybrid_attention import (HybridAttention,
                                                   WeightedEnsembleAttention,
                                                   _extract_valid_mask)


def test_basic_construction():
    """Basic construction with minimal args."""
    block = HybridAttention(config=None, hidden_size=64, num_heads=4)
    assert block.hidden_size == 64
    assert block.num_heads == 4
    assert block.head_dim == 16
    print("[OK] test_basic_construction")


def test_forward_shape():
    """Forward produces correct output shape."""
    block = HybridAttention(config=None, hidden_size=64, num_heads=4)
    x = torch.randn(2, 16, 64)
    out, _ = block(x)
    assert out.shape == (2, 16, 64), f"Got {out.shape}"
    print("[OK] test_forward_shape")


def test_gate_init_is_softmax_dominant():
    """At init, gate ≈ 1 so model ≈ standard attention."""
    block = HybridAttention(config=None, hidden_size=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    gate = torch.sigmoid(block.ensemble_gate(x))
    assert (gate > 0.99).all(), f"Gate min: {gate.min()}"
    print(f"[OK] test_gate_init_is_softmax_dominant (min={gate.min():.5f})")


def test_causal_mask_no_nan():
    """Causal mode produces valid output."""
    block = HybridAttention(
        config=None, hidden_size=64, num_heads=4, is_decoder=True,
    )
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    print("[OK] test_causal_mask_no_nan")


def test_causal_future_does_not_affect_past():
    """In causal mode, changing future tokens does not affect past."""
    block = HybridAttention(
        config=None, hidden_size=64, num_heads=4, is_decoder=True,
    )
    block.eval()
    x = torch.randn(1, 16, 64)
    x_modified = x.clone()
    x_modified[:, 8:, :] = torch.randn(1, 8, 64)

    with torch.no_grad():
        out1, _ = block(x)
        out2, _ = block(x_modified)

    err = (out1[:, :8] - out2[:, :8]).abs().max().item()
    assert err < 1e-5, f"Causal violation: err={err}"
    print(f"[OK] test_causal_future_does_not_affect_past (err={err:.2e})")


def test_long_sequence_memory_bounded():
    """Chunked scan keeps memory sub-quadratic."""
    if not torch.cuda.is_available():
        print("[SKIP] test_long_sequence_memory_bounded (no CUDA)")
        return
    block = HybridAttention(
        config=None, hidden_size=256, num_heads=4, is_decoder=True,
    ).cuda()
    x = torch.randn(1, 1024, 256).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out, _ = block(x)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    assert peak_mb < 500, f"Memory too high: {peak_mb} MB"
    print(f"[OK] test_long_sequence_memory_bounded (peak={peak_mb:.0f} MB)")


def test_parameter_count():
    """Parameter count is within expected range."""
    block = HybridAttention(config=None, hidden_size=64, num_heads=4)
    total = sum(p.numel() for p in block.parameters())
    assert 16_500 < total < 17_500, f"Unexpected: {total}"
    print(f"[OK] test_parameter_count ({total})")


def test_valid_mask_extraction():
    """_extract_valid_mask handles all dims and dtypes."""
    B, N = 2, 8

    # None
    m = _extract_valid_mask(None, B, N, torch.device("cpu"))
    assert m.shape == (B, N) and m.all()

    # (B, N) bool
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -2:] = False
    m = _extract_valid_mask(mask, B, N, torch.device("cpu"))
    assert m.shape == (B, N)
    assert m[:, -2:].sum() == 0

    # (B, N) int
    mask = torch.ones(B, N, dtype=torch.long)
    mask[:, 0] = 0
    m = _extract_valid_mask(mask, B, N, torch.device("cpu"))
    assert m[:, 0].sum() == 0

    # (B, N) float additive
    mask = torch.zeros(B, N)
    mask[:, -1] = -float("inf")
    m = _extract_valid_mask(mask, B, N, torch.device("cpu"))
    assert m[:, -1].sum() == 0

    # (B, 1, N)
    mask = torch.ones(B, 1, N)
    m = _extract_valid_mask(mask, B, N, torch.device("cpu"))
    assert m.shape == (B, N)

    # (B, H, N, N)
    mask = torch.ones(B, 4, N, N)
    m = _extract_valid_mask(mask, B, N, torch.device("cpu"))
    assert m.shape == (B, N)

    print("[OK] test_valid_mask_extraction")


def test_nested_encoder_detection():
    """BertSelfAttention-like class is detected as nested."""
    class FakeBertSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)

    src = FakeBertSelfAttention()
    block = HybridAttention(config=None, hidden_size=64, num_heads=4)
    block.copy_weights(src)
    assert block.is_nested_encoder is True
    print("[OK] test_nested_encoder_detection")


def test_alias_class():
    """WeightedEnsembleAttention is an alias for HybridAttention."""
    assert WeightedEnsembleAttention is HybridAttention
    print("[OK] test_alias_class")


def test_registry_keys_preserved():
    """Old registry keys still work."""
    from neuralops.core.registry import ATTENTION_REGISTRY
    assert ATTENTION_REGISTRY.get("weighted_ensemble") is not None
    assert ATTENTION_REGISTRY.get("smart_weighted_ensemble") is not None
    assert ATTENTION_REGISTRY.get("hybrid_attention") is not None
    print("[OK] test_registry_keys_preserved")


def test_no_layernorm_added():
    """HybridAttention should not add internal LayerNorm."""
    block = HybridAttention(config=None, hidden_size=64, num_heads=4)
    for m in block.modules():
        assert not isinstance(m, nn.LayerNorm), "Block should not contain LayerNorm"
    print("[OK] test_no_layernorm_added")


def test_forward_with_mask():
    """Forward with attention mask works."""
    block = HybridAttention(config=None, hidden_size=64, num_heads=4)
    x = torch.randn(2, 16, 64)
    mask = torch.ones(2, 16, dtype=torch.long)
    mask[:, -4:] = 0
    out, _ = block(x, attention_mask=mask)
    assert not torch.isnan(out).any()
    print("[OK] test_forward_with_mask")


if __name__ == "__main__":
    tests = [
        test_basic_construction,
        test_forward_shape,
        test_gate_init_is_softmax_dominant,
        test_causal_mask_no_nan,
        test_causal_future_does_not_affect_past,
        test_long_sequence_memory_bounded,
        test_parameter_count,
        test_valid_mask_extraction,
        test_nested_encoder_detection,
        test_alias_class,
        test_registry_keys_preserved,
        test_no_layernorm_added,
        test_forward_with_mask,
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
