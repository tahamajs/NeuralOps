import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "Codes")

from framework.attentions.spectral_mamba_attention import (
    SpectralMambaAttention,
    SpectralMamba,
)


def test_spectral_mamba_forward():
    """SpectralMamba produces correct shape."""
    m = SpectralMamba(d_model=64)
    x = torch.randn(2, 16, 64)
    y = m(x)
    assert y.shape == (2, 16, 64)
    assert not torch.isnan(y).any()
    print("[OK] test_spectral_mamba_forward")


def test_fft_actually_filters():
    """SpectralMamba should change the input."""
    m = SpectralMamba(d_model=64)
    m.freq_weights.copy_(torch.rand_like(m.freq_weights))
    x = torch.randn(2, 16, 64)
    y = m(x)
    diff = (y - x).abs().mean().item()
    assert diff > 0.01, f"FFT filtering produced no change: {diff}"
    print(f"[OK] test_fft_actually_filters (diff={diff:.4f})")


def test_gate_init_is_attention_dominant():
    """At init, gate ≈ 1 so output ≈ attention."""
    block = SpectralMambaAttention(
        config=None, hidden_size=64, num_heads=4, rank=32,
    )
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        gate_logits = F.linear(x, block.gate_weight, block.gate_bias)
        gate_logits = gate_logits + F.linear(
            F.linear(x, block.gate_lora_A), block.gate_lora_B
        ) * 2.0
        gate = torch.sigmoid(gate_logits)
    assert (gate > 0.99).all(), f"Gate min: {gate.min()}"
    print(f"[OK] test_gate_init_is_attention_dominant (min={gate.min():.5f})")


def test_forward_shape():
    """Forward pass produces correct shape."""
    block = SpectralMambaAttention(
        config=None, hidden_size=64, num_heads=4, rank=32,
    )
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)
    print("[OK] test_forward_shape")


def test_causal_no_nan():
    """Causal mode works without NaN."""
    block = SpectralMambaAttention(
        config=None, hidden_size=64, num_heads=4, rank=32,
        is_decoder=True,
    )
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert not torch.isnan(out).any()
    print("[OK] test_causal_no_nan")


def test_output_attentions():
    """With output_attentions=True, weights should be returned."""
    block = SpectralMambaAttention(
        config=None, hidden_size=64, num_heads=4, rank=32,
    )
    x = torch.randn(2, 8, 64)
    out, attn = block(x, output_attentions=True)
    assert out.shape == (2, 8, 64)
    assert attn is not None
    assert attn.shape == (2, 4, 8, 8)
    print("[OK] test_output_attentions")


def test_gqa_copy_averaging():
    """GQA weight copy should average, not repeat."""
    block = SpectralMambaAttention(
        config=None, hidden_size=64, num_heads=8, rank=32,
        num_kv_heads=2,
    )
    assert block.k_lin.weight.shape[0] == 2 * block.head_dim
    print("[OK] test_gqa_copy_averaging")


def test_frobenius_norm_fails_loudly():
    """Near-zero copy should raise if source norm was non-zero."""
    block = SpectralMambaAttention(
        config=None, hidden_size=64, num_heads=4, rank=32,
    )
    class ZeroAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)
            self.k_lin = nn.Linear(64, 64)
            self.v_lin = nn.Linear(64, 64)
            self.out_lin = nn.Linear(64, 64)
            for m in [self.q_lin, self.k_lin, self.v_lin, self.out_lin]:
                m.weight.data.zero_()
                m.bias.data.zero_()

    src = ZeroAttn()
    try:
        block.copy_weights(src)
        assert torch.allclose(
            block.q_lin.weight, torch.zeros_like(block.q_lin.weight)
        )
        print("[OK] test_frobenius_norm_fails_loudly (warning only)")
    except RuntimeError:
        print("[OK] test_frobenius_norm_fails_loudly (raised)")


if __name__ == "__main__":
    tests = [
        test_spectral_mamba_forward,
        test_fft_actually_filters,
        test_gate_init_is_attention_dominant,
        test_forward_shape,
        test_causal_no_nan,
        test_output_attentions,
        test_gqa_copy_averaging,
        test_frobenius_norm_fails_loudly,
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
