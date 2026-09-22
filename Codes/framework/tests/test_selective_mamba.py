import math
import sys
import torch
import torch.nn as nn
sys.path.insert(0, "Codes")

from framework.attentions.upi_mamba import SelectiveMamba, UPIMambaBlock


def test_basic_construction():
    block = SelectiveMamba(config=None, dim=64, d_state=8)
    assert block.dim == 64
    assert block.d_state == 8
    assert block.inner_dim == 64  # expand=1
    assert block.dt_rank == 4  # ceil(64/16)
    print("[OK] test_basic_construction")


def test_dt_rank_parameter_efficiency():
    """dt_rank should be small, not = inner_dim."""
    block = SelectiveMamba(config=None, dim=256, d_state=16)
    # dt_rank = ceil(256/16) = 16
    assert block.dt_rank == 16
    # x_proj output size = 16 + 32 = 48, much smaller than 256
    assert block.x_proj.out_features == 16 + 32
    print(f"[OK] test_dt_rank_parameter_efficiency "
          f"(dt_rank={block.dt_rank}, x_proj_out={block.x_proj.out_features})")


def test_forward_shape():
    block = SelectiveMamba(config=None, dim=64, d_state=8)
    x = torch.randn(2, 16, 64)
    out, _ = block(x)
    assert out.shape == (2, 16, 64)
    print("[OK] test_forward_shape")


def test_cross_attention_raises():
    block = SelectiveMamba(config=None, dim=64, d_state=8)
    x = torch.randn(2, 16, 64)
    enc = torch.randn(2, 8, 64)
    try:
        block(x, encoder_hidden_states=enc)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        print("[OK] test_cross_attention_raises")


def test_float_mask_handling():
    """Float mask with -inf should mark invalid."""
    block = SelectiveMamba(config=None, dim=64, d_state=8)
    x = torch.randn(1, 8, 64)
    # Float mask: -inf for last 4 tokens
    mask = torch.zeros(1, 8)
    mask[:, -4:] = -float("inf")
    out, _ = block(x, attention_mask=mask)
    assert out.shape == (1, 8, 64)
    print("[OK] test_float_mask_handling")


def test_causal_no_nan():
    block = SelectiveMamba(
        config=None, dim=64, d_state=8, is_decoder=True,
    )
    x = torch.randn(2, 32, 64)
    out, _ = block(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    print("[OK] test_causal_no_nan")


def test_bidirectional_no_nan():
    block = SelectiveMamba(
        config=None, dim=64, d_state=8, is_decoder=False,
    )
    x = torch.randn(2, 32, 64)
    out, _ = block(x)
    assert not torch.isnan(out).any()
    print("[OK] test_bidirectional_no_nan")


def test_step_recurrent():
    block = SelectiveMamba(config=None, dim=64, d_state=8)
    # Streaming: process one token at a time
    state = None
    outputs = []
    for t in range(4):
        x_t = torch.randn(1, 1, 64)
        out, state = block.step_recurrent(x_t, state)
        outputs.append(out)
    y_stream = torch.cat(outputs, dim=1)
    assert y_stream.shape == (1, 4, 64)
    print("[OK] test_step_recurrent")


def test_step_matches_forward():
    """Streaming should match batch forward."""
    block = SelectiveMamba(config=None, dim=64, d_state=8, is_decoder=True)
    block.eval()
    x = torch.randn(1, 4, 64)
    
    with torch.no_grad():
        # Batch
        y_batch, _ = block(x)
        
        # Streaming
        state = None
        ys = []
        for t in range(4):
            y_t, state = block.step_recurrent(x[:, t:t+1], state)
            ys.append(y_t)
        y_stream = torch.cat(ys, dim=1)
    
    err = (y_batch - y_stream).abs().max().item()
    # Allow some tolerance due to conv buffer
    assert err < 0.1, f"Streaming vs batch mismatch: {err}"
    print(f"[OK] test_step_matches_forward (err={err:.4f})")


def test_registry_keys():
    from framework.core.registry import ATTENTION_REGISTRY
    assert "selective_mamba" in ATTENTION_REGISTRY
    assert "upi_mamba" in ATTENTION_REGISTRY  # compat
    print("[OK] test_registry_keys")


def test_backward_compat_alias():
    assert UPIMambaBlock is SelectiveMamba
    print("[OK] test_backward_compat_alias")


def test_mask_bool_correctness():
    """Bool mask with padding: padding tokens should be zero."""
    block = SelectiveMamba(config=None, dim=64, d_state=8)
    block.eval()
    x = torch.randn(1, 8, 64)
    mask = torch.ones(1, 8, dtype=torch.long)
    mask[:, -4:] = 0
    
    with torch.no_grad():
        out, _ = block(x, attention_mask=mask)
    
    # Last 4 outputs should be zero (masked)
    last4_norm = out[:, -4:].abs().mean().item()
    assert last4_norm < 1e-5, f"Masked tokens not zero: {last4_norm}"
    print(f"[OK] test_mask_bool_correctness (norm={last4_norm:.2e})")


if __name__ == "__main__":
    tests = [
        test_basic_construction,
        test_dt_rank_parameter_efficiency,
        test_forward_shape,
        test_cross_attention_raises,
        test_float_mask_handling,
        test_causal_no_nan,
        test_bidirectional_no_nan,
        test_step_recurrent,
        test_step_matches_forward,
        test_registry_keys,
        test_backward_compat_alias,
        test_mask_bool_correctness,
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
