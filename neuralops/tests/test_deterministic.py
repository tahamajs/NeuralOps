"""Tests for deterministic attention mode."""
from neuralops.core.attention_utils import enable_deterministic_attention, is_deterministic_attention


def test_deterministic_attention_toggle():
    assert not is_deterministic_attention()
    prev = enable_deterministic_attention(True)
    assert not prev
    assert is_deterministic_attention()
    prev2 = enable_deterministic_attention(False)
    assert prev2
    assert not is_deterministic_attention()
