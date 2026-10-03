"""Part 6 Verification: Capability Declarations & Early Incompatibility Skipping (RCA-6)."""
import pytest
from neuralops.core.registry import (get_module_capabilities,
                                     is_module_declared_compatible)


def test_causal_only_incompatible_on_roberta():
    is_compat, reason = is_module_declared_compatible("adaptive_span", "roberta")
    assert not is_compat
    assert "causal_only" in reason

def test_causal_only_compatible_on_gpt2():
    is_compat, _ = is_module_declared_compatible("adaptive_span", "gpt2")
    assert is_compat

def test_bidirectional_compatible_on_all():
    is_compat_enc, _ = is_module_declared_compatible("adaptive_svd", "roberta")
    is_compat_dec, _ = is_module_declared_compatible("adaptive_svd", "gpt2")
    assert is_compat_enc
    assert is_compat_dec
