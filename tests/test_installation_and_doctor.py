"""
Comprehensive tests for post-installation environment integrity, verification,
and system diagnostics (doctor & installation checkers).
"""

import sys
import subprocess
from pathlib import Path
import pytest
import torch
import neuralops
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from neuralops.cli.doctor import cmd_doctor


def test_installation_smoke_and_version():
    """Verify neuralops top-level imports and semantic version."""
    assert hasattr(neuralops, "__version__")
    assert neuralops.__version__ == "1.3.0"
    if callable(getattr(neuralops, "get_version", None)):
        assert neuralops.get_version() == "1.3.0"


def test_doctor_diagnostics_runs_cleanly(capsys):
    """Verify that `neuralops doctor` runs to completion and exits with code 0."""
    code = cmd_doctor()
    assert code == 0
    captured = capsys.readouterr()
    assert "NeuralOps Production Environment & Hardware Diagnostics" in captured.out
    assert "Attention Mechanism Registry" in captured.out
    assert "Compression Engine Registry" in captured.out


def test_registry_exact_canonical_counts():
    """Strictly verify canonical framework counts."""
    assert len(ATTENTION_REGISTRY) == 131, f"Expected 131, got {len(ATTENTION_REGISTRY)}"
    assert len(COMPRESSOR_REGISTRY) == 72, f"Expected 72, got {len(COMPRESSOR_REGISTRY)}"


def test_flagship_fla_aliases_available():
    """Verify that flagship FLA mechanisms are globally instantiable via aliases."""
    for alias in ["pgdn", "gdn2", "delta_product"]:
        attn_cls = ATTENTION_REGISTRY.get(alias)
        assert attn_cls is not None, f"Alias '{alias}' should resolve in ATTENTION_REGISTRY"
        # Test basic instantiation
        instance = attn_cls(dim=128, num_heads=4)
        assert isinstance(instance, torch.nn.Module)


def test_triton_or_pytorch_fallback_execution():
    """Verify that core kernels execute without errors on current device (GPU or CPU/MPS fallback)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from neuralops.core.triton_kernels.elementwise import fused_swiglu
    x = torch.randn(2, 16, 64, device=device)
    gate = torch.randn(2, 16, 64, device=device)
    y = fused_swiglu(x, gate)
    assert y.shape == (2, 16, 64)
    assert not torch.isnan(y).any()



def test_public_api_completeness():
    """Verify all top-level public API functions exist and are callable."""
    expected_exports = [
        "replace_attention",
        "compress",
        "create_attention",
        "create_compressor",
        "list_attentions",
        "list_compressors",
        "get_version",
        "info",
    ]
    for fn_name in expected_exports:
        assert hasattr(neuralops, fn_name), f"Missing public export '{fn_name}'"
        assert callable(getattr(neuralops, fn_name)), f"Export '{fn_name}' must be callable"
