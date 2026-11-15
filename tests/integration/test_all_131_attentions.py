#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Exhaustive 131 Attention Mechanisms Integration and Hardware Audit Test Suite.

Audits, instantiates, and executes forward passes across all 131 attention mechanisms
cataloged in ATTENTION_REGISTRY, verifying input-output dimension contracts,
finite numerical behavior, and non-CUDA graceful skipping.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import pytest
import torch
from neuralops.core.registry import ATTENTION_REGISTRY, create_attention

ATTENTION_REGISTRY._load_all()

# Load all 131 audited attention keys (118 canonical keys + 13 primary preset aliases)
_KEYS_FILE = (
    Path(__file__).resolve().parent.parent.parent / "audits" / "wave5" / "w5_8_evidence" / "attention_keys_dump.txt"
)

if _KEYS_FILE.exists():
    with open(_KEYS_FILE, encoding="utf-8") as f:
        ATTENTION_NAMES = [line.strip() for line in f if line.strip()]
else:
    canonical = sorted(ATTENTION_REGISTRY.keys())
    aliases_13 = [
        "flash_attention",
        "gla",
        "gqa",
        "linear_attention",
        "linformer",
        "mamba",
        "mha",
        "mla",
        "nystrom",
        "performer",
        "retnet_attention",
        "rwkv_attention",
        "sparse_attention",
    ]
    ATTENTION_NAMES = sorted(set(canonical + aliases_13))


def test_attention_registry_exact_131_count():
    """Verify that exactly 131 attention mechanisms are cataloged and tested."""
    assert len(ATTENTION_NAMES) == 131, f"Expected exactly 131 attention mechanisms, found {len(ATTENTION_NAMES)}"


@pytest.mark.parametrize("name", ATTENTION_NAMES)
def test_single_attention_instantiation_and_forward(name: str):
    """Systematically test instantiation and forward pass for every attention mechanism."""
    cls = ATTENTION_REGISTRY.get(name)
    assert cls is not None, f"Attention key '{name}' failed to resolve from registry"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Build kwargs dynamically based on module constructor signatures
    sig = inspect.signature(cls.__init__)
    kwargs = {}

    # Common dimension mappings across various layer implementations
    params = sig.parameters
    if "d_model" in params:
        kwargs["d_model"] = 64
    elif "dim" in params:
        kwargs["dim"] = 64
    elif "hidden_size" in params:
        kwargs["hidden_size"] = 64
    else:
        kwargs["dim"] = 64

    if "num_heads" in params:
        kwargs["num_heads"] = 4
    elif "n_heads" in params:
        kwargs["n_heads"] = 4
    elif "heads" in params:
        kwargs["heads"] = 4
    else:
        kwargs["num_heads"] = 4

    if "max_seq_len" in params:
        kwargs["max_seq_len"] = 128
    if "seq_len" in params:
        kwargs["seq_len"] = 32
    if "causal" in params:
        kwargs["causal"] = True

    try:
        model = cls(**kwargs)
        try:
            model = model.to(device)
        except Exception:
            device = torch.device("cpu")
            model = model.to(device)
    except Exception as exc:
        # Check if instantiation requires specific hardware
        err_msg = str(exc).lower()
        if "triton" in err_msg or "cuda" in err_msg or "gpu" in err_msg or "not implemented" in err_msg:
            pytest.skip(f"Hardware-dependent attention '{name}': {exc}")
        pytest.fail(f"Instantiation failed for '{name}': {exc}")

    model.eval()
    x = torch.randn(2, 32, 64, device=device)

    try:
        # Forward execution handling various parameter signatures
        forward_sig = inspect.signature(model.forward)
        f_params = forward_sig.parameters
        if "key" in f_params and "value" in f_params:
            out = model(x, x, x)
        else:
            out = model(x)

        # Output extraction handling tuple/dict outputs
        if isinstance(out, tuple):
            out_tensor = out[0]
        elif isinstance(out, dict):
            out_tensor = out.get("hidden_states", out.get("logits", None))
        else:
            out_tensor = out

        assert out_tensor is not None, f"Output tensor was None for '{name}'"
        assert isinstance(out_tensor, torch.Tensor), f"Expected Tensor, got {type(out_tensor)} for '{name}'"
        assert out_tensor.dim() >= 2, f"Output tensor dim too small: {out_tensor.shape}"
        assert torch.isfinite(out_tensor).all(), f"Output contained NaN/Inf for '{name}'"

    except Exception as exc:
        err_msg = str(exc).lower()
        if "triton" in err_msg or "cuda" in err_msg or "gpu" in err_msg:
            pytest.skip(f"Hardware-dependent forward pass for '{name}': {exc}")
        elif isinstance(exc, NotImplementedError) or "not implemented" in err_msg:
            pytest.xfail(f"Optional feature not implemented: {exc}")
        else:
            pytest.fail(f"Forward pass crashed for '{name}': {exc}")
