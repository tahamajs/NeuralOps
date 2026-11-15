#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Wave 4 Subagent W4-L1: Linear Family Test Suite.

Audits and verifies the complete set of Linear & SSM Attention Modules:
1. Module imports and backward-compatible entrypoints.
2. Instantiation, forward execution, finite outputs (no NaN/Inf), and shape preservation.
3. Strict causal masking invariance: mutating future tokens does not leak into past states.
4. Autograd backward stability and non-vanishing gradients.
5. Code hygiene: absence of deprecated torch.cuda.amp, time.time() on GPU, or silent except: pass.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig

from neuralops.layers.based import BasedAttention
from neuralops.layers.bimamba import BiMambaBlock
from neuralops.layers.hedgehog import HedgehogAttention
from neuralops.layers.lru import LRUAttention
from neuralops.layers.mamba2 import Mamba2Block
from neuralops.layers.recurrent_linear import RecurrentLinearAttention
from neuralops.layers.s5 import S5Block
from neuralops.layers.t2r import T2RAttention
from neuralops.layers.unlimiformer import UnlimiformerAttention

W4_L1_MODULE_NAMES = [
    "neuralops.layers.cosformer",
    "neuralops.layers.cosformer_linear",
    "neuralops.layers.gla",
    "neuralops.layers.gla_attention",
    "neuralops.layers.gdn",
    "neuralops.layers.retnet",
    "neuralops.layers.rwkv",
    "neuralops.layers.rwkv4",
    "neuralops.layers.rwkv6",
    "neuralops.layers.linear_attention",
    "neuralops.layers.hgrn",
    "neuralops.layers.hgrn2",
    "neuralops.layers.fms",
    "neuralops.layers.based",
    "neuralops.layers.recurrent_linear",
    "neuralops.layers.performer_linear",
    "neuralops.layers.linformer_linear",
    "neuralops.layers.t2r",
    "neuralops.layers.delta_net",
    "neuralops.layers.gated_deltanet",
    "neuralops.layers.hedgehog",
    "neuralops.layers.mamba",
    "neuralops.layers.mamba2",
    "neuralops.layers.ssm",
    "neuralops.layers.s4_attention",
    "neuralops.layers.s5",
    "neuralops.layers.lru",
    "neuralops.layers.unlimiformer",
    "neuralops.layers.kda",
    "neuralops.layers.s4",
    "neuralops.layers.state_space",
    "neuralops.layers.bimamba",
    "neuralops.layers.linear_rnn",
    "neuralops.layers.lrnn",
]

NEW_LINEAR_CLASSES = [
    BasedAttention,
    Mamba2Block,
    BiMambaBlock,
    S5Block,
    LRUAttention,
    HedgehogAttention,
    T2RAttention,
    RecurrentLinearAttention,
    UnlimiformerAttention,
]

OWNED_FILE_PATHS = [
    Path("neuralops/layers/cosformer.py"),
    Path("neuralops/layers/cosformer_linear.py"),
    Path("neuralops/layers/gla.py"),
    Path("neuralops/layers/gla_attention.py"),
    Path("neuralops/layers/gdn.py"),
    Path("neuralops/layers/retnet.py"),
    Path("neuralops/layers/rwkv.py"),
    Path("neuralops/layers/rwkv4.py"),
    Path("neuralops/layers/rwkv6.py"),
    Path("neuralops/layers/linear_attention.py"),
    Path("neuralops/layers/hgrn.py"),
    Path("neuralops/layers/hgrn2.py"),
    Path("neuralops/layers/fms.py"),
    Path("neuralops/layers/based.py"),
    Path("neuralops/layers/recurrent_linear.py"),
    Path("neuralops/layers/performer_linear.py"),
    Path("neuralops/layers/linformer_linear.py"),
    Path("neuralops/layers/t2r.py"),
    Path("neuralops/layers/delta_net.py"),
    Path("neuralops/layers/gated_deltanet.py"),
    Path("neuralops/layers/hedgehog.py"),
    Path("neuralops/layers/mamba.py"),
    Path("neuralops/layers/mamba2.py"),
    Path("neuralops/layers/ssm.py"),
    Path("neuralops/layers/s4_attention.py"),
    Path("neuralops/layers/s5.py"),
    Path("neuralops/layers/lru.py"),
    Path("neuralops/layers/unlimiformer.py"),
    Path("neuralops/layers/kda.py"),
    Path("neuralops/layers/s4.py"),
    Path("neuralops/layers/state_space.py"),
    Path("neuralops/layers/bimamba.py"),
    Path("neuralops/layers/linear_rnn.py"),
    Path("neuralops/layers/lrnn.py"),
]


@pytest.fixture(autouse=True)
def reset_rng():
    """Reset PyTorch RNG seed before each test."""
    torch.manual_seed(42)


@pytest.mark.parametrize("mod_name", W4_L1_MODULE_NAMES)
def test_w4_l1_module_import_and_all(mod_name: str) -> None:
    """Verify that every owned layer module can be imported cleanly and exports non-empty __all__."""
    mod = importlib.import_module(mod_name)
    assert mod is not None, f"Failed to import {mod_name}"
    all_symbols = getattr(mod, "__all__", None)
    assert all_symbols is not None and len(all_symbols) > 0, f"{mod_name} missing __all__ exports"


@pytest.mark.parametrize("layer_cls", NEW_LINEAR_CLASSES)
def test_new_linear_layers_forward_shape_and_finite(layer_cls: type[nn.Module]) -> None:
    """Verify forward execution, output shapes (B, N, D), and finite outputs (no NaN/Inf)."""
    B, N, D = 2, 64, 64
    H = 4
    cfg = BertConfig(hidden_size=D, num_attention_heads=H, is_decoder=True)
    cfg.is_causal = True

    try:
        mod = layer_cls(config=cfg, dim=D, num_heads=H)
    except TypeError:
        mod = layer_cls(config=cfg, d_model=D)

    mod.eval()
    x = torch.randn(B, N, D, dtype=torch.float32)

    with torch.no_grad():
        out = mod(x)
        if isinstance(out, tuple):
            out = out[0]

    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)} for {layer_cls.__name__}"
    assert out.shape == (B, N, D), f"Expected shape {(B, N, D)}, got {out.shape} for {layer_cls.__name__}"
    assert torch.isfinite(out).all(), f"NaN or Inf found in output of {layer_cls.__name__}"


@pytest.mark.parametrize(
    "layer_cls",
    [
        BasedAttention,
        Mamba2Block,
        S5Block,
        LRUAttention,
        HedgehogAttention,
        T2RAttention,
        RecurrentLinearAttention,
        UnlimiformerAttention,
    ],
)
def test_new_linear_layers_causal_masking(layer_cls: type[nn.Module]) -> None:
    """Verify that mutating future tokens (at step N-1) does not alter output at step 0."""
    B, N, D = 2, 32, 64
    H = 4
    cfg = BertConfig(hidden_size=D, num_attention_heads=H, is_decoder=True)
    cfg.is_causal = True

    try:
        mod = layer_cls(config=cfg, dim=D, num_heads=H)
    except TypeError:
        mod = layer_cls(config=cfg, d_model=D)

    mod.eval()
    x1 = torch.randn(B, N, D, dtype=torch.float32)
    x2 = x1.clone()
    x2[:, -1, :] = x2[:, -1, :] + 10.0  # Mutate final token

    with torch.no_grad():
        out1 = mod(x1)
        out2 = mod(x2)
        if isinstance(out1, tuple):
            out1 = out1[0]
        if isinstance(out2, tuple):
            out2 = out2[0]

    step0_diff = (out1[:, 0, :] - out2[:, 0, :]).abs().max().item()
    assert step0_diff < 1e-4, (
        f"Causal violation in {layer_cls.__name__}: step 0 altered by {step0_diff:.6e} when mutating token at step N-1"
    )


@pytest.mark.parametrize("layer_cls", NEW_LINEAR_CLASSES)
def test_new_linear_layers_autograd_backward(layer_cls: type[nn.Module]) -> None:
    """Verify backward gradient propagation and stability."""
    B, N, D = 2, 16, 32
    H = 2
    cfg = BertConfig(hidden_size=D, num_attention_heads=H, is_decoder=True)

    try:
        mod = layer_cls(config=cfg, dim=D, num_heads=H)
    except TypeError:
        mod = layer_cls(config=cfg, d_model=D)

    mod.train()
    x = torch.randn(B, N, D, dtype=torch.float32, requires_grad=True)

    out = mod(x)
    if isinstance(out, tuple):
        out = out[0]

    loss = out.sum()
    loss.backward()

    assert x.grad is not None, f"No gradient computed for input in {layer_cls.__name__}"
    assert torch.isfinite(x.grad).all(), f"NaN or Inf gradient in {layer_cls.__name__}"
    assert x.grad.norm().item() > 0.0, f"Vanishing gradient in {layer_cls.__name__}"


def test_w4_l1_code_hygiene() -> None:
    """Audit all 34 owned files for code hygiene: no torch.cuda.amp, no time.time(), no silent pass."""
    amp_regex = re.compile(r"torch\.cuda\.amp")
    time_regex = re.compile(r"time\.time\(\)")
    except_regex = re.compile(r"except\s*(Exception)?\s*:\s*pass")
    cuda0_regex = re.compile(r"cuda:0")

    violations: list[str] = []

    for path in OWNED_FILE_PATHS:
        assert path.exists(), f"File does not exist: {path}"
        content = path.read_text(encoding="utf-8")
        if amp_regex.search(content):
            violations.append(f"{path}: contains deprecated torch.cuda.amp")
        if time_regex.search(content):
            violations.append(f"{path}: contains time.time() for timing")
        if except_regex.search(content):
            violations.append(f"{path}: contains silent except: pass")
        if cuda0_regex.search(content):
            violations.append(f"{path}: contains hardcoded cuda:0")

    assert not violations, "Code hygiene violations found in owned files:\n" + "\n".join(violations)
