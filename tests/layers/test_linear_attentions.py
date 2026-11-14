#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Smoke and Regression Test Suite for Linear / Recurrent / SSM Attention Mechanisms.

Audits 35 linear family attention modules verifying:
1. Instantiation contract with B=2, H=8, N=512, D=64, dtype=torch.float32.
2. Forward execution, finite outputs (no NaN / Inf), and shape preservation (B, N, D_model).
3. Strict causal masking invariance: mutating future tokens does not contaminate past tokens.
4. Autograd backward stability and gradient preservation.
5. Code hygiene: absence of deprecated torch.cuda.amp, time.time() on GPU, or silent except: pass.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import BertConfig

from neuralops.core.registry import ATTENTION_REGISTRY, create_attention

LINEAR_ATTENTION_MODULES = [
    # Linear & Kernel Attention (neuralops/layers/linear/)
    "cosformer",
    "gla_attention",
    "performer_attention",
    "linformer_attention",
    "galerkin_attention",
    "nasc_attention",
    "nystrom_attention",
    "primal_attention",
    "rala_attention",
    "rpc_attention",
    "spectra_v2",
    "spectraformer_attention",
    "spectron",
    "tlinformer_attention",
    "tpa_attention",
    "triton_cosformer",
    "chela_slay",
    # Gated Delta & Recurrent Attention (neuralops/layers/gated/)
    "gated_delta_net",
    "gated_deltanet_block",
    "hgrn2_gated_layer",
    "hgrn2_gated_deltanet",
    "hgrn2_gated_outer",
    "decay_linear_attention",
    "hla_attention",
    "kda_attention",
    # SVD & Factorized State Attention (neuralops/layers/svd/)
    "factorised_matrix_state",
    "svd_linear_attention",
    "svd_linear_decay",
    # SSM & Mamba Recurrence (neuralops/layers/ssm/)
    "simple_ssm",
    "selective_ssm",
    "mamba_block",
    "mamba_svd_flash",
    "griffin_ssd",
    # Hybrid Linear & Kernel (neuralops/layers/hybrid/)
    "positive_kernel_linear_attention",
    "regla_attention",
]

LAYER_FILE_PATHS = [
    Path("neuralops/layers/linear/cosformer.py"),
    Path("neuralops/layers/linear/gla_attention.py"),
    Path("neuralops/layers/linear/performer_attention.py"),
    Path("neuralops/layers/linear/linformer_attention.py"),
    Path("neuralops/layers/linear/galerkin_attention.py"),
    Path("neuralops/layers/linear/nasc_attention.py"),
    Path("neuralops/layers/linear/nystrom_attention.py"),
    Path("neuralops/layers/linear/primal_attention.py"),
    Path("neuralops/layers/linear/rala_attention.py"),
    Path("neuralops/layers/linear/rpc_attention.py"),
    Path("neuralops/layers/linear/spectra_v2.py"),
    Path("neuralops/layers/linear/spectraformer_attention.py"),
    Path("neuralops/layers/linear/spectron.py"),
    Path("neuralops/layers/linear/tlinformer_attention.py"),
    Path("neuralops/layers/linear/tpa_attention.py"),
    Path("neuralops/layers/linear/triton_cosformer.py"),
    Path("neuralops/layers/linear/chela_slay.py"),
    Path("neuralops/layers/gated/gated_delta_net.py"),
    Path("neuralops/layers/gated/gated_deltanet_block.py"),
    Path("neuralops/layers/gated/hgrn2_gated_layer.py"),
    Path("neuralops/layers/gated/hgrn2_gated_deltanet.py"),
    Path("neuralops/layers/gated/hgrn2_gated_outer.py"),
    Path("neuralops/layers/gated/decay_linear_attention.py"),
    Path("neuralops/layers/gated/hla_attention.py"),
    Path("neuralops/layers/gated/kda_attention.py"),
    Path("neuralops/layers/svd/factorised_matrix_state.py"),
    Path("neuralops/layers/svd/svd_linear_attention.py"),
    Path("neuralops/layers/svd/svd_linear_decay.py"),
    Path("neuralops/layers/ssm/simple_ssm.py"),
    Path("neuralops/layers/ssm/selective_ssm.py"),
    Path("neuralops/layers/ssm/mamba_block.py"),
    Path("neuralops/layers/ssm/mamba_svd_flash.py"),
    Path("neuralops/layers/ssm/griffin_ssd.py"),
    Path("neuralops/layers/hybrid/positive_kernel_linear_attention.py"),
    Path("neuralops/layers/hybrid/regla_attention.py"),
]


@pytest.fixture(autouse=True)
def reset_seed():
    """Reset RNG state to ensure reproducible evaluations."""
    torch.manual_seed(42)


@pytest.mark.parametrize("mod_name", LINEAR_ATTENTION_MODULES)
def test_linear_attention_instantiation(mod_name: str) -> None:
    """Protocol 1: Verify instantiation with B=2, H=8, N=512, D=64, dtype=torch.float32."""
    H, D = 8, 64
    hidden_size = H * D  # 512
    cfg = BertConfig(hidden_size=hidden_size, num_attention_heads=H, is_decoder=True)
    cfg.is_causal = True

    mod = create_attention(mod_name, config=cfg, dim=hidden_size, num_heads=H)
    assert mod is not None, f"Failed to instantiate {mod_name}"
    assert isinstance(mod, torch.nn.Module), f"{mod_name} must be a torch.nn.Module"


@pytest.mark.parametrize("mod_name", LINEAR_ATTENTION_MODULES)
def test_linear_attention_forward(mod_name: str) -> None:
    """Protocol 2: Run forward with fixed seed, check no NaN/Inf, shape (B, N, D_model)."""
    B, H, N, D = 2, 8, 512, 64
    hidden_size = H * D
    cfg = BertConfig(hidden_size=hidden_size, num_attention_heads=H, is_decoder=True)
    cfg.is_causal = True

    mod = create_attention(mod_name, config=cfg, dim=hidden_size, num_heads=H)
    mod.eval()

    torch.manual_seed(42)
    x = torch.randn(B, N, hidden_size, dtype=torch.float32)

    with torch.no_grad():
        out = mod(x)
        if isinstance(out, tuple):
            out = out[0]

    assert isinstance(out, torch.Tensor), f"Expected Tensor output, got {type(out)}"
    assert out.shape == (B, N, hidden_size), f"Expected shape {(B, N, hidden_size)}, got {out.shape}"
    assert torch.isfinite(out).all(), f"Found NaN or Inf in {mod_name} forward output"


@pytest.mark.parametrize("mod_name", LINEAR_ATTENTION_MODULES)
def test_linear_attention_causal_masking(mod_name: str) -> None:
    """Protocol 3: Check causal masking correctness.

    Mutating the token at step N-1 must NOT alter the output at step 0 in causal mode.
    """
    B, H, N, D = 2, 8, 256, 64
    hidden_size = H * D
    cfg = BertConfig(hidden_size=hidden_size, num_attention_heads=H, is_decoder=True)
    cfg.is_causal = True

    mod = create_attention(mod_name, config=cfg, dim=hidden_size, num_heads=H)
    mod.eval()

    torch.manual_seed(42)
    x1 = torch.randn(B, N, hidden_size, dtype=torch.float32)
    x2 = x1.clone()
    x2[:, -1, :] = x2[:, -1, :] + 10.0  # Mutate final token

    with torch.no_grad():
        out1 = mod(x1)
        if isinstance(out1, tuple):
            out1 = out1[0]
        out2 = mod(x2)
        if isinstance(out2, tuple):
            out2 = out2[0]

    step0_diff = (out1[:, 0, :] - out2[:, 0, :]).abs().max().item()
    assert step0_diff < 1e-5, (
        f"Causal violation in {mod_name}: step 0 changed by {step0_diff:.6e} "
        "when mutating token at step N-1."
    )


@pytest.mark.parametrize("mod_name", [
    "cosformer", "gla_attention", "performer_attention", "linformer_attention",
    "gated_delta_net", "hgrn2_gated_layer", "factorised_matrix_state", "svd_linear_attention",
    "simple_ssm", "mamba_block"
])
def test_linear_attention_autograd_backward(mod_name: str) -> None:
    """Verify autograd backward stability and gradient flow."""
    B, H, N, D = 2, 8, 64, 64
    hidden_size = H * D
    cfg = BertConfig(hidden_size=hidden_size, num_attention_heads=H, is_decoder=True)
    cfg.is_causal = True

    mod = create_attention(mod_name, config=cfg, dim=hidden_size, num_heads=H)
    mod.train()

    torch.manual_seed(42)
    x = torch.randn(B, N, hidden_size, dtype=torch.float32, requires_grad=True)

    out = mod(x)
    if isinstance(out, tuple):
        out = out[0]

    loss = out.sum()
    loss.backward()

    assert x.grad is not None, f"Gradient not received for {mod_name}"
    assert torch.isfinite(x.grad).all(), f"NaN or Inf detected in gradients for {mod_name}"
    assert x.grad.norm().item() > 0.0, f"Vanishing gradient (0 norm) for {mod_name}"


def test_code_hygiene_rules() -> None:
    """Protocol 4: Ensure no deprecated torch.cuda.amp, time.time() on GPU, or bare except: pass."""
    amp_regex = re.compile(r"torch\.cuda\.amp")
    time_regex = re.compile(r"time\.time\(\)")
    except_regex = re.compile(r"except\s*(Exception)?\s*:\s*pass")

    violations: list[str] = []

    for path in LAYER_FILE_PATHS:
        assert path.exists(), f"Layer file not found: {path}"
        content = path.read_text(encoding="utf-8")
        if amp_regex.search(content):
            violations.append(f"{path}: contains deprecated torch.cuda.amp")
        if time_regex.search(content):
            violations.append(f"{path}: contains time.time() for timing")
        if except_regex.search(content):
            violations.append(f"{path}: contains silent except: pass")

    assert not violations, "Code hygiene violations detected:\n" + "\n".join(violations)
