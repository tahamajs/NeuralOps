#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Sampled Attention Mechanisms Smoke Test Suite (Wave 1 Audit).

Verifies forward execution, output shapes, numerical stability (NaN/Inf checks),
and autograd backward stability across the 20 representative attention mechanisms:
1. full_attention
2. cosformer
3. gla
4. mha
5. linear_attention
6. sliding_window
7. flash_attention
8. gqa
9. sparse_attention
10. nystrom
11. performer
12. linformer
13. rwkv
14. mamba
15. mla
16. retnet
17. gated_deltanet
18. seesaw
19. softmax_attention
20. favor_plus
"""

from __future__ import annotations

import pytest
import torch
from transformers import BertConfig

# Import layer modules to guarantee registry population
import neuralops.layers.dense.full_attention  # noqa: F401
import neuralops.layers.fast_kernels.flash_attention3  # noqa: F401
import neuralops.layers.gated.gated_delta_net  # noqa: F401
import neuralops.layers.linear.cosformer  # noqa: F401
import neuralops.layers.linear.gla_attention  # noqa: F401
import neuralops.layers.linear.linformer_attention  # noqa: F401
import neuralops.layers.linear.nystrom_attention  # noqa: F401
import neuralops.layers.linear.performer_attention  # noqa: F401
import neuralops.layers.linear.retnet_attention  # noqa: F401
import neuralops.layers.modern.dynamic_conv_attention  # noqa: F401
import neuralops.layers.modern.gqa_attention  # noqa: F401
import neuralops.layers.modern.mla_attention  # noqa: F401
import neuralops.layers.sparse.adaptive_sparse_attention  # noqa: F401
import neuralops.layers.ssm.hymba_transx  # noqa: F401
import neuralops.layers.ssm.mamba_block  # noqa: F401
import neuralops.layers.ssm.rwkv_attention  # noqa: F401
import neuralops.layers.svd.salt_attention  # noqa: F401
import neuralops.layers.svd.svd_linear_attention  # noqa: F401
from neuralops.core.registry import ATTENTION_REGISTRY

SAMPLED_ATTENTIONS: list[str] = [
    "full_attention",
    "cosformer",
    "gla",
    "mha",
    "linear_attention",
    "sliding_window",
    "flash_attention",
    "gqa",
    "sparse_attention",
    "nystrom",
    "performer",
    "linformer",
    "rwkv",
    "mamba",
    "mla",
    "retnet",
    "gated_deltanet",
    "seesaw",
    "softmax_attention",
    "favor_plus",
]


@pytest.fixture(autouse=True)
def set_seed() -> None:
    """Set deterministic RNG seed for test repeatability."""
    torch.manual_seed(42)


@pytest.mark.parametrize("attn_name", SAMPLED_ATTENTIONS)
def test_sampled_attention_forward_and_numerical_safety(attn_name: str) -> None:
    """Test forward execution and numerical finiteness on standard input dimensions.

    Standard Dimensions:
        Batch (B): 2
        Heads (H): 4
        Sequence Length (N): 64
        Head Dim (D): 32
        Hidden Dim: H * D = 128

    Args:
        attn_name: Registered identifier of the attention mechanism.
    """
    factory = ATTENTION_REGISTRY.get(attn_name)
    assert factory is not None, f"Attention mechanism '{attn_name}' not found in ATTENTION_REGISTRY"

    b, h, n, d = 2, 4, 64, 32
    hidden_dim = h * d

    config = BertConfig(
        hidden_size=hidden_dim,
        num_attention_heads=h,
        num_heads=h,
        head_dim=d,
        d_model=hidden_dim,
    )

    try:
        mod = factory(config)
    except TypeError:
        mod = factory(config, rank=8)

    assert isinstance(mod, torch.nn.Module), f"{attn_name} factory did not return nn.Module"

    x = torch.randn(b, n, hidden_dim, dtype=torch.float32, requires_grad=True)
    out = mod(x)

    if isinstance(out, tuple):
        out = out[0]

    assert isinstance(out, torch.Tensor), f"Expected Tensor output, got {type(out)}"
    assert out.shape == (b, n, hidden_dim), f"Shape mismatch: expected {(b, n, hidden_dim)}, got {out.shape}"
    assert torch.isfinite(out).all(), f"Non-finite values (NaN or Inf) detected in {attn_name} forward pass"

    # Backward pass smoke check
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, f"Gradients were not computed for {attn_name}"
    assert torch.isfinite(x.grad).all(), f"Non-finite gradients detected in {attn_name} backward pass"
