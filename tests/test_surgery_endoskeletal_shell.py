#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for Phase 2 UniversalAttentionBridge, warm-start weight transfer, and HF registration."""

import torch
import torch.nn as nn
import pytest

from neuralops.surgery.scanner import UniversalArchitectureScanner
from neuralops.surgery.endoskeletal_shell import (
    UniversalAttentionBridge,
    transfer_pretrained_qkv_weights,
    register_neuralops_with_huggingface,
)
from neuralops.attentions.linear.cosformer import CosformerAttention


class MockLlamaBlock(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        nn.init.constant_(self.q_proj.weight, 1.0)
        nn.init.constant_(self.k_proj.weight, 2.0)
        nn.init.constant_(self.v_proj.weight, 3.0)
        nn.init.constant_(self.o_proj.weight, 4.0)

    def forward(self, x, **kwargs):
        return x


def test_warm_start_weight_transfer():
    host_block = MockLlamaBlock(d=64, heads=4)
    specs = UniversalArchitectureScanner.scan(host_block)
    assert len(specs) == 1
    spec = specs[0]

    cosformer = CosformerAttention(hidden_size=64, num_heads=4)
    assert not torch.allclose(cosformer.q_proj.weight, torch.tensor(1.0))

    transfer_pretrained_qkv_weights(spec, cosformer)
    assert torch.allclose(cosformer.q_proj.weight, torch.tensor(1.0))
    assert torch.allclose(cosformer.k_proj.weight, torch.tensor(2.0))
    assert torch.allclose(cosformer.v_proj.weight, torch.tensor(3.0))
    assert torch.allclose(cosformer.out_proj.weight, torch.tensor(4.0))


def test_universal_attention_bridge_execution():
    host_block = MockLlamaBlock(d=64, heads=4)
    specs = UniversalArchitectureScanner.scan(host_block)
    spec = specs[0]

    cosformer = CosformerAttention(hidden_size=64, num_heads=4)
    bridge = UniversalAttentionBridge(
        orig_module=host_block,
        spec=spec,
        new_attention=cosformer,
        inherit_weights=True,
        attach_sidecar=True,
    )

    B, T, D = 2, 8, 64
    x = torch.randn(B, T, D, requires_grad=True)
    out = bridge(x)
    if isinstance(out, tuple):
        assert out[0].shape == (B, T, D)
        loss = out[0].sum()
    else:
        assert out.shape == (B, T, D)
        loss = out.sum()

    loss.backward()
    assert x.grad is not None


def test_register_neuralops_with_huggingface():
    register_neuralops_with_huggingface()
    try:
        import transformers.modeling_utils as hf_utils
        if hasattr(hf_utils, "ALL_ATTENTION_FUNCTIONS"):
            assert "neuralops_cosformer" in hf_utils.ALL_ATTENTION_FUNCTIONS
    except ImportError:
        pass


def test_transfer_pretrained_qkv_weights_isometric_expansion_and_contraction():
    """Verify padded isometric zero-extension when target is larger and logged warning on contraction."""
    host_block = MockLlamaBlock(d=64, heads=4)
    specs = UniversalArchitectureScanner.scan(host_block)
    assert len(specs) == 1
    spec = specs[0]

    # Target module with larger projection (d=128 > d=64)
    larger_mod = nn.Module()
    larger_mod.q_proj = nn.Linear(128, 128, bias=False)
    larger_mod.k_proj = nn.Linear(128, 128, bias=False)
    larger_mod.v_proj = nn.Linear(128, 128, bias=False)
    larger_mod.out_proj = nn.Linear(128, 128, bias=False)

    # In strict mode, mismatch raises ValueError
    with pytest.raises(ValueError, match="shape mismatch"):
        transfer_pretrained_qkv_weights(spec, larger_mod, strict=True)

    # In non-strict mode, padded isometric mapping embeds source and zeroes remaining
    transfer_pretrained_qkv_weights(spec, larger_mod, strict=False)
    assert torch.allclose(larger_mod.q_proj.weight[:64, :64], torch.tensor(1.0))
    assert torch.all(larger_mod.q_proj.weight[64:, 64:] == 0.0)

    # Target module with smaller projection (d=32 < d=64)
    smaller_mod = nn.Module()
    smaller_mod.q_proj = nn.Linear(32, 32, bias=False)
    smaller_mod.k_proj = nn.Linear(32, 32, bias=False)
    smaller_mod.v_proj = nn.Linear(32, 32, bias=False)
    smaller_mod.out_proj = nn.Linear(32, 32, bias=False)

    transfer_pretrained_qkv_weights(spec, smaller_mod, strict=False)
    assert torch.allclose(smaller_mod.q_proj.weight, torch.tensor(1.0))


def test_universal_attention_bridge_preserves_lora_adapters():
    """Verify UniversalAttentionBridge unwraps base layer and preserves adapter references."""
    host_block = MockLlamaBlock(d=64, heads=4)
    # Wrap with mock LoRA adapter attributes
    host_block.lora_A = nn.Parameter(torch.randn(8, 64))
    host_block.lora_B = nn.Parameter(torch.randn(64, 8))
    host_block.scaling = 2.0

    specs = UniversalArchitectureScanner.scan(host_block)
    spec = specs[0]

    cosformer = CosformerAttention(hidden_size=64, num_heads=4)
    bridge = UniversalAttentionBridge(
        orig_module=host_block,
        spec=spec,
        new_attention=cosformer,
        inherit_weights=False,
    )

    assert bridge.has_lora_adapters is True
    adapters = bridge.get_lora_adapters()
    assert "lora_A" in adapters
    assert "lora_B" in adapters
    assert bridge.base_module is host_block


def test_universal_attention_bridge_keyword_dispatch():
    """Verify forward call using keyword hidden_states does not trigger argument collisions."""
    host_block = MockLlamaBlock(d=64, heads=4)
    specs = UniversalArchitectureScanner.scan(host_block)
    spec = specs[0]

    cosformer = CosformerAttention(hidden_size=64, num_heads=4)
    bridge = UniversalAttentionBridge(
        orig_module=host_block,
        spec=spec,
        new_attention=cosformer,
        inherit_weights=False,
    )

    B, T, D = 2, 8, 64
    x = torch.randn(B, T, D)
    out = bridge(hidden_states=x)
    core = out[0] if isinstance(out, tuple) else out
    assert core.shape == (B, T, D)


def test_transfer_pretrained_qkv_weights_fused_target():
    """Verify weight transfer succeeds when target module uses a single fused qkv linear layer."""
    host_block = MockLlamaBlock(d=64, heads=4)
    specs = UniversalArchitectureScanner.scan(host_block)
    spec = specs[0]

    fused_mod = nn.Module()
    fused_mod.qkv = nn.Linear(64, 3 * 64, bias=False)
    fused_mod.out_proj = nn.Linear(64, 64, bias=False)

    transfer_pretrained_qkv_weights(spec, fused_mod, strict=True)
    # W_q is 1.0, W_k is 2.0, W_v is 3.0
    assert torch.allclose(fused_mod.qkv.weight[:64, :], torch.tensor(1.0))
    assert torch.allclose(fused_mod.qkv.weight[64:128, :], torch.tensor(2.0))
    assert torch.allclose(fused_mod.qkv.weight[128:, :], torch.tensor(3.0))
    assert torch.allclose(fused_mod.out_proj.weight, torch.tensor(4.0))
