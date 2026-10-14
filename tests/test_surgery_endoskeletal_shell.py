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
        # Give weights deterministic values
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

    # Instantiate CosformerAttention
    cosformer = CosformerAttention(hidden_size=64, num_heads=4)
    # Check weights before transfer (random / default)
    assert not torch.allclose(cosformer.q_proj.weight, torch.tensor(1.0))

    # Apply transfer
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
    # Forward pass works and matches input tensor dimensions
    if isinstance(out, tuple):
        assert out[0].shape == (B, T, D)
        loss = out[0].sum()
    else:
        assert out.shape == (B, T, D)
        loss = out.sum()

    # Differentiable backward pass
    loss.backward()
    assert x.grad is not None


def test_register_neuralops_with_huggingface():
    # Calling register_neuralops_with_huggingface() executes without error
    register_neuralops_with_huggingface()
    try:
        import transformers.modeling_utils as hf_utils
        if hasattr(hf_utils, "ALL_ATTENTION_FUNCTIONS"):
            assert "neuralops_cosformer" in hf_utils.ALL_ATTENTION_FUNCTIONS
    except ImportError:
        pass
