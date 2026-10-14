"""Unit tests for Phase 4 ShadowTwin distillation, homotopy alpha, and finalization."""

import torch
import torch.nn as nn
import pytest

from neuralops.surgery.scanner import UniversalArchitectureScanner
from neuralops.surgery.endoskeletal_shell import UniversalAttentionBridge
from neuralops.surgery.shadow_twin import (
    enable_shadow_twin,
    patch_bridge_for_shadow_twin,
    set_surgery_alpha,
    set_surgery_stage,
    collect_surgery_distillation_loss,
    finalize_surgery,
    compute_operator_cloning_loss,
    compute_linear_cka_loss,
)
from neuralops.attentions.linear.cosformer import CosformerAttention


class MockSelfAttentionBlock(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, **kwargs):
        return x * 1.5


def test_shadow_twin_distillation_and_finalization():
    host_block = MockSelfAttentionBlock(d=64)
    specs = UniversalArchitectureScanner.scan(host_block)
    spec = specs[0]

    cosformer = CosformerAttention(hidden_size=64, num_heads=4)
    bridge = UniversalAttentionBridge(
        orig_module=host_block,
        spec=spec,
        new_attention=cosformer,
        inherit_weights=True,
    )

    # Enable shadow twin & patch forward
    enable_shadow_twin(bridge, stage="stage2_block_cloning")
    patch_bridge_for_shadow_twin(bridge)
    assert bridge.teacher_branch is not None

    # Forward with shadow twin active
    B, T, D = 2, 8, 64
    x = torch.randn(B, T, D, requires_grad=True)

    out = bridge(x)
    loss = collect_surgery_distillation_loss(bridge)
    assert loss is not None
    assert loss.item() > 0.0

    # Backprop through distillation loss
    loss.backward()
    assert x.grad is not None

    # Test homotopy alpha change
    set_surgery_alpha(bridge, alpha=0.5)
    assert bridge.alpha == 0.5

    # Test finalization
    finalize_surgery(bridge)
    assert bridge.teacher_branch is None
    assert bridge.alpha == 1.0
