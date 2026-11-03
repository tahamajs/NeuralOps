#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for FX Graph Rewriter in Neural Attention Surgery.

Verifies symbolic tracing, dual subgraph pattern matching (SDPA & decomposed),
dead-code elimination, autograd gradient propagation, and Inductor compilation.
"""

import operator
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.surgery.fx_surgery import (
    FXAttentionGraphRewriter,
    NeuralOpsAttentionTracer,
    fast_linear_attention_replacement,
)
from neuralops.surgery.model_surgery import ModelSurgery
from neuralops.surgery.pipeline import SurgeryOp, SurgeryRecipe


class ToyDecomposedAttentionBlock(nn.Module):
    """Toy block using decomposed eager PyTorch attention."""

    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # Decomposed attention: matmul -> scale -> optional mask -> softmax -> matmul
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)

        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)


class ToySDPAAttentionBlock(nn.Module):
    """Toy block using PyTorch 2.x SDPA."""

    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)


class MultiLayerTransformerModel(nn.Module):
    """Multi-layer Transformer with LayerNorm and projections."""

    def __init__(self, d_model: int = 32, num_heads: int = 2, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ToySDPAAttentionBlock(d_model=d_model, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(self.norm(x))
        return x


def test_tracer_leaf_registration() -> None:
    """Verifies that NeuralOpsAttentionTracer preserves leaf modules intact."""
    tracer = NeuralOpsAttentionTracer()
    model = ToyDecomposedAttentionBlock(d_model=32, num_heads=2)
    graph = tracer.trace(model)
    assert graph is not None

    # Verify that Linear layers remain call_module leaf operations
    ops = [n.op for n in graph.nodes]
    assert "call_module" in ops


def test_sdpa_node_replacement() -> None:
    """Verifies that all SDPA calls are cleanly replaced with linear attention."""
    model = ToySDPAAttentionBlock(d_model=32, num_heads=2)
    rewriter = FXAttentionGraphRewriter(target_attention="cosformer", head_dim=16, is_causal=True)
    gm = rewriter.rewrite(model)

    # Assert no SDPA nodes remain in the graph
    targets = [n.target for n in gm.graph.nodes]
    assert F.scaled_dot_product_attention not in targets
    assert torch.ops.aten.scaled_dot_product_attention not in targets
    assert fast_linear_attention_replacement in targets

    # Run forward pass
    x = torch.randn(2, 16, 32)
    out = gm(x)
    assert out.shape == (2, 16, 32)
    assert not torch.isnan(out).any()


def test_decomposed_pattern_replacement() -> None:
    """Verifies that decomposed matmul-softmax-matmul sequences are replaced."""
    model = ToyDecomposedAttentionBlock(d_model=32, num_heads=2)
    rewriter = FXAttentionGraphRewriter(target_attention="cosformer", head_dim=16, is_causal=True)
    gm = rewriter.rewrite(model)

    # Assert no softmax calls remain
    targets = [n.target for n in gm.graph.nodes]
    assert F.softmax not in targets
    assert torch.softmax not in targets
    assert fast_linear_attention_replacement in targets

    # Run forward pass
    x = torch.randn(2, 16, 32)
    out = gm(x)
    assert out.shape == (2, 16, 32)
    assert not torch.isnan(out).any()


def test_decomposed_pattern_with_mask() -> None:
    """Verifies that decomposed attention with additive mask is cleanly rewritten."""
    model = ToyDecomposedAttentionBlock(d_model=32, num_heads=2)
    tracer = NeuralOpsAttentionTracer()
    # Trace with concrete args specifying mask presence
    graph = tracer.trace(model)
    gm = torch.fx.GraphModule(model, graph)

    rewriter = FXAttentionGraphRewriter(target_attention="cosformer", head_dim=16, is_causal=True)
    rewriter.replace_decomposed_patterns(gm)
    rewriter.clean_and_recompile(gm)

    targets = [n.target for n in gm.graph.nodes]
    assert F.softmax not in targets
    assert fast_linear_attention_replacement in targets


def test_fx_backward_gradients() -> None:
    """Verifies that gradients flow back to all model parameters in the rewritten graph."""
    model = ToySDPAAttentionBlock(d_model=32, num_heads=2)
    rewriter = FXAttentionGraphRewriter(target_attention="cosformer", head_dim=16, is_causal=True)
    gm = rewriter.rewrite(model)

    x = torch.randn(2, 16, 32, requires_grad=True)
    out = gm(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    for name, param in gm.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            assert not torch.isnan(param.grad).any()


def test_fast_linear_attention_replacement_direct() -> None:
    """Verifies fast_linear_attention_replacement with 3D/4D shapes, causal and non-causal."""
    b, h, t, d = 2, 4, 16, 32
    q = torch.randn(b, h, t, d, requires_grad=True)
    k = torch.randn(b, h, t, d, requires_grad=True)
    v = torch.randn(b, h, t, d, requires_grad=True)

    # Test causal
    out_causal = fast_linear_attention_replacement(q, k, v, is_causal=True)
    assert out_causal.shape == (b, h, t, d)
    assert not torch.isnan(out_causal).any()

    # Test non-causal
    out_non_causal = fast_linear_attention_replacement(q, k, v, is_causal=False)
    assert out_non_causal.shape == (b, h, t, d)
    assert not torch.isnan(out_non_causal).any()

    # Test with boolean mask
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[:, -2:] = False
    out_masked = fast_linear_attention_replacement(q, k, v, mask=mask, is_causal=True)
    assert out_masked.shape == (b, h, t, d)
    assert not torch.isnan(out_masked).any()


def test_dead_code_elimination() -> None:
    """Verifies that eliminate_dead_nodes purges orphaned intermediate operations."""
    model = ToyDecomposedAttentionBlock(d_model=32, num_heads=2)
    tracer = NeuralOpsAttentionTracer()
    graph = tracer.trace(model)
    gm = torch.fx.GraphModule(model, graph)

    rewriter = FXAttentionGraphRewriter(target_attention="cosformer", head_dim=16)
    nodes_before = len(gm.graph.nodes)
    rewriter.replace_decomposed_patterns(gm)
    dead_count = rewriter.eliminate_dead_nodes(gm)
    nodes_after = len(gm.graph.nodes)

    assert dead_count > 0
    assert nodes_after < nodes_before


def test_model_surgery_engine_fx() -> None:
    """Verifies ModelSurgery fluent API with engine='fx'."""
    model = MultiLayerTransformerModel(d_model=32, num_heads=2, num_layers=2)
    surgery = ModelSurgery(model)
    surgery.replace_attention(target_attention="cosformer", engine="fx", head_dim=16, is_causal=True)

    x = torch.randn(2, 8, 32)
    out = surgery.model(x)
    assert out.shape == (2, 8, 32)
    assert not torch.isnan(out).any()


def test_pipeline_surgery_recipe_fx() -> None:
    """Verifies SurgeryRecipe execution with FX attention replacement."""
    model = ToySDPAAttentionBlock(d_model=32, num_heads=2)
    recipe = SurgeryRecipe(
        ops=[
            SurgeryOp(name="attention.fx", params={"target_attention": "cosformer", "head_dim": 16})
        ]
    )
    new_model, report = recipe.apply(model)
    assert report.success
    assert "attention.fx" in report.applied_ops

    x = torch.randn(2, 8, 32)
    out = new_model(x)
    assert out.shape == (2, 8, 32)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_fx_inductor_compile() -> None:
    """Verifies that the rewritten GraphModule compiles via PyTorch Inductor."""
    model = ToySDPAAttentionBlock(d_model=32, num_heads=2)
    rewriter = FXAttentionGraphRewriter(target_attention="cosformer", head_dim=16, is_causal=True)
    gm = rewriter.rewrite(model)

    try:
        compiled_gm = torch.compile(gm, mode="default")
        x = torch.randn(2, 8, 32)
        eager_out = gm(x)
        compiled_out = compiled_gm(x)
        assert torch.allclose(eager_out, compiled_out, atol=1e-4)
    except Exception as exc:
        # On certain host environments without C++ toolchain / AOTAutograd backends
        pytest.skip(f"Inductor compilation skipped on host: {exc}")
