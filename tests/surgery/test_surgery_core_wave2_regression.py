#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Wave 2 Regression Tests: Surgery Core Subsystem.

Validates fixes for:
  - P0-6: director.py::certify_model — rigorous law validation without hardcoded True
  - P0-10: surgery/__init__.py apply_backbone_surgery_hooks — real hook adapter
  - P0-19: fx_surgery.py::causal_prefix_scan — O(C^2 + D^2) bounded chunk memory
  - P1: endoskeletal_shell.py::apply_moa_schedule — dict and empty attr resilience
  - P1: layer_replacement.py::_instantiate_replacement — internal TypeError propagation
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.surgery import apply_backbone_surgery_hooks
from neuralops.surgery.attention_replace import (
    apply_backbone_surgery_hooks as apply_backbone_surgery_hooks_alt,
)
from neuralops.surgery.director import (
    ArchitectureProfile,
    InvariantPreservingLinearBridge,
    UniversalSurgeryDirector,
)
from neuralops.surgery.endoskeletal_shell import EndoskeletalShell
from neuralops.surgery.fx_surgery import causal_prefix_scan
from neuralops.surgery.layer_replacement import _instantiate_replacement
from neuralops.surgery.scanner import AttentionTargetSpec


# ==============================================================================
# Helpers & Mocks
# ==============================================================================


class SimpleAttentionLeaf(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Intentionally takes only hidden_states and returns single tensor (not 3-tuple)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        return self.out_proj(torch.matmul(attn, v))


class MockDecoderLayer(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.self_attn = SimpleAttentionLeaf(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        out = self.self_attn(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out + self.mlp(x)


class MockTransformerBackbone(nn.Module):
    def __init__(self, num_layers: int = 2, dim: int = 32):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, dim)
        self.layers = nn.ModuleList([MockDecoderLayer(dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(dim, 100, bias=False)
        # Tie embeddings
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(h)


# ==============================================================================
# P0-6: director.py::certify_model
# ==============================================================================


def test_certify_model_all_laws_pass():
    model = MockTransformerBackbone(num_layers=2, dim=32)
    director = UniversalSurgeryDirector(model)
    # Ensure profile has tied embeddings matched
    director.profile.has_tied_embeddings = True
    report = director.certify_model(strict=True)

    assert report["certified"] is True
    assert report["score"] == "10/10"
    laws = report["laws_certified"]
    for k in (
        "Law_1_QK_Norm",
        "Law_2_Softcapping",
        "Law_3_Zero_Bleed_Padding",
        "Law_4_Static_Cross_Attention",
        "Law_5_RoPE_Pre_Projection",
        "Law_6_Decoupled_Latent_Router",
        "Law_7_Causal_Monotonicity_State_Decay",
        "Law_8_Tied_Embeddings",
        "Law_9_FP32_Accumulation",
        "Law_10_Master_Integration",
    ):
        assert laws[k] is True, f"Law {k} should have certified True"


def test_certify_model_catches_qk_norm_violation():
    model = MockTransformerBackbone(num_layers=2, dim=32)
    director = UniversalSurgeryDirector(model)
    director.profile.has_qk_norm = True  # Model configured with QK-norm but modules lack it

    with pytest.raises(RuntimeError, match="Law 1 Violation"):
        director.certify_model(strict=True)

    report = director.certify_model(strict=False)
    assert report["certified"] is False
    assert report["laws_certified"]["Law_1_QK_Norm"] is False


def test_certify_model_catches_cross_attention_violation():
    model = MockTransformerBackbone(num_layers=2, dim=32)
    director = UniversalSurgeryDirector(model)
    director.profile.is_encoder_decoder = True  # Model claimed to be enc-dec but no cross-attention bridge

    with pytest.raises(RuntimeError, match="Law 4 Violation"):
        director.certify_model(strict=True)


def test_certify_model_catches_deepseek_mla_violation():
    model = MockTransformerBackbone(num_layers=2, dim=32)
    director = UniversalSurgeryDirector(model)
    director.profile.family = "deepseek_mla"
    director.profile.is_mla = True  # Claimed MLA but no MLA bridge attached

    with pytest.raises(RuntimeError, match="Law 6 Violation"):
        director.certify_model(strict=True)


def test_certify_model_catches_invalid_decay_gamma():
    model = MockTransformerBackbone(num_layers=2, dim=32)
    # Attach a bridge with an invalid decay gamma >= 1.0
    bad_bridge = InvariantPreservingLinearBridge(head_dim=32, decay=1.5)
    model.layers[0].bad_bridge = bad_bridge

    director = UniversalSurgeryDirector(model)
    report = director.certify_model(strict=False)
    assert report["laws_certified"]["Law_7_Causal_Monotonicity_State_Decay"] is False
    assert any("Law 7 Violation" in v for v in report["violations"])


def test_certify_model_catches_nan_weights():
    model = MockTransformerBackbone(num_layers=2, dim=32)
    # Inject NaN into a parameter
    with torch.no_grad():
        model.layers[0].mlp.weight[0, 0] = float("nan")

    director = UniversalSurgeryDirector(model)
    report = director.certify_model(strict=False)
    assert report["laws_certified"]["Law_10_Master_Integration"] is False
    assert any("Law 10 Violation" in v for v in report["violations"])


# ==============================================================================
# P0-10: surgery/__init__.py apply_backbone_surgery_hooks
# ==============================================================================


def test_apply_backbone_surgery_hooks_export_and_execution():
    assert callable(apply_backbone_surgery_hooks)
    assert callable(apply_backbone_surgery_hooks_alt)

    model = MockTransformerBackbone(num_layers=2, dim=32)
    hooked = apply_backbone_surgery_hooks(model, model_type="llama")
    assert len(hooked) == 2

    leaf = model.layers[0].self_attn
    assert getattr(leaf, "supports_gqa", False) is True
    assert getattr(leaf, "supports_rope", False) is True

    # Call with standard HF decoder kwargs that SimpleAttentionLeaf does not accept
    x = torch.randn(2, 4, 32)
    dummy_pos = torch.arange(4).unsqueeze(0).expand(2, -1)
    res = leaf(
        x,
        position_ids=dummy_pos,
        position_embeddings=torch.randn(2, 4, 32),
        past_key_value=None,
        use_cache=True,
        output_attentions=True,
    )
    # Verify return value is 3-tuple (attn_out, attn_weights, past_kv)
    assert isinstance(res, tuple)
    assert len(res) == 3
    assert res[0].shape == (2, 4, 32)


# ==============================================================================
# P0-19: fx_surgery.py::causal_prefix_scan
# ==============================================================================


def _reference_causal_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
    decay: float | None = None,
) -> torch.Tensor:
    b, h, t, d = q.shape
    e = v.shape[-1]
    res = torch.zeros(b, h, t, e, dtype=q.dtype, device=q.device)

    for i in range(b):
        for j in range(h):
            s = torch.zeros(d, e, dtype=q.dtype, device=q.device)
            z = torch.zeros(d, dtype=q.dtype, device=q.device)
            for step in range(t):
                qt = q[i, j, step]
                kt = k[i, j, step]
                vt = v[i, j, step]
                if decay is not None:
                    s = s * decay + torch.outer(kt, vt)
                    z = z * decay + kt
                else:
                    s = s + torch.outer(kt, vt)
                    z = z + kt
                num = torch.matmul(qt, s)
                den = torch.dot(qt, z) + eps
                res[i, j, step] = num / den
    return res


def test_causal_prefix_scan_numerical_parity_no_decay():
    torch.manual_seed(42)
    b, h, t, d = 2, 2, 64, 16
    q = torch.randn(b, h, t, d).abs() + 0.1
    k = torch.randn(b, h, t, d).abs() + 0.1
    v = torch.randn(b, h, t, d)

    ref = _reference_causal_linear_attention(q, k, v, decay=None)
    out = causal_prefix_scan(q, k, v, chunk_size=16)

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_causal_prefix_scan_numerical_parity_with_decay():
    torch.manual_seed(42)
    b, h, t, d = 2, 2, 64, 16
    decay = 0.95
    q = torch.randn(b, h, t, d).abs() + 0.1
    k = torch.randn(b, h, t, d).abs() + 0.1
    v = torch.randn(b, h, t, d)

    ref = _reference_causal_linear_attention(q, k, v, decay=decay)
    out = causal_prefix_scan(q, k, v, decay=decay, chunk_size=16)

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_causal_prefix_scan_3d_input_and_edge_lengths():
    torch.manual_seed(42)
    # 3D [B, T, D] input
    q = torch.randn(2, 33, 16).abs() + 0.1
    k = torch.randn(2, 33, 16).abs() + 0.1
    v = torch.randn(2, 33, 16)

    out = causal_prefix_scan(q, k, v, chunk_size=16)
    assert out.shape == (2, 33, 16)

    # Empty T=0 edge case
    q_empty = torch.randn(2, 0, 16)
    k_empty = torch.randn(2, 0, 16)
    v_empty = torch.randn(2, 0, 16)
    out_empty = causal_prefix_scan(q_empty, k_empty, v_empty)
    assert out_empty.shape == (2, 0, 16)


# ==============================================================================
# P1: endoskeletal_shell.py::apply_moa_schedule
# ==============================================================================


class DictLayersModel(nn.Module):
    """Model whose layers are stored in an nn.ModuleDict with non-digit keys."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.blocks = nn.ModuleDict(
            {
                "layer_a": MockDecoderLayer(dim),
                "layer_b": MockDecoderLayer(dim),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks.values():
            x = block(x)
        return x


def test_apply_moa_schedule_dict_and_empty_attr():
    model = DictLayersModel(dim=32)
    report = EndoskeletalShell.apply_moa_schedule(model)
    assert report["num_layers"] >= 1
    assert "wrapped_bridges" in report
    assert len(report["wrapped_bridges"]) >= 1

    x = torch.randn(2, 4, 32)
    out = report["model"](x)
    assert out.shape == (2, 4, 32)


# ==============================================================================
# P1: layer_replacement.py::_instantiate_replacement
# ==============================================================================


class FaultyAttention(nn.Module):
    """Attention class that deliberately raises TypeError during its internal __init__."""

    def __init__(self, **kwargs: object):
        super().__init__()
        # Internal TypeError that must NOT be silently swallowed by callers
        _ = "string" + 10  # type: ignore[operator]


class ValidCustomAttention(nn.Module):
    def __init__(self, hidden_size: int = 32, num_heads: int = 4, **kwargs: object):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_instantiate_replacement_propagates_internal_type_error():
    target = SimpleAttentionLeaf(dim=32)

    with pytest.raises(TypeError, match="can only concatenate str"):
        _instantiate_replacement(
            target_module=target,
            new_attention_cls=FaultyAttention,
        )


def test_instantiate_replacement_succeeds_with_valid_class():
    target = SimpleAttentionLeaf(dim=32)
    inst = _instantiate_replacement(
        target_module=target,
        new_attention_cls=ValidCustomAttention,
        hidden_size=64,
        num_heads=8,
    )
    assert isinstance(inst, ValidCustomAttention)
    assert inst.hidden_size == 64
    assert inst.num_heads == 8
