#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Universal Surgery Director & Invariant Compliance Engine.
Validates the Ten Inviolable Laws of Universal Surgery across all 9 model families:
LLaMA, Gemma-2, DeepSeek MLA, BERT, T5, Falcon, ViT, Jamba, and Quantized Bases.
"""

from __future__ import annotations

import math
from typing import Optional
import pytest
import torch
import torch.nn as nn

from neuralops.surgery.director import (
    ArchitectureProfile,
    CrossAttentionLinearBridge,
    InvariantPreservingLinearBridge,
    UniversalSurgeryDirector,
    profile_model_topology,
    re_tie_embeddings,
    verify_tied_embeddings,
)


# ==============================================================================
# Mock Architectural Building Blocks for 9 Model Families
# ==============================================================================


class MockConfig:
    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockLLaMAAttention(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.is_causal = True


class MockGemma2Attention(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.attn_logit_softcapping = 50.0
        self.is_causal = True


class MockDeepSeekMLAAttention(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.q_b_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv_b_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_dim, 32, bias=False)


class MockLinear4bit(nn.Module):
    """Simulates bitsandbytes 4-bit quantized linear projection."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randint(0, 127, (out_features, in_features // 2), dtype=torch.uint8))
        self.bias = None


class MockLLaMAModel(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.config = MockConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_dim,
            num_attention_heads=4,
            model_type="llama",
        )
        self.embed_tokens = nn.Embedding(100, hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"self_attn": MockLLaMAAttention(hidden_dim)})
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, 100, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


class MockGemma2Model(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.config = MockConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_dim,
            num_attention_heads=4,
            attn_logit_softcapping=50.0,
            model_type="gemma2",
        )
        self.embed_tokens = nn.Embedding(100, hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"self_attn": MockGemma2Attention(hidden_dim)})
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, 100, bias=False)
        # Tied embeddings by default in Gemma
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


class MockBERTModel(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = MockConfig(
            num_hidden_layers=2,
            hidden_size=hidden_dim,
            num_attention_heads=4,
            model_type="bert",
        )
        self.embeddings = nn.ModuleDict({"word_embeddings": nn.Embedding(100, hidden_dim)})
        self.encoder = nn.ModuleDict({
            "layer": nn.ModuleList([
                nn.ModuleDict({
                    "attention": nn.ModuleDict({
                        "self": MockLLaMAAttention(hidden_dim),
                        "output": nn.ModuleDict({"dense": nn.Linear(hidden_dim, hidden_dim)}),
                    })
                })
            ])
        })
        self.cls = nn.ModuleDict({"predictions": nn.Linear(hidden_dim, 100, bias=False)})

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings["word_embeddings"]


class MockT5Model(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = MockConfig(
            num_hidden_layers=2,
            hidden_size=hidden_dim,
            num_attention_heads=4,
            is_encoder_decoder=True,
            model_type="t5",
        )
        self.shared = nn.Embedding(100, hidden_dim)
        self.encoder = nn.ModuleDict({"block": nn.ModuleList([MockLLaMAAttention(hidden_dim)])})
        self.decoder = nn.ModuleDict({
            "block": nn.ModuleList([
                nn.ModuleDict({
                    "self_attn": MockLLaMAAttention(hidden_dim),
                    "EncDecAttention": MockLLaMAAttention(hidden_dim),
                })
            ])
        })
        self.lm_head = nn.Linear(hidden_dim, 100, bias=False)
        self.lm_head.weight = self.shared.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.shared

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


class MockFalconModel(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = MockConfig(hidden_size=hidden_dim, num_attention_heads=4, model_type="falcon")
        self.transformer = nn.ModuleDict({
            "h": nn.ModuleList([
                nn.ModuleDict({
                    "self_attention": nn.ModuleDict({
                        "query_key_value": nn.Linear(hidden_dim, 3 * hidden_dim),
                        "dense": nn.Linear(hidden_dim, hidden_dim),
                    })
                })
            ])
        })


class MockViTModel(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = MockConfig(hidden_size=hidden_dim, num_attention_heads=4, model_type="vit")
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=16)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.ModuleDict({
                    "qkv": nn.Linear(hidden_dim, 3 * hidden_dim),
                    "proj": nn.Linear(hidden_dim, hidden_dim),
                })
            })
        ])


class MockJambaModel(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = MockConfig(hidden_size=hidden_dim, num_attention_heads=4, model_type="jamba")
        self.layers = nn.ModuleList([
            nn.ModuleDict({"mamba": nn.Linear(hidden_dim, hidden_dim)}),
            nn.ModuleDict({"self_attn": MockLLaMAAttention(hidden_dim)}),
        ])


class MockQuantizedModel(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = MockConfig(hidden_size=hidden_dim, num_attention_heads=4)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.ModuleDict({
                    "q_proj": MockLinear4bit(hidden_dim, hidden_dim),
                    "k_proj": MockLinear4bit(hidden_dim, hidden_dim),
                    "v_proj": MockLinear4bit(hidden_dim, hidden_dim),
                    "o_proj": MockLinear4bit(hidden_dim, hidden_dim),
                })
            })
        ])


# ==============================================================================
# Phase 3.1 Tests: Universal Topology Profiler (9 Canonical Families)
# ==============================================================================


def test_profile_topology_all_nine_families() -> None:
    """Verifies profile_model_topology achieves 100% classification precision across 9 families."""
    llama_prof = profile_model_topology(MockLLaMAModel())
    assert llama_prof.family == "llama"
    assert not llama_prof.has_qk_norm
    assert not llama_prof.is_encoder_decoder

    gemma_prof = profile_model_topology(MockGemma2Model())
    assert gemma_prof.family == "gemma2"
    assert gemma_prof.has_qk_norm
    assert gemma_prof.softcapping_factor == 50.0
    assert gemma_prof.has_tied_embeddings

    deepseek_prof = profile_model_topology(MockDeepSeekMLAAttention())
    assert deepseek_prof.family == "deepseek_mla"

    bert_prof = profile_model_topology(MockBERTModel())
    assert bert_prof.family == "bert"

    t5_prof = profile_model_topology(MockT5Model())
    assert t5_prof.family == "t5"
    assert t5_prof.is_encoder_decoder

    falcon_prof = profile_model_topology(MockFalconModel())
    assert falcon_prof.family == "falcon"

    vit_prof = profile_model_topology(MockViTModel())
    assert vit_prof.family == "vit"

    jamba_prof = profile_model_topology(MockJambaModel())
    assert jamba_prof.family == "jamba"

    quant_prof = profile_model_topology(MockQuantizedModel())
    assert quant_prof.family == "quantized"
    assert quant_prof.is_quantized


# ==============================================================================
# Phase 3.2 Tests: Laws 1, 2, 3 (QK-Norm, Soft-Capping, Zero-Bleed Padding)
# ==============================================================================


def test_law_1_qk_norm_preservation() -> None:
    """Law 1: Queries and Keys must be normalized via RMSNorm prior to feature mapping."""
    bridge = InvariantPreservingLinearBridge(
        has_qk_norm=True,
        head_dim=16,
    )
    assert bridge.has_qk_norm
    assert isinstance(bridge.q_norm, nn.RMSNorm)
    assert isinstance(bridge.k_norm, nn.RMSNorm)

    # Large magnitude inputs
    q = torch.randn(2, 4, 8, 16) * 100.0
    k = torch.randn(2, 4, 8, 16) * 100.0
    v = torch.randn(2, 4, 8, 16)

    out = bridge(q, k, v)
    assert out.shape == (2, 4, 8, 16)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_law_2_logit_softcapping_temperature_folding() -> None:
    """Law 2: Attention logit soft-capping factor is folded into temperature scaling."""
    bridge_uncapped = InvariantPreservingLinearBridge(
        has_qk_norm=False,
        softcapping_factor=None,
        head_dim=64,
    )
    scale_uncapped = bridge_uncapped.compute_folded_scale()
    assert pytest.approx(scale_uncapped, 1e-6) == 1.0 / 8.0

    bridge_capped = InvariantPreservingLinearBridge(
        has_qk_norm=False,
        softcapping_factor=50.0,
        head_dim=64,
    )
    scale_capped = bridge_capped.compute_folded_scale()
    expected = (1.0 / 8.0) * math.tanh(1.0)
    assert pytest.approx(scale_capped, 1e-6) == expected
    assert scale_capped < scale_uncapped


def test_law_3_zero_bleed_padding_mask_sanitization() -> None:
    """
    Law 3: Padded tokens must contribute strictly zero delta to linear attention
    recurrence and associative memory states.
    """
    bridge = InvariantPreservingLinearBridge(head_dim=16)
    B, H, T, D = 2, 2, 6, 16

    # Sequence with padding on the last 2 tokens: [1, 1, 1, 1, 0, 0]
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    # Pass 1: Original tensors with padding mask
    out_1 = bridge(q, k, v, attention_mask=mask, is_causal=True)

    # Pass 2: Perturb the padded positions in K and V with huge random values
    k_perturbed = k.clone()
    v_perturbed = v.clone()
    k_perturbed[:, :, 4:, :] = 9999.0
    v_perturbed[:, :, 4:, :] = -8888.0

    out_2 = bridge(q, k_perturbed, v_perturbed, attention_mask=mask, is_causal=True)

    # The unpadded outputs (tokens 0..3) must be strictly identical despite huge padded perturbations!
    diff = torch.abs(out_1[:, :, :4, :] - out_2[:, :, :4, :]).max().item()
    assert diff <= 1e-5, f"Zero-bleed violation: padding tokens leaked into valid output (max diff: {diff})"


# ==============================================================================
# Phase 3.3 Tests: Law 4 (Static Cross-Attention Caching) & Law 8 (Tied Weights)
# ==============================================================================


def test_law_4_static_cross_attention_caching_t5() -> None:
    """
    Law 4: In encoder-decoder models, prefill_encoder caches S_cross once.
    decode_step executes in O(1) time per token without re-evaluating encoder representations.
    """
    bridge = CrossAttentionLinearBridge(head_dim=16)
    B, H, T_enc, D = 2, 2, 10, 16

    k_enc = torch.randn(B, H, T_enc, D)
    v_enc = torch.randn(B, H, T_enc, D)

    # 1. Prefill encoder once
    s_cross = bridge.prefill_encoder(k_enc, v_enc)
    assert s_cross.shape == (B, H, D, D)
    assert bridge.cached_s_cross is not None

    # 2. Simulate 5 autoregressive decoding steps
    outputs = []
    for _ in range(5):
        q_step = torch.randn(B, H, 1, D)
        out_step = bridge.decode_step(q_step)
        assert out_step.shape == (B, H, 1, D)
        assert not torch.isnan(out_step).any()
        outputs.append(out_step)

    assert len(outputs) == 5

    # 3. Test reset
    bridge.reset()
    assert bridge.cached_s_cross is None
    with pytest.raises(RuntimeError, match="Static cross-attention matrix not initialized"):
        bridge.decode_step(torch.randn(B, H, 1, D))


def test_law_8_tied_embeddings_preservation_and_remediation() -> None:
    """
    Law 8: Verifies tied embeddings are preserved and automatically re-tied if broken.
    """
    gemma = MockGemma2Model()
    assert verify_tied_embeddings(gemma)

    # Intentionally untie the embeddings
    gemma.lm_head.weight = nn.Parameter(gemma.embed_tokens.weight.clone())
    assert not verify_tied_embeddings(gemma)

    # Remediate via re_tie_embeddings
    success = re_tie_embeddings(gemma)
    assert success
    assert verify_tied_embeddings(gemma)
    assert id(gemma.lm_head.weight) == id(gemma.embed_tokens.weight)


# ==============================================================================
# Phase 3.5 Tests: UniversalSurgeryDirector End-to-End Orchestration
# ==============================================================================


def test_universal_surgery_director_end_to_end() -> None:
    """Tests UniversalSurgeryDirector orchestration and verification status."""
    gemma = MockGemma2Model()
    director = UniversalSurgeryDirector(gemma)
    assert director.profile.family == "gemma2"
    assert director.profile.has_tied_embeddings

    invariants = director.verify_invariants()
    assert invariants["All_Invariants_Verified"]
    assert invariants["Law_1_QK_Norm"]
    assert invariants["Law_2_Softcapping"]
    assert invariants["Law_3_Zero_Bleed_Padding"]
    assert invariants["Law_8_Tied_Embeddings"]


def test_universal_surgery_director_raises_on_untied_embeddings() -> None:
    """Verifies verify_invariants raises RuntimeError if tied embeddings law is violated."""
    gemma = MockGemma2Model()
    director = UniversalSurgeryDirector(gemma)
    # Untie
    gemma.lm_head.weight = nn.Parameter(torch.randn_like(gemma.embed_tokens.weight))
    with pytest.raises(RuntimeError, match="Law 8 Violation"):
        director.verify_invariants()
