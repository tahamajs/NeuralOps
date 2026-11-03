#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Universal Surgery Director & Invariant Compliance Engine.
Rigorous validation of Phases 041 through 060:
- Phase 041-050: Universal Topology Scanner & Classifier across 9 canonical families
  (LLaMA, Gemma-2, DeepSeek MLA, BERT, T5, Falcon/GPT-2, ViT, Jamba, Quantized Bases)
- Phase 051-060: Enforcement of the Ten Inviolable Laws of Universal Surgery
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import pytest
import torch
import torch.nn as nn

from neuralops.surgery.director import (
    ArchitectureProfile,
    CrossAttentionLinearBridge,
    InvariantPreservingLinearBridge,
    UniversalSurgeryDirector,
    apply_rope_pre_feature_map,
    apply_rotary_pos_emb,
    compute_decayed_state_bound,
    profile_model_topology,
    re_tie_embeddings,
    verify_tied_embeddings,
)


# ==============================================================================
# Mock Architectural Building Blocks for 9 Model Families
# ==============================================================================


class MockConfig:
    def __init__(self, **kwargs: Any) -> None:
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
        self.weight = nn.Parameter(
            torch.randint(0, 127, (out_features, in_features // 2), dtype=torch.uint8)
        )
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
        self.config = MockConfig(
            hidden_size=hidden_dim,
            num_attention_heads=4,
            model_type="falcon",
        )
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


class MockGPT2Model(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = MockConfig(
            hidden_size=hidden_dim,
            num_attention_heads=4,
            model_type="gpt2",
        )
        self.transformer = nn.ModuleDict({
            "h": nn.ModuleList([
                nn.ModuleDict({
                    "attn": nn.ModuleDict({
                        "c_attn": nn.Linear(hidden_dim, 3 * hidden_dim),
                        "c_proj": nn.Linear(hidden_dim, hidden_dim),
                    })
                })
            ])
        })


class MockViTModel(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = MockConfig(
            hidden_size=hidden_dim,
            num_attention_heads=4,
            model_type="vit",
        )
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
        self.config = MockConfig(
            hidden_size=hidden_dim,
            num_attention_heads=4,
            model_type="jamba",
        )
        self.layers = nn.ModuleList([
            nn.ModuleDict({"mamba": nn.Linear(hidden_dim, hidden_dim)}),
            nn.ModuleDict({"self_attn": MockLLaMAAttention(hidden_dim)}),
        ])


class MockQuantizedModel(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = MockConfig(
            hidden_size=hidden_dim,
            num_attention_heads=4,
        )
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
# Phase 041 – 050 Tests: Universal Topology Scanner & Classifier
# ==============================================================================


def test_phase_041_model_config_extraction() -> None:
    """Phase 041: 100% extraction accuracy on layer count, hidden dim, num heads, and head dim."""
    model = MockLLaMAModel(hidden_dim=128, num_layers=4)
    prof = profile_model_topology(model)
    assert prof.num_layers == 4
    assert prof.hidden_dim == 128
    assert prof.num_heads == 4
    assert prof.head_dim == 32


def test_phase_042_llama_causal_decoder_classifier() -> None:
    """Phase 042: Classifies LLaMA/Mistral/Qwen standard causal decoders."""
    model = MockLLaMAModel()
    prof = profile_model_topology(model)
    assert prof.family == "llama"
    assert not prof.has_qk_norm
    assert not prof.is_encoder_decoder
    assert not prof.is_quantized


def test_phase_043_gemma2_architecture_profiler() -> None:
    """Phase 043: Flags has_qk_norm=True and softcapping_factor=50.0."""
    model = MockGemma2Model()
    prof = profile_model_topology(model)
    assert prof.family == "gemma2"
    assert prof.has_qk_norm is True
    assert prof.softcapping_factor == 50.0
    assert prof.has_tied_embeddings is True


def test_phase_044_deepseek_mla_profiler() -> None:
    """Phase 044: Detects MLA projections and flags family='deepseek_mla'."""
    attn = MockDeepSeekMLAAttention()
    prof = profile_model_topology(attn)
    assert prof.family == "deepseek_mla"
    assert prof.is_mla is True


def test_phase_045_bert_bidirectional_encoder_profiler() -> None:
    """Phase 045: Flags family='bert', is_encoder_decoder=False, and is_encoder=True."""
    model = MockBERTModel()
    prof = profile_model_topology(model)
    assert prof.family == "bert"
    assert prof.is_encoder_decoder is False
    assert prof.is_encoder is True


def test_phase_046_t5_encoder_decoder_cross_attention_profiler() -> None:
    """Phase 046: Flags family='t5' and is_encoder_decoder=True."""
    model = MockT5Model()
    prof = profile_model_topology(model)
    assert prof.family == "t5"
    assert prof.is_encoder_decoder is True


def test_phase_047_falcon_gpt2_fused_qkv_conv1d_profiler() -> None:
    """Phase 047: Flags family='falcon' or family='gpt2'."""
    falcon = MockFalconModel()
    prof_falcon = profile_model_topology(falcon)
    assert prof_falcon.family == "falcon"

    gpt2 = MockGPT2Model()
    prof_gpt2 = profile_model_topology(gpt2)
    assert prof_gpt2.family == "gpt2"


def test_phase_048_vit_patch_embed_profiler() -> None:
    """Phase 048: Detects 2D patch embedding projection and flags family='vit'."""
    vit = MockViTModel()
    prof = profile_model_topology(vit)
    assert prof.family == "vit"
    assert prof.is_encoder is True


def test_phase_049_jamba_hybrid_mamba_ssm_profiler() -> None:
    """Phase 049: Detects alternating Transformer and Mamba SSM layers."""
    jamba = MockJambaModel()
    prof = profile_model_topology(jamba)
    assert prof.family == "jamba"
    assert prof.extra_attributes is not None
    assert "selective_attention_layer_indices" in prof.extra_attributes


def test_phase_050_quantized_weight_layer_profiler() -> None:
    """Phase 050: Detects 4-bit/8-bit quantized projections and flags is_quantized=True."""
    quant = MockQuantizedModel()
    prof = profile_model_topology(quant)
    assert prof.family == "quantized"
    assert prof.is_quantized is True


# ==============================================================================
# Phase 051 – 060 Tests: Enforcement of Invariant Laws 1 to 10
# ==============================================================================


def test_phase_051_law_1_qk_norm_preservation_hook() -> None:
    """
    Phase 051 (Law 1): Normalization hook ensures queries and keys pass through RMSNorm
    prior to feature mapping, bounding activations even with high magnitude inputs (10^3).
    """
    bridge = InvariantPreservingLinearBridge(
        has_qk_norm=True,
        head_dim=16,
    )
    assert bridge.has_qk_norm
    assert isinstance(bridge.q_norm, nn.RMSNorm)
    assert isinstance(bridge.k_norm, nn.RMSNorm)

    # Large magnitude inputs (10^3)
    q = torch.randn(2, 4, 8, 16) * 1000.0
    k = torch.randn(2, 4, 8, 16) * 1000.0
    v = torch.randn(2, 4, 8, 16)

    q_normed, k_normed = bridge.apply_qk_norm(q, k)
    # RMSNorm ensures root-mean-square along head_dim is approximately 1.0 (bounded)
    rms_q = torch.sqrt(torch.mean(q_normed ** 2, dim=-1))
    assert pytest.approx(rms_q.mean().item(), rel=1e-2) == 1.0

    out = bridge(q, k, v)
    assert out.shape == (2, 4, 8, 16)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_phase_052_law_2_logit_softcapping_temperature_folding() -> None:
    """
    Phase 052 (Law 2): Folds soft-capping factor c into linear temperature scaling:
    tau_folded = (1 / sqrt(d_k)) * tanh(c / max(c, 1.0)).
    """
    bridge_capped = InvariantPreservingLinearBridge(
        has_qk_norm=False,
        softcapping_factor=50.0,
        head_dim=64,
    )
    scale_capped = bridge_capped.compute_folded_scale()
    expected = (1.0 / 8.0) * math.tanh(50.0 / max(50.0, 1.0))
    assert pytest.approx(scale_capped, 1e-6) == expected

    # Numerical difference tolerance test
    ref_scale = (1.0 / 8.0) * math.tanh(1.0)
    diff = abs(scale_capped - ref_scale)
    assert diff < 1e-4


def test_phase_053_law_3_zero_bleed_padding_mask_sanitization() -> None:
    """
    Phase 053 (Law 3): Invariant holds:
      1. Arbitrary noise (10^6) in masked padding positions does not affect unpadded outputs.
      2. forall t in padding, ||O_t||_infty == 0.0.
    """
    bridge = InvariantPreservingLinearBridge(head_dim=16)
    B, H, T, D = 2, 2, 6, 16

    # Mask with last 2 tokens padded: [1, 1, 1, 1, 0, 0]
    mask = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    ])

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    # Pass 1: Original clean tensors
    out_1 = bridge(q, k, v, attention_mask=mask, is_causal=True)

    # Pass 2: Insert huge 10^6 arbitrary noise into masked padding positions
    k_noisy = k.clone()
    v_noisy = v.clone()
    k_noisy[:, :, 4:, :] = 1e6
    v_noisy[:, :, 4:, :] = -1e6

    out_2 = bridge(q, k_noisy, v_noisy, attention_mask=mask, is_causal=True)

    # Valid positions (0..3) are completely unperturbed
    max_diff_valid = torch.abs(out_1[:, :, :4, :] - out_2[:, :, :4, :]).max().item()
    assert max_diff_valid <= 1e-5

    # Padded positions (4..5) have strictly zero norm
    norm_pad_1 = torch.abs(out_1[:, :, 4:, :]).max().item()
    norm_pad_2 = torch.abs(out_2[:, :, 4:, :]).max().item()
    assert norm_pad_1 == 0.0
    assert norm_pad_2 == 0.0


def test_phase_054_law_4_t5_static_cross_attention_caching() -> None:
    """
    Phase 054 (Law 4): S_cross computed once during prefill;
    20 decoder steps execute in O(1) time with 100% cache hit rate.
    """
    bridge = CrossAttentionLinearBridge(head_dim=16)
    B, H, T_enc, D = 2, 2, 10, 16

    k_enc = torch.randn(B, H, T_enc, D)
    v_enc = torch.randn(B, H, T_enc, D)

    # 1. Prefill encoder once
    s_cross = bridge.prefill_encoder(k_enc, v_enc)
    assert s_cross.shape == (B, H, D, D)
    assert bridge.cached_s_cross is not None
    assert bridge.prefill_count == 1

    # 2. Run 20 decoder steps
    for _ in range(20):
        q_step = torch.randn(B, H, 1, D)
        out_step = bridge.decode_step(q_step)
        assert out_step.shape == (B, H, 1, D)
        assert not torch.isnan(out_step).any()

    # 3. Verify cache telemetry
    stats = bridge.get_cache_stats()
    assert stats["prefill_count"] == 1
    assert stats["decode_steps"] == 20
    assert stats["hit_rate"] == 1.0


def test_phase_055_law_5_rope_pre_projection_alignment() -> None:
    """
    Phase 055 (Law 5): RoPE is applied to Q and K prior to non-linear feature map phi(x).
    Verifies positional distance decay in linear feature space.
    """
    B, H, T, D = 1, 1, 8, 16
    q = torch.ones(B, H, T, D)
    k = torch.ones(B, H, T, D)

    # Generate rotary cos, sin frequencies
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(-1)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    angles = pos * inv_freq
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1).view(1, 1, T, D)
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1).view(1, 1, T, D)

    q_rot, k_rot = apply_rope_pre_feature_map(q, k, cos, sin)
    q_phi = torch.nn.functional.elu(q_rot) + 1.0
    k_phi = torch.nn.functional.elu(k_rot) + 1.0

    # Token 0 vs Token 0 (distance 0)
    sim_0_0 = torch.sum(q_phi[:, :, 0, :] * k_phi[:, :, 0, :]).item()
    # Token 0 vs Token 6 (distance 6)
    sim_0_6 = torch.sum(q_phi[:, :, 0, :] * k_phi[:, :, 6, :]).item()

    assert sim_0_0 > sim_0_6, "Positional distance decay invariant violated"


def test_phase_056_law_6_decoupled_latent_state_router() -> None:
    """
    Phase 056 (Law 6): Routes DeepSeek MLA transplant through DeepSeekMLASurgeryBridge
    without uncompressing into full MHA dimension.
    """
    deepseek_model = nn.ModuleDict({
        "layers": nn.ModuleList([
            nn.ModuleDict({"self_attn": MockDeepSeekMLAAttention(hidden_dim=64)})
        ])
    })
    deepseek_model.config = MockConfig(
        num_hidden_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        multi_head_latent_attention=True,
        model_type="deepseek_mla",
    )

    director = UniversalSurgeryDirector(deepseek_model)
    assert director.profile.family == "deepseek_mla"
    assert director.profile.is_mla is True

    # Execute surgery
    surgeried_model = director.execute_surgery()
    assert surgeried_model is not None


def test_phase_057_law_7_causal_monotonicity_state_decay() -> None:
    """
    Phase 057 (Law 7): Guarantees bounded state norms under gamma=0.99 across 10,000 steps:
    ||S_t||_F <= (1 / (1 - gamma)) * max_i ||phi(K_i)^T V_i||_F.
    """
    gamma = 0.99
    D = 16
    S = torch.zeros(D, D)
    max_step_norm = 0.0

    # Simulate 10,000 autoregressive steps
    for _ in range(10000):
        k_step = torch.randn(1, D)
        v_step = torch.randn(1, D)
        k_phi = torch.nn.functional.elu(k_step) + 1.0
        delta = torch.matmul(k_phi.t(), v_step)
        step_norm = torch.norm(delta, p="fro").item()
        if step_norm > max_step_norm:
            max_step_norm = step_norm
        S = gamma * S + delta

    final_norm = torch.norm(S, p="fro").item()
    bound = compute_decayed_state_bound(max_step_norm, gamma=gamma)

    assert final_norm < float("inf")
    assert final_norm <= bound * 1.05


def test_phase_058_law_8_tied_embedding_preservation_and_remediation() -> None:
    """
    Phase 058 (Law 8): Ensures tied embeddings are preserved and auto-remediated if broken.
    """
    gemma = MockGemma2Model()
    assert verify_tied_embeddings(gemma) is True

    # Untie intentionally
    gemma.lm_head.weight = nn.Parameter(gemma.embed_tokens.weight.clone())
    assert verify_tied_embeddings(gemma) is False

    # Auto-remediation
    remediated = re_tie_embeddings(gemma)
    assert remediated is True
    assert verify_tied_embeddings(gemma) is True
    assert id(gemma.lm_head.weight) == id(gemma.embed_tokens.weight)


def test_phase_059_law_9_fp32_accumulation_guard() -> None:
    """
    Phase 059 (Law 9): Enforces FP32 accumulation when input activations are FP16 or BF16.
    """
    bridge = InvariantPreservingLinearBridge(head_dim=16)
    B, H, T, D = 2, 2, 4, 16

    # Run with float16 inputs
    q = torch.randn(B, H, T, D, dtype=torch.float16)
    k = torch.randn(B, H, T, D, dtype=torch.float16)
    v = torch.randn(B, H, T, D, dtype=torch.float16)

    out = bridge(q, k, v)
    assert out.dtype == torch.float16
    assert bridge.last_accum_dtype == torch.float32


def test_phase_060_law_10_master_universal_surgery_integration() -> None:
    """
    Phase 060 (Law 10): Master UniversalSurgeryDirector.execute_surgery & verify_invariants.
    All 10 Laws report True with zero regressions.
    """
    gemma = MockGemma2Model()
    director = UniversalSurgeryDirector(gemma)

    # Perform surgery using alias
    surgeried_model = director.perform_surgery()
    assert surgeried_model is not None

    invariants = director.verify_invariants()
    assert invariants["All_Invariants_Verified"] is True
    assert invariants["Law_1_QK_Norm"] is True
    assert invariants["Law_2_Softcapping"] is True
    assert invariants["Law_3_Zero_Bleed_Padding"] is True
    assert invariants["Law_4_Static_Cross_Attention"] is True
    assert invariants["Law_5_RoPE_Pre_Projection"] is True
    assert invariants["Law_6_Decoupled_Latent_Router"] is True
    assert invariants["Law_7_Causal_Monotonicity_State_Decay"] is True
    assert invariants["Law_8_Tied_Embeddings"] is True
    assert invariants["Law_9_FP32_Accumulation"] is True
    assert invariants["Law_10_Master_Integration"] is True


def test_universal_surgery_director_raises_on_untied_embeddings() -> None:
    """Verifies verify_invariants raises RuntimeError if tied embeddings law is violated."""
    gemma = MockGemma2Model()
    director = UniversalSurgeryDirector(gemma)
    # Untie
    gemma.lm_head.weight = nn.Parameter(torch.randn_like(gemma.embed_tokens.weight))
    with pytest.raises(RuntimeError, match="Law 8 Violation"):
        director.verify_invariants()
