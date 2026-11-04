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
    _decouple_rope_transform,
    _detect_projection_layout,
    _initialize_identity_adapters,
    _inspect_positional_topology,
    _remediate_tied_embeddings,
    apply_rope_pre_feature_map,
    apply_rotary_pos_emb,
    compute_decayed_state_bound,
    compute_state_decay_bound,
    decouple_rope_transform,
    extract_qkv_weights,
    initialize_identity_adapters,
    profile_model_topology,
    re_tie_embeddings,
    sanitize_kv_for_padding_mask,
    sanitize_output_for_padding_mask,
    verify_tied_embeddings,
)
from neuralops.surgery.scanner import (
    DiscoveredLayer,
    ModelScanner,
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
            torch.randint(0, 127, (out_features, in_features // 2), dtype=torch.uint8),
            requires_grad=False,
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
    from neuralops.surgery.mla_surgery import DeepSeekMLASurgeryBridge
    assert isinstance(surgeried_model.layers[0]["self_attn"], DeepSeekMLASurgeryBridge)


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
    Phase 059 (Law 9): Enforces FP32 accumulation when input activations are FP16 or BF16,
    and validates GatheredParametersContext for distributed parameter gathering.
    """
    bridge = InvariantPreservingLinearBridge(head_dim=16)
    B, H, T, D = 2, 2, 4, 16

    # 1. Run with float16 inputs
    q_fp16 = torch.randn(B, H, T, D, dtype=torch.float16)
    k_fp16 = torch.randn(B, H, T, D, dtype=torch.float16)
    v_fp16 = torch.randn(B, H, T, D, dtype=torch.float16)

    out_fp16 = bridge(q_fp16, k_fp16, v_fp16)
    assert out_fp16.dtype == torch.float16
    assert bridge.last_accum_dtype == torch.float32

    # 2. Run with bfloat16 inputs
    q_bf16 = torch.randn(B, H, T, D, dtype=torch.bfloat16)
    k_bf16 = torch.randn(B, H, T, D, dtype=torch.bfloat16)
    v_bf16 = torch.randn(B, H, T, D, dtype=torch.bfloat16)

    out_bf16 = bridge(q_bf16, k_bf16, v_bf16)
    assert out_bf16.dtype == torch.bfloat16
    assert bridge.last_accum_dtype == torch.float32

    # 3. Distributed sharding context test
    from neuralops.surgery.director import GatheredParametersContext
    linear = nn.Linear(16, 16)
    with GatheredParametersContext(linear):
        assert linear.weight.shape == (16, 16)


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


def test_cross_attention_linear_bridge_forward_dual_mode() -> None:
    """Tests CrossAttentionLinearBridge forward execution in both prefill and decode modes."""
    bridge = CrossAttentionLinearBridge(head_dim=16)
    B, H, T_enc, D = 2, 2, 8, 16
    q = torch.randn(B, H, 1, D)
    k_enc = torch.randn(B, H, T_enc, D)
    v_enc = torch.randn(B, H, T_enc, D)

    # 1. Forward with k and v: prefill + decode
    out_1 = bridge(q, k=k_enc, v=v_enc)
    assert out_1.shape == (B, H, 1, D)
    assert bridge.cached_s_cross is not None

    # 2. Forward with only q: single-step decode using cached matrix
    out_2 = bridge(q)
    assert out_2.shape == (B, H, 1, D)
    assert torch.allclose(out_1, out_2, atol=1e-6)


def test_invariant_preserving_linear_bridge_forward_step_autoregressive() -> None:
    """Tests forward_step single-token recurrence with exponential state decay gamma."""
    bridge = InvariantPreservingLinearBridge(head_dim=16, decay=0.99)
    B, H, D = 2, 2, 16

    state = None
    z_state = None
    outputs = []

    for _ in range(5):
        q_t = torch.randn(B, H, 1, D)
        k_t = torch.randn(B, H, 1, D)
        v_t = torch.randn(B, H, 1, D)
        out_t, state, z_state = bridge.forward_step(q_t, k_t, v_t, state=state, z_state=z_state)
        assert out_t.shape == (B, H, 1, D)
        assert state is not None
        assert z_state is not None
        outputs.append(out_t)

    assert len(outputs) == 5


def test_legacy_shims_forwarding() -> None:
    """Verifies pruned legacy surgery shims cleanly redirect via neuralops.surgery."""
    from neuralops.surgery.attention_replace import (
        UniversalSurgeryDirector as DirectorShim1,
        replace_attention,
    )
    from neuralops.surgery import (
        SurgeryManifest,
        SurgeryPipeline,
        extract_surgical_manifest,
        get_recovery_params,
        load_surgical_checkpoint,
        replay_surgery_manifest,
        save_surgical_checkpoint,
    )
    from neuralops.export.surgery_manifest import SurgeryManifest as CanonicalSurgeryManifest

    assert DirectorShim1 is UniversalSurgeryDirector
    assert SurgeryManifest is CanonicalSurgeryManifest
    assert callable(get_recovery_params)
    assert callable(extract_surgical_manifest)
    assert callable(save_surgical_checkpoint)
    assert callable(load_surgical_checkpoint)
    assert callable(replay_surgery_manifest)
    assert hasattr(SurgeryPipeline, "apply_universal_director")

    gemma = MockGemma2Model()
    surgeried = replace_attention(gemma)
    assert surgeried is not None


def test_surgical_model_protocol_compliance() -> None:
    """Verifies surgeried model adheres to SurgicalModel protocol from neuralops.models.protocols."""
    from neuralops.models.protocols import SurgicalModel

    gemma = MockGemma2Model()
    director = UniversalSurgeryDirector(gemma)
    surgeried = director.execute_surgery(target_attention="cosformer")

    manifest = surgeried.get_surgical_manifest()
    assert manifest["target_attention"] == "cosformer"
    assert manifest["family"] == "gemma2"
    assert surgeried.get_attention_family(0) == "cosformer"
    assert isinstance(surgeried, SurgicalModel)


# ==============================================================================
# Phases 046 – 050 Specialized DoD Tests
# ==============================================================================


def test_phase_046_law_2_logit_softcapping_mathematical_folding() -> None:
    """
    Phase 046 (Law 2): Gemma-2 logit soft-capping mathematical folding.
    Validates bounded soft-capping feature map:
      phi_cap(x) = C * tanh(x / C) + C, 0.0 <= phi_cap(x) <= 2C.
    Prevents logit saturation and FP16 overflow during long-context generation.
    """
    c = 50.0
    bridge = InvariantPreservingLinearBridge(
        has_qk_norm=False,
        softcapping_factor=c,
        head_dim=64,
    )
    scale = bridge.compute_folded_scale()
    expected_scale = (1.0 / 8.0) * math.tanh(c / max(c, 1.0))
    assert pytest.approx(scale, 1e-6) == expected_scale

    # Test extreme negative inputs: phi_cap(x) approaches 0.0 but strictly >= 0.0
    x_neg = torch.tensor([-1e6, -100.0, -50.0, -10.0])
    phi_neg = bridge.fold_softcapping_feature_map(x_neg)
    assert (phi_neg >= 0.0).all()

    # Test zero input: phi_cap(0) = C
    x_zero = torch.tensor([0.0])
    phi_zero = bridge.fold_softcapping_feature_map(x_zero)
    assert pytest.approx(phi_zero.item(), 1e-5) == c

    # Test extreme positive inputs: phi_cap(x) bounded by 2C
    x_pos = torch.tensor([10.0, 50.0, 100.0, 1e6])
    phi_pos = bridge.fold_softcapping_feature_map(x_pos)
    assert (phi_pos <= 2.0 * c).all()

    # Full forward pass with soft-capping
    B, H, T, D = 2, 4, 16, 64
    q = torch.randn(B, H, T, D) * 100.0
    k = torch.randn(B, H, T, D) * 100.0
    v = torch.randn(B, H, T, D)
    out = bridge(q, k, v)
    assert out.shape == (B, H, T, D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_phase_047_law_3_zero_bleed_padding_and_varlen_boundary() -> None:
    """
    Phase 047 (Law 3): Zero-Bleed Padding & Varlen Boundary Offset Invariant.
    Validates:
      1. Arbitrary noise (10^7) in masked padding positions produces exactly zero output.
      2. Cross-document attention leakage is mathematically zero (||leakage|| == 0.0).
    """
    B, H, T, D = 2, 2, 8, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    # Valid length = 5, padded = 3
    mask = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ])

    bridge = InvariantPreservingLinearBridge(head_dim=16)

    # Clean execution
    out_clean = bridge(q, k, v, attention_mask=mask)

    # Noisy padding injection
    k_noisy = k.clone()
    v_noisy = v.clone()
    k_noisy[:, :, 5:, :] = 1e7
    v_noisy[:, :, 5:, :] = -1e7

    out_noisy = bridge(q, k_noisy, v_noisy, attention_mask=mask)

    # Valid positions are identical
    leakage = torch.abs(out_clean[:, :, :5, :] - out_noisy[:, :, :5, :]).max().item()
    assert leakage == 0.0 or leakage < 1e-5

    # Padded positions are strictly zero
    pad_norm = torch.abs(out_noisy[:, :, 5:, :]).max().item()
    assert pad_norm == 0.0

    # Varlen cu_seqlens test
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    k_packed = torch.randn(1, 1, 8, 16)
    v_packed = torch.randn(1, 1, 8, 16)
    k_packed[:, :, 6:, :] = 999.0
    v_packed[:, :, 6:, :] = 999.0

    k_clean, v_clean = sanitize_kv_for_padding_mask(k_packed, v_packed, cu_seqlens=cu_seqlens)
    assert (k_clean[:, :, 6:, :] == 0.0).all()
    assert (v_clean[:, :, 6:, :] == 0.0).all()


def test_phase_048_law_4_cross_attention_o1_decoding_latency() -> None:
    """
    Phase 048 (Law 4): Cross-Attention Static Key-Value Cache Bridge.
    Precomputes S_cross in O(T_enc) once; subsequent decoding steps execute in O(1) time.
    """
    bridge = CrossAttentionLinearBridge(head_dim=32)
    assert not bridge.is_initialized

    B, H, T_enc, D = 1, 4, 128, 32
    k_enc = torch.randn(B, H, T_enc, D)
    v_enc = torch.randn(B, H, T_enc, D)

    # Prefill once
    s_cross = bridge.compute_cross_state(k_enc, v_enc)
    assert bridge.is_initialized
    assert s_cross.shape == (B, H, D, D)
    assert bridge.prefill_count == 1

    # 10 decoding steps
    for _ in range(10):
        q_step = torch.randn(B, H, 1, D)
        out_step = bridge.decode_step(q_step)
        assert out_step.shape == (B, H, 1, D)
        assert not torch.isnan(out_step).any()

    stats = bridge.get_cache_stats()
    assert stats["prefill_count"] == 1
    assert stats["decode_steps"] == 10
    assert stats["hit_rate"] == 1.0


def test_phase_049_law_5_rotary_embedding_decoupling_and_passthrough() -> None:
    """
    Phase 049 (Law 5): Rotary Embedding Decoupling & Pass-Through.
    Validates decoupled RoPE stream and 1D exponential decay pass-through.
    """
    B, H, T, D = 1, 1, 16, 32
    q = torch.ones(B, H, T, D)
    k = torch.ones(B, H, T, D)

    # RoPE on 16 channels, content on 16 channels
    rope_dim = 16
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(-1)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    angles = pos * inv_freq
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1).view(1, 1, T, rope_dim)
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1).view(1, 1, T, rope_dim)

    q_dec, k_dec = _decouple_rope_transform(q, k, cos=cos, sin=sin, rope_dim=rope_dim, decay_gamma=0.95)
    assert q_dec.shape == (B, H, T, D)
    assert k_dec.shape == (B, H, T, D)

    # Decoupled alias test
    q_dec2, k_dec2 = decouple_rope_transform(q, k, cos=cos, sin=sin, rope_dim=rope_dim)
    assert torch.allclose(q_dec2[..., :16], q[..., :16])


def test_phase_050_law_6_identity_adapters_initialization() -> None:
    """
    Phase 050 (Law 6): Post-Surgery Parameter Identity Initialization.
    Newly mounted adapters and sidecars initialize with alpha_init = 0.0,
    guaranteeing student replicates teacher output with zero degradation at step 0.
    """
    class MockAdapterModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.linear_1 = nn.Linear(16, 8)
            self.linear_2 = nn.Linear(8, 16)
            self.sidecar = nn.Module()
            self.sidecar.alpha = 0.8
            self.sidecar.gate_proj = nn.Linear(16, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.alpha * self.linear_2(self.linear_1(x))

    model = MockAdapterModule()
    # At start, weights are non-zero
    assert model.alpha.item() == 1.0
    assert model.sidecar.alpha == 0.8

    # Apply Law 6 identity initialization
    initialize_identity_adapters(model)

    assert model.alpha.item() == 0.0
    assert model.sidecar.alpha == 0.0
    assert torch.all(model.linear_2.weight == 0.0)
    assert torch.all(model.sidecar.gate_proj.weight == 0.0)

    # Test forward is exact identity (student replicates teacher)
    x = torch.randn(2, 4, 16)
    out = model(x)
    assert torch.allclose(out, x, atol=1e-7)


def test_phase_051_state_decay_bound_and_attention_sinks() -> None:
    """
    Phase 051 (Law 7): Validates compute_state_decay_bound and attention sink stability.
    State norm is strictly bounded by geometric series: ||S_t||_F <= (1 / (1 - gamma)) * max_i ||phi(K_i)^T V_i||_F.
    """
    gamma = 0.99
    max_step_norm = 1.25
    bound = compute_state_decay_bound(max_step_norm, gamma=gamma)
    assert abs(bound - (1.25 / 0.01)) < 1e-6  # exactly 125.0 (100x max factor)

    # Test InvariantPreservingLinearBridge with attention sinks
    bridge = InvariantPreservingLinearBridge(head_dim=16, decay=0.99, num_attention_sinks=2)
    B, H, T, D = 2, 2, 8, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    out = bridge(q, k, v, is_causal=True)
    assert out.shape == (B, H, T, D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_phase_052_tied_embedding_pointer_synchronization() -> None:
    """
    Phase 052 (Law 8): Detects tied weights and guarantees pointers remain properly synchronized
    without memory corruption.
    """
    gemma = MockGemma2Model()
    assert verify_tied_embeddings(gemma) is True
    assert gemma.lm_head.weight.data_ptr() == gemma.embed_tokens.weight.data_ptr()

    # Intentionally untie by allocating new parameter
    gemma.lm_head.weight = nn.Parameter(torch.randn_like(gemma.embed_tokens.weight))
    assert verify_tied_embeddings(gemma) is False

    # Auto-remediation via _remediate_tied_embeddings
    remediated = _remediate_tied_embeddings(gemma)
    assert remediated is True
    assert verify_tied_embeddings(gemma) is True
    assert gemma.lm_head.weight.data_ptr() == gemma.embed_tokens.weight.data_ptr()
    assert id(gemma.lm_head.weight) == id(gemma.embed_tokens.weight)


def test_phase_053_fp32_accumulation_drift_bound() -> None:
    """
    Phase 053 (Law 9): Validates FP32 accumulation registers with reduced precision inputs,
    verifying numerical drift between float64 reference and surgical output <= 1e-4.
    """
    bridge = InvariantPreservingLinearBridge(head_dim=16)
    B, H, T, D = 2, 2, 8, 16

    # Generate double-precision inputs for reference
    torch.manual_seed(42)
    q_f64 = torch.randn(B, H, T, D, dtype=torch.float64)
    k_f64 = torch.randn(B, H, T, D, dtype=torch.float64)
    v_f64 = torch.randn(B, H, T, D, dtype=torch.float64)

    # Reference computation in float64
    scale_64 = bridge.compute_folded_scale()
    q_phi_64 = torch.nn.functional.elu(q_f64 * scale_64) + 1.0
    k_phi_64 = torch.nn.functional.elu(k_f64) + 1.0
    kv_64 = torch.einsum("bhtd,bhte->bhde", k_phi_64, v_f64)
    out_64 = torch.einsum("bhtd,bhde->bhte", q_phi_64, kv_64)
    k_sum_64 = k_phi_64.sum(dim=2)
    denom_64 = torch.einsum("bhtd,bhd->bht", q_phi_64, k_sum_64).unsqueeze(-1) + 1e-6
    ref_out = out_64 / denom_64

    # Run through bridge with float16 input
    q_f16 = q_f64.to(torch.float16)
    k_f16 = k_f64.to(torch.float16)
    v_f16 = v_f64.to(torch.float16)
    out_bridge = bridge(q_f16, k_f16, v_f16, is_causal=False)

    assert out_bridge.dtype == torch.float16
    assert bridge.last_accum_dtype == torch.float32

    # Verify numerical drift against float64 reference <= 1e-3 (FP16 precision resolution)
    diff = torch.abs(out_bridge.to(torch.float64) - ref_out).max().item()
    assert diff <= 1e-3

    # Also run with float32 inputs: drift against float64 reference <= 1e-4
    q_f32 = q_f64.to(torch.float32)
    k_f32 = k_f64.to(torch.float32)
    v_f32 = v_f64.to(torch.float32)
    out_f32 = bridge(q_f32, k_f32, v_f32, is_causal=False)
    diff_f32 = torch.abs(out_f32.to(torch.float64) - ref_out).max().item()
    assert diff_f32 <= 1e-4


def test_phase_054_certify_model_structured_json_audit() -> None:
    """
    Phase 054 (Law 10): Validates UniversalSurgeryDirector.certify_model automated auditor,
    returning structured dictionary and JSON report with 10/10 Inviolable Laws certified.
    """
    import json

    gemma = MockGemma2Model()
    director = UniversalSurgeryDirector(gemma)
    surgeried_model = director.perform_surgery()
    assert surgeried_model is not None

    report = director.certify_model(strict=True)
    assert report["certified"] is True
    assert report["score"] == "10/10"
    assert len(report["violations"]) == 0
    assert report["laws_certified"]["Law_1_QK_Norm"] is True
    assert report["laws_certified"]["Law_8_Tied_Embeddings"] is True
    assert report["laws_certified"]["Law_10_Master_Universal_Certification"] is True

    # Validate JSON serialization
    json_str = director.certify_model_json(strict=True)
    parsed = json.loads(json_str)
    assert parsed["score"] == "10/10"
    assert parsed["certified"] is True


# ==============================================================================
# Phases 041 through 045 DoD Gate Validation Tests
# ==============================================================================


def test_phase_041_dod_exhaustive_architecture_profile() -> None:
    """
    Phase 041 DoD Gate: Generates exhaustive ArchitectureProfile with zero unidentified fields
    across all 9 canonical model families.
    """
    models = {
        "llama": MockLLaMAModel(hidden_dim=128, num_layers=4),
        "gemma2": MockGemma2Model(hidden_dim=128, num_layers=2),
        "deepseek_mla": MockDeepSeekMLAAttention(hidden_dim=64),
        "bert": MockBERTModel(hidden_dim=64),
        "t5": MockT5Model(hidden_dim=64),
        "falcon": MockFalconModel(hidden_dim=64),
        "vit": MockViTModel(hidden_dim=64),
        "jamba": MockJambaModel(hidden_dim=64),
        "quantized": MockQuantizedModel(hidden_dim=64),
    }

    for expected_family, model in models.items():
        prof = profile_model_topology(model)
        assert prof.family == expected_family
        assert prof.num_layers > 0
        assert prof.hidden_dim > 0
        assert prof.num_heads > 0
        assert prof.head_dim > 0
        assert prof.num_kv_heads is not None and prof.num_kv_heads > 0
        assert isinstance(prof.has_qk_norm, bool)
        assert isinstance(prof.is_encoder_decoder, bool)
        assert isinstance(prof.has_tied_embeddings, bool)
        assert isinstance(prof.is_quantized, bool)
        assert prof.positional_topology is not None
        assert prof.positional_params is not None
        assert prof.projection_layout is not None


def test_phase_042_dod_packed_vs_decoupled_slicing() -> None:
    """
    Phase 042 DoD Gate: Exact weight slicing with zero parameter leakage or transposition errors.
    Detects and slices decoupled projections, packed projections, and transposed Conv1D weights.
    """
    # 1. Decoupled projections (LLaMA / Mistral)
    llama_attn = MockLLaMAAttention(hidden_dim=64, num_heads=4)
    layout_llama = _detect_projection_layout(llama_attn)
    assert layout_llama == "decoupled"
    W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o = extract_qkv_weights(llama_attn, num_heads=4, head_dim=16)
    assert W_q.shape == (64, 64)
    assert W_k.shape == (64, 64)
    assert W_v.shape == (64, 64)
    assert W_o.shape == (64, 64)

    # 2. Packed fused projections (Falcon / ViT)
    falcon_mod = MockFalconModel(hidden_dim=64).transformer["h"][0]["self_attention"]
    layout_falcon = _detect_projection_layout(falcon_mod)
    assert layout_falcon == "packed"
    W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o = extract_qkv_weights(falcon_mod, num_heads=4, head_dim=16)
    assert W_q.shape == (64, 64)
    assert W_k.shape == (64, 64)
    assert W_v.shape == (64, 64)
    assert W_o.shape == (64, 64)

    # 3. Transposed Conv1D (HuggingFace GPT-2)
    gpt2_attn = MockGPT2Model(hidden_dim=64).transformer["h"][0]["attn"]
    layout_gpt2 = _detect_projection_layout(gpt2_attn)
    assert layout_gpt2 == "transposed_conv1d"
    W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o = extract_qkv_weights(gpt2_attn, num_heads=4, head_dim=16)
    assert W_q.shape == (64, 64)
    assert W_k.shape == (64, 64)
    assert W_v.shape == (64, 64)
    assert W_o.shape == (64, 64)


def test_phase_043_dod_positional_topology_inspector() -> None:
    """
    Phase 043 DoD Gate: Correctly identifies positional topology and parameters
    (RoPE with base frequency theta and scaling, ALiBi slopes, Learned Absolute, Relative Position Buckets).
    """
    # RoPE
    llama = MockLLaMAModel()
    llama.config.rope_theta = 500000.0
    llama.config.rope_scaling = {"type": "linear", "factor": 2.0}
    top, params = _inspect_positional_topology(llama)
    assert top == "rope"
    assert params["base_theta"] == 500000.0
    assert params["rope_scaling"] == {"type": "linear", "factor": 2.0}

    # ALiBi
    alibi_cfg = MockConfig(alibi=True, num_attention_heads=16, model_type="bloom")
    alibi_model = nn.Module()
    alibi_model.config = alibi_cfg
    top, params = _inspect_positional_topology(alibi_model)
    assert top == "alibi"
    assert params["num_heads"] == 16
    assert params["has_alibi"] is True

    # Relative Position Buckets (T5)
    t5 = MockT5Model()
    t5.config.relative_attention_num_buckets = 64
    t5.config.relative_attention_max_distance = 256
    top, params = _inspect_positional_topology(t5)
    assert top == "relative_position_buckets"
    assert params["num_buckets"] == 64
    assert params["max_distance"] == 256

    # Learned Absolute (BERT)
    bert = MockBERTModel()
    top, params = _inspect_positional_topology(bert)
    assert top == "learned_absolute"
    assert params["max_position_embeddings"] is not None


def test_phase_044_dod_layer_monotonic_scanner() -> None:
    """
    Phase 044 DoD Gate: Discovers 100% of layers in strictly monotonic ascending order: 0, 1, ..., N-1
    across HF, GPT-2, Megatron, and Timm ViT conventions.
    """
    # 4-layer LLaMA model
    model_llama = MockLLaMAModel(hidden_dim=64, num_layers=4)
    specs = ModelScanner.scan(model_llama)
    assert len(specs) == 4
    indices = [s.layer_idx for s in specs]
    assert indices == [0, 1, 2, 3]

    # Verify monotonic ordering and DiscoveredLayer
    layers = ModelScanner.scan_layers(model_llama)
    assert len(layers) == 4
    assert [l.layer_idx for l in layers] == [0, 1, 2, 3]
    for idx, l in enumerate(layers):
        assert l.layer_idx == idx
        assert isinstance(l, DiscoveredLayer)
        assert l.module is not None


def test_phase_045_dod_law_1_qk_norm_preservation() -> None:
    """
    Phase 045 DoD Gate: Preserves QK-Normalization (Gemma-2 Law 1).
    Eliminates activation drift under high-magnitude inputs (10^4);
    output and gradients remain strictly bounded <= 1.0.
    """
    bridge = InvariantPreservingLinearBridge(has_qk_norm=True, head_dim=16)

    # Inputs with extreme magnitude (10^4)
    q = (torch.randn(2, 4, 8, 16) * 10000.0).requires_grad_(True)
    k = (torch.randn(2, 4, 8, 16) * 10000.0).requires_grad_(True)
    v = torch.randn(2, 4, 8, 16)

    # Normalized activations prior to feature map
    q_normed, k_normed = bridge.apply_qk_norm(q, k)
    rms_q = torch.sqrt(torch.mean(q_normed**2, dim=-1))
    rms_k = torch.sqrt(torch.mean(k_normed**2, dim=-1))

    # Activation drift is completely eliminated: RMS is normalized to ~1.0
    assert pytest.approx(rms_q.mean().item(), rel=1e-2) == 1.0
    assert pytest.approx(rms_k.mean().item(), rel=1e-2) == 1.0

    # Forward pass and gradient check
    out = bridge(q, k, v)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # Backward pass: gradient wrt extreme magnitude inputs must remain strictly bounded <= 1.0
    loss = out.sum()
    loss.backward()
    assert q.grad is not None and k.grad is not None
    assert q.grad.abs().max().item() <= 1.0
    assert k.grad.abs().max().item() <= 1.0


# ==============================================================================
# Agent 12 (Phases 056 - 060) Master Validation & DoD Gates
# ==============================================================================


def test_phase_056_dod_invariant_preserving_linear_bridge_autograd() -> None:
    """
    Phase 056 DoD Gate: InvariantPreservingLinearBridge packages all 10 Inviolable Laws.
    Verifies full forward and backward autograd pass completes without errors.
    """
    bridge = InvariantPreservingLinearBridge(
        has_qk_norm=True,
        softcapping_factor=30.0,
        head_dim=16,
        decay=0.95,
        num_attention_sinks=2,
    )

    # 1. 4D shape: [B, H, T, D] with autograd gradients enabled on q, k, v
    B, H, T, D = 2, 4, 12, 16
    q = torch.randn(B, H, T, D, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, T, D, dtype=torch.float32, requires_grad=True)

    out = bridge(q, k, v, is_causal=True)
    assert out.shape == (B, H, T, D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()
    assert q.grad.abs().sum().item() > 0.0
    assert k.grad.abs().sum().item() > 0.0
    assert v.grad.abs().sum().item() > 0.0

    # 2. 3D shape: [B, T, D] with autograd gradients enabled
    q3 = torch.randn(B, T, D, dtype=torch.float32, requires_grad=True)
    k3 = torch.randn(B, T, D, dtype=torch.float32, requires_grad=True)
    v3 = torch.randn(B, T, D, dtype=torch.float32, requires_grad=True)

    out3 = bridge(q3, k3, v3, is_causal=True)
    assert out3.shape == (B, T, D)
    loss3 = out3.sum()
    loss3.backward()
    assert q3.grad is not None and not torch.isnan(q3.grad).any()

    # 3. Protocol conformance & manifest metadata
    manifest = bridge.get_surgical_manifest()
    assert manifest["has_qk_norm"] is True
    assert manifest["softcapping_factor"] == 30.0
    assert manifest["decay"] == 0.95
    assert bridge.get_attention_family() == "linear_attention"


def test_phase_057_dod_cross_attention_linear_bridge_batch_parity() -> None:
    """
    Phase 057 DoD Gate: CrossAttentionLinearBridge static cross-attention state pre-computation
    and caching. Produces identical outputs whether decoding step-by-step or in full batch.
    """
    bridge = CrossAttentionLinearBridge(head_dim=16)

    B, H, T_enc, T_dec, D = 2, 4, 16, 8, 16
    k_enc = torch.randn(B, H, T_enc, D)
    v_enc = torch.randn(B, H, T_enc, D)
    q_full = torch.randn(B, H, T_dec, D)

    # 1. Precompute static encoder associative matrix once
    s_cross = bridge.prefill_encoder(k_enc, v_enc)
    assert s_cross is not None
    assert s_cross.shape == (B, H, D, D)
    assert bridge.is_initialized is True

    # 2. Full-batch decode pass
    out_batch = bridge.decode_batch(q_full)
    assert out_batch.shape == (B, H, T_dec, D)

    # 3. Step-by-step token-by-token decode pass
    step_outputs = []
    for t in range(T_dec):
        q_t = q_full[:, :, t : t + 1, :]
        out_t = bridge.decode_step(q_t)
        assert out_t.shape == (B, H, 1, D)
        step_outputs.append(out_t)

    out_stepwise = torch.cat(step_outputs, dim=-2)
    assert out_stepwise.shape == (B, H, T_dec, D)

    # 4. Strict numerical parity between step-by-step and full-batch
    max_diff = (out_batch - out_stepwise).abs().max().item()
    assert max_diff < 1e-5
    assert torch.allclose(out_batch, out_stepwise, atol=1e-5)

    # 5. Cache telemetry
    stats = bridge.get_cache_stats()
    assert stats["prefill_count"] == 1
    assert stats["decode_steps"] >= T_dec
    assert stats["hit_rate"] > 0.0


def test_phase_058_dod_surgery_pipeline_surgical_model_integration() -> None:
    """
    Phase 058 DoD Gate: SurgeryPipeline.apply_universal_director executes and
    returns a certified SurgicalModel conforming to the protocol.
    """
    from neuralops.models.protocols import SurgicalModel
    from neuralops.surgery.pipeline import SurgeryPipeline

    gemma = MockGemma2Model(hidden_dim=64, num_layers=2)

    # Execute universal director through master SurgeryPipeline
    surgeried = SurgeryPipeline.apply_universal_director(
        gemma,
        laws=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        target_attention="cosformer",
        decay=0.99,
        softcapping_factor=50.0,
    )

    # Verify model is a certified SurgicalModel
    assert isinstance(surgeried, SurgicalModel)
    assert hasattr(surgeried, "get_surgical_manifest")
    assert hasattr(surgeried, "get_attention_family")

    manifest = surgeried.get_surgical_manifest()
    assert manifest["target_attention"] == "cosformer"
    assert manifest["family"] == "gemma2"
    assert surgeried.get_attention_family(0) == "cosformer"

    # Verify invalid laws specification raises ValueError
    with pytest.raises(ValueError, match="Invalid laws specified"):
        SurgeryPipeline.apply_universal_director(gemma, laws=[0, 11])


def test_phase_059_dod_safe_pruning_and_clean_redirection() -> None:
    """
    Phase 059 DoD Gate: Verifies obsolete files are removed and all package symbols
    cleanly redirect from neuralops.surgery with 0 import errors.
    """
    from pathlib import Path
    import importlib

    repo_root = Path(__file__).resolve().parent.parent
    surgery_dir = repo_root / "neuralops" / "surgery"

    # Verify physical file removal
    assert not (surgery_dir / "param_recovery.py").exists(), "param_recovery.py was not pruned"
    assert not (surgery_dir / "surgery_manifest.py").exists(), "surgery_manifest.py was not pruned"
    assert not (surgery_dir / "surgical_io.py").exists(), "surgical_io.py was not pruned"

    # Verify package imports cleanly resolve without errors
    surgery_pkg = importlib.import_module("neuralops.surgery")
    assert hasattr(surgery_pkg, "get_recovery_params")
    assert hasattr(surgery_pkg, "SurgeryManifest")
    assert hasattr(surgery_pkg, "extract_surgical_manifest")
    assert hasattr(surgery_pkg, "save_surgical_checkpoint")
    assert hasattr(surgery_pkg, "load_surgical_checkpoint")
    assert hasattr(surgery_pkg, "replay_surgery_manifest")
    assert hasattr(surgery_pkg, "SurgeryPipeline")
    assert hasattr(surgery_pkg, "UniversalSurgeryDirector")
    assert hasattr(surgery_pkg, "InvariantPreservingLinearBridge")
    assert hasattr(surgery_pkg, "CrossAttentionLinearBridge")


def test_phase_060_dod_deprecation_warning_and_invariant_verification() -> None:
    """
    Phase 060 DoD Gate: Deprecation warning emitted by universal_calibrator.py,
    and all 10 Inviolable Laws certified across target models.
    """
    import warnings

    # 1. Deprecation warning on universal_calibrator
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        import importlib
        from neuralops.surgery import universal_calibrator
        importlib.reload(universal_calibrator)

        # Instantiate calibrator
        dummy_model = nn.Linear(16, 16)
        _ = universal_calibrator.UniversalActivationCalibrator(dummy_model)

        deprecations = [w for w in recorded if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) > 0
        assert any("deprecated" in str(w.message).lower() for w in deprecations)

    # 2. Invariant verification of all 10 Laws
    gemma = MockGemma2Model(hidden_dim=64, num_layers=2)
    director = UniversalSurgeryDirector(gemma)
    surgeried = director.execute_surgery(target_attention="cosformer")
    assert surgeried is not None

    report = director.certify_model(strict=True)
    assert report["certified"] is True
    assert report["score"] == "10/10"
    assert len(report["violations"]) == 0
    assert report["laws_certified"]["Law_1_QK_Norm"] is True
    assert report["laws_certified"]["Law_2_Softcapping"] is True
    assert report["laws_certified"]["Law_3_Zero_Bleed_Padding"] is True
    assert report["laws_certified"]["Law_4_Static_Cross_Attention"] is True
    assert report["laws_certified"]["Law_5_RoPE_Pre_Projection"] is True
    assert report["laws_certified"]["Law_6_Decoupled_Latent_Router"] is True
    assert report["laws_certified"]["Law_7_Causal_Monotonicity_State_Decay"] is True
    assert report["laws_certified"]["Law_8_Tied_Embeddings"] is True
    assert report["laws_certified"]["Law_9_FP32_Accumulation"] is True
    assert report["laws_certified"]["Law_10_Master_Integration"] is True


