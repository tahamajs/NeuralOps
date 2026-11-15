#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive test suite for W4-L4 Specialized Attention Layers."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

# Specialized Layers
import neuralops.layers.adaptive_span as adaptive_span
import neuralops.layers.attention_wrapper as attention_wrapper
import neuralops.layers.bitnet_attention as bitnet_attn
import neuralops.layers.cosformer_hybrid as cosformer_hybrid
import neuralops.layers.cross_attention_hybrid as cross_attn_hybrid
import neuralops.layers.deepseek_mla as deepseek_mla
import neuralops.layers.falcon_attention as falcon_attn
import neuralops.layers.gemma2_attention as gemma2_attn
import neuralops.layers.gemma_attention as gemma_attn
import neuralops.layers.gla_hybrid as gla_hybrid
import neuralops.layers.gpt2_attention as gpt2_attn
import neuralops.layers.griffin as griffin
import neuralops.layers.griffin_ssd as griffin_ssd
import neuralops.layers.hawk as hawk
import neuralops.layers.hgrn_bitnet as hgrn_bitnet
import neuralops.layers.hybrid_attention as hybrid_attention
import neuralops.layers.hymba as hymba
import neuralops.layers.jamba as jamba
import neuralops.layers.jamba_hybrid_gate as jamba_hybrid_gate
import neuralops.layers.llama_attention as llama_attn
import neuralops.layers.mamba_inject as mamba_inject
import neuralops.layers.megalodon as megalodon
import neuralops.layers.mistral_attention as mistral_attn
import neuralops.layers.mixture_of_attentions as mixture_of_attentions
import neuralops.layers.mla as mla
import neuralops.layers.moa as moa
import neuralops.layers.moe_attention as moe_attention
import neuralops.layers.moe_deltakv_paged as moe_deltakv_paged
import neuralops.layers.perceiver as perceiver
import neuralops.layers.perceiver_resampler as perceiver_resampler
import neuralops.layers.persimmon as persimmon
import neuralops.layers.phimoe as phimoe
import neuralops.layers.qwen2_attention as qwen2_attn
import neuralops.layers.qwen_attention as qwen_attn
import neuralops.layers.recurrent_gemma as recurrent_gemma
import neuralops.layers.router_attention as router_attention
import neuralops.layers.samba as samba
import neuralops.layers.sambamoe as sambamoe
import neuralops.layers.t5_attention as t5_attn
import neuralops.layers.ternary_attention as ternary_attn
import neuralops.layers.transformer_mamba as transformer_mamba
import neuralops.layers.universal_adapter as universal_adapter
import neuralops.layers.zamba as zamba


class MockConfig(PretrainedConfig):
    """Synthetic configuration for layer testing."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        head_dim: int = 16,
        is_decoder: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.d_model = hidden_size
        self.is_decoder = is_decoder
        for k, v in kwargs.items():
            setattr(self, k, v)


def _extract_tensor(out: Any) -> torch.Tensor:
    """Helper to extract primary tensor from output."""
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict):
        return out.get("context", out.get("out", out.get("hidden_states")))
    return out


# =============================================================================
# 1. MLA and DeepSeek MLA Verification Suite
# =============================================================================


class TestMLALayers:
    def test_mla_imports_and_forward(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        layer = mla.MLAAttention(config=cfg, rank=16)
        x = torch.randn(2, 8, 64)
        out = _extract_tensor(layer(x))
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_deepseek_mla_bridge_threading(self) -> None:
        bridge = deepseek_mla.DeepSeekMLASurgeryBridge(latent_kv_dim=64, decoupled_rope_dim=32, q_latent_dim=128)
        state = bridge.init_latent_state(batch_size=2, device=torch.device("cpu"))
        assert isinstance(state, deepseek_mla.DualStreamLatentState)
        assert state.state_c.shape == (2, 64, 128)
        assert state.state_r.shape == (2, 32, 32)


# =============================================================================
# 2. Jamba and Jamba Hybrid Gating Suite
# =============================================================================


class TestJambaLayers:
    def test_jamba_blocks(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        block = jamba.Jamba2Block(config=cfg, dim=64)
        x = torch.randn(2, 8, 64)
        out = _extract_tensor(block(x))
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_jamba_hybrid_gate(self) -> None:
        gate = jamba_hybrid_gate.JambaHybridGate(num_heads=4)
        attn_out = torch.randn(2, 8, 64)
        ssm_out = torch.randn(2, 8, 64)
        out = gate(attn_out, ssm_out)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()


# =============================================================================
# 3. MoA and MoE Attention Suite
# =============================================================================


class TestMoAAndMoELayers:
    def test_moa_and_mixture_of_attentions(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        moa_layer = moa.MixtureOfAttentions(config=cfg, layer_idx=0)
        x = torch.randn(2, 8, 64)
        out = _extract_tensor(moa_layer(x))
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_moe_attention_facade(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        moe_layer = moe_attention.MoEAttention(config=cfg, hidden_size=64)
        x = torch.randn(2, 8, 64)
        out = _extract_tensor(moe_layer(x))
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_phimoe_attention(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        phi_layer = phimoe.PhiMoEAttention(config=cfg, hidden_size=64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        out = phi_layer(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()
        # Verify aux_loss is a valid finite tensor
        assert phi_layer.aux_loss >= 0.0


# =============================================================================
# 4. Hybrid Attention Variants Suite
# =============================================================================


class TestHybridVariants:
    def test_cosformer_hybrid(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        layer = cosformer_hybrid.CosformerHybridAttention(config=cfg)
        x = torch.randn(2, 8, 64)
        out = _extract_tensor(layer(x))
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_gla_hybrid(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        layer = gla_hybrid.GLAHybridAttention(config=cfg)
        x = torch.randn(2, 8, 64)
        out = _extract_tensor(layer(x))
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_transformer_mamba(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        layer = transformer_mamba.TransformerMambaBlock(config=cfg, hidden_size=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_samba_and_sambamoe(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        samba_block = samba.SambaBlock(config=cfg, d_model=64)
        sambamoe_block = sambamoe.SambaMoE(config=cfg, d_model=64, n_experts=2)
        x = torch.randn(2, 8, 64)

        out_samba = _extract_tensor(samba_block(x))
        assert out_samba.shape == (2, 8, 64)
        assert torch.isfinite(out_samba).all()

        out_moe = _extract_tensor(sambamoe_block(x))
        assert out_moe.shape == (2, 8, 64)
        assert torch.isfinite(out_moe).all()

    def test_zamba(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        block = zamba.ZambaBlock(config=cfg, hidden_size=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()


# =============================================================================
# 5. Griffin, Hawk, and RecurrentGemma Suite
# =============================================================================


class TestGriffinHawkRecurrentGemma:
    def test_griffin(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        block = griffin.GriffinBlock(config=cfg, dim=64)
        x = torch.randn(2, 8, 64)
        out = _extract_tensor(block(x))
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_hawk_recurrence(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        layer = hawk.HawkAttention(config=cfg, hidden_size=64)
        x = torch.randn(2, 8, 64)
        out, state = layer(x, use_cache=True)
        assert out.shape == (2, 8, 64)
        assert state.shape == (2, layer.d_inner)
        assert torch.isfinite(out).all()

    def test_recurrent_gemma(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        layer = recurrent_gemma.RecurrentGemmaAttention(config=cfg, hidden_size=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()


# =============================================================================
# 6. Megalodon Complex EMA Suite
# =============================================================================


class TestMegalodon:
    def test_megalodon_forward(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        layer = megalodon.MegalodonAttention(config=cfg, hidden_size=64, num_heads=4, ema_dim=16)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()


# =============================================================================
# 7. BitNet, Ternary, and HGRN BitNet Suite (with STE Verification)
# =============================================================================


class TestBitNetAndTernary:
    def test_bitnet_quantization_and_ste_gradients(self) -> None:
        w = nn.Parameter(torch.randn(32, 32))
        x = torch.randn(2, 8, 32)
        w_quant = bitnet_attn.weight_quant_ste(w)

        # Verify values are ternary {-1, 0, 1} scaled
        scale = w.abs().mean().clamp(min=1e-8)
        ternary_vals = torch.round(w_quant / scale)
        unique_vals = torch.unique(ternary_vals)
        for u in unique_vals:
            assert u.item() in (-1.0, 0.0, 1.0)

        # Verify STE gradient flow
        loss = (w_quant * 2.0).sum()
        loss.backward()
        assert w.grad is not None
        assert torch.isfinite(w.grad).all()
        assert (w.grad != 0.0).any()

    def test_bitnet_and_ternary_attention_forward(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        bit_layer = bitnet_attn.BitNetAttention(config=cfg, hidden_size=64, num_heads=4)
        ternary_layer = ternary_attn.TernaryAttention(config=cfg, hidden_size=64, num_heads=4)
        hgrn_bit_layer = hgrn_bitnet.HGRNBitNetAttention(config=cfg, hidden_size=64)

        x = torch.randn(2, 8, 64)
        assert bit_layer(x).shape == (2, 8, 64)
        assert ternary_layer(x).shape == (2, 8, 64)
        assert hgrn_bit_layer(x).shape == (2, 8, 64)


# =============================================================================
# 8. Adaptive Span, Persimmon, Perceiver, Cross Attention Hybrid Suite
# =============================================================================


class TestSpecializedArchitectures:
    def test_adaptive_span(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        layer = adaptive_span.AdaptiveSpanAttention(config=cfg)
        x = torch.randn(2, 8, 64)
        out = _extract_tensor(layer(x))
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_persimmon(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        layer = persimmon.PersimmonAttention(config=cfg, hidden_size=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_perceiver_and_resampler(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        perc = perceiver.PerceiverAttention(config=cfg, dim=64, num_heads=4, num_latents=16)
        resampler = perceiver_resampler.PerceiverResampler(config=cfg, dim=64, num_heads=4, num_latents=16)

        context = torch.randn(2, 32, 64)
        out_p = perc(context)
        assert out_p.shape == (2, 16, 64)
        assert torch.isfinite(out_p).all()

        out_r = resampler(context)
        assert out_r.shape == (2, 16, 64)
        assert torch.isfinite(out_r).all()

    def test_cross_attention_hybrid(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        layer = cross_attn_hybrid.CrossAttentionHybrid(config=cfg, hidden_size=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        enc = torch.randn(2, 12, 64)
        out = layer(x, encoder_hidden_states=enc)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()


# =============================================================================
# 9. Model-Specific Attentions Suite (LLaMA, Mistral, Qwen, Gemma, Falcon, GPT2, T5)
# =============================================================================


class TestModelSpecificAttentions:
    def test_llama_attention_contract(self) -> None:
        # LLaMA-3 style GQA: 32 query heads, 8 kv heads, head_dim=128, hidden_size=4096
        cfg = MockConfig(hidden_size=64, num_attention_heads=8, num_key_value_heads=2, head_dim=8)
        layer = llama_attn.LlamaAttention(config=cfg)
        assert layer.num_heads == 8
        assert layer.num_kv_heads == 2
        assert layer.head_dim == 8

        x = torch.randn(2, 8, 64)
        out, attn_w, kv = layer(x, use_cache=True)
        assert out.shape == (2, 8, 64)
        assert kv is not None
        assert kv[0].shape == (2, 2, 8, 8)  # [B, num_kv_heads, L, head_dim]

    def test_mistral_attention_sliding_window(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=8, num_key_value_heads=2, sliding_window=4)
        layer = mistral_attn.MistralAttention(config=cfg, hidden_size=64, num_heads=8, num_kv_heads=2, sliding_window=4)
        assert layer.sliding_window == 4

        x = torch.randn(2, 8, 64)
        out, _, _ = layer(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_qwen2_attention_contract(self) -> None:
        # Qwen2.5-0.5B style: 14 query heads, 2 kv heads, head_dim=64, hidden_size=896
        cfg = MockConfig(hidden_size=896, num_attention_heads=14, num_key_value_heads=2, head_dim=64)
        layer = qwen2_attn.Qwen2Attention(config=cfg)
        assert layer.num_heads == 14
        assert layer.num_kv_heads == 2
        assert layer.head_dim == 64

        x = torch.randn(2, 4, 896)
        out, _, kv = layer(x, use_cache=True)
        assert out.shape == (2, 4, 896)
        assert kv is not None

    def test_gemma2_attention_softcapping(self) -> None:
        cfg = MockConfig(
            hidden_size=128, num_attention_heads=4, num_key_value_heads=1, head_dim=32, attn_logit_softcapping=50.0
        )
        layer = gemma2_attn.Gemma2Attention(config=cfg)
        assert layer.soft_capping == 50.0

        x = torch.randn(2, 8, 128)
        out, _, _ = layer(x)
        assert out.shape == (2, 8, 128)
        assert torch.isfinite(out).all()

    def test_falcon_attention(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=8, num_kv_heads=1)
        layer = falcon_attn.FalconAttention(config=cfg, hidden_size=64, num_heads=8, num_kv_heads=1)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_gpt2_attention(self) -> None:
        cfg = MockConfig(hidden_size=64, num_attention_heads=4)
        layer = gpt2_attn.GPT2Attention(config=cfg, hidden_size=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out, _, _ = layer(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_t5_attention_relative_bias(self) -> None:
        cfg = MockConfig(d_model=64, num_heads=4, is_decoder=False)
        layer = t5_attn.T5Attention(config=cfg, d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out, pos_bias = layer(x)
        assert out.shape == (2, 8, 64)
        assert pos_bias is not None
        assert pos_bias.shape == (1, 4, 8, 8)
        assert torch.isfinite(out).all()


# =============================================================================
# 10. Shims and Re-exports Suite
# =============================================================================


class TestShimsAndReexports:
    def test_universal_adapter_and_wrapper_shims(self) -> None:
        assert hasattr(universal_adapter, "UniversalAdapter")
        assert hasattr(attention_wrapper, "ReplaceAttention")
        assert hasattr(attention_wrapper, "UniversalAttentionWrapper")

    def test_sparse_and_ssm_shims(self) -> None:
        assert hasattr(moe_deltakv_paged, "MoEDeltaKVPagedAttention")
        assert hasattr(router_attention, "RoutingAttention")
        assert hasattr(mamba_inject, "MambaBlock")
        assert hasattr(hymba, "HymbaTransXBlock")
        assert hasattr(griffin_ssd, "GriffinSSDBlock")
        assert hasattr(hybrid_attention, "HybridAttention")
