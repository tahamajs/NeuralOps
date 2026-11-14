#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Smoke and unit test suite for Specialized Attentions & Adapters (Wave 2 - Section 3.12)."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

# Adapters
from neuralops.adapters.base import ArchitectureAdapter
from neuralops.adapters.distilbert import DistilBertAdapter
from neuralops.adapters.generic import GenericAdapter
from neuralops.adapters.gpt2 import Gpt2Adapter
from neuralops.adapters.llama import LlamaAdapter
from neuralops.core._attention_wrapper import ReplaceAttention
from neuralops.core.adapters import UniversalAdapter
from neuralops.layers.bert_adapter import BertAttentionAdapter
from neuralops.layers.hybrid.gated_hybrid_kernel_attention import GatedHybridKernelAttention
from neuralops.layers.hybrid.granite_hybrid import GraniteHybrid, GraniteHybridBlock
from neuralops.layers.hybrid.halo_regla import HALOReGLAAttention, HaloReGLAAttention
from neuralops.layers.hybrid.hybrid_attention import HybridAttention
from neuralops.layers.hybrid.mixattention_moska import MixAttentionMoSKA, MixAttentionMoska
from neuralops.layers.hybrid.optimized_hybrid_attention import OptimizedHybridAttention
from neuralops.layers.hybrid.positive_kernel_linear_attention import PositiveKernelLinearAttention
from neuralops.layers.hybrid.regla_attention import ReGLAAttention
from neuralops.layers.hybrid.split_head_hybrid_attention import SplitHeadHybridAttention
from neuralops.layers.hybrid.tri_branch_hybrid_attention import TriBranchHybridAttention
from neuralops.layers.linear.cosformer import CosformerAttention
from neuralops.layers.linear.gla_attention import GLAttention, GatedLinearAttention

# MLA & Surgery
from neuralops.layers.modern.mla_attention import MLAAttention
from neuralops.layers.qwen_adapter import QwenAttentionAdapter
from neuralops.layers.sparse.bam_umoe import BAMUMoEBlock
from neuralops.layers.sparse.expert_choice_moe import ExpertChoiceMoEAttention
from neuralops.layers.sparse.moe_deltakv_paged import MoEDeltaKVPagedAttention
from neuralops.layers.sparse.reap_moe import REAPMoEAttention
from neuralops.layers.ssm.griffin_ssd import GriffinSSDBlock
from neuralops.layers.ssm.hymba_transx import HymbaTransXBlock

# SSM & Hybrid Blocks
from neuralops.layers.ssm.jamba import Jamba2Block
from neuralops.layers.ssm.jamba_info import JambaInfoBlock
from neuralops.layers.ssm.samba_m3 import SAMBAM3Block
from neuralops.layers.ssm.samba_moe import SambaMoE, SambaMoEBlock
from neuralops.surgery.mla_surgery import (
    DeepSeekMLASurgeryBridge,
    DualStreamLatentState,
    apply_rotary_pos_emb,
)

# MoA & Routing
from neuralops.surgery.moa_router import (
    MoARouter,
    compute_attention_entropy,
    profile_layer_sensitivity,
)
from neuralops.utils.replace_attention import UniversalAttentionWrapper


class DummyConfig(PretrainedConfig):
    """Standard synthetic configuration for specialized attention testing."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 4,
        is_decoder: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.d_model = hidden_size
        self.is_decoder = is_decoder
        for k, v in kwargs.items():
            setattr(self, k, v)


def _unpack(out: Any) -> torch.Tensor:
    """Helper unpacking layer outputs across tuple/dict formats."""
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict):
        return out.get("context", out.get("out", out.get("hidden_states")))
    return out


@pytest.fixture(autouse=True)
def set_seed() -> None:
    """Ensure deterministic random initialization across all tests."""
    torch.manual_seed(42)


# =============================================================================
# 1. Multi-Head Latent Attention (MLA) Verification Suite
# =============================================================================


class TestMLASpecialized:
    """Tests for MLAAttention and DeepSeekMLASurgeryBridge."""

    def test_mla_attention_causal_forward(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4, is_decoder=True)
        layer = MLAAttention(config=cfg, rank=32)
        x = torch.randn(2, 16, 64)
        out = _unpack(layer(x, causal=True))
        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()

    def test_mla_attention_bidirectional_forward(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4, is_decoder=False)
        layer = MLAAttention(config=cfg, rank=32)
        x = torch.randn(2, 8, 64)
        out = _unpack(layer(x, causal=False))
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_deepseek_mla_bridge_compression_ratio(self) -> None:
        bridge = DeepSeekMLASurgeryBridge(latent_kv_dim=512, decoupled_rope_dim=64)
        bytes_per_token = bridge.get_mla_cache_bytes_per_token(bytes_per_elem=2)
        assert bytes_per_token == (512 + 64) * 2  # 1152 bytes
        ratio = bridge.get_theoretical_compression_ratio(standard_mha_bytes_per_token=8192)
        assert abs(ratio - 0.859375) < 1e-4
        assert bridge.verify_phase_061_compression_dod(standard_mha_bytes_per_token=8192)

    def test_deepseek_mla_bridge_dual_stream_state(self) -> None:
        bridge = DeepSeekMLASurgeryBridge(latent_kv_dim=512, decoupled_rope_dim=64, q_latent_dim=128)
        summary = bridge.get_recurrent_state_memory_summary()
        assert summary["recurrent_state_reduction_ratio"] >= 0.50
        assert bridge.verify_phase_062_recurrent_state_dod()

        init_state = bridge.init_latent_state(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
        assert init_state.content_state.shape == (2, 512, 128)
        assert init_state.rope_state.shape == (2, 64, 64)

    def test_deepseek_mla_bridge_rope_norm_preservation(self) -> None:
        batch_size, seq_len, dim = 2, 8, 64
        x = torch.randn(batch_size, seq_len, dim)
        freqs = torch.arange(0, dim // 2, dtype=torch.float32)
        angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), freqs)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        out = apply_rotary_pos_emb(x, cos, sin)
        norm_orig = torch.norm(x, dim=-1)
        norm_rot = torch.norm(out, dim=-1)
        torch.testing.assert_close(norm_rot, norm_orig, rtol=1e-4, atol=1e-4)

    def test_deepseek_mla_bridge_prefill_and_step_decode(self) -> None:
        latent_kv_dim = 64
        decoupled_rope_dim = 32
        bridge = DeepSeekMLASurgeryBridge(
            latent_kv_dim=latent_kv_dim,
            decoupled_rope_dim=decoupled_rope_dim,
            decay=0.99,
        )
        batch_size, seq_len, chunk_size = 2, 16, 8
        c_kv = torch.randn(batch_size, seq_len, latent_kv_dim)
        k_rope = torch.randn(batch_size, seq_len, decoupled_rope_dim)
        qc = torch.randn(batch_size, seq_len, latent_kv_dim)
        qr = torch.randn(batch_size, seq_len, decoupled_rope_dim)

        out, final_state = bridge.forward_chunk_scan(
            c_kv=c_kv,
            k_rope=k_rope,
            q_content=qc,
            q_rope=qr,
            chunk_size=chunk_size,
        )
        assert out.shape == (batch_size, seq_len, latent_kv_dim)
        assert torch.isfinite(out).all()
        assert isinstance(final_state, DualStreamLatentState)

        # Single step decode
        step_out, next_state = bridge.forward_step(
            c_kv[:, 0:1],
            k_rope[:, 0:1],
            qc[:, 0:1],
            qr[:, 0:1],
            final_state.state_c,
            final_state.state_r,
        )
        assert step_out.shape == (batch_size, 1, latent_kv_dim)
        assert torch.isfinite(step_out).all()


# =============================================================================
# 2. SSM and Hybrid Attention Recurrence Suite
# =============================================================================


class TestSSMAndHybridRecurrence:
    """Tests for hybrid SSM and recurrent layers."""

    @pytest.mark.parametrize(
        "cls_name,cls_type,kwargs",
        [
            ("Jamba2Block", Jamba2Block, {"dim": 64}),
            ("JambaInfoBlock", JambaInfoBlock, {"dim": 64}),
            ("SAMBAM3Block", SAMBAM3Block, {"d_model": 64}),
            ("SambaMoE", SambaMoE, {"d_model": 64, "n_experts": 2}),
            ("GriffinSSDBlock", GriffinSSDBlock, {"dim": 64}),
            ("HymbaTransXBlock", HymbaTransXBlock, {"hidden_size": 64}),
        ],
    )
    def test_ssm_hybrid_block_forward(self, cls_name: str, cls_type: type[nn.Module], kwargs: dict[str, Any]) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        block = cls_type(config=cfg, **kwargs)
        x = torch.randn(2, 12, 64)
        out = _unpack(block(x))
        assert out.shape == (2, 12, 64)
        assert torch.isfinite(out).all()


# =============================================================================
# 3. Hybrid Attention Family Suite
# =============================================================================


class TestHybridAttentionFamily:
    """Tests for various hybrid attention variants."""

    def test_hybrid_attention_convex_blend(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        layer = HybridAttention(config=cfg)
        x = torch.randn(2, 16, 64)
        out = _unpack(layer(x))
        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()

    def test_split_head_hybrid_attention(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        layer = SplitHeadHybridAttention(config=cfg)
        x = torch.randn(2, 16, 64)
        out = _unpack(layer(x))
        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()

    def test_tri_branch_hybrid_attention(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        layer = TriBranchHybridAttention(config=cfg)
        x = torch.randn(2, 16, 64)
        out = _unpack(layer(x))
        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()

    def test_gated_hybrid_kernel_attention(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        layer = GatedHybridKernelAttention(config=cfg)
        x = torch.randn(2, 16, 64)
        out = _unpack(layer(x))
        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()

    def test_optimized_hybrid_attention(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        layer = OptimizedHybridAttention(config=cfg)
        x = torch.randn(2, 16, 64)
        out = _unpack(layer(x))
        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()

    def test_granite_hybrid(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        layer = GraniteHybrid(config=cfg, dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = _unpack(layer(x))
        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()

    def test_regla_and_halo_regla(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        regla = ReGLAAttention(config=cfg)
        halo = HaloReGLAAttention(config=cfg)
        x = torch.randn(2, 16, 64)

        out_regla = _unpack(regla(x))
        assert out_regla.shape == (2, 16, 64)
        assert torch.isfinite(out_regla).all()

        out_halo = _unpack(halo(x))
        assert out_halo.shape == (2, 16, 64)
        assert torch.isfinite(out_halo).all()

    def test_positive_kernel_and_mixattention(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        pos = PositiveKernelLinearAttention(config=cfg)
        mix = MixAttentionMoska(config=cfg)
        x = torch.randn(2, 16, 64)

        out_pos = _unpack(pos(x))
        assert out_pos.shape == (2, 16, 64)
        assert torch.isfinite(out_pos).all()

        out_mix = _unpack(mix(x))
        assert out_mix.shape == (2, 16, 64)
        assert torch.isfinite(out_mix).all()

    def test_cosformer_and_gla(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        cos = CosformerAttention(config=cfg)
        gla = GLAttention(config=cfg, hidden_size=64, num_heads=4)
        x = torch.randn(2, 16, 64)

        out_cos = _unpack(cos(x))
        assert out_cos.shape == (2, 16, 64)
        assert torch.isfinite(out_cos).all()

        out_gla = _unpack(gla(x))
        assert out_gla.shape == (2, 16, 64)
        assert torch.isfinite(out_gla).all()


# =============================================================================
# 4. MoE, MoA & Routing Suite
# =============================================================================


class TestMoEAndRouting:
    """Tests for Mixture of Attentions (MoA) and MoE layers."""

    def test_moa_router_canonical_partition(self) -> None:
        router = MoARouter(num_layers=12)
        assert router.num_layers == 12
        assert 0 in router.full_attention_layers
        assert 11 in router.full_attention_layers
        assert len(router.get_linear_attention_layers()) > 0

        # Boundary layers: 0-3 and 10-11
        for idx in [0, 1, 2, 3, 10, 11]:
            assert router.is_full_attention(idx)

        # Intermediate layers: 4-9
        for idx in range(4, 10):
            assert router.is_linear_attention(idx)

    def test_moa_entropy_profiling(self) -> None:
        B, H, T = 2, 4, 16
        # Uniform attention -> maximum entropy ln(T)
        uniform_attn = torch.full((B, H, T, T), 1.0 / T)
        entropy_uniform = compute_attention_entropy(uniform_attn)
        expected_entropy = math.log(T)
        assert abs(entropy_uniform - expected_entropy) < 1e-3

        # One-hot attention -> zero entropy
        one_hot_attn = torch.eye(T).unsqueeze(0).unsqueeze(0).expand(B, H, T, T)
        entropy_one_hot = compute_attention_entropy(one_hot_attn)
        assert abs(entropy_one_hot - 0.0) < 1e-3

        matrices = [uniform_attn for _ in range(8)]
        profile = profile_layer_sensitivity(matrices)
        assert profile.num_layers == 8
        assert len(profile.entropies) == 8

    def test_sparse_moe_attentions(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        expert_choice = ExpertChoiceMoEAttention(config=cfg, d_model=64)
        reap_moe = REAPMoEAttention(config=cfg, hidden_size=64)
        moe_deltakv = MoEDeltaKVPagedAttention(config=cfg, hidden_size=64)
        bam_umoe = BAMUMoEBlock(config=cfg, hidden_size=64, svd_rank=8)

        x = torch.randn(2, 16, 64)

        out_ec = _unpack(expert_choice(x))
        assert out_ec.shape == (2, 16, 64)
        assert torch.isfinite(out_ec).all()

        out_reap = _unpack(reap_moe(x))
        assert out_reap.shape == (2, 16, 64)
        assert torch.isfinite(out_reap).all()

        out_delta = _unpack(moe_deltakv(x))
        assert out_delta.shape == (2, 16, 64)
        assert torch.isfinite(out_delta).all()

        out_bam = _unpack(bam_umoe(x))
        assert out_bam.shape == (2, 16, 64)
        assert torch.isfinite(out_bam).all()


# =============================================================================
# 5. Adapters & Forward Signature Contracts Suite
# =============================================================================


class TestAdaptersAndContracts:
    """Tests verifying adapter layers strictly preserve forward contracts."""

    def test_qwen_adapter_forward_contract(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        core = HybridAttention(config=cfg)
        adapter = QwenAttentionAdapter(core_attention=core, embed_dim=64, num_heads=4, num_kv_heads=2)

        x = torch.randn(2, 12, 64)
        mask = torch.ones(2, 12, dtype=torch.long)
        out = adapter(x, attention_mask=mask)

        # Expected return: 3-tuple (context, attn_weights, kv_cache)
        assert isinstance(out, tuple)
        assert len(out) == 3
        context, attn_weights, kv = out
        assert context.shape == (2, 12, 64)
        assert torch.isfinite(context).all()

    def test_bert_adapter_forward_contract(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        core = HybridAttention(config=cfg)
        adapter = BertAttentionAdapter(core_attention=core, embed_dim=64, num_heads=4)

        x = torch.randn(2, 12, 64)
        mask = torch.ones(2, 12, dtype=torch.long)
        out = adapter(x, attention_mask=mask)

        # Expected return: 2-tuple (context, attn_weights)
        assert isinstance(out, tuple)
        assert len(out) == 2
        context, attn_weights = out
        assert context.shape == (2, 12, 64)
        assert torch.isfinite(context).all()

    def test_universal_attention_wrapper(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        core = HybridAttention(config=cfg)
        wrapper = UniversalAttentionWrapper(core)

        x = torch.randn(2, 12, 64)
        out, _ = wrapper(x)
        assert out.shape == (2, 12, 64)
        assert torch.isfinite(out).all()

        # Test with query, key, value keyword arguments
        out_qkv, _ = wrapper(query=x, key=x, value=x)
        assert out_qkv.shape == (2, 12, 64)
        assert torch.isfinite(out_qkv).all()

    def test_replace_attention(self) -> None:
        cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
        rep = ReplaceAttention(config=cfg, rank=16)
        x = torch.randn(2, 12, 64)
        out = _unpack(rep(x))
        assert out.shape == (2, 12, 64)
        assert torch.isfinite(out).all()

    def test_universal_adapter_projection_extraction(self) -> None:
        class MockAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 32)
                self.v_proj = nn.Linear(64, 32)
                self.o_proj = nn.Linear(64, 64)

        mock = MockAttn()
        wq, bq = UniversalAdapter.get_weight_bias(mock, "q")
        wk, bk = UniversalAdapter.get_weight_bias(mock, "k")
        wv, bv = UniversalAdapter.get_weight_bias(mock, "v")
        wo, bo = UniversalAdapter.get_weight_bias(mock, "o")

        assert wq.shape == (64, 64)
        assert wk.shape == (32, 64)
        assert wv.shape == (32, 64)
        assert wo.shape == (64, 64)

    def test_architecture_adapters_introspect(self) -> None:
        class MockLlamaModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = PretrainedConfig()
                self.config.model_type = "llama"
                self.config.hidden_size = 64
                self.config.num_hidden_layers = 2
                self.config.num_attention_heads = 4

        model = MockLlamaModel()
        adapter = LlamaAdapter()
        assert adapter.can_handle(model)
        assert adapter.detect_family(model) == "llama"
        info = adapter.inspect(model)
        assert info["family"] == "llama"
        assert info["hidden_size"] == 64
