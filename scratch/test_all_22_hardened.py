"""
Unified Test Suite for All 23 Hardened Attention Modules Across 4 Architectures:
- DistilBERT (Flat MHA)
- RoBERTa (Nested Encoder)
- GPT-2 (Causal Conv1D)
- Qwen (GQA with partial RoPE)
"""

import math
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, "Codes")

# Mock sources
class MockDistilBertAttn(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)


class MockRobertaSelf(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)


class MockRobertaOutput(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)


class MockRobertaAttn(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.self = MockRobertaSelf(d_model)
        self.output = MockRobertaOutput(d_model)


class MockGPT2Attn(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)


class MockQwenAttn(nn.Module):
    def __init__(self, d_model=768, num_heads=12, num_kv_heads=4):
        super().__init__()
        head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, num_heads * head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model)


class MockConfig:
    def __init__(self, hidden_size=768, num_attention_heads=12, num_key_value_heads=None, is_decoder=False):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.is_decoder = is_decoder
        self.dropout = 0.1
        self.n_heads = num_attention_heads
        self.dim = hidden_size


def test_all():
    from framework.attentions.chela_slay import ChelaSlayAttention
    from framework.attentions.gated_delta_net import GatedDeltaNet
    from framework.attentions.galerkin_attention import GalerkinAttention
    from framework.attentions.fusion_attn import FUSIONATTN
    from framework.attentions.flash_attention3_fp8 import FlashAttention3FP8
    from framework.attentions.flash_attention3 import FlashAttention3
    from framework.attentions.factorised_matrix_state import FactorisedMatrixState
    from framework.attentions.expert_choice_moe import ExpertChoiceMoEAttention
    from framework.attentions.decay_linear_attention import DecayLinearAttention
    from framework.attentions.cosformer import CosformerAttention
    from framework.attentions.wgqa_attention import WeightedGroupedQueryAttention
    from framework.attentions.svd_projection_attention import SVDProjectionAttention
    from framework.attentions.svd_llm_v2_attention import SVDLLMV2Attention
    from framework.attentions.svd_linear_attention import SVDLinearAttention as SVDLinearAttention1
    from framework.attentions.svd_compressed import SVDCompressedAttention
    from framework.attentions.svd_attention import SVDLinearAttention as SVDLinearAttention2
    from framework.attentions.squeezing_heads_distill import SqueezingHeadsDistiller
    from framework.attentions.spectron import SpectronAttention
    from framework.attentions.soa_attn import SOAAttention
    from framework.attentions.simple_ssm import SimpleSSM
    from framework.attentions.samba_moe import SambaMoEBlock
    from framework.attentions.hla_attention import HLAAttention
    from framework.attentions.hgrn2_outer import HGRN2OuterBlock

    all_modules = [
        ("ChelaSlayAttention", ChelaSlayAttention),
        ("GatedDeltaNet", GatedDeltaNet),
        ("GalerkinAttention", GalerkinAttention),
        ("FUSIONATTN", FUSIONATTN),
        ("FlashAttention3FP8", FlashAttention3FP8),
        ("FlashAttention3", FlashAttention3),
        ("FactorisedMatrixState", FactorisedMatrixState),
        ("ExpertChoiceMoEAttention", ExpertChoiceMoEAttention),
        ("DecayLinearAttention", DecayLinearAttention),
        ("CosformerAttention", CosformerAttention),
        ("WeightedGroupedQueryAttention", WeightedGroupedQueryAttention),
        ("SVDProjectionAttention", SVDProjectionAttention),
        ("SVDLLMV2Attention", SVDLLMV2Attention),
        ("SVDLinearAttention (decay)", SVDLinearAttention1),
        ("SVDCompressedAttention", SVDCompressedAttention),
        ("SVDLinearAttention (QR)", SVDLinearAttention2),
        ("SqueezingHeadsDistiller", SqueezingHeadsDistiller),
        ("SpectronAttention", SpectronAttention),
        ("SOAAttention", SOAAttention),
        ("SimpleSSM", SimpleSSM),
        ("SambaMoEBlock", SambaMoEBlock),
        ("HLAAttention", HLAAttention),
        ("HGRN2OuterBlock", HGRN2OuterBlock),
    ]

    B, N, D = 2, 8, 768
    x = torch.randn(B, N, D)
    mask = torch.ones(B, N)
    head_mask = torch.ones(12)

    rotary_dim = 64
    cos = torch.cos(torch.randn(1, N, rotary_dim))
    sin = torch.sin(torch.randn(1, N, rotary_dim))
    pos_emb = (cos, sin)

    total_passed = 0
    total_tests = 0

    print("=" * 80)
    print(f"VERIFYING ALL {len(all_modules)} HARDENED MODULES ACROSS 4 ARCHITECTURES")
    print("=" * 80)

    for name, cls in all_modules:
        print(f"\n>> {name}")

        # 1. DistilBERT
        try:
            old_db = MockDistilBertAttn(d_model=D)
            cfg_db = MockConfig(hidden_size=D, num_attention_heads=12, is_decoder=False)
            mod_db = cls(config=cfg_db)
            mod_db.copy_weights(old_db)
            out_db, _ = mod_db(x, attention_mask=mask, head_mask=head_mask)
            assert out_db.shape == (B, N, D), f"Bad shape: {out_db.shape}"
            assert not torch.isnan(out_db).any(), "NaN detected"
            print("   [PASS] DistilBERT")
            total_passed += 1
        except Exception as e:
            print(f"   [FAIL] DistilBERT: {e}")
        total_tests += 1

        # 2. RoBERTa Nested
        try:
            old_rob = MockRobertaAttn(d_model=D)
            cfg_rob = MockConfig(hidden_size=D, num_attention_heads=12, is_decoder=False)
            mod_rob = cls(config=cfg_rob)
            mod_rob.copy_weights(old_rob)
            out_rob, _ = mod_rob(x, attention_mask=mask)
            assert out_rob.shape == (B, N, D), f"Bad shape: {out_rob.shape}"
            assert not torch.isnan(out_rob).any(), "NaN detected"
            print("   [PASS] RoBERTa Nested")
            total_passed += 1
        except Exception as e:
            print(f"   [FAIL] RoBERTa Nested: {e}")
        total_tests += 1

        # 3. GPT-2 Causal Conv1D
        try:
            old_gpt2 = MockGPT2Attn(d_model=D)
            cfg_gpt2 = MockConfig(hidden_size=D, num_attention_heads=12, is_decoder=True)
            mod_gpt2 = cls(config=cfg_gpt2)
            mod_gpt2.copy_weights(old_gpt2)
            out_gpt2, _ = mod_gpt2(x, attention_mask=mask)
            assert out_gpt2.shape == (B, N, D), f"Bad shape: {out_gpt2.shape}"
            assert not torch.isnan(out_gpt2).any(), "NaN detected"
            print("   [PASS] GPT-2 Causal")
            total_passed += 1
        except Exception as e:
            print(f"   [FAIL] GPT-2 Causal: {e}")
        total_tests += 1

        # 4. Qwen GQA with partial RoPE
        try:
            old_qwen = MockQwenAttn(d_model=D, num_heads=12, num_kv_heads=4)
            cfg_qwen = MockConfig(hidden_size=D, num_attention_heads=12, num_key_value_heads=4, is_decoder=True)
            mod_qwen = cls(config=cfg_qwen)
            mod_qwen.copy_weights(old_qwen)
            out_qwen, _ = mod_qwen(x, attention_mask=mask, position_embeddings=pos_emb)
            assert out_qwen.shape == (B, N, D), f"Bad shape: {out_qwen.shape}"
            assert not torch.isnan(out_qwen).any(), "NaN detected"
            print("   [PASS] Qwen GQA + RoPE")
            total_passed += 1
        except Exception as e:
            print(f"   [FAIL] Qwen GQA + RoPE: {e}")
        total_tests += 1

    print("\n" + "=" * 80)
    print(f"OVERALL SUMMARY: {total_passed} / {total_tests} PASSED ({total_passed/total_tests*100:.1f}%)")
    print("=" * 80)
    return total_passed == total_tests


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
