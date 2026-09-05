import sys
import torch
import torch.nn as nn

sys.path.insert(0, "Codes")
sys.path.insert(0, ".")

from scratch.test_all_22_hardened import (
    MockDistilBertAttn,
    MockRobertaAttn,
    MockGPT2Attn,
    MockQwenAttn,
    MockConfig,
)

from framework.attentions.moba_attention import MoBAAttention
from framework.attentions.salt_attention import SALTAttention
from framework.attentions.svd_attention import SVDLinearAttention
from framework.attentions.gated_linear_attention import GatedLinearAttention
from framework.attentions.mla_attention import MLAAttention
from framework.attentions.mixattention_moska import MixAttentionMoSKA
from framework.attentions.nasc_attention import NASCAttention
from framework.attentions.nexus import NEXUSAttention


def test_group9():
    modules = [
        ("MoBAAttention", MoBAAttention),
        ("SALTAttention", SALTAttention),
        ("SVDLinearAttention", SVDLinearAttention),
        ("GatedLinearAttention", GatedLinearAttention),
        ("MLAAttention", MLAAttention),
        ("MixAttentionMoSKA", MixAttentionMoSKA),
        ("NASCAttention", NASCAttention),
        ("NEXUSAttention", NEXUSAttention),
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
    print(f"VERIFYING GROUP 9: {len(modules)} MODULES ACROSS 4 ARCHITECTURES")
    print("=" * 80)

    for name, cls in modules:
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

        # 4. Qwen GQA + RoPE
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
    print(f"SUMMARY: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
    print("=" * 80)
    return total_passed == total_tests


if __name__ == "__main__":
    success = test_group9()
    sys.exit(0 if success else 1)
