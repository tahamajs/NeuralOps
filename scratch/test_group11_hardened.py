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

from framework.attentions.aegis_v1 import AegisAttentionV1
from framework.attentions.aegis_v2 import AegisAttentionV2
from framework.attentions.aegis_v3 import AegisAttention as AegisAttentionV3
from framework.attentions.aegis_v4 import AegisAttention as AegisAttentionV4
from framework.attentions.aegis_v5 import AegisAttention as AegisAttentionV5
from framework.attentions.aegis_v6 import AegisAttentionV6
from framework.attentions.aegis_hybrid import AegisHybridAttention
from framework.attentions.aegis_spectron_fp8 import AegisSpectronFP8Attention


def test_group11():
    modules = [
        ("AegisAttentionV1", AegisAttentionV1),
        ("AegisAttentionV2", AegisAttentionV2),
        ("AegisAttentionV3", AegisAttentionV3),
        ("AegisAttentionV4", AegisAttentionV4),
        ("AegisAttentionV5", AegisAttentionV5),
        ("AegisAttentionV6", AegisAttentionV6),
        ("AegisHybridAttention", AegisHybridAttention),
        ("AegisSpectronFP8Attention", AegisSpectronFP8Attention),
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
    print(f"VERIFYING GROUP 11 (AEGIS FAMILY): {len(modules)} MODULES ACROSS 4 ARCHITECTURES")
    print("=" * 80)

    for name, cls in modules:
        print(f"\n>> {name}")

        # 1. DistilBERT
        try:
            old_db = MockDistilBertAttn(d_model=D)
            cfg_db = MockConfig(hidden_size=D, num_attention_heads=12, is_decoder=False)
            try:
                mod_db = cls(config=cfg_db)
            except TypeError:
                mod_db = cls(model_config=cfg_db)
            if hasattr(mod_db, "copy_weights"):
                mod_db.copy_weights(old_db)
            elif hasattr(mod_db, "_copy_pretrained"):
                mod_db._copy_pretrained(old_db)
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
            try:
                mod_rob = cls(config=cfg_rob)
            except TypeError:
                mod_rob = cls(model_config=cfg_rob)
            if hasattr(mod_rob, "copy_weights"):
                mod_rob.copy_weights(old_rob)
            elif hasattr(mod_rob, "_copy_pretrained"):
                mod_rob._copy_pretrained(old_rob)
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
            try:
                mod_gpt2 = cls(config=cfg_gpt2)
            except TypeError:
                mod_gpt2 = cls(model_config=cfg_gpt2)
            if hasattr(mod_gpt2, "copy_weights"):
                mod_gpt2.copy_weights(old_gpt2)
            elif hasattr(mod_gpt2, "_copy_pretrained"):
                mod_gpt2._copy_pretrained(old_gpt2)
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
            try:
                mod_qwen = cls(config=cfg_qwen)
            except TypeError:
                mod_qwen = cls(model_config=cfg_qwen)
            if hasattr(mod_qwen, "copy_weights"):
                mod_qwen.copy_weights(old_qwen)
            elif hasattr(mod_qwen, "_copy_pretrained"):
                mod_qwen._copy_pretrained(old_qwen)
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
    success = test_group11()
    sys.exit(0 if success else 1)
