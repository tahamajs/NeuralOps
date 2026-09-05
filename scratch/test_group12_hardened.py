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

from framework.attentions.eat import EATBlock
from framework.attentions.goldfinch_ahn import GoldFinchAHNBlock
from framework.attentions.granite_hybrid import GraniteHybridBlock
from framework.attentions.griffin_ssd import GriffinSSDBlock
from framework.attentions.hgrn2_gated_deltanet import HGRN2GatedDeltaNet
from framework.attentions.hgrn2_layer import HGRN2Layer
from framework.attentions.mamba import Mamba
from framework.attentions.mamba_svd_flash import MambaSVDFlash
from framework.attentions.moe_deltakv_paged import MoEDeltaKVPagedAttention
from framework.attentions.samba_m3 import SAMBAM3Block


def test_group12():
    modules = [
        ("EATBlock", EATBlock),
        ("GoldFinchAHNBlock", GoldFinchAHNBlock),
        ("GraniteHybridBlock", GraniteHybridBlock),
        ("GriffinSSDBlock", GriffinSSDBlock),
        ("HGRN2GatedDeltaNet", HGRN2GatedDeltaNet),
        ("HGRN2Layer", HGRN2Layer),
        ("Mamba", Mamba),
        ("MambaSVDFlash", MambaSVDFlash),
        ("MoEDeltaKVPagedAttention", MoEDeltaKVPagedAttention),
        ("SAMBAM3Block", SAMBAM3Block),
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
    print(f"VERIFYING GROUP 12 (HYBRID SSM/MAMBA): {len(modules)} MODULES ACROSS 4 ARCHITECTURES")
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
            res_db = mod_db(x, attention_mask=mask, head_mask=head_mask)
            out_db = res_db[0] if isinstance(res_db, (tuple, list)) else res_db
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
            res_rob = mod_rob(x, attention_mask=mask)
            out_rob = res_rob[0] if isinstance(res_rob, (tuple, list)) else res_rob
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
            res_gpt2 = mod_gpt2(x, attention_mask=mask)
            out_gpt2 = res_gpt2[0] if isinstance(res_gpt2, (tuple, list)) else res_gpt2
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
            res_qwen = mod_qwen(x, attention_mask=mask, position_embeddings=pos_emb)
            out_qwen = res_qwen[0] if isinstance(res_qwen, (tuple, list)) else res_qwen
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
    success = test_group12()
    sys.exit(0 if success else 1)
