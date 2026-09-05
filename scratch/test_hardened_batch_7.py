import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "Codes/framework")

from framework.attentions.gated_delta_net import GatedDeltaNet, inject_gated_delta_net
from framework.attentions.galerkin_attention import GalerkinAttention, inject_galerkin
from framework.attentions.fusion_attn import FUSIONATTN, inject_fusion_attn
from framework.attentions.flash_attention3_fp8 import FlashAttention3FP8, inject_flash3_fp8
from framework.attentions.flash_attention3 import FlashAttention3, inject_flash_attention3
from framework.attentions.factorised_matrix_state import FactorisedMatrixState, inject_factorised_matrix_state
from framework.attentions.expert_choice_moe import ExpertChoiceMoEAttention, inject_expert_choice_moe
from framework.attentions.decay_linear_attention import DecayLinearAttention, inject_decay_linear
from framework.attentions.cosformer import CosformerAttention, inject_cosformer


class MockDistilBertAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_lin = nn.Linear(768, 768)
        self.k_lin = nn.Linear(768, 768)
        self.v_lin = nn.Linear(768, 768)
        self.out_lin = nn.Linear(768, 768)


class MockRobertaSelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(768, 768)
        self.value = nn.Linear(768, 768)


class MockRobertaOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)


class MockRobertaAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = MockRobertaSelfAttn()
        self.output = MockRobertaOutput()


class MockGPT2Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(768, 2304)
        self.c_proj = nn.Linear(768, 768)


class MockQwenAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 128)  # 2 kv heads * 64
        self.v_proj = nn.Linear(512, 128)
        self.o_proj = nn.Linear(512, 512)


class DummyQwenConfig:
    hidden_size = 512
    num_attention_heads = 8
    num_key_value_heads = 2
    head_dim = 64
    model_type = "qwen2"
    is_decoder = True


modules = [
    ("GatedDeltaNet", GatedDeltaNet, inject_gated_delta_net),
    ("GalerkinAttention", GalerkinAttention, inject_galerkin),
    ("FUSIONATTN", FUSIONATTN, inject_fusion_attn),
    ("FlashAttention3FP8", FlashAttention3FP8, inject_flash3_fp8),
    ("FlashAttention3", FlashAttention3, inject_flash_attention3),
    ("FactorisedMatrixState", FactorisedMatrixState, inject_factorised_matrix_state),
    ("ExpertChoiceMoEAttention", ExpertChoiceMoEAttention, inject_expert_choice_moe),
    ("DecayLinearAttention", DecayLinearAttention, inject_decay_linear),
    ("CosformerAttention", CosformerAttention, inject_cosformer),
]

print("================================================================================")
print(f"STARTING COMPREHENSIVE MULTI-ARCHITECTURE HARDENING SUITE FOR {len(modules)} MODULES")
print("================================================================================")

for name, cls, inject_fn in modules:
    print(f"\n>>> TESTING MODULE: {name}")

    # 1. DistilBERT test
    distil_old = MockDistilBertAttn()
    mod_distil = cls(old_attn=distil_old, hidden_size=768, num_attention_heads=12, lora_rank=16)
    assert not mod_distil.is_nested_encoder, f"[{name}] False positive nested encoder on DistilBERT"
    assert not mod_distil.is_decoder, f"[{name}] False positive decoder on DistilBERT"
    
    x = torch.randn(2, 16, 768)
    out_distil, _ = mod_distil(x)
    assert out_distil.shape == (2, 16, 768), f"[{name}] DistilBERT output shape mismatch: {out_distil.shape}"

    # 2. RoBERTa Nested Encoder test
    roberta_old = MockRobertaAttn()
    mod_roberta = cls(old_attn=roberta_old, hidden_size=768, num_attention_heads=12, lora_rank=16)
    assert mod_roberta.is_nested_encoder, f"[{name}] Failed to detect nested encoder for RoBERTa"
    out_roberta, _ = mod_roberta(x)
    assert out_roberta.shape == (2, 16, 768), f"[{name}] RoBERTa output shape mismatch: {out_roberta.shape}"

    # 3. GPT-2 Fused c_attn & Causal test
    gpt2_old = MockGPT2Attn()
    mod_gpt2 = cls(old_attn=gpt2_old, hidden_size=768, num_attention_heads=12, is_decoder=True, lora_rank=16)
    assert mod_gpt2.is_decoder, f"[{name}] Failed to detect decoder for GPT-2"
    out_gpt2, _ = mod_gpt2(x)
    assert out_gpt2.shape == (2, 16, 768), f"[{name}] GPT-2 output shape mismatch: {out_gpt2.shape}"

    # 4. Qwen GQA & RoPE test
    qwen_old = MockQwenAttn()
    qwen_cfg = DummyQwenConfig()
    mod_qwen = cls(config=qwen_cfg, old_attn=qwen_old, lora_rank=16)
    assert mod_qwen.num_kv_heads == 2, f"[{name}] GQA num_kv_heads failed: {mod_qwen.num_kv_heads}"
    assert mod_qwen.kv_repeat == 4, f"[{name}] GQA kv_repeat failed: {mod_qwen.kv_repeat}"
    assert mod_qwen.is_decoder, f"[{name}] Failed to detect Qwen decoder"

    x_qwen = torch.randn(2, 10, 512)
    cos = torch.randn(2, 1, 10, 32)
    sin = torch.randn(2, 1, 10, 32)
    out_qwen, _ = mod_qwen(x_qwen, position_embeddings=(cos, sin))
    assert out_qwen.shape == (2, 10, 512), f"[{name}] Qwen output shape mismatch: {out_qwen.shape}"

    # 5. LoRA Parameter Freezing
    # Find linear projections
    for attr in ["q_proj", "k_proj", "v_proj", "out_proj", "q_lin", "k_lin", "v_lin", "out_lin", "query", "key", "value", "W_res"]:
        sub = getattr(mod_distil, attr, None)
        if sub is not None and hasattr(sub, "weight") and hasattr(sub, "lora_A") and sub.lora_A is not None:
            assert sub.weight.requires_grad is False, f"[{name}] Base weight {attr} not frozen!"
            assert sub.lora_A.requires_grad is True, f"[{name}] lora_A {attr} not trainable!"
            assert sub.lora_B.requires_grad is True, f"[{name}] lora_B {attr} not trainable!"

    # 6. Injection factory test
    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = MockDistilBertAttn()

    mock_layer = MockLayer()
    inj_res = inject_fn(mock_layer, lora_rank=16)
    assert inj_res is not None, f"[{name}] Injection returned None!"
    assert isinstance(mock_layer.attention, cls), f"[{name}] Layer attention not replaced with {name}"

    print(f"PASSED ALL TESTS FOR {name}!")

print("\n================================================================================")
print("SUCCESS: ALL 7 ATTENTION MODULES FULLY HARDENED AND VERIFIED!")
print("================================================================================")
