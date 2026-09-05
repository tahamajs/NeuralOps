import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "Codes/framework")

from framework.attentions.chela_slay import ChelaSlayAttention, inject_chela_slay, LoRALinear

print("=== 1. Testing DistilBERT Attention Compatibility ===")
class MockDistilBertAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_lin = nn.Linear(768, 768)
        self.k_lin = nn.Linear(768, 768)
        self.v_lin = nn.Linear(768, 768)
        self.out_lin = nn.Linear(768, 768)

distil_old = MockDistilBertAttn()
chela_distil = ChelaSlayAttention(pretrained_attn=distil_old, hidden_size=768, num_attention_heads=12, lora_rank=16, gate_init_bias=12.0)
assert not chela_distil.is_nested_encoder
assert not chela_distil.is_decoder

x = torch.randn(2, 16, 768)
distil_old.eval()
chela_distil.eval()
with torch.no_grad():
    q = distil_old.q_lin(x).view(2, 16, 12, 64).transpose(1, 2)
    k = distil_old.k_lin(x).view(2, 16, 12, 64).transpose(1, 2)
    v = distil_old.v_lin(x).view(2, 16, 12, 64).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / 8.0
    probs = torch.softmax(scores, dim=-1)
    orig_out = distil_old.out_lin(torch.matmul(probs, v).transpose(1, 2).contiguous().view(2, 16, 768))

    chela_out, _ = chela_distil(x)
    diff = (orig_out - chela_out).abs().max().item()
    print(f"DistilBERT Step-0 max absolute diff: {diff:.6e}")
    assert diff < 1e-4, f"DistilBERT Step-0 diff too large: {diff}"
print("DistilBERT compatibility PASSED!")

print("\n=== 2. Testing RoBERTa Nested Encoder Compatibility ===")
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

roberta_old = MockRobertaAttn()
chela_roberta = ChelaSlayAttention(pretrained_attn=roberta_old, hidden_size=768, num_attention_heads=12, lora_rank=16, gate_init_bias=12.0)
assert chela_roberta.is_nested_encoder, "Must detect nested RoBERTa encoder!"
roberta_old.eval()
chela_roberta.eval()
with torch.no_grad():
    q = roberta_old.self.query(x).view(2, 16, 12, 64).transpose(1, 2)
    k = roberta_old.self.key(x).view(2, 16, 12, 64).transpose(1, 2)
    v = roberta_old.self.value(x).view(2, 16, 12, 64).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / 8.0
    probs = torch.softmax(scores, dim=-1)
    orig_self_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(2, 16, 768)
    chela_out, _ = chela_roberta(x)
    diff = (orig_self_out - chela_out).abs().max().item()
    print(f"RoBERTa Self-Attn Step-0 max absolute diff (bypassing out_proj): {diff:.6e}")
    assert diff < 1e-4, f"RoBERTa nested diff too large: {diff}"
print("RoBERTa compatibility PASSED!")

print("\n=== 3. Testing GPT-2 Fused c_attn & Causal Masking ===")
class MockGPT2Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(768, 2304)
        self.c_proj = nn.Linear(768, 768)

gpt2_old = MockGPT2Attn()
chela_gpt2 = ChelaSlayAttention(pretrained_attn=gpt2_old, hidden_size=768, num_attention_heads=12, is_decoder=True, lora_rank=16, gate_init_bias=12.0)
assert chela_gpt2.is_decoder, "Must detect GPT-2 decoder!"
gpt2_old.eval()
chela_gpt2.eval()
with torch.no_grad():
    c_w = gpt2_old.c_attn.weight.data
    c_b = gpt2_old.c_attn.bias.data
    q_w, k_w, v_w = c_w[:768], c_w[768:1536], c_w[1536:]
    q_b, k_b, v_b = c_b[:768], c_b[768:1536], c_b[1536:]
    q = F.linear(x, q_w, q_b).view(2, 16, 12, 64).transpose(1, 2)
    k = F.linear(x, k_w, k_b).view(2, 16, 12, 64).transpose(1, 2)
    v = F.linear(x, v_w, v_b).view(2, 16, 12, 64).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / 8.0
    causal = torch.triu(torch.full((16, 16), -1e4), diagonal=1)
    scores = scores + causal.unsqueeze(0).unsqueeze(1)
    probs = torch.softmax(scores, dim=-1)
    orig_out = gpt2_old.c_proj(torch.matmul(probs, v).transpose(1, 2).contiguous().view(2, 16, 768))

    chela_out, _ = chela_gpt2(x)
    diff = (orig_out - chela_out).abs().max().item()
    print(f"GPT-2 Step-0 max absolute diff: {diff:.6e}")
    assert diff < 1e-4, f"GPT-2 diff too large: {diff}"
print("GPT-2 compatibility PASSED!")

print("\n=== 4. Testing Yat-Kernel Causality Leakage in Linear Stream ===")
# Set gate bias to -12.0 to isolate the pure linear stream
class DummyCausalConfig:
    hidden_size = 256
    num_attention_heads = 4
    is_decoder = True

causal_cfg = DummyCausalConfig()
chela_linear = ChelaSlayAttention(config=causal_cfg, is_causal=True, gate_init_bias=-12.0)
chela_linear.eval()

x_seq = torch.randn(1, 16, 256)
with torch.no_grad():
    out1, _ = chela_linear(x_seq)
    # Modify future tokens (from position 8 onward)
    x_seq_mod = x_seq.clone()
    x_seq_mod[:, 8:, :] = torch.randn(1, 8, 256)
    out2, _ = chela_linear(x_seq_mod)

# Positions 0..7 must be identical
leakage = (out1[:, :8, :] - out2[:, :8, :]).abs().max().item()
print(f"Max leakage from future tokens to past tokens in linear stream: {leakage:.6e}")
assert leakage < 1e-5, f"Causal violation in linear stream! Leakage: {leakage}"
print("Causality strictly preserved in linear stream!")

print("\n=== 5. Testing Qwen3.5 GQA & Partial RoPE ===")
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

qwen_cfg = DummyQwenConfig()
qwen_old = MockQwenAttn()
chela_qwen = ChelaSlayAttention(config=qwen_cfg, pretrained_attn=qwen_old, lora_rank=16)
assert chela_qwen.num_kv_heads == 2
assert chela_qwen.kv_repeat == 4
assert chela_qwen.is_causal

# Test forward with partial RoPE (rotary_dim=32, head_dim=64)
x_qwen = torch.randn(2, 12, 512)
cos = torch.randn(2, 1, 12, 32)
sin = torch.randn(2, 1, 12, 32)
out_qwen, _ = chela_qwen(x_qwen, position_embeddings=(cos, sin))
assert out_qwen.shape == (2, 12, 512)
print("Qwen3.5 GQA & Partial RoPE PASSED!")

print("\n=== 6. Testing LoRA Freezing ===")
assert chela_distil.W_q.weight.requires_grad is False
assert chela_distil.W_q.lora_A.requires_grad is True
assert chela_distil.W_q.lora_B.requires_grad is True
print("LoRA parameter freezing PASSED!")

print("\n=== 7. Testing inject_chela_slay ===")
class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MockDistilBertAttn()

layer = MockLayer()
injected = inject_chela_slay(layer, lora_rank=16)
assert isinstance(layer.attention, ChelaSlayAttention)
assert layer.attention.lora_rank == 16
print("Injection factory PASSED!")

print("\n========================================================")
print("ALL 7 CHELASLAY HARDENING TESTS PASSED SUCCESSFULLY!")
print("========================================================")
