import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.core.attention_utils import inject_attention
from framework.attentions.adaptive_span_attention import AdaptiveSpanAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ds = load_from_disk("Codes/framework/data/glue_data/sst2")
val = ds["validation"]
tok = AutoTokenizer.from_pretrained("models/roberta-base")

m1 = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
ckpt = torch.load("results/unified_asvd/roberta/finetuned_model.pt", map_location=device)
m1.load_state_dict(ckpt, strict=False)
m1.eval()

m2 = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
m2.load_state_dict(ckpt, strict=False)
inject_attention(m2, "adaptive_span", rank=16)
m2.eval()

# Sample 2 sentences
enc = tok(val[:2]["sentence"], padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(device)
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]

print("attention_mask sum per sequence:", attention_mask.sum(dim=-1).tolist())

emb1 = m1.roberta.embeddings(input_ids)
ext_mask = m1.roberta.get_extended_attention_mask(attention_mask, input_ids.shape)

l1 = m1.roberta.encoder.layer[0]
l2 = m2.roberta.encoder.layer[0]

s1 = l1.attention.self
s2 = l2.attention.self
w2 = s2._wrapped_attention

# 1. Check Q, K, V
q1 = s1.query(emb1)
q2 = w2.q_proj(emb1)
print("q diff:", (q1 - q2).abs().max().item())

k1 = s1.key(emb1)
k2 = w2.k_proj(emb1)
print("k diff:", (k1 - k2).abs().max().item())

v1 = s1.value(emb1)
v2 = w2.v_proj(emb1)
print("v diff:", (v1 - v2).abs().max().item())

# 2. Check attention output of s1 vs w2 directly
with torch.no_grad():
    out_s1 = s1(emb1, attention_mask=ext_mask)[0]
    out_w2 = w2(emb1, attention_mask=ext_mask)[0]
    out_s2 = s2(emb1, attention_mask=ext_mask)[0]

print("out_s1 vs out_w2 diff:", (out_s1 - out_w2).abs().max().item())
print("out_s1 vs out_s2 diff:", (out_s1 - out_s2).abs().max().item())
print("out_s1 norm:", out_s1.norm().item())
print("out_w2 norm:", out_w2.norm().item())

# Check token 0 (CLS) diff
print("out_s1 token 0 vs out_w2 token 0 diff:", (out_s1[:, 0] - out_w2[:, 0]).abs().max().item())

# Let's inspect w2 internals:
# Manually run the scores and softmax inside w2
B, N_q, D = emb1.shape
H, d = w2.num_heads, w2.head_dim
Q = q2.view(B, N_q, H, d).transpose(1, 2)
K = k2.view(B, N_q, H, d).transpose(1, 2)
V = v2.view(B, N_q, H, d).transpose(1, 2)
scores = torch.matmul(Q, K.transpose(-2, -1)) * (d ** -0.5)

print("scores diff vs s1 scores:")
s1_Q = s1.transpose_for_scores(q1)
s1_K = s1.transpose_for_scores(k1)
s1_scores = torch.matmul(s1_Q, s1_K.transpose(-1, -2)) / (d ** 0.5)
print("  raw scores diff:", (scores - s1_scores).abs().max().item())

scores_masked = scores + ext_mask.to(scores.dtype)
s1_scores_masked = s1_scores + ext_mask
print("  masked scores diff:", (scores_masked - s1_scores_masked).abs().max().item())

attn_prob = torch.softmax(scores_masked, dim=-1)
s1_prob = torch.softmax(s1_scores_masked, dim=-1)
print("  attn_prob diff:", (attn_prob - s1_prob).abs().max().item())

# Now check ramp!
span = torch.sigmoid(w2.span_logits) * w2.max_span
print("w2.max_span:", w2.max_span, "N_k:", N_q, "span:", span[0].item())
pos_q = torch.arange(N_q, device=device, dtype=scores.dtype).unsqueeze(1)
pos_k = torch.arange(N_q, device=device, dtype=scores.dtype).unsqueeze(0)
dist = (pos_q - pos_k).abs()
span_h = span.view(1, H, 1, 1).to(scores.dtype)
dist_mat = dist.view(1, 1, N_q, N_q)
ramp = torch.clamp((w2.ramp_size + span_h - dist_mat) / w2.ramp_size, 0.0, 1.0)
print("ramp min:", ramp.min().item(), "ramp max:", ramp.max().item())
print("ramp at [0, 0, 0, :10]:", ramp[0, 0, 0, :10].tolist())
print("ramp at [0, 0, 0, 90:100]:", ramp[0, 0, 0, 90:100].tolist())
print("ramp at [0, 0, 0, 120:]: ", ramp[0, 0, 0, 120:].tolist())

# Now with ramp applied:
attn_ramped = attn_prob * ramp
attn_sum = attn_ramped.sum(dim=-1, keepdim=True).clamp(min=1e-4)
attn_renorm = attn_ramped / attn_sum
print("attn_renorm diff vs s1_prob:", (attn_renorm - s1_prob).abs().max().item())

ctx_orig = torch.matmul(s1_prob, s1.transpose_for_scores(v1)).transpose(1, 2).contiguous().view(B, N_q, D)
ctx_ramped = torch.matmul(attn_renorm, V).transpose(1, 2).contiguous().view(B, N_q, D)
print("ctx_orig vs ctx_ramped diff:", (ctx_orig - ctx_ramped).abs().max().item())
print("ctx_orig token 0 vs ctx_ramped token 0 diff:", (ctx_orig[:, 0] - ctx_ramped[:, 0]).abs().max().item())
