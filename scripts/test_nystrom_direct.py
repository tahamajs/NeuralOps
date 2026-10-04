import torch
from transformers import AutoModelForCausalLM
from neuralops.attentions.nystrom_attention import NystromAttention

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
orig_attn = model.transformer.h[0].attn
x = torch.randn(1, 64, 768)

with torch.no_grad():
    orig_out = orig_attn(x)[0]

print("Initializing NystromAttention...")
new_attn = NystromAttention(config=model.config, old_attn=orig_attn)
with torch.no_grad():
    new_out = new_attn(x)[0]

diff = (orig_out - new_out).abs().max()
mean_diff = (orig_out - new_out).abs().mean()
print(f"Max diff: {diff.item():.8e}")
print(f"Mean diff: {mean_diff.item():.8e}")
print(f"Is identical (< 1e-4)? {diff.item() < 1e-4}")
print(f"new_attn.is_decoder: {new_attn.is_decoder}")
print(f"new_attn.is_causal: {new_attn.is_causal}")
