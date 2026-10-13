import torch
from transformers import GPT2LMHeadModel
from neuralops.attentions.linear.gla_attention import GLAttention

device = "cuda"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
old_attn = model.transformer.h[0].attn

gla = GLAttention(
    hidden_size=model.config.n_embd,
    num_heads=model.config.n_head,
    head_dim=model.config.n_embd // model.config.n_head,
    old_attn=old_attn,
).to(device)

x = torch.randn(1, 128, 768, device=device)
with torch.no_grad():
    out_orig, _ = old_attn(x)
    out_gla, _ = gla(x)

orig_mean_abs = out_orig.abs().mean().item()
orig_max_abs = out_orig.abs().max().item()
orig_std = out_orig.std().item()

gla_mean_abs = out_gla.abs().mean().item()
gla_max_abs = out_gla.abs().max().item()
gla_std = out_gla.std().item()

print(f"GPT-2 layer 0:  mean_abs ≈ {orig_mean_abs:.4f}, max_abs ≈ {orig_max_abs:.4f}, std ≈ {orig_std:.4f}")
print(f"GLA layer 0:    mean_abs ≈ {gla_mean_abs:.4f}, max_abs ≈ {gla_max_abs:.4f}, std ≈ {gla_std:.4f}")
print(f"Ratio:          {gla_mean_abs / orig_mean_abs:.4f}")
