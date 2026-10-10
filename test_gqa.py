import torch
import traceback
from transformers import GPT2LMHeadModel, GPT2Config
from neuralops.core.registry import ATTENTION_REGISTRY
import neuralops.attentions

# Build tiny GPT-2
cfg = GPT2Config(vocab_size=1000, n_embd=64, n_layer=2, n_head=4)
model = GPT2LMHeadModel(cfg)
old_attn = model.transformer.h[0].attn

print(f"old_attn type: {type(old_attn).__name__}")
print(f"c_attn.weight.shape: {old_attn.c_attn.weight.shape}")
print(f"c_attn class: {old_attn.c_attn.__class__.__name__}")

# Get GQA
GQA = ATTENTION_REGISTRY.get("gqa_attention")
print(f"\nGQA class: {GQA}")

# Try instantiate
try:
    gqa = GQA(config=cfg, rank=16, pretrained_attn=old_attn)
    print(f"Created: {gqa}")
    print(f"hidden_size: {gqa.hidden_size}")
    print(f"num_heads: {gqa.num_heads}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
    traceback.print_exc()