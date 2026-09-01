import torch
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.mla_attention import MLAAttention

class DummyConfig:
    hidden_size = 896
    num_attention_heads = 14
    dropout = 0.1
    mla_rank = 128

config = DummyConfig()
mla = MLAAttention(config, rank=128)
x = torch.randn(16, 41, 896)
out, _ = mla(x)
print("Out shape:", out.shape)
