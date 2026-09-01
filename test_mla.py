import torch
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.mla_attention import MLAAttention

class DummyConfig:
    hidden_size = 512
    num_attention_heads = 8
    dropout = 0.1
    mla_rank = 128

config = DummyConfig()
mla = MLAAttention(config, rank=128)
x = torch.randn(2, 64, 512)
try:
    mla(x)
    print("MLA forward success")
except Exception as e:
    print(f"Error: {e}")
