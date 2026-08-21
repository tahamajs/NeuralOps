import torch
from transformers import DistilBertConfig
from framework.tests.conftest import build_attention
from framework.core.registry import ATTENTION_REGISTRY

config = DistilBertConfig(hidden_size=768, num_attention_heads=12)
x = torch.randn(2, 128, 768)

for key in ATTENTION_REGISTRY.keys():
    try:
        attn = build_attention(key, config)
        out = attn(x)
    except Exception as e:
        err_str = str(e)
        if "must match" in err_str:
            print(f"FAILED: {key} - {err_str}")
