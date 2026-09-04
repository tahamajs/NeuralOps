import torch
import torch.nn as nn
from framework.attentions.aegis_v1 import AegisAttentionV1, AegisConfigV1
from framework.attentions.aegis_v4 import AegisAttention, AegisConfig
from framework.attentions.aegis_v6 import AegisAttentionV6, AegisConfigV3

class DummyConfig:
    hidden_size = 768
    num_attention_heads = 12
    num_key_value_heads = 12
    head_dim = 64

cfg = DummyConfig()
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_mod(mod_cls, conf):
    mod = mod_cls(cfg, conf).to(dev)
    x = torch.randn(2, 32, 768, device=dev)
    mask = torch.ones(2, 32, device=dev)
    try:
        y = mod(x, attention_mask=mask)
        print(f"{mod_cls.__name__} passed.")
    except Exception as e:
        print(f"{mod_cls.__name__} failed: {e}")

test_mod(AegisAttentionV1, AegisConfigV1())
test_mod(AegisAttention, AegisConfig())
test_mod(AegisAttentionV6, AegisConfigV3(hidden_size=768, num_heads=12))
