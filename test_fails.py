import torch
import inspect
import sys
import traceback
from framework.core.registry import ATTENTION_REGISTRY

fails = [
  "fused_svd_kernel",
  "svd_attention",
  "span_expanded_attention",
  "vig_attention",
  "mla_attention",
  "samba_moe"
]
for f in fails:
    __import__(f"framework.attentions.{f}")

config = type("Config", (), {"hidden_size": 128, "num_attention_heads": 4})()

for name, attn_cls in ATTENTION_REGISTRY.items():
    if not inspect.isclass(attn_cls): continue
    print(f"--- {name} ---")
    try:
        attn = attn_cls(config)
        hidden_states = torch.randn(2, 10, 128, requires_grad=True)
        attn(hidden_states)
        print("Success")
    except Exception as e:
        traceback.print_exc()
