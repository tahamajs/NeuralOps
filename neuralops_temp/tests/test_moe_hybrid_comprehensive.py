import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from neuralops.core.registry import ATTENTION_REGISTRY


def test_attentions():
    B = 2
    L = 128
    D = 768
    num_heads = 12
    
    class Config:
        hidden_size = D
        num_attention_heads = num_heads
        num_key_value_heads = 4
        dropout = 0.1
        is_causal = True

    config = Config()
    
    names = [
        "bam_umoe", "moe_deltakv_paged", "compressed_mha", "mixattention_moska",
        "wgqa", "gqa", "oha", "block_transformer_hiba", "progressive_token_pruning",
        "nasc_attention", "nha_attention", "token_compressed_attention",
        "reap_moe", "samba_moe", "expert_choice_moe"
    ]
    
    for name in names:
        try:
            cls = ATTENTION_REGISTRY.get(name)
            if cls is None:
                continue
            print(f"Testing {name}...")
            model = cls(config)
            model.train()
            x = torch.randn(B, L, D, requires_grad=True)
            out, attn = model(x, is_causal=True)
            if out.shape != (B, L, D):
                print(f"[{name}] FAILED shape: {out.shape}")
                continue
            
            loss = out.mean()
            loss.backward()
            
            if x.grad is None or torch.isnan(x.grad).any():
                print(f"[{name}] FAILED gradient")
                continue
                
            print(f"[{name}] PASSED")
        except Exception as e:
            print(f"[{name}] ERROR: {e}")

if __name__ == '__main__':
    test_attentions()
