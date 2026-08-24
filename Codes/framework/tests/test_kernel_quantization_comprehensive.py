import torch
import sys
import os
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from framework.attentions.cluster_attention import ClusterAttention
from framework.attentions.adasplash_attention import AdaSplashAttention
from framework.attentions.hamming_attention import HammingAttention
from framework.attentions.galerkin_attention import GalerkinAttention
from framework.attentions.performer_attention import PerformerAttention

class MockConfig:
    def __init__(self, embed_dim=768, num_heads=8, max_seq_len=128):
        self.embed_dim = embed_dim
        self.hidden_size = embed_dim
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.max_position_embeddings = max_seq_len
        self.max_seq_len = max_seq_len
        self.attention_dropout = 0.0
        self.dropout = 0.0
        self.layer_norm_eps = 1e-12

def test_attention(module, name):
    print(f"Testing {name}...")
    try:
        x = torch.randn(2, 128, 768, requires_grad=True)
        out = module(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and not torch.isnan(x.grad).any(), "Gradient is NaN or None"
        print(f"{name} passed.")
    except Exception as e:
        print(f"{name} failed: {e}")
        traceback.print_exc()

def main():
    print("Testing Attentions...")
    config = MockConfig()
    try:
        test_attention(ClusterAttention(config), "ClusterAttention")
    except Exception as e:
        print("ClusterAttention failed to initialize", e)

    try:
        test_attention(AdaSplashAttention(config), "AdaSplashAttention")
    except Exception as e:
        print("AdaSplashAttention failed to initialize", e)
        
    try:
        test_attention(HammingAttention(config), "HammingAttention")
    except Exception as e:
        print("HammingAttention failed to initialize", e)

    try:
        test_attention(GalerkinAttention(config), "GalerkinAttention")
    except Exception as e:
        print("GalerkinAttention failed to initialize", e)

    try:
        test_attention(PerformerAttention(config), "PerformerAttention")
    except Exception as e:
        print("PerformerAttention failed to initialize", e)

if __name__ == '__main__':
    main()
