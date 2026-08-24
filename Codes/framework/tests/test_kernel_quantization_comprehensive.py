import torch
import sys
import os
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from attentions.cluster_attention import ClusterAttention
from attentions.adasplash_attention import AdaSplashAttention
from attentions.hamming_attention import HammingAttention
from attentions.galerkin_attention import GalerkinAttention
from attentions.performer_attention import PerformerAttention

from compression_engine.astro_quantizer import AstroQuantizer
from compression_engine.faq_quantizer import FAQQuantizer
from compression_engine.coopq_quantizer import CoopQQuantizer
from compression_engine.modegpt import ModeGPTQuantizer
from compression_engine.zerokv_neo import ZeroKVNeo

def test_attention(module, name):
    print(f"Testing {name}...")
    try:
        x = torch.randn(2, 128, 768, requires_grad=True)
        out = module(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and not torch.isnan(x.grad).any(), "Gradient is NaN or None"
        print(f"{name} passed.")
    except Exception as e:
        print(f"{name} failed: {e}")
        traceback.print_exc()

def main():
    print("Testing Attentions...")
    try:
        test_attention(ClusterAttention(embed_dim=768, num_heads=8), "ClusterAttention")
    except Exception as e:
        print("ClusterAttention failed to initialize", e)

    try:
        test_attention(AdaSplashAttention(embed_dim=768, num_heads=8), "AdaSplashAttention")
    except Exception as e:
        print("AdaSplashAttention failed to initialize", e)
        
    try:
        test_attention(HammingAttention(embed_dim=768, num_heads=8), "HammingAttention")
    except Exception as e:
        print("HammingAttention failed to initialize", e)

    try:
        test_attention(GalerkinAttention(embed_dim=768, num_heads=8), "GalerkinAttention")
    except Exception as e:
        print("GalerkinAttention failed to initialize", e)

    try:
        test_attention(PerformerAttention(embed_dim=768, num_heads=8), "PerformerAttention")
    except Exception as e:
        print("PerformerAttention failed to initialize", e)

if __name__ == '__main__':
    main()
