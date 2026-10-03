import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from framework.attentions.adasplash_attention import AdaSplashAttention
from framework.attentions.cluster_attention import ClusterAttention
from framework.attentions.galerkin_attention import GalerkinAttention
from framework.attentions.hamming_attention import HammingAttention
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

import pytest


def _run_attention_test(module, name):
    print(f"Testing {name}...")
    x = torch.randn(2, 128, 768, requires_grad=True)
    out = module(x)
    if isinstance(out, tuple):
        out = out[0]
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any(), "Gradient is NaN or None"
    print(f"{name} passed.")

@pytest.mark.parametrize("attn_cls,name", [
    (ClusterAttention, "ClusterAttention"),
    (AdaSplashAttention, "AdaSplashAttention"),
    (HammingAttention, "HammingAttention"),
    (GalerkinAttention, "GalerkinAttention"),
    (PerformerAttention, "PerformerAttention"),
])
def test_all_attentions(attn_cls, name):
    config = MockConfig()
    module = attn_cls(config)
    _run_attention_test(module, name)

def main():
    print("Testing Attentions...")
    config = MockConfig()
    for attn_cls, name in [
        (ClusterAttention, "ClusterAttention"),
        (AdaSplashAttention, "AdaSplashAttention"),
        (HammingAttention, "HammingAttention"),
        (GalerkinAttention, "GalerkinAttention"),
        (PerformerAttention, "PerformerAttention"),
    ]:
        try:
            _run_attention_test(attn_cls(config), name)
        except Exception as e:
            print(f"{name} failed: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    main()

