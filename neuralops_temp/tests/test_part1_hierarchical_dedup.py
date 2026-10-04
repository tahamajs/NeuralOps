"""Part 1 Verification: Hierarchical Prefix Deduplication (RCA-1)."""
import pytest
import torch
import torch.nn as nn
from neuralops.core.architecture import get_attention_modules
from transformers import (GPT2Config, GPT2ForSequenceClassification,
                          RobertaConfig, RobertaForSequenceClassification,
                          T5Config, T5ForConditionalGeneration)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)

def test_gpt2_dedup():
    cfg = GPT2Config(n_layer=4, n_head=4, n_embd=64, vocab_size=100)
    model = GPT2ForSequenceClassification(cfg)
    attns = get_attention_modules(model)
    assert len(attns) == 4
    for name, mod in attns:
        assert name.endswith(".attn")

def test_roberta_leaf_only():
    cfg = RobertaConfig(num_hidden_layers=4, num_attention_heads=4, hidden_size=64, vocab_size=100)
    model = RobertaForSequenceClassification(cfg)
    attns = get_attention_modules(model)
    assert len(attns) == 4
    for name, mod in attns:
        assert name.endswith(".attention.self")

def test_t5_encoder_decoder_counts():
    cfg = T5Config(num_layers=2, num_decoder_layers=2, num_heads=2, d_model=32, d_kv=16, d_ff=64, vocab_size=100)
    model = T5ForConditionalGeneration(cfg)
    attns = get_attention_modules(model)
    # 2 self in encoder + 2 self + 2 cross in decoder = 6
    assert len(attns) == 6

def test_nested_container_prefix_drop():
    class DeepLeaf(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)
    class Middle(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DeepLeaf()
    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.middle = Middle()
            self.dense = nn.Linear(32, 32)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Outer() for _ in range(3)])

    m = Model()
    leaves = get_attention_modules(m)
    assert len(leaves) == 3
    for name, mod in leaves:
        assert name.endswith(".middle.self_attn")
