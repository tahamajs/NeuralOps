"""Part 4 Verification: Dynamic LoRA Target Discovery & Canonical Aliases (RCA-4, RCA-10)."""
import pytest
import torch
import torch.nn as nn
from neuralops.core.adapters import (CANONICAL_PROJECTION_ALIASES,
                                     UniversalAdapter)
from neuralops.utils.lora import find_all_linear_submodule_names
from transformers import (GPT2Config, GPT2ForSequenceClassification,
                          RobertaConfig, RobertaForSequenceClassification,
                          T5Config, T5EncoderModel)


def test_dynamic_lora_targets_gpt2():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100)
    m = GPT2ForSequenceClassification(cfg)
    targets = find_all_linear_submodule_names(m)
    assert "c_attn" in targets
    assert "c_proj" in targets

def test_dynamic_lora_targets_roberta():
    cfg = RobertaConfig(num_hidden_layers=2, num_attention_heads=2, hidden_size=32, vocab_size=100)
    m = RobertaForSequenceClassification(cfg)
    targets = find_all_linear_submodule_names(m)
    assert "query" in targets
    assert "key" in targets
    assert "value" in targets

def test_canonical_aliases_coverage():
    for k in ("q", "k", "v", "o"):
        assert k in CANONICAL_PROJECTION_ALIASES
        assert len(CANONICAL_PROJECTION_ALIASES[k]) >= 3
