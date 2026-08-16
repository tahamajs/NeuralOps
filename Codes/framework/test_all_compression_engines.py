"""
Comprehensive validation runner for all compression engines in Codes/framework/compression_engine/
"""

import os
import sys
import math
import copy
import inspect
import importlib
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import framework base
import framework
from framework.core.base import BaseCompressor

class DummyTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
    def encode(self, text, *args, **kwargs):
        return [1, 2, 3, 4]
    def decode(self, tokens, *args, **kwargs):
        return "sample text"
    def save_pretrained(self, path, *args, **kwargs):
        pass
    def __call__(self, text, *args, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }

class DummyHFLikeBatch:
    def __init__(self, B=2, T=16, C=768):
        self.input_ids = torch.randint(0, 1000, (B, T))
        self.attention_mask = torch.ones(B, T, dtype=torch.long)
        self.inputs_embeds = torch.randn(B, T, C)
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)
    def items(self):
        return [("input_ids", self.input_ids), ("attention_mask", self.attention_mask), ("inputs_embeds", self.inputs_embeds)]
    def keys(self):
        return ["input_ids", "attention_mask", "inputs_embeds"]

class SingleLinearModel(nn.Module):
    def __init__(self, in_features=768, out_features=768, bias=True):
        super().__init__()
        self.config = type("Config", (), {
            "hidden_size": in_features,
            "intermediate_size": out_features,
            "dim": in_features,
            "num_attention_heads": 12,
            "n_heads": 12,
            "num_hidden_layers": 1,
            "n_layers": 1,
            "vocab_size": 1000,
            "max_position_embeddings": 128,
            "rank": 32,
        })()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.proj = self.linear
        self.fc = self.linear
        self.q_proj = self.linear
        self.k_proj = self.linear
        self.v_proj = self.linear
        self.out_proj = self.linear

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            if "inputs_embeds" in x:
                x = x["inputs_embeds"]
            elif "input_ids" in x:
                x = torch.randn(x["input_ids"].shape[0], x["input_ids"].shape[1], self.config.hidden_size, device=x["input_ids"].device)
        elif torch.is_tensor(x) and x.dtype in (torch.int64, torch.int32):
            x = torch.randn(x.shape[0], x.shape[1], self.config.hidden_size, device=x.device)
        return self.linear(x)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or p.numel() > 0)

print("Test setup initialized.")
