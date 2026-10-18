"""Part 8 Verification: Precise Recovery Trainable Parameter Selection."""
import pytest
import torch
import torch.nn as nn
from neuralops.utils.lora import LoRALinear


def test_targeted_parameter_freezing():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 32)
            self.norm = nn.LayerNorm(32)
            lin = nn.Linear(32, 32)
            self.attn_proj = LoRALinear(lin, r=4)
            self.classifier = nn.Linear(32, 2)

    m = Model()
    for p_name, p in m.named_parameters():
        p_low = p_name.lower()
        is_head = any(h in p_low for h in ("score", "classifier", "lm_head"))
        is_lora = ".lora_a." in p_low or ".lora_b." in p_low or p_low.endswith((".lora_a", ".lora_b"))
        is_backbone_norm_emb = any(h in p_low for h in ("emb", "norm", "layernorm"))
        if (is_head or is_lora) and not is_backbone_norm_emb:
            p.requires_grad = True
        else:
            p.requires_grad = False

    assert not m.emb.weight.requires_grad
    assert not m.norm.weight.requires_grad
    assert m.attn_proj.lora_A.weight.requires_grad
    assert m.attn_proj.lora_B.weight.requires_grad
    assert m.classifier.weight.requires_grad
