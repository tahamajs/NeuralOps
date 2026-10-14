import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from neuralops.api import transplant_attention
from neuralops.attentions.linear.cosformer import CosformerAttention


class MultiLayerTransformer(nn.Module):
    def __init__(self, d_model=32, num_heads=2, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=64,
                dropout=0.0,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, use_checkpoint=True, use_reentrant=False):
        for layer in self.layers:
            if use_checkpoint:
                # Test with user-specified reentrancy
                x = checkpoint(layer, x, use_reentrant=use_reentrant)
            else:
                x = layer(x)
        return x


def test_gradient_checkpointing_compatibility():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiLayerTransformer(d_model=32, num_heads=2, num_layers=4).to(device)
    model.train()

    # 1. Non-reentrant checkpointing (PyTorch modern standard)
    x = torch.randn(2, 8, 32, device=device, requires_grad=True)
    out1 = model(x, use_checkpoint=True, use_reentrant=False)
    loss1 = out1.sum()
    loss1.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), "NaN in gradients with use_reentrant=False!"
    assert not torch.isinf(x.grad).any(), "Inf in gradients with use_reentrant=False!"

    # 2. Re-entrant checkpointing (classic standard)
    model.zero_grad()
    x2 = torch.randn(2, 8, 32, device=device, requires_grad=True)
    out2 = model(x2, use_checkpoint=True, use_reentrant=True)
    loss2 = out2.sum()
    loss2.backward()

    assert x2.grad is not None
    assert not torch.isnan(x2.grad).any(), "NaN in gradients with use_reentrant=True!"
    assert not torch.isinf(x2.grad).any(), "Inf in gradients with use_reentrant=True!"


def test_gradient_checkpointing_transplanted_cosformer():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiLayerTransformer(d_model=32, num_heads=2, num_layers=4).to(device)
    model = transplant_attention(model, target_attention="cosformer", d_model=32, num_heads=2)
    model.train()

    x = torch.randn(2, 8, 32, device=device, requires_grad=True)
    out = model(x, use_checkpoint=True, use_reentrant=False)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), "NaN in gradients with transplanted Cosformer under checkpointing!"
    assert not torch.isinf(x.grad).any(), "Inf in gradients with transplanted Cosformer under checkpointing!"
