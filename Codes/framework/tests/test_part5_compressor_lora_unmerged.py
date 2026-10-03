"""Part 5 Verification: Preserving Unmerged LoRA Adapters for Compressor Recovery (RCA-5)."""
import pytest
import torch
import torch.nn as nn
from framework.utils.lora import LoRALinear


def test_unmerged_lora_gradient_flow():
    base = nn.Linear(32, 32, bias=False)
    lora = LoRALinear(base, r=4, alpha=8.0)
    assert not lora.merged

    # Simulate compressor factorizing the base_layer
    class SVDMock(nn.Module):
        def __init__(self):
            super().__init__()
            self.U = nn.Parameter(torch.randn(32, 4))
            self.V = nn.Parameter(torch.randn(4, 32))
        def forward(self, x):
            return x @ self.V.t() @ self.U.t()

    lora.base_layer = SVDMock()
    assert not lora.merged

    opt = torch.optim.AdamW(lora.parameters(), lr=1e-3)
    x = torch.randn(2, 32)
    for _ in range(2):
        opt.zero_grad()
        loss = lora(x).sum()
        loss.backward()
        opt.step()

    assert lora.lora_A.weight.grad is not None
    assert lora.lora_B.weight.grad is not None
    assert lora.lora_A.weight.grad.norm().item() > 0
