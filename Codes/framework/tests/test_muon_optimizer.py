import pytest
import torch
import torch.nn as nn
from framework.trainer.optimizers import (OPTIMIZER_REGISTRY, Muon, MuonAdamW,
                                          create_muon_adamw_optimizer,
                                          zeropower_via_newtonschulz5)


def test_zeropower_via_newtonschulz5():
    """Verify Newton-Schulz iteration produces approximately orthonormal matrices with singular values near 1."""
    torch.manual_seed(42)
    # Square matrix
    G = torch.randn(32, 32)
    ortho = zeropower_via_newtonschulz5(G, steps=5)
    assert ortho.shape == G.shape
    s = torch.linalg.svdvals(ortho)
    assert s.min().item() > 0.5
    assert s.max().item() < 1.5

    # Rectangular tall matrix (m > n)
    G_tall = torch.randn(64, 16)
    ortho_tall = zeropower_via_newtonschulz5(G_tall, steps=5)
    assert ortho_tall.shape == G_tall.shape
    s_tall = torch.linalg.svdvals(ortho_tall)
    assert s_tall.min().item() > 0.5
    assert s_tall.max().item() < 1.5

    # Rectangular wide matrix (m < n)
    G_wide = torch.randn(16, 64)
    ortho_wide = zeropower_via_newtonschulz5(G_wide, steps=5)
    assert ortho_wide.shape == G_wide.shape
    s_wide = torch.linalg.svdvals(ortho_wide)
    assert s_wide.min().item() > 0.5
    assert s_wide.max().item() < 1.5


def test_muon_parameter_validation():
    """Verify Muon strictly enforces 2D matrix parameters."""
    bias = nn.Parameter(torch.randn(16))
    with pytest.raises(ValueError, match="Muon only supports 2-D matrix parameters"):
        Muon([bias], lr=0.02)

    weight_2d = nn.Parameter(torch.randn(32, 16))
    opt = Muon([weight_2d], lr=0.02)
    assert len(opt.param_groups[0]["params"]) == 1


def test_muon_step_and_gradient_flow():
    """Verify Muon updates 2D parameters and reduces loss."""
    torch.manual_seed(42)
    linear = nn.Linear(16, 8, bias=False)
    opt = Muon(linear.parameters(), lr=0.02, momentum=0.95, weight_decay=0.01)

    initial_param = linear.weight.clone()
    x = torch.randn(4, 16)
    y = torch.randn(4, 8)

    for _ in range(5):
        opt.zero_grad()
        loss = ((linear(x) - y) ** 2).mean()
        loss.backward()
        opt.step()

    assert not torch.allclose(linear.weight, initial_param)


def test_create_muon_adamw_optimizer_split():
    """Verify create_muon_adamw_optimizer cleanly partitions 2D vs 1D parameters."""
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 16, bias=True)  # weight: 2D, bias: 1D
            self.norm = nn.LayerNorm(16)                # weight: 1D, bias: 1D
            self.lora_a = nn.Parameter(torch.randn(8, 32))  # 2D
            self.lora_b = nn.Parameter(torch.randn(16, 8))  # 2D

    model = ToyModel()
    opt = create_muon_adamw_optimizer(model, lr_muon=0.02, lr_adam=5e-5)

    assert isinstance(opt, MuonAdamW)
    assert len(opt.param_groups) == 2

    # Group 0 is Muon (2D matrices: linear.weight, lora_a, lora_b)
    muon_group = opt.param_groups[0]
    assert muon_group["optimizer"] == "muon"
    assert muon_group["lr"] == 0.02
    assert len(muon_group["params"]) == 3
    for p in muon_group["params"]:
        assert p.ndim == 2

    # Group 1 is AdamW (1D vectors: linear.bias, norm.weight, norm.bias)
    adam_group = opt.param_groups[1]
    assert adam_group["optimizer"] == "adamw"
    assert adam_group["lr"] == 5e-5
    assert len(adam_group["params"]) == 3
    for p in adam_group["params"]:
        assert p.ndim != 2


def test_muon_adamw_training_loop():
    """Verify end-to-end training loop with MuonAdamW hybrid optimizer."""
    torch.manual_seed(42)
    class ModelWithLoRA(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16, bias=True)
            self.lora_A = nn.Parameter(torch.randn(4, 16))
            self.lora_B = nn.Parameter(torch.randn(16, 4))
            self.bias = nn.Parameter(torch.zeros(16))

        def forward(self, x):
            base = self.fc1(x)
            lora = (x @ self.lora_A.t()) @ self.lora_B.t()
            return base + lora + self.bias

    model = ModelWithLoRA()
    opt = create_muon_adamw_optimizer(model, lr_muon=0.02, lr_adam=1e-4)

    x = torch.randn(8, 16)
    target = torch.randn(8, 16)

    initial_loss = ((model(x) - target) ** 2).mean().item()
    for _ in range(10):
        opt.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

    final_loss = ((model(x) - target) ** 2).mean().item()
    assert final_loss < initial_loss


def test_optimizer_registry_contains_muon():
    """Verify Muon and MuonAdamW are registered in OPTIMIZER_REGISTRY."""
    assert "muon" in OPTIMIZER_REGISTRY
    assert "muon_adamw" in OPTIMIZER_REGISTRY

    p = nn.Parameter(torch.randn(10, 10))
    opt = OPTIMIZER_REGISTRY.get("muon")([p], lr=0.01)
    assert isinstance(opt, Muon)
