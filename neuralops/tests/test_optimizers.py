"""Unit tests for neuralops.trainer.optimizers package."""

import torch
import torch.nn as nn
from neuralops.trainer.optimizers import (
    create_muon,
    create_adamw,
    create_sgd,
    RiemannianStiefelOptimizer,
    Muon,
    create_muon_adamw_reg,
)


def test_adamw_and_sgd_factory():
    p = nn.Parameter(torch.randn(4, 4))
    opt_adamw = create_adamw([p], lr=1e-3)
    assert opt_adamw is not None

    opt_sgd = create_sgd([p], lr=1e-2)
    assert opt_sgd is not None


def test_muon_optimizer():
    p = nn.Parameter(torch.randn(4, 4))
    opt = create_muon([p], lr=0.01)
    assert isinstance(opt, Muon)

    loss = (p ** 2).sum()
    loss.backward()
    opt.step()
    assert p.grad is not None


def test_muon_adamw_hybrid():
    m = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    opt = create_muon_adamw_reg(m)
    assert opt is not None

    x = torch.randn(2, 4)
    loss = m(x).sum()
    loss.backward()
    opt.step()


def test_riemannian_stiefel_optimizer():
    w = nn.Parameter(torch.randn(6, 4))
    opt = RiemannianStiefelOptimizer([w], lr=1e-3)
    loss = (w ** 2).sum()
    loss.backward()
    opt.step()
    err = opt.check_orthogonality_error(w)
    assert err is not None
