"""Unit tests for neuralops.trainer.optimizers package."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from neuralops.trainer.optimizers import (
    OPTIMIZER_REGISTRY,
    Muon,
    RiemannianStiefelOptimizer,
    create_adabelief,
    create_adam,
    create_adamw,
    create_lamb,
    create_lars,
    create_muon,
    create_muon_adamw_reg,
    create_ranger,
    create_riemannian_stiefel,
    create_sgd,
)


class TestOptimizerFactories:
    """Test standard optimizer factory methods and registry registrations."""

    def test_optimizer_registry_contains_core_optimizers(self):
        registered = OPTIMIZER_REGISTRY.keys()
        for expected in ["adamw", "sgd", "adam", "muon", "muon_adamw", "stiefel"]:
            assert expected in registered

    def test_adamw_and_sgd_factory(self):
        p = nn.Parameter(torch.randn(4, 4))
        opt_adamw = create_adamw([p], lr=1e-3)
        assert opt_adamw is not None
        assert opt_adamw.defaults["lr"] == 1e-3

        opt_sgd = create_sgd([p], lr=1e-2, momentum=0.9)
        assert opt_sgd is not None
        assert opt_sgd.defaults["momentum"] == 0.9

    def test_adam_factory(self):
        p = nn.Parameter(torch.randn(4, 4))
        opt_adam = create_adam([p], lr=2e-3)
        assert opt_adam is not None
        assert opt_adam.defaults["lr"] == 2e-3

    def test_lamb_lars_ranger_adabelief_factories(self):
        p = nn.Parameter(torch.randn(4, 4))
        try:
            import torch_optimizer
            opt_lamb = create_lamb([p], lr=1e-3)
            assert opt_lamb is not None

            opt_lars = create_lars([p], lr=1e-3)
            assert opt_lars is not None

            opt_ranger = create_ranger([p], lr=1e-3)
            assert opt_ranger is not None

            opt_adabelief = create_adabelief([p], lr=1e-3)
            assert opt_adabelief is not None
        except ModuleNotFoundError:
            pytest.skip("torch_optimizer not installed")

    def test_one_optimizer_step_reduces_loss(self):
        model = nn.Linear(4, 1)
        opt = create_adamw(model.parameters(), lr=0.1)
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        
        # Initial loss
        loss1 = ((model(x) - y) ** 2).mean()
        loss1.backward()
        opt.step()
        opt.zero_grad()

        # Step 2
        for _ in range(5):
            l = ((model(x) - y) ** 2).mean()
            l.backward()
            opt.step()
            opt.zero_grad()
        loss_final = ((model(x) - y) ** 2).mean()
        assert loss_final.item() < loss1.item()

    def test_muon_optimizer_and_step(self):
        p = nn.Parameter(torch.randn(4, 4))
        opt = create_muon([p], lr=0.01)
        assert isinstance(opt, Muon)

        loss = (p ** 2).sum()
        loss.backward()
        opt.step()
        assert p.grad is not None

    def test_muon_adamw_hybrid_step(self):
        m = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        opt = create_muon_adamw_reg(m)
        assert opt is not None

        x = torch.randn(2, 4)
        loss = m(x).sum()
        loss.backward()
        opt.step()

    def test_riemannian_stiefel_optimizer(self):
        w = nn.Parameter(torch.randn(6, 4))
        opt = create_riemannian_stiefel([w], lr=1e-3)
        assert isinstance(opt, RiemannianStiefelOptimizer)

        loss = (w ** 2).sum()
        loss.backward()
        opt.step()
        err = opt.check_orthogonality_error(w)
        assert err is not None

    def test_invalid_config_raises(self):
        with pytest.raises(Exception):
            create_adamw([], lr=-1.0)


class TestSchedulerFactories:
    """Test standard learning rate scheduler factory methods."""

    def test_schedulers(self):
        from neuralops.trainer.schedulers import (
            create_linear,
            create_cosine,
            create_rop,
            create_onecycle,
            create_cosine_warmup,
            create_poly,
        )

        p = nn.Parameter(torch.randn(2, 2))
        opt = torch.optim.SGD([p], lr=0.1)

        s_lin = create_linear(opt, num_training_steps=100)
        assert s_lin is not None

        s_cos = create_cosine(opt, T_max=100)
        assert s_cos is not None

        s_rop = create_rop(opt)
        assert s_rop is not None

        s_oc = create_onecycle(opt, max_lr=0.1, steps_per_epoch=10, epochs=2)
        assert s_oc is not None

        s_cw = create_cosine_warmup(opt, num_warmup_steps=10, num_training_steps=100)
        assert s_cw is not None

        s_poly = create_poly(opt, total_iters=100)
        assert s_poly is not None
