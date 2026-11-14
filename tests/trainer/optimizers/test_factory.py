#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and factory contract tests for NeuralOps optimizers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.trainer.optimizers import (
    OPTIMIZER_REGISTRY,
    AnyPrecisionAdamW,
    Lion,
    Muon,
    MuonAdamW,
    RiemannianStiefelOptimizer,
    create_adam,
    create_adamw,
    create_anyprecision,
    create_lion,
    create_muon,
    create_muon_adamw_optimizer,
    create_muon_adamw_reg,
    create_riemannian_stiefel,
    create_sgd,
)


class TestOptimizerFactoryInstantiations:
    """Verifies that all registered optimizers instantiate with standard parameters."""

    def test_lion_instantiation(self):
        p = [nn.Parameter(torch.randn(8, 8))]
        opt = create_lion(p, lr=1e-4, weight_decay=0.01)
        assert isinstance(opt, Lion)

    def test_muon_instantiation(self):
        p = [nn.Parameter(torch.randn(16, 16))]
        opt = create_muon(p, lr=0.02, momentum=0.95)
        assert isinstance(opt, Muon)

    def test_muon_adamw_reg_instantiation(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        opt = create_muon_adamw_reg(model, lr=0.02, lr_adamw=1e-3)
        assert isinstance(opt, (MuonAdamW, torch.optim.Optimizer))

    def test_anyprecision_instantiation(self):
        p = [nn.Parameter(torch.randn(8, 8))]
        opt = create_anyprecision(p, lr=1e-3, use_kahan_summation=True)
        assert isinstance(opt, AnyPrecisionAdamW)

    def test_riemannian_stiefel_instantiation(self):
        model = nn.Linear(8, 8)
        opt = create_riemannian_stiefel(model, lr=1e-3)
        assert isinstance(opt, RiemannianStiefelOptimizer)

    def test_standard_optimizers_instantiation(self):
        p = [nn.Parameter(torch.randn(4, 4))]
        assert isinstance(create_adamw(p, lr=1e-3), torch.optim.AdamW)
        assert isinstance(create_adam(p, lr=1e-3), torch.optim.Adam)
        assert isinstance(create_sgd(p, lr=1e-2), torch.optim.SGD)

    @pytest.mark.parametrize("name", ["lion", "muon", "anyprecision", "adamw", "sgd", "adam"])
    def test_registry_lookup_and_creation(self, name: str):
        factory_fn = OPTIMIZER_REGISTRY.get(name)
        assert callable(factory_fn)
        p = [nn.Parameter(torch.randn(8, 8))]
        opt = factory_fn(p)
        assert isinstance(opt, torch.optim.Optimizer)

    @pytest.mark.parametrize("name", ["adabelief", "ranger", "lamb", "lars"])
    def test_optional_third_party_optimizers(self, name: str):
        """Verifies optional 3rd party optimizers instantiate or raise clean ImportError."""
        factory_fn = OPTIMIZER_REGISTRY.get(name)
        assert callable(factory_fn)
        p = [nn.Parameter(torch.randn(4, 4))]
        try:
            opt = factory_fn(p)
            assert isinstance(opt, torch.optim.Optimizer)
        except (ImportError, ModuleNotFoundError):
            pass  # Expected when 3rd party dependency is not installed in environment


class TestBenchmarkTrainingCallersSignatures:
    """Verifies that create_muon_adamw_optimizer and create_riemannian_stiefel match callers in benchmarks/core/training.py."""

    def test_create_muon_adamw_optimizer_caller_signature(self):
        """Matches: benchmarks/core/training.py line 54:
        optimizer = create_muon_adamw_optimizer(model, lr_muon=lr, lr_adamw=lr)
        """
        model = nn.Sequential(nn.Linear(16, 8, bias=True), nn.LayerNorm(8))
        lr = 0.01
        opt = create_muon_adamw_optimizer(model, lr_muon=lr, lr_adamw=lr)
        assert isinstance(opt, MuonAdamW)

        # Confirm lr_adamw took effect
        adam_group = [g for g in opt.param_groups if g["optimizer"] == "adamw"][0]
        assert adam_group["lr"] == lr

    def test_create_riemannian_stiefel_caller_signature(self):
        """Matches: benchmarks/core/training.py line 58:
        optimizer = create_riemannian_stiefel(model, lr=lr)
        """
        model = nn.Linear(16, 16, bias=False)
        lr = 0.005
        opt = create_riemannian_stiefel(model, lr=lr)
        assert isinstance(opt, RiemannianStiefelOptimizer)
        assert opt.param_groups[0]["lr"] == lr


class TestLionSparseGradientHandling:
    """Verifies Lion optimizer behavior on sparse gradients."""

    def test_sparse_gradient_raises_runtime_error(self):
        import warnings

        p = nn.Parameter(torch.randn(4, 4))
        opt = Lion([p], lr=1e-3)
        # Fabricate a sparse gradient
        indices = torch.tensor([[0, 1], [2, 3]])
        values = torch.tensor([1.0, 2.0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            p.grad = torch.sparse_coo_tensor(indices, values, (4, 4))
        with pytest.raises(RuntimeError, match="Lion does not support sparse gradients"):
            opt.step()
