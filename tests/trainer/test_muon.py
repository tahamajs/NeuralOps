#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for Muon and AnyPrecision optimizers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.trainer.optimizers import (
    AnyPrecisionAdamW,
    Muon,
    MuonAdamW,
    OPTIMIZER_REGISTRY,
    create_anyprecision,
    create_muon,
    create_muon_adamw_optimizer,
    zeropower_via_newtonschulz5,
)


class TestMuonNewtonSchulz:
    """Verifies zeroth-power orthogonalization via quintic Newton-Schulz iteration."""

    def test_zeropower_orthogonality(self):
        torch.manual_seed(42)
        G = torch.randn(32, 32)
        O = zeropower_via_newtonschulz5(G, steps=5)
        assert O.shape == (32, 32)

        # Orthogonality invariant: singular values of O should be near 1.0
        s = torch.linalg.svdvals(O)
        assert s.min().item() > 0.5
        assert s.max().item() < 1.5

    def test_rectangular_matrices(self):
        torch.manual_seed(42)
        # Tall matrix
        tall = torch.randn(64, 16)
        O_tall = zeropower_via_newtonschulz5(tall, steps=5)
        assert O_tall.shape == (64, 16)

        # Wide matrix
        wide = torch.randn(16, 64)
        O_wide = zeropower_via_newtonschulz5(wide, steps=5)
        assert O_wide.shape == (16, 64)


class TestMuonOptimizer:
    """Verifies Muon optimizer parameter handling and step updates."""

    def test_parameter_dimension_validation(self):
        bias = nn.Parameter(torch.randn(16))
        with pytest.raises(ValueError, match="Muon only supports 2-D matrix parameters"):
            Muon([bias], lr=0.02)

        weight_2d = nn.Parameter(torch.randn(16, 16))
        opt = Muon([weight_2d], lr=0.02)
        assert len(opt.param_groups[0]["params"]) == 1

    def test_dict_param_groups(self):
        w1 = nn.Parameter(torch.randn(16, 16))
        w2 = nn.Parameter(torch.randn(32, 16))
        opt = Muon([{"params": [w1, w2], "lr": 0.01}], lr=0.02)
        assert len(opt.param_groups) == 1
        assert len(opt.param_groups[0]["params"]) == 2

    def test_optimization_step_reduces_loss(self):
        torch.manual_seed(42)
        model = nn.Linear(16, 8, bias=False)
        opt = Muon(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0)

        x = torch.randn(4, 16)
        target = torch.randn(4, 8)

        # Check that loss decreases over several steps
        initial_loss = nn.functional.mse_loss(model(x), target).item()
        for _ in range(5):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), target)
            loss.backward()
            opt.step()

        final_loss = nn.functional.mse_loss(model(x), target).item()
        assert final_loss < initial_loss


class TestMuonAdamWHybrid:
    """Verifies MuonAdamW hybrid optimizer parameter grouping."""

    def test_hybrid_creation_from_module(self):
        model = nn.Sequential(
            nn.Linear(16, 8, bias=True),
            nn.LayerNorm(8),
        )
        opt = create_muon_adamw_optimizer(model, lr_muon=0.02, lr_adam=1e-3)
        assert isinstance(opt, MuonAdamW)

        # Weight matrices go to Muon, biases and LayerNorm go to AdamW
        muon_group = [g for g in opt.param_groups if g["optimizer"] == "muon"][0]
        adam_group = [g for g in opt.param_groups if g["optimizer"] == "adamw"][0]
        assert len(muon_group["params"]) >= 1
        assert len(adam_group["params"]) >= 1


class TestAnyPrecisionAdamW:
    """Verifies AnyPrecisionAdamW reduced-precision optimizer states."""

    def test_anyprecision_loss_reduction(self):
        torch.manual_seed(42)
        model = nn.Linear(16, 8)
        opt = AnyPrecisionAdamW(
            model.parameters(),
            lr=0.05,
            momentum_dtype=torch.float32,
            variance_dtype=torch.float32,
            use_kahan_summation=True,
        )

        x = torch.randn(4, 16)
        y = torch.randn(4, 8)

        init_loss = nn.functional.mse_loss(model(x), y).item()
        for _ in range(5):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()

        final_loss = nn.functional.mse_loss(model(x), y).item()
        assert final_loss < init_loss

    def test_optimizer_registry_resolution(self):
        params = [nn.Parameter(torch.randn(8, 8))]
        opt_muon = create_muon(params, lr=0.01)
        assert isinstance(opt_muon, Muon)

        opt_anyprec = create_anyprecision(params, lr=0.001)
        assert isinstance(opt_anyprec, AnyPrecisionAdamW)
