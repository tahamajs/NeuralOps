#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and mathematical verification tests for Muon optimizer step."""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from neuralops.trainer.optimizers.muon import (
    Muon,
    MuonAdamW,
    create_muon_adamw_optimizer,
    zeropower_via_newtonschulz5,
)


class TestMuonMathematicalStep:
    """Verifies that Muon step precisely matches the Newton-Schulz mathematical formulation."""

    def test_single_step_mathematical_parity(self):
        torch.manual_seed(42)
        N, M = 16, 24
        W_init = torch.randn(N, M, dtype=torch.float32)
        G = torch.randn(N, M, dtype=torch.float32)

        p = nn.Parameter(W_init.clone())
        p.grad = G.clone()

        lr = 0.02
        momentum = 0.95
        weight_decay = 0.01
        ns_steps = 5

        opt = Muon([p], lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        opt.step()

        # Manual analytical reference
        B1 = G.clone()
        ortho = zeropower_via_newtonschulz5(B1, steps=ns_steps)
        scale = max(1.0, math.sqrt(N / M))
        ortho_scaled = ortho * scale

        W_expected = W_init.clone() * (1.0 - lr * weight_decay) - lr * ortho_scaled

        assert torch.allclose(p.data, W_expected, atol=1e-5), f"Max diff: {(p.data - W_expected).abs().max().item()}"

    def test_second_step_momentum_accumulation(self):
        torch.manual_seed(42)
        N, M = 8, 8
        W_init = torch.randn(N, M, dtype=torch.float32)
        G1 = torch.randn(N, M, dtype=torch.float32)
        G2 = torch.randn(N, M, dtype=torch.float32)

        p = nn.Parameter(W_init.clone())
        lr = 0.01
        momentum = 0.9
        wd = 0.05
        steps = 5

        opt = Muon([p], lr=lr, momentum=momentum, weight_decay=wd, ns_steps=steps)

        # Step 1
        p.grad = G1.clone()
        opt.step()

        # Step 2
        p.grad = G2.clone()
        opt.step()

        # Manual calculation of step 1 & 2
        buf = G1.clone()
        ortho1 = zeropower_via_newtonschulz5(buf, steps=steps) * max(1.0, math.sqrt(N / M))
        w1 = W_init.clone() * (1.0 - lr * wd) - lr * ortho1

        buf = buf * momentum + G2
        ortho2 = zeropower_via_newtonschulz5(buf, steps=steps) * max(1.0, math.sqrt(N / M))
        w2_expected = w1 * (1.0 - lr * wd) - lr * ortho2

        assert torch.allclose(p.data, w2_expected, atol=1e-5)

    def test_zeropower_singular_values_unitary(self):
        torch.manual_seed(42)
        G = torch.randn(32, 32)
        O = zeropower_via_newtonschulz5(G, steps=5)

        # Singular values of zeropower output must be close to 1.0
        s = torch.linalg.svdvals(O)
        assert s.min().item() > 0.5
        assert s.max().item() < 1.5

    def test_compile_compatibility(self):
        compiled_fn = torch.compile(zeropower_via_newtonschulz5, backend="aot_eager")
        x = torch.randn(16, 32)
        out = compiled_fn(x)
        assert out.shape == (16, 32)
        assert torch.isfinite(out).all()

    def test_muon_adamw_hybrid_routing(self):
        torch.manual_seed(42)
        w2d = nn.Parameter(torch.randn(16, 16))
        b1d = nn.Parameter(torch.zeros(16))

        opt = create_muon_adamw_optimizer(
            [w2d, b1d],
            lr_muon=0.02,
            lr_adam=1e-3,
            lr_adamw=1e-3,
        )

        w2d.grad = torch.randn(16, 16)
        b1d.grad = torch.randn(16)
        opt.step()

        # Both parameters should have updated
        assert not torch.allclose(b1d.data, torch.zeros(16))
        assert not torch.isnan(w2d.data).any()

    def test_invalid_dimension_raises(self):
        bias = nn.Parameter(torch.zeros(8))
        with pytest.raises(ValueError, match="Muon only supports 2-D matrix parameters"):
            Muon([bias])
