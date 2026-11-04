#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit verification suite for Pre-Benchmark Micro-Training & Surgical Healing Regimes (Phases 21-40).

Verifies:
1. Closed-form Ridge calibration with dynamic Tikhonov damping escalation.
2. 6-pillar multi-level distillation loss (Logit KL, MSE, MiniLM relation, CKA, Frobenius, Attention Stability).
3. MOHAWK Attention Homotopy scheduler.
4. Muon 5-step Newton-Schulz matrix optimizer and parameter splitting (Muon for 2D, AdamW for 1D).
5. WSD (Warmup-Stable-Decay) learning rate scheduler.
6. F-Micro natural gradient curvature damping.
7. DoRA magnitude/direction weight decomposition and merging.
8. TTT-Online test-time adaptation on evaluation prompts.
9. Homotopic Spectral SVD Annealing (HSA) singular value interpolation.
10. TrainAndBenchmarkPipeline orchestrator.
"""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuralops.benchmarks.eval_with_training import TrainAndBenchmarkPipeline
from neuralops.surgery.calibration import (
    align_output_projections,
    compute_ridge_calibration,
    ridge_regression_solve,
)
from neuralops.trainer.distillation.multi_level_kd import (
    LinearCKALoss,
    LogitDistillationLoss,
    MiniLMValueRelationKDLoss,
    MultiLevelKDTrainer,
)
from neuralops.trainer.optimizers.muon import (
    Muon,
    create_muon_adamw_optimizer,
    zeropower_via_newtonschulz5,
)
from neuralops.trainer.regimes.dora_healing import DoRAHealingHealer, DoRAProjectionWrapper
from neuralops.trainer.regimes.fisher_micro import (
    FisherMicroHealer,
    FMicroOptimizer,
    compute_empirical_diagonal_fisher,
)
from neuralops.trainer.regimes.spectral_annealing import SpectralAnnealer
from neuralops.trainer.regimes.ttt_online import TTTOnlineAdapter
from neuralops.trainer.schedulers import WSDScheduler
from neuralops.trainer.surgery_schedules.mohawk import AttentionHomotopyScheduler


class SimpleToyModel(nn.Module):
    """Synthetic toy model with 2D projections, 1D norm/bias, and gate parameter."""

    def __init__(self, in_features: int = 16, hidden: int = 32, num_classes: int = 8) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.gate = nn.Parameter(torch.zeros(hidden))
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor, output_hidden_states: bool = False):
        h0 = torch.relu(self.in_proj(x))
        q = self.q_proj(h0)
        k = self.k_proj(h0)
        v = self.v_proj(h0)
        # Simple dot-product style interaction
        attn = torch.sigmoid(q @ k.transpose(-1, -2) / (q.size(-1) ** 0.5))
        ctx = torch.matmul(attn, v)
        h1 = self.norm(self.out_proj(ctx) + h0)
        logits = self.classifier(h1)
        if output_hidden_states:
            return logits, [h0, h1]
        return logits


def test_ridge_calibration_and_damping_escalation():
    """Verify closed-form Ridge solver with condition number damping escalation."""
    torch.manual_seed(42)
    # 1. Normal well-conditioned inputs
    x = torch.randn(64, 16)
    w_true = torch.randn(16, 8)
    b_true = torch.randn(8)
    y = x @ w_true + b_true + 0.01 * torch.randn(64, 8)

    w_hat, b_hat = compute_ridge_calibration(x, y, ridge_lambda=1e-4, device="cpu")
    assert w_hat.shape == (16, 8)
    assert b_hat.shape == (8,)
    pred = (x - x.mean(0)) @ w_hat + b_hat + x.mean(0) @ w_hat
    assert torch.allclose(pred, y, atol=0.2)

    # 2. Ill-conditioned collinear matrix to trigger damping escalation
    x_ill = torch.ones(64, 16)
    x_ill[:, 1:] = x_ill[:, :1] + 1e-8 * torch.randn(64, 15)
    y_ill = torch.randn(64, 8)

    w_ill, b_ill = ridge_regression_solve(x_ill, y_ill, ridge_lambda=1e-5, device="cpu")
    assert not torch.isnan(w_ill).any()
    assert not torch.isnan(b_ill).any()

    # 3. align_output_projections execution test
    teacher = SimpleToyModel()
    student = copy.deepcopy(teacher)
    # perturb student classifier
    student.classifier.weight.data.add_(0.5 * torch.randn_like(student.classifier.weight.data))
    calib_batches = [torch.randn(8, 16) for _ in range(3)]
    align_output_projections(teacher, student, calib_batches, damping=1e-4, device="cpu")
    assert not torch.isnan(student.classifier.weight).any()


def test_6pillar_multilevel_kd_loss():
    """Verify MultiLevelKDTrainer coordinates all 6 pillars without NaN."""
    torch.manual_seed(42)
    teacher = SimpleToyModel()
    student = copy.deepcopy(teacher)

    kd_engine = MultiLevelKDTrainer(
        teacher=teacher,
        student=student,
        device="cpu",
        temperature=2.0,
        lambda_kd=1.0,
        lambda_mse=0.5,
        lambda_rel=0.1,
        lambda_cka=0.2,
        lambda_frob=0.05,
        lambda_stab=0.01,
    )

    batch = torch.randn(4, 10, 16)
    loss_dict = kd_engine.training_step(batch)

    assert "total_loss" in loss_dict
    assert "kd_loss" in loss_dict
    assert "mse_loss" in loss_dict
    assert "cka_loss" in loss_dict
    assert "rel_loss" in loss_dict
    assert "frob_loss" in loss_dict
    assert "stab_loss" in loss_dict

    total_loss = loss_dict["total_loss"]
    assert total_loss.item() > 0.0
    assert not torch.isnan(total_loss)

    total_loss.backward()
    for p in student.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_mohawk_attention_homotopy():
    """Verify MOHAWK cosine curriculum alpha transition."""
    scheduler = AttentionHomotopyScheduler(total_steps=100, schedule="cosine")
    assert pytest.approx(scheduler.get_alpha(0), abs=1e-5) == 0.0
    assert pytest.approx(scheduler.get_alpha(50), abs=1e-5) == 0.5
    assert pytest.approx(scheduler.get_alpha(100), abs=1e-5) == 1.0


def test_muon_optimizer_and_parameter_splitting():
    """Verify 5-step Newton-Schulz iteration, parameter routing, and optimizer steps."""
    torch.manual_seed(42)
    g = torch.randn(32, 16)
    ortho = zeropower_via_newtonschulz5(g, steps=5)
    assert ortho.shape == (32, 16)
    assert not torch.isnan(ortho).any()

    model = SimpleToyModel()
    opt = create_muon_adamw_optimizer(model, lr_muon=0.02, lr_adam=1e-3)
    assert len(opt.param_groups) == 2

    # Verify 2D routed to Muon, 1D routed to AdamW
    muon_group = next(g for g in opt.param_groups if g["optimizer"] == "muon")
    adam_group = next(g for g in opt.param_groups if g["optimizer"] == "adamw")

    for p in muon_group["params"]:
        assert p.ndim == 2
    for p in adam_group["params"]:
        assert p.ndim != 2

    x = torch.randn(4, 10, 16)
    loss = model(x).sum()
    loss.backward()
    opt.step()


def test_wsd_scheduler():
    """Verify Warmup-Stable-Decay schedule phases."""
    model = nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = WSDScheduler(
        opt,
        warmup_steps=10,
        stable_steps=70,
        decay_steps=20,
    )

    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        opt.step()
        scheduler.step()

    # Warmup increases
    assert lrs[0] < lrs[9]
    # Stable phase holds peak
    assert pytest.approx(lrs[15], abs=1e-5) == 1.0
    assert pytest.approx(lrs[75], abs=1e-5) == 1.0
    # Decay phase decreases
    assert lrs[99] < lrs[80]


def test_fisher_micro_regime():
    """Verify empirical diagonal Fisher computation and natural gradient stepping."""
    torch.manual_seed(42)
    model = SimpleToyModel()
    dataset = TensorDataset(torch.randn(16, 5, 16))
    loader = DataLoader(dataset, batch_size=4)

    # Test standalone compute_empirical_diagonal_fisher
    f_diag_standalone = compute_empirical_diagonal_fisher(model, loader, num_samples=16, device="cpu")
    assert len(f_diag_standalone) > 0

    healer = FisherMicroHealer(model, damping=1e-3, device="cpu")
    f_diag = healer.compute_empirical_fisher(loader, num_samples=16)
    assert len(f_diag) > 0

    opt = healer.create_optimizer(lr=1e-3, damping=1e-3)
    x = torch.randn(2, 5, 16)
    out = model(x)
    loss = out.sum()
    loss.backward()
    opt.step()


def test_dora_healing_regime():
    """Verify DoRA magnitude/direction split, forward pass, and weight collapse."""
    torch.manual_seed(42)
    model = SimpleToyModel()
    healer = DoRAHealingHealer(model, rank=4, lora_alpha=8.0)
    injected_count = healer.inject_dora_adapters()
    assert injected_count > 0

    # Test single wrapper collapse_adapters
    wrapper = next(iter(healer.injected_wrappers.values()))
    collapsed_linear = wrapper.collapse_adapters()
    assert isinstance(collapsed_linear, nn.Linear)

    x = torch.randn(2, 5, 16)
    out_adapted = model(x)
    assert out_adapted.shape == (2, 5, 8)

    healer.collapse_adapters()
    assert len(healer.injected_wrappers) == 0
    out_collapsed = model(x)
    assert out_collapsed.shape == (2, 5, 8)


def test_ttt_online_regime():
    """Verify TTT-Online adaptation on prompt batches."""
    torch.manual_seed(42)
    model = SimpleToyModel()
    ttt = TTTOnlineAdapter(model, inner_lr=1e-3, inner_steps=1)
    prompt = torch.randn(2, 6, 16)
    res = ttt.adapt_on_prompt(prompt)
    assert "ttt_loss" in res


def test_spectral_annealing_regime():
    """Verify Homotopic Spectral SVD singular value annealing."""
    torch.manual_seed(42)
    teacher = SimpleToyModel()
    student = copy.deepcopy(teacher)

    annealer = SpectralAnnealer(teacher, student, total_steps=50)
    assert len(annealer.spectral_data) > 0

    gamma_0 = annealer.get_gamma(0)
    gamma_half = annealer.get_gamma(25)
    gamma_1 = annealer.get_gamma(50)

    assert pytest.approx(gamma_0, abs=1e-5) == 0.0
    assert pytest.approx(gamma_half, abs=1e-5) == 0.5
    assert pytest.approx(gamma_1, abs=1e-5) == 1.0

    annealer.step(25)


def test_train_and_benchmark_pipeline_micro_train():
    """Verify TrainAndBenchmarkPipeline.micro_train_healer executes across steps."""
    torch.manual_seed(42)
    teacher = SimpleToyModel()
    student = copy.deepcopy(teacher)

    dataset = TensorDataset(torch.randn(20, 5, 16))
    loader = DataLoader(dataset, batch_size=4)

    pipeline = TrainAndBenchmarkPipeline(output_dir="/tmp/test_healed_sweep", device="cpu")
    telemetry = pipeline.micro_train_healer(
        teacher=teacher,
        student=student,
        train_loader=loader,
        steps=5,
        lr=1e-3,
        optimizer_type="muon",
        regime="kd_standard",
    )

    assert "initial_loss" in telemetry
    assert "final_healed_loss" in telemetry
    assert "loss_reduction_pct" in telemetry
    assert telemetry["pre_train_steps"] == 5
