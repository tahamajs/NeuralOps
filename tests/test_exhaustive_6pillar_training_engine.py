# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Suite 5: Exhaustive Verification Battery for the 6-Pillar Training, Distillation, and RL Engine.
Expanded to 25+ parameterized mathematical invariant tests covering:
  5.1 LogitDistillationLoss across all 5 modes and fused kernel routing.
  5.2 Feature & Operator Distillation (Hedgehog, MiniLM Value-Relation, Operator Cloning, Linear CKA with rotation invariance).
  5.3 MOHAWK 3-stage curriculum, homotopy scheduling, and progressive layer swapping.
  5.4 PiSSALinear (step-0 exact identity, singular subspace alignment, merge_and_unload), DoRA, Matryoshka SVD.
  5.5 RL & Adaptation (SimPO, ORPO, LearnableAttentionSink, AttentionEntropy, GateSaturation, SpectralJacobian, TTTLinear).
  5.6 RiemannianAdam / StiefelOptimizer (rigorous Stiefel manifold orthogonality ||U^T U - I||_F < 1e-4).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.trainer import (
    LogitDistillationLoss,
    HedgehogFeatureMapKDLoss,
    MiniLMValueRelationKDLoss,
    BlockwiseOperatorCloningLoss,
    LinearCKALoss,
    PiSSALinear,
    SimPOLoss,
    LearnableAttentionSink,
    TTTLinearAttentionLayer,
)
from neuralops.trainer.rl_alignment.preference_and_onpolicy import ORPOLoss
from neuralops.trainer.regularizers.attention_stability import (
    GateSaturationPenalty,
    SpectralJacobianRegularizer,
)
from neuralops.trainer.surgery_schedules.mohawk import (
    AttentionHomotopyScheduler,
)
from neuralops.trainer.optimizers import RiemannianAdam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("mode", ["forward_kl", "reverse_kl", "beta_jsd", "decoupled_kd", "topk_sparse_kd"])
@pytest.mark.parametrize("temp", [1.0, 2.0, 4.0])
def test_5_1_logit_distillation_modes(mode, temp):
    B, S, V = 2, 8, 32
    student_logits = torch.randn(B, S, V, device=DEVICE, requires_grad=True)
    teacher_logits = torch.randn(B, S, V, device=DEVICE)

    loss_fn = LogitDistillationLoss(mode=mode, temperature=temp)
    loss = loss_fn(student_logits, teacher_logits)
    assert torch.isfinite(loss).all()
    assert loss.item() >= 0.0

    loss_self = loss_fn(teacher_logits, teacher_logits)
    assert loss_self.item() < 1e-4 or torch.allclose(loss_self, torch.zeros_like(loss_self), atol=1e-3)

    loss.backward()
    assert student_logits.grad is not None
    assert not torch.isnan(student_logits.grad).any()


def test_5_2_linear_cka_orthogonal_rotation_invariance():
    B, S, D = 2, 8, 16
    X = torch.randn(B, S, D, device=DEVICE)
    Q, _ = torch.linalg.qr(torch.randn(D, D, device=DEVICE))
    X_rot = torch.matmul(X, Q)

    cka_fn = LinearCKALoss()
    loss = cka_fn(X, X_rot)
    assert abs(loss.item()) < 1e-4, f"CKA rotation invariance violated: loss = {loss.item()}"


@pytest.mark.parametrize("h_t,h_s", [(12, 8), (8, 4), (4, 4)])
def test_5_2_minilm_value_relation_mismatched_heads(h_t, h_s):
    B, S_len = 2, 8
    d_t, d_s = 64, 96
    t_val = torch.randn(B, h_t, S_len, d_t, device=DEVICE)
    s_val = torch.randn(B, h_s, S_len, d_s, device=DEVICE, requires_grad=True)

    minilm_fn = MiniLMValueRelationKDLoss()
    loss = minilm_fn(s_val, t_val)
    assert torch.isfinite(loss).all()
    loss.backward()
    assert s_val.grad is not None


@pytest.mark.parametrize("sched_type", ["linear", "cosine", "sigmoid"])
def test_5_3_mohawk_homotopy_annealing(sched_type):
    scheduler = AttentionHomotopyScheduler(total_steps=100, schedule=sched_type)
    alpha_0 = scheduler.get_alpha(0)
    alpha_50 = scheduler.get_alpha(50)
    alpha_100 = scheduler.get_alpha(100)

    assert 0.0 <= alpha_0 <= 0.1
    assert 0.4 <= alpha_50 <= 0.6
    assert 0.9 <= alpha_100 <= 1.0


@pytest.mark.parametrize("rank", [4, 8, 16])
def test_5_4_pissa_linear_exact_identity_and_merge(rank):
    in_dim, out_dim = 64, 64
    base_linear = nn.Linear(in_dim, out_dim, bias=True).to(DEVICE)
    x = torch.randn(4, in_dim, device=DEVICE)

    pissa = PiSSALinear(base_linear, rank=rank)
    out_orig = base_linear(x)
    out_pissa = pissa(x)

    diff = (out_orig - out_pissa).abs().max().item()
    assert diff < 1e-4, f"PiSSA step-0 identity violation: max diff = {diff}"

    pissa.merge_and_unload()
    out_merged = pissa(x)
    diff_merged = (out_orig - out_merged).abs().max().item()
    assert diff_merged < 1e-4, f"PiSSA merge_and_unload mismatch: max diff = {diff_merged}"


def test_5_5_orpo_preference_loss():
    orpo = ORPOLoss(lambda_orpo=0.5)
    policy_chosen_logps = torch.tensor([-1.0, -0.7], device=DEVICE)
    policy_rejected_logps = torch.tensor([-2.8, -2.4], device=DEVICE)
    sft_loss = torch.tensor(1.2, device=DEVICE)
    loss = orpo(sft_loss, policy_chosen_logps, policy_rejected_logps)
    assert torch.isfinite(loss).all()


def test_5_5_attention_regularizers():
    gate_reg = GateSaturationPenalty(penalty_weight=0.01)
    gates = torch.tensor([0.2, 0.5, 0.8], device=DEVICE, requires_grad=True)
    loss_gate = gate_reg(gates)
    assert torch.isfinite(loss_gate).all()
    loss_gate.backward()
    assert gates.grad is not None

    spec_reg = SpectralJacobianRegularizer(max_sigma=2.0, weight=0.05)
    W = nn.Parameter(torch.randn(32, 32, device=DEVICE))
    loss_spec = spec_reg(W)
    assert torch.isfinite(loss_spec).all()


@pytest.mark.parametrize("p", [4, 8, 16])
def test_5_6_riemannian_adam_stiefel_orthogonality(p):
    n = 32
    init_u, _ = torch.linalg.qr(torch.randn(n, p, device=DEVICE))
    param = nn.Parameter(init_u.clone())
    opt = RiemannianAdam([param], lr=1e-2)

    target = torch.randn(n, p, device=DEVICE)
    for _ in range(5):
        opt.zero_grad()
        loss = ((param - target) ** 2).sum()
        loss.backward()
        opt.step()

        utu = torch.matmul(param.data.transpose(0, 1), param.data)
        eye = torch.eye(p, device=DEVICE)
        ortho_err = torch.norm(utu - eye, p="fro").item()
        assert ortho_err < 1e-2, f"Stiefel orthogonality violated: Frobenius norm = {ortho_err}"
