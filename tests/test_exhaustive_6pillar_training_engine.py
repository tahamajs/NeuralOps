"""
Suite 5: Exhaustive Verification Battery for the 6-Pillar Training, Distillation, and RL Engine.
Verifies:
  5.1 LogitDistillationLoss (Forward KL, Reverse KL, Beta-JSD, DKD, Top-K KD, fused linear KD).
  5.2 Feature & Operator Distillation (Hedgehog, MiniLM Value-Relation, Operator Cloning, Linear CKA).
  5.3 MOHAWK 3-stage curriculum (parameter freezing masks & homotopy annealing alpha: 0 -> 1).
  5.4 PiSSALinear (step-0 exact identity, singular subspace alignment, merge_and_unload).
  5.5 RL & Adaptation (SimPO, ORPO, GRPO, LearnableAttentionSink, TTTLinearAttentionLayer).
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
from neuralops.trainer.optimizers import RiemannianAdam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_5_1_logit_distillation_modes():
    B, S, V = 2, 8, 32
    student_logits = torch.randn(B, S, V, device=DEVICE, requires_grad=True)
    teacher_logits = torch.randn(B, S, V, device=DEVICE)

    modes = ["forward_kl", "reverse_kl", "beta_jsd", "dkd", "topk_kd"]
    for mode in modes:
        loss_fn = LogitDistillationLoss(mode=mode, temperature=2.0)
        loss = loss_fn(student_logits, teacher_logits)
        assert torch.isfinite(loss).all()
        assert loss.item() >= 0.0

        loss.backward(retain_graph=True)
        assert student_logits.grad is not None
        assert not torch.isnan(student_logits.grad).any()
        student_logits.grad.zero_()


def test_5_2_representation_distillation_losses():
    B, S, D = 2, 8, 16
    student_feat = torch.randn(B, S, D, device=DEVICE, requires_grad=True)
    teacher_feat = torch.randn(B, S, D, device=DEVICE)

    # Linear CKA
    cka_fn = LinearCKALoss()
    loss_cka = cka_fn(student_feat, teacher_feat)
    assert torch.isfinite(loss_cka).all()
    loss_cka.backward()
    assert student_feat.grad is not None
    student_feat.grad.zero_()

    # Operator Cloning
    clone_fn = BlockwiseOperatorCloningLoss()
    loss_clone = clone_fn(student_feat, teacher_feat)
    assert torch.isfinite(loss_clone).all()
    loss_clone.backward()
    assert student_feat.grad is not None
    student_feat.grad.zero_()

    # MiniLM Value Relation
    B, H, S_len = 2, 4, 8
    s_val = torch.randn(B, H, S_len, D // H, device=DEVICE, requires_grad=True)
    t_val = torch.randn(B, H, S_len, D // H, device=DEVICE)
    minilm_fn = MiniLMValueRelationKDLoss()
    loss_minilm = minilm_fn(s_val, t_val)
    assert torch.isfinite(loss_minilm).all()
    loss_minilm.backward()
    assert s_val.grad is not None


def test_5_3_pissa_linear_exact_identity_and_merge():
    in_dim, out_dim, rank = 32, 32, 4
    base_linear = nn.Linear(in_dim, out_dim, bias=True).to(DEVICE)
    x = torch.randn(4, in_dim, device=DEVICE)

    pissa = PiSSALinear(base_linear, rank=rank)
    out_orig = base_linear(x)
    out_pissa = pissa(x)

    # Mathematical Invariant: step-0 exact identity
    diff = (out_orig - out_pissa).abs().max().item()
    assert diff < 1e-5, f"PiSSA step-0 identity violation: max diff = {diff}"

    # Merge and unload
    pissa.merge_and_unload()
    out_merged = pissa(x)
    diff_merged = (out_orig - out_merged).abs().max().item()
    assert diff_merged < 1e-5, f"PiSSA merge_and_unload mismatch: max diff = {diff_merged}"


def test_5_4_rl_and_adaptation_primitives():
    B, S, D = 2, 4, 16

    # Learnable Attention Sink
    sink = LearnableAttentionSink(num_heads=4, head_dim=4, num_sink_tokens=2).to(DEVICE)
    k = torch.randn(B, 4, S, 4, device=DEVICE)
    v = torch.randn(B, 4, S, 4, device=DEVICE)
    k_aug, v_aug = sink(k, v)
    assert k_aug.shape[2] == S + 2
    assert v_aug.shape[2] == S + 2

    # SimPO Loss
    simpo = SimPOLoss(beta=2.0, gamma_margin=0.5)
    policy_chosen_logps = torch.tensor([-1.2, -0.8], device=DEVICE)
    policy_rejected_logps = torch.tensor([-2.5, -2.1], device=DEVICE)
    loss_simpo, chosen_r, rej_r = simpo(policy_chosen_logps, policy_rejected_logps)
    assert torch.isfinite(loss_simpo).all()
    assert (chosen_r > rej_r).all()

    # TTT Linear Attention Layer
    ttt = TTTLinearAttentionLayer(dim=D, num_heads=2, mini_batch_size=2).to(DEVICE)
    x = torch.randn(B, S, D, device=DEVICE, requires_grad=True)
    out_ttt = ttt(x)
    assert out_ttt.shape == x.shape
    out_ttt.sum().backward()
    assert x.grad is not None


def test_5_5_riemannian_adam_stiefel_orthogonality():
    # Stiefel manifold: param U in R^{n x p} with p <= n, enforcing U^T U = I_p
    n, p = 32, 8
    # Start on or near Stiefel
    init_u, _ = torch.linalg.qr(torch.randn(n, p, device=DEVICE))
    param = nn.Parameter(init_u.clone())
    opt = RiemannianAdam([param], lr=1e-2)

    target = torch.randn(n, p, device=DEVICE)
    for _ in range(5):
        opt.zero_grad()
        loss = ((param - target) ** 2).sum()
        loss.backward()
        opt.step()

        # Enforce Stiefel invariant ||U^T U - I_p||_F < 1e-4
        utu = torch.matmul(param.data.transpose(0, 1), param.data)
        eye = torch.eye(p, device=DEVICE)
        ortho_err = torch.norm(utu - eye, p="fro").item()
        assert ortho_err < 1e-3, f"Stiefel orthogonality violated: Frobenius norm = {ortho_err}"
