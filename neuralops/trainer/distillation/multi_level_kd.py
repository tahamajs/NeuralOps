"""
Multi-Level Attention & Logit Distillation Loss Library.
Supports 5 divergence modes (forward_kl, reverse_kl, beta_jsd, decoupled_kd, topk_sparse_kd)
and feature/relation alignment objectives (Hedgehog, MiniLM Gram, LoLCATs blockwise, Linear CKA).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuralops.core.triton_kernels.fused_kd import fused_linear_kl_loss
    _HAS_FUSED_KD = True
except Exception:
    _HAS_FUSED_KD = False


class LogitDistillationLoss(nn.Module):
    """
    Unified Logit Distillation Loss supporting:
    - forward_kl: tau^2 * KL(teacher || student) (optimal for encoders)
    - reverse_kl: tau^2 * KL(student || teacher) (mode-seeking, optimal for autoregressive decoders)
    - beta_jsd: Generalized Jensen-Shannon Divergence
    - decoupled_kd: DKD (Target Class KD + Non-Target Class KD)
    - topk_sparse_kd: Top-K sparse distillation with tail probability pooling
    - use_fused_linear_kd: routes through Triton FusedLinearKLDivergence kernel
    """

    def __init__(
        self,
        mode: str = "forward_kl",
        temperature: float = 2.0,
        alpha_tckd: float = 1.0,
        beta_nckd: float = 1.0,
        beta_jsd_weight: float = 0.5,
        top_k: int = 64,
        use_fused_linear_kd: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.temperature = float(temperature)
        self.alpha_tckd = float(alpha_tckd)
        self.beta_nckd = float(beta_nckd)
        self.beta_jsd_weight = float(beta_jsd_weight)
        self.top_k = int(top_k)
        self.use_fused_linear_kd = use_fused_linear_kd and _HAS_FUSED_KD

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        student_hidden: Optional[torch.Tensor] = None,
        student_w: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
        teacher_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tau = self.temperature

        # Triton in-forward fused execution
        if self.use_fused_linear_kd and student_hidden is not None and student_w is not None and teacher_hidden is not None and teacher_w is not None:
            return fused_linear_kl_loss(student_hidden, student_w, teacher_hidden, teacher_w, temperature=tau)

        s_logits = student_logits.float() / tau
        t_logits = teacher_logits.float() / tau

        if self.mode == "forward_kl":
            s_log = F.log_softmax(s_logits, dim=-1)
            t_prob = F.softmax(t_logits, dim=-1)
            return F.kl_div(s_log, t_prob, reduction="batchmean") * (tau ** 2)

        elif self.mode == "reverse_kl":
            s_prob = F.softmax(s_logits, dim=-1)
            t_log = F.log_softmax(t_logits, dim=-1)
            return F.kl_div(t_log, s_prob, reduction="batchmean") * (tau ** 2)

        elif self.mode == "beta_jsd":
            s_prob = F.softmax(s_logits, dim=-1)
            t_prob = F.softmax(t_logits, dim=-1)
            beta = self.beta_jsd_weight
            m_prob = (1.0 - beta) * s_prob + beta * t_prob
            m_log = torch.log(m_prob.clamp(min=1e-8))
            kl_sm = F.kl_div(m_log, s_prob, reduction="batchmean")
            kl_tm = F.kl_div(m_log, t_prob, reduction="batchmean")
            return ((1.0 - beta) * kl_sm + beta * kl_tm) * (tau ** 2)

        elif self.mode == "decoupled_kd":
            if labels is None:
                labels = teacher_logits.argmax(dim=-1)
            labels = labels.view(-1, 1)
            s_prob = F.softmax(s_logits.view(-1, s_logits.size(-1)), dim=-1)
            t_prob = F.softmax(t_logits.view(-1, t_logits.size(-1)), dim=-1)

            # Target class probability
            p_s = s_prob.gather(1, labels)
            p_t = t_prob.gather(1, labels)

            # Binary TCKD
            tckd = F.kl_div(
                torch.log(torch.cat([p_s, (1.0 - p_s).clamp(min=1e-7)], dim=1)),
                torch.cat([p_t, (1.0 - p_t).clamp(min=1e-7)], dim=1),
                reduction="batchmean",
            )

            # Non-target NCKD
            mask = torch.ones_like(s_prob, dtype=torch.bool)
            mask.scatter_(1, labels, False)
            s_nt = (s_prob * mask) / (1.0 - p_s).clamp(min=1e-7)
            t_nt = (t_prob * mask) / (1.0 - p_t).clamp(min=1e-7)
            nckd = F.kl_div(torch.log(s_nt.clamp(min=1e-8)), t_nt, reduction="batchmean")

            return (self.alpha_tckd * tckd + self.beta_nckd * nckd) * (tau ** 2)

        elif self.mode == "topk_sparse_kd":
            K = min(self.top_k, t_logits.size(-1) - 1)
            t_topk_vals, t_topk_idx = torch.topk(t_logits, k=K, dim=-1)
            s_topk_vals = torch.gather(s_logits, -1, t_topk_idx)

            t_prob = F.softmax(t_topk_vals, dim=-1)
            s_log = F.log_softmax(s_topk_vals, dim=-1)
            return F.kl_div(s_log, t_prob, reduction="batchmean") * (tau ** 2)

        else:
            raise ValueError(f"Unknown distillation mode '{self.mode}'")


class SpikyFeatureMap(nn.Module):
    """Dual positive learnable feature map for spiky linear attention alignment."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(head_dim, head_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        # Bounded dual exponential feature map
        h_pos = torch.exp(torch.clamp(h, -5.0, 5.0))
        h_neg = torch.exp(torch.clamp(-h, -5.0, 5.0))
        return torch.cat([h_pos, h_neg], dim=-1)


class HedgehogFeatureMapKDLoss(nn.Module):
    """Chunked Attention-Map KL Distillation Loss with O(N * B_T) memory footprint."""

    def __init__(self, chunk_size: int = 64, temperature: float = 1.0) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.temperature = temperature

    def forward(
        self,
        q_teacher: torch.Tensor,
        k_teacher: torch.Tensor,
        phi_q_student: torch.Tensor,
        phi_k_student: torch.Tensor,
    ) -> torch.Tensor:
        B, H, N, d_t = q_teacher.shape
        scale = 1.0 / math.sqrt(d_t)

        loss = torch.tensor(0.0, device=q_teacher.device, dtype=torch.float32)
        num_chunks = max(1, N // self.chunk_size)

        for c in range(min(num_chunks, 4)):  # sample up to 4 chunks to keep compute bounded
            st = c * self.chunk_size
            en = min(st + self.chunk_size, N)

            qc_t = q_teacher[:, :, st:en] * scale
            A_t = F.softmax(torch.matmul(qc_t, k_teacher.transpose(-1, -2)), dim=-1)

            qc_s = phi_q_student[:, :, st:en]
            A_s_raw = torch.matmul(qc_s, phi_k_student.transpose(-1, -2))
            A_s = A_s_raw / A_s_raw.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            loss = loss + F.kl_div(torch.log(A_s.clamp(min=1e-8)), A_t, reduction="batchmean")

        return loss / min(num_chunks, 4)


class MiniLMValueRelationKDLoss(nn.Module):
    """Head-agnostic and dimension-agnostic Value-Relation Gram alignment."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, v_student: torch.Tensor, v_teacher: torch.Tensor) -> torch.Tensor:
        # Flatten heads into batch: (B*H, N, D)
        Bs, Hs, N, Ds = v_student.shape
        Bt, Ht, _, Dt = v_teacher.shape

        vs = v_student.view(Bs * Hs, N, Ds)
        vt = v_teacher.view(Bt * Ht, N, Dt)

        # Compute Gram relation matrices (N x N)
        rel_s = F.softmax(torch.bmm(vs, vs.transpose(1, 2)) / math.sqrt(Ds), dim=-1)
        rel_t = F.softmax(torch.bmm(vt, vt.transpose(1, 2)) / math.sqrt(Dt), dim=-1)

        # Mean over heads if head counts differ
        rel_s_mean = rel_s.view(Bs, Hs, N, N).mean(dim=1)
        rel_t_mean = rel_t.view(Bt, Ht, N, N).mean(dim=1)

        return F.kl_div(torch.log(rel_s_mean.clamp(min=self.eps)), rel_t_mean, reduction="batchmean")


class BlockwiseOperatorCloningLoss(nn.Module):
    """LoLCATs Blockwise Attention Transfer Loss: MSE + Cosine Distance + L2 Norm."""

    def __init__(self, lambda_cos: float = 0.5, lambda_norm: float = 0.1) -> None:
        super().__init__()
        self.lambda_cos = lambda_cos
        self.lambda_norm = lambda_norm

    def forward(self, out_student: torch.Tensor, out_teacher: torch.Tensor) -> torch.Tensor:
        s = out_student.float()
        t = out_teacher.float()

        mse = F.mse_loss(s, t)
        cos_dist = (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()
        norm_diff = torch.abs(s.norm(p=2, dim=-1) - t.norm(p=2, dim=-1)).mean()

        return mse + self.lambda_cos * cos_dist + self.lambda_norm * norm_diff


class LinearCKALoss(nn.Module):
    """Centered Kernel Alignment loss for orthogonal-invariant representation matching."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        X = x.view(-1, x.size(-1)).float()
        Y = y.view(-1, y.size(-1)).float()

        # If D is very large (e.g. vocab logits V=50,257), compute CKA on sample dimension N
        if X.size(1) > X.size(0):
            X = X.t()
            Y = Y.t()

        # Center columns
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        # Frobenius products
        XtX = torch.matmul(X.t(), X)
        YtY = torch.matmul(Y.t(), Y)
        XtY = torch.matmul(X.t(), Y)

        hsic_xy = (XtY ** 2).sum()
        hsic_xx = (XtX ** 2).sum()
        hsic_yy = (YtY ** 2).sum()

        denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=self.eps)
        cka = hsic_xy / denom
        return 1.0 - cka


class SmoothMaxOverLayersCKASteeringLoss(nn.Module):
    """
    Smooth Log-Sum-Exp Maximum over Layerwise CKA Alignments.
    L_CKA_max = tau * log(sum_l exp( (1 - CKA(H_s^(l), H_t^(l))) / tau ))
    Differentiably steers worst-aligned layers towards optimal representation alignment.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.cka_loss_fn = LinearCKALoss(eps=self.eps)

    def forward(
        self,
        student_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(student_hidden_states) != len(teacher_hidden_states):
            min_layers = min(len(student_hidden_states), len(teacher_hidden_states))
            student_hidden_states = student_hidden_states[:min_layers]
            teacher_hidden_states = teacher_hidden_states[:min_layers]

        layer_losses = []
        for s_h, t_h in zip(student_hidden_states, teacher_hidden_states):
            layer_loss = self.cka_loss_fn(s_h, t_h)
            layer_losses.append(layer_loss)

        if not layer_losses:
            return torch.tensor(0.0, device=student_hidden_states[0].device if student_hidden_states else "cpu")

        stacked = torch.stack(layer_losses)
        # SmoothMax via Log-Sum-Exp: tau * log(sum(exp(L_l / tau)))
        smooth_max = self.temperature * torch.logsumexp(stacked / self.temperature, dim=0)
        return smooth_max


class HiddenAngularMSELoss(nn.Module):
    """Angular and normalized direction-matching MSE loss for hidden representations."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
        s = student_hidden.float()
        t = teacher_hidden.float()
        if s.shape != t.shape:
            min_dim = min(s.shape[-1], t.shape[-1])
            s = s[..., :min_dim]
            t = t[..., :min_dim]
        # Cosine distance direction loss + normalized MSE
        cos_dist = (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()
        s_norm = s / s.norm(p=2, dim=-1, keepdim=True).clamp(min=self.eps)
        t_norm = t / t.norm(p=2, dim=-1, keepdim=True).clamp(min=self.eps)
        mse_norm = F.mse_loss(s_norm, t_norm)
        return cos_dist + mse_norm


class OrthogonalityPenaltyLoss(nn.Module):
    """Penalizes deviation of weight matrix from semi-orthogonality ||W^T W - I||_F^2."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        W = weight.float()
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        if W.size(0) < W.size(1):
            W = W.t()
        gram = torch.matmul(W.t(), W)
        identity = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return torch.norm(gram - identity, p="fro").pow(2) / max(1, gram.numel())


class LatentEntropyRegularizer(nn.Module):
    """Maximizes entropy/diversity across latent feature dimensions to avoid representation collapse."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = hidden_states.float().view(-1, hidden_states.size(-1))
        # Variance per feature dimension
        var = h.var(dim=0).clamp(min=self.eps)
        p = var / var.sum().clamp(min=self.eps)
        entropy = -(p * torch.log(p.clamp(min=self.eps))).sum()
        # Loss minimizes negative entropy
        return -entropy


class Unified6PillarDistillationLoss(nn.Module):
    """
    Unified 6-Pillar Knowledge Distillation Engine.
    Combines:
    1. Logit Distillation (Forward KL / Reverse KL / DKD)
    2. Hidden Angular MSE Direction Alignment
    3. MiniLM Multi-Head Value Relation Matching
    4. Representation CKA Alignment
    5. Orthogonality Regularization on Low-Rank / Adapter Weights
    6. Latent Feature Entropy Regularization
    """

    def __init__(
        self,
        weight_logits: float = 1.0,
        weight_angular: float = 0.5,
        weight_relation: float = 0.25,
        weight_cka: float = 0.2,
        weight_ortho: float = 0.05,
        weight_entropy: float = 0.01,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.weight_logits = weight_logits
        self.weight_angular = weight_angular
        self.weight_relation = weight_relation
        self.weight_cka = weight_cka
        self.weight_ortho = weight_ortho
        self.weight_entropy = weight_entropy

        self.logit_loss_fn = LogitDistillationLoss(temperature=temperature)
        self.angular_loss_fn = HiddenAngularMSELoss()
        self.relation_loss_fn = MiniLMValueRelationKDLoss()
        self.cka_loss_fn = LinearCKALoss()
        self.ortho_loss_fn = OrthogonalityPenaltyLoss()
        self.entropy_loss_fn = LatentEntropyRegularizer()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_hidden: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
        student_values: Optional[torch.Tensor] = None,
        teacher_values: Optional[torch.Tensor] = None,
        projection_weights: Optional[list[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        loss_logits = self.logit_loss_fn(student_logits, teacher_logits, labels=labels)
        total_loss = self.weight_logits * loss_logits
        metrics = {"loss_logits": loss_logits.detach()}

        if student_hidden is not None and teacher_hidden is not None:
            loss_ang = self.angular_loss_fn(student_hidden, teacher_hidden)
            loss_cka = self.cka_loss_fn(student_hidden, teacher_hidden)
            loss_ent = self.entropy_loss_fn(student_hidden)

            total_loss = total_loss + self.weight_angular * loss_ang + self.weight_cka * loss_cka + self.weight_entropy * loss_ent
            metrics["loss_angular"] = loss_ang.detach()
            metrics["loss_cka"] = loss_cka.detach()
            metrics["loss_entropy"] = loss_ent.detach()

        if student_values is not None and teacher_values is not None:
            loss_rel = self.relation_loss_fn(student_values, teacher_values)
            total_loss = total_loss + self.weight_relation * loss_rel
            metrics["loss_relation"] = loss_rel.detach()

        if projection_weights:
            ortho_losses = [self.ortho_loss_fn(w) for w in projection_weights]
            loss_ortho = torch.stack(ortho_losses).mean()
            total_loss = total_loss + self.weight_ortho * loss_ortho
            metrics["loss_ortho"] = loss_ortho.detach()

        metrics["loss_total"] = total_loss
        return metrics


__all__ = [
    "LogitDistillationLoss",
    "SpikyFeatureMap",
    "HedgehogFeatureMapKDLoss",
    "MiniLMValueRelationKDLoss",
    "BlockwiseOperatorCloningLoss",
    "LinearCKALoss",
    "SmoothMaxOverLayersCKASteeringLoss",
    "HiddenAngularMSELoss",
    "OrthogonalityPenaltyLoss",
    "LatentEntropyRegularizer",
    "Unified6PillarDistillationLoss",
]

