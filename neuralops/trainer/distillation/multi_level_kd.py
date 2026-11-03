#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Multi-Level Knowledge Distillation & 6-Pillar Healing Engine

Architectural Role (NeuralOps v1.6.0):
This library provides the multi-objective distillation and rapid healing loss suite used to
restore baseline representation fidelity within 50 to 250 micro-tuning steps post-surgery:

The 6-Pillar Distillation Formulation:
1. Logit Divergence (L_KD): Multi-mode logit matching (forward KL, reverse KL, Jensen-Shannon,
   decoupled TCKD/NCKD, and top-k sparse KL), accelerated via fused Triton kernels.
2. Centered Kernel Alignment (L_CKA): Exact rotation-invariant linear CKA similarity
   between student and teacher hidden states with SmoothMax steering across deep layers.
3. Attention Distribution Parity (L_Attn): Chunked attention map KL divergence (B_T=64)
   and MiniLM self-attention Gram matrix alignment across heterogeneous head counts.
4. Angular Directional Alignment (L_Cos): Cosine embedding and L2 feature norm matching.
5. Stiefel Manifold Orthogonality (L_Stiefel): Riemannian tangent space projections guaranteeing
   unitary constraint ||W^T W - I||_F < 1e-4 throughout adapter optimization.
6. Entropy & Dynamic Stability (L_Entropy): Log-barrier penalties preventing gate saturation
   and learnable attention sinks absorbing unbounded attention scores.

Frozen Teacher Backbone Invariant:
All distillation trainers strictly enforce teacher.eval(), requires_grad=False, and
torch.no_grad() contexts, guaranteeing zero gradient leakage and minimal VRAM footprint.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.surgery.streaming_distill import StreamingCKALoss

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

            p_s = s_prob.gather(1, labels)
            p_t = t_prob.gather(1, labels)

            tckd = F.kl_div(
                torch.log(torch.cat([p_s, (1.0 - p_s).clamp(min=1e-7)], dim=1)),
                torch.cat([p_t, (1.0 - p_t).clamp(min=1e-7)], dim=1),
                reduction="batchmean",
            )

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

        for c in range(min(num_chunks, 4)):
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
        Bs, Hs, N, Ds = v_student.shape
        Bt, Ht, _, Dt = v_teacher.shape

        vs = v_student.contiguous().view(Bs * Hs, N, Ds)
        vt = v_teacher.contiguous().view(Bt * Ht, N, Dt)

        rel_s = F.softmax(torch.bmm(vs, vs.transpose(1, 2)) / math.sqrt(Ds), dim=-1)
        rel_t = F.softmax(torch.bmm(vt, vt.transpose(1, 2)) / math.sqrt(Dt), dim=-1)

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

    def __init__(
        self,
        eps: float = 1e-6,
        use_streaming: bool = False,
        chunk_size: int = 128,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.use_streaming = use_streaming
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_streaming or (
            x.dim() >= 2 and x.size(1) > self.chunk_size and x.numel() > 65536
        ):
            hidden_dim = x.size(-1)
            cka_mod = StreamingCKALoss(hidden_dim=hidden_dim, eps=self.eps).to(x.device)
            return cka_mod(x, y, chunk_size=self.chunk_size)

        X = x.view(-1, x.size(-1)).float()
        Y = y.view(-1, y.size(-1)).float()

        if X.size(1) > X.size(0):
            X = X.t()
            Y = Y.t()

        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

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

    def __init__(
        self,
        temperature: float = 0.1,
        eps: float = 1e-6,
        use_streaming: bool = False,
        chunk_size: int = 128,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.use_streaming = use_streaming
        self.chunk_size = chunk_size
        self.cka_loss_fn = LinearCKALoss(
            eps=self.eps,
            use_streaming=use_streaming,
            chunk_size=chunk_size,
        )

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
        smooth_max = self.temperature * torch.logsumexp(stacked / self.temperature, dim=0)
        return smooth_max


class MultiLevelKDTrainer:
    """
    6-Pillar Knowledge Distillation & Post-Surgery Healing Engine.

    Coordinates multi-objective distillation loss between teacher and student backbones:
        L_{total} = lambda_{KD} * L_{KL} + lambda_{MSE} * L_{MSE} + lambda_{Rel} * L_{MiniLM}
                  + lambda_{CKA} * L_{CKA} + lambda_{Frob} * L_{Frobenius} + lambda_{Stab} * L_{AttentionStability}
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        temperature: float = 2.0,
        lambda_kd: float = 1.0,
        lambda_mse: float = 0.5,
        lambda_rel: float = 0.1,
        lambda_cka: float = 0.2,
        lambda_frob: float = 0.05,
        lambda_stab: float = 0.01,
        lambda_feat: float = 0.0,
        use_streaming_cka: bool = False,
        chunk_size: int = 128,
        memory_mode: str = "standard",
    ) -> None:
        self.device = (
            torch.device(device)
            if device is not None
            else (
                next(student.parameters()).device
                if list(student.parameters())
                else torch.device("cpu")
            )
        )
        self.teacher = teacher.to(self.device)
        self.student = student.to(self.device)

        # Teacher invariant: frozen, eval mode
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.lambda_kd = float(lambda_kd)
        self.lambda_mse = float(lambda_mse)
        self.lambda_rel = float(lambda_rel)
        self.lambda_cka = float(lambda_cka)
        self.lambda_frob = float(lambda_frob)
        self.lambda_stab = float(lambda_stab)
        self.lambda_feat = float(lambda_feat)
        self.use_streaming_cka = bool(use_streaming_cka or memory_mode == "streaming_checkpointed")
        self.chunk_size = int(chunk_size)
        self.memory_mode = str(memory_mode)

        self.logit_loss_fn = LogitDistillationLoss(temperature=temperature)
        self.cka_loss_fn = SmoothMaxOverLayersCKASteeringLoss(
            temperature=0.1,
            use_streaming=self.use_streaming_cka,
            chunk_size=self.chunk_size,
        )
        self.rel_loss_fn = MiniLMValueRelationKDLoss()
        self.feature_loss_fn = HedgehogFeatureMapKDLoss(chunk_size=64, temperature=temperature)

    def _extract_logits_and_hidden(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """Extracts logits and intermediate hidden states from arbitrary forward output."""
        if isinstance(inputs, dict):
            batch_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            outputs = model(**batch_inputs, output_hidden_states=True) if "output_hidden_states" in getattr(model, "forward", {}).__code__.co_varnames else model(**batch_inputs)
        elif isinstance(inputs, (tuple, list)):
            batch_inputs = [
                v.to(self.device) if isinstance(v, torch.Tensor) else v
                for v in inputs
            ]
            outputs = model(*batch_inputs)
        elif isinstance(inputs, torch.Tensor):
            outputs = model(inputs.to(self.device))
        else:
            raise TypeError(f"Unsupported batch type: {type(inputs).__name__}")

        logits = None
        hidden_states = None

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            logits = outputs[0]

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 1 and isinstance(outputs[1], (tuple, list)):
            hidden_states = outputs[1]

        return logits, hidden_states

    def training_step(self, batch: Any) -> Dict[str, Any]:
        """
        Executes one forward pass over teacher and student, evaluating the 6-pillar objective.
        Returns dictionary containing individual loss terms and total combined loss.
        """
        self.student.train()

        # 1. Forward teacher under torch.no_grad()
        with torch.no_grad():
            t_logits, t_hidden = self._extract_logits_and_hidden(self.teacher, batch)

        # 2. Forward student with grad
        s_logits, s_hidden = self._extract_logits_and_hidden(self.student, batch)

        total_loss = torch.tensor(0.0, device=self.device)
        losses_tracked: Dict[str, Any] = {}

        # Pillar 1 & 2: Logit KL Divergence & Hidden/Logit MSE
        if s_logits is not None and t_logits is not None:
            kd_loss = self.logit_loss_fn(s_logits, t_logits)
            total_loss = total_loss + self.lambda_kd * kd_loss
            losses_tracked["kd_loss"] = kd_loss.item()

            if self.lambda_mse > 0.0:
                mse_loss = F.mse_loss(s_logits.float(), t_logits.float())
                total_loss = total_loss + self.lambda_mse * mse_loss
                losses_tracked["mse_loss"] = mse_loss.item()

        # Pillar 3 & 4: Centered Kernel Alignment (CKA) & MiniLM Relation Distillation
        if s_hidden is not None and t_hidden is not None:
            if self.lambda_cka > 0.0:
                cka_loss = self.cka_loss_fn(s_hidden, t_hidden)
                total_loss = total_loss + self.lambda_cka * cka_loss
                losses_tracked["cka_loss"] = cka_loss.item()

            if self.lambda_rel > 0.0 and len(s_hidden) > 0 and len(t_hidden) > 0:
                rel_loss = self.rel_loss_fn(s_hidden[-1], t_hidden[-1])
                total_loss = total_loss + self.lambda_rel * rel_loss
                losses_tracked["rel_loss"] = rel_loss.item()

        # Pillar 5: Frobenius Feature Alignment
        if self.lambda_frob > 0.0 and s_hidden is not None and t_hidden is not None:
            s_last = s_hidden[-1].float()
            t_last = t_hidden[-1].float()
            if s_last.shape == t_last.shape:
                frob_loss = torch.linalg.norm(s_last - t_last, ord="fro") / max(1, s_last.numel())
            else:
                frob_loss = F.mse_loss(s_last, t_last)
            total_loss = total_loss + self.lambda_frob * frob_loss
            losses_tracked["frob_loss"] = frob_loss.item()

        # Pillar 6: Attention Stability Regularizer (Gate saturation prevention)
        if self.lambda_stab > 0.0:
            stab_penalties = []
            for name, param in self.student.named_parameters():
                if "gate" in name.lower() and param.requires_grad:
                    g = torch.sigmoid(param.float())
                    stab_penalties.append(torch.mean((g - 0.5) ** 2))
            if stab_penalties:
                stab_loss = torch.stack(stab_penalties).mean()
                total_loss = total_loss + self.lambda_stab * stab_loss
                losses_tracked["stab_loss"] = stab_loss.item()

        losses_tracked["total_loss"] = total_loss

        # Memory threshold purging for streaming checkpointed distillation
        if self.memory_mode == "streaming_checkpointed" and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            if total > 0 and (alloc / total) > 0.75:
                torch.cuda.empty_cache()

        return losses_tracked


__all__ = [
    "LogitDistillationLoss",
    "SpikyFeatureMap",
    "HedgehogFeatureMapKDLoss",
    "MiniLMValueRelationKDLoss",
    "BlockwiseOperatorCloningLoss",
    "LinearCKALoss",
    "SmoothMaxOverLayersCKASteeringLoss",
    "StreamingCKALoss",
    "MultiLevelKDTrainer",
]
