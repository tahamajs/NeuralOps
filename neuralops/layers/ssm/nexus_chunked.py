# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Chunked causal NEXUS attention computation."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

def _chunked_causal_nexus(
    self,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    token_valid: torch.Tensor | None = None,
    chunk_size: int = 256,
    num_landmarks: int = 32,
) -> torch.Tensor:
    """
    Chunked Causal NEXUS with Nyström-style inter-chunk summary.

    Algorithm:
      For each chunk c = [start, end):
        1. Intra-chunk: XAttention block-sparse causal attention + softpick.
        2. Inter-chunk (c > 0): Nyström with landmarks pooled from ALL
           previous chunks.
           F1 = softmax(Q_c @ K_landmarks^T)        (B, H, C, m)
           F2 = softmax(Q_landmarks @ K_landmarks^T) (B, H, m, m)
           F3 = softmax(Q_landmarks @ K_prev^T)      (B, H, m, start)
           out_cross = F1 @ pinv(F2) @ F3 @ V_prev
        3. Combine intra + cross.

    Args:
        Q, K, V: (B, H, N, D)
        token_valid: (B, N) boolean mask or None
        chunk_size: int, size of each causal chunk
        num_landmarks: int, landmarks per chunk
    Returns:
        context: (B, H, N, D)
    """
    original_dtype = Q.dtype
    B, H, N, D = Q.shape
    device = Q.device

    with torch.amp.autocast(device_type=device.type, enabled=False):
        Q = Q.float()
        K = K.float()
        V = V.float()

        scale = 1.0 / math.sqrt(max(int(D), 1))
        m = min(num_landmarks, N)

        if N <= chunk_size:
            scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale
            causal = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)
            if self.seer_gate is not None and getattr(self.cfg, "seer_enabled", False):
                bs = max(1, self.cfg.block_size)
                nb = (N + bs - 1) // bs
                if nb > 1:
                    seer_mask = self.seer_gate(Q, K, nb, training=self.training).float()
                    expanded_seer = seer_mask.repeat_interleave(bs, dim=2).repeat_interleave(bs, dim=3)[:, :, :N, :N]
                    scores = scores + torch.tanh(self.seer_beta.float()).to(scores.dtype) * (expanded_seer - 1.0) * 10000.0
            if token_valid is not None:
                scores = scores.masked_fill((~token_valid.bool()).unsqueeze(1).unsqueeze(2), -1e4)
            if self.cfg.softpick_enabled:
                attn = softpick(scores, dim=-1, temperature=self.cfg.softpick_temperature)
            else:
                scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
                attn = F.softmax(scores, dim=-1)
            context = torch.matmul(attn, V).to(original_dtype)
            return self.dropout(context)

        num_chunks = (N + chunk_size - 1) // chunk_size
        outputs: list[torch.Tensor] = []

        for c_idx in range(num_chunks):
            start = c_idx * chunk_size
            end = min(start + chunk_size, N)
            C_len = end - start

            Q_c = Q[:, :, start:end, :]
            K_c = K[:, :, start:end, :]
            V_c = V[:, :, start:end, :]
            mask_c = token_valid[:, start:end] if token_valid is not None else None

            scores_intra = torch.matmul(Q_c, K_c.transpose(-2, -1).contiguous()) * scale
            causal_intra = torch.tril(torch.ones(C_len, C_len, device=device, dtype=torch.bool))
            scores_intra = scores_intra.masked_fill(~causal_intra.unsqueeze(0).unsqueeze(0), -1e4)
            if self.seer_gate is not None and getattr(self.cfg, "seer_enabled", False):
                bs = max(1, self.cfg.block_size)
                nb_intra = (C_len + bs - 1) // bs
                if nb_intra > 1:
                    seer_mask = self.seer_gate(Q_c, K_c, nb_intra, training=self.training).float()
                    expanded_seer = seer_mask.repeat_interleave(bs, dim=2).repeat_interleave(bs, dim=3)[
                        :, :, :C_len, :C_len
                    ]
                    scores_intra = (
                        scores_intra
                        + torch.tanh(self.seer_beta.float()).to(scores_intra.dtype) * (expanded_seer - 1.0) * 10000.0
                    )
            if mask_c is not None:
                scores_intra = scores_intra.masked_fill((~mask_c.bool()).unsqueeze(1).unsqueeze(2), -1e4)

            if self.cfg.softpick_enabled:
                attn_intra = softpick(scores_intra, dim=-1, temperature=self.cfg.softpick_temperature)
            else:
                scores_intra = torch.nan_to_num(scores_intra, nan=-1e4, posinf=1e4, neginf=-1e4)
                attn_intra = F.softmax(scores_intra, dim=-1)

            out_intra = torch.matmul(attn_intra, V_c)

            if c_idx == 0:
                outputs.append(out_intra)
                continue

            K_prev = K[:, :, :start, :]
            V_prev = V[:, :, :start, :]
            mask_prev = token_valid[:, :start] if token_valid is not None else None
            prev_len = start

            m_prev = min(m, prev_len)

            def _pool(X, n_landmarks):
                Bx, Hx, Nx, Dx = X.shape
                if n_landmarks >= Nx:
                    return X
                X_flat = X.reshape(Bx * Hx, Nx, Dx).transpose(1, 2).contiguous()
                pooled = F.adaptive_avg_pool1d(X_flat, n_landmarks)
                return pooled.transpose(1, 2).contiguous().reshape(Bx, Hx, n_landmarks, Dx)

            K_landmarks = _pool(K_prev, m_prev)

            scores1 = torch.matmul(Q_c, K_landmarks.transpose(-2, -1).contiguous()) * scale
            F1 = F.softmax(scores1, dim=-1)

            scores2 = torch.matmul(K_landmarks, K_landmarks.transpose(-2, -1).contiguous()) * scale
            F2 = F.softmax(scores2, dim=-1)

            scores3 = torch.matmul(K_landmarks, K_prev.transpose(-2, -1).contiguous()) * scale
            if mask_prev is not None:
                scores3 = scores3.masked_fill((~mask_prev.bool()).unsqueeze(1).unsqueeze(2), -1e4)
            F3 = F.softmax(scores3, dim=-1)

            try:
                F2_pinv = torch.linalg.pinv(F2, rcond=1e-3)
            except Exception:
                mx = F2.shape[-1]
                F2_reg = F2 + 1e-2 * torch.eye(mx, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                F2_pinv = torch.linalg.inv(F2_reg)

            if F1.is_cuda and V_prev.is_cuda:
                try:
                    from neuralops.ops.triton.low_rank import fused_nystrom_landmark_pinv_step_pytorch

                    out_cross = fused_nystrom_landmark_pinv_step_pytorch(F1, F2_pinv, F3, V_prev)
                except Exception:
                    step1 = torch.matmul(F3, V_prev)
                    step2 = torch.matmul(F2_pinv, step1)
                    out_cross = torch.matmul(F1, step2)
            else:
                step1 = torch.matmul(F3, V_prev)
                step2 = torch.matmul(F2_pinv, step1)
                out_cross = torch.matmul(F1, step2)
            out_cross = torch.nan_to_num(out_cross, nan=0.0, posinf=50.0, neginf=-50.0)

            out_c = out_intra + 0.1 * out_cross
            outputs.append(out_c)

        context = torch.cat(outputs, dim=2)
        context = self.dropout(context)
        return context.to(original_dtype)
