#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recurrent step, streaming execution, and checkpointing mixin for SVDLinearAttention.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.utils.gqa import repeat_kv
from neuralops.layers.svd._svd_linear_kernels import compute_exponential_decay

logger = logging.getLogger(__name__)


class SVDLinearRecurrentMixin:
    """Mixin providing recurrent stepping, KV cache, streaming forward, and persistence."""

    def step_recurrent(
        self,
        x_t: torch.Tensor,
        past_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)

        B = x_t.shape[0]
        self.sync_inherited_weights()
        q = self.q_proj(x_t).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x_t).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x_t).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        phi_q = F.elu(q.float()) + 1.0
        phi_k = F.elu(k.float()) + 1.0

        _, H, _, D_k = q.shape
        D_v = v.shape[-1]

        if self.use_decay_gating:
            decay_logits = self.decay_gate(x_t)
            if self.decay_channel_wise:
                decay_logits = decay_logits.view(B, 1, H, D_k).permute(0, 2, 1, 3)
                lam_t = (
                    compute_exponential_decay(decay_logits, activation=self.decay_activation).unsqueeze(-1).squeeze(2)
                )
            else:
                decay_logits = decay_logits.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
                lam_t = compute_exponential_decay(decay_logits, activation=self.decay_activation).squeeze(2)
        else:
            lam_t = torch.ones(B, H, 1, 1, device=x_t.device, dtype=torch.float32)

        if past_state is not None:
            S_prev, Z_prev = past_state
        else:
            S_prev = torch.zeros(B, H, D_k, D_v, device=x_t.device, dtype=torch.float32)
            Z_prev = torch.zeros(B, H, D_k, 1, device=x_t.device, dtype=torch.float32)

        kt_t = phi_k.transpose(-1, -2).contiguous()
        kv_t = torch.matmul(kt_t, v.float())

        S_t = lam_t * S_prev + kv_t
        Z_t = lam_t * Z_prev + kt_t

        num_t = torch.matmul(phi_q, S_t)
        den_t = torch.matmul(phi_q, Z_t) + self.eps
        y_t = num_t / den_t

        context = y_t.transpose(1, 2).contiguous().view(B, 1, self.all_head_size).to(x_t.dtype)
        if self.is_nested_encoder:
            out_t = context
        else:
            out_t = self.out_proj(context)

        if out_t.shape[-1] != x_t.shape[-1]:
            raise RuntimeError(
                f"SVDLinearAttention.step_recurrent: output dim " f"{out_t.shape[-1]} != input dim {x_t.shape[-1]}"
            )

        return out_t, (S_t, Z_t)

    def get_orthogonality_loss(self) -> torch.Tensor:
        return (
            self.q_proj.get_orthogonality_loss()
            + self.k_proj.get_orthogonality_loss()
            + self.v_proj.get_orthogonality_loss()
            + self.out_proj.get_orthogonality_loss()
        )

    def init_kv_cache(
        self,
        batch_size: int = 1,
        max_len: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        """Initializes circular buffer KV-cache storing compressed latents C_k, C_v."""
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        r = self.rank
        return {
            "C_k": torch.zeros(batch_size, max_len, r, device=device, dtype=dtype),
            "C_v": torch.zeros(batch_size, max_len, r, device=device, dtype=dtype),
            "step": 0,
            "max_len": max_len,
            "batch_size": batch_size,
        }

    def streaming_forward(
        self,
        x_t: torch.Tensor,
        kv_cache: dict[str, Any],
        max_rank: int = 64,
    ) -> torch.Tensor:
        """Streaming token-by-token forward pass with latent circular buffer KV cache.

        Computes compressed latent C_t = x_t @ V_r^T, records into circular buffer,
        and decompresses K, V via U_r @ C_t in attention computation.
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        B, seq_len, _ = x_t.shape
        assert seq_len == 1, "streaming_forward processes single tokens at a time"

        step = kv_cache["step"]
        max_len = kv_cache["max_len"]
        slot = step % max_len
        self.sync_inherited_weights()

        q = self.q_proj(x_t).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        c_k_t = F.linear(x_t, self.k_proj.V)
        c_v_t = F.linear(x_t, self.v_proj.V)

        if self.k_proj.norm_weight is not None:
            norm_k = torch.rsqrt(c_k_t.pow(2).mean(-1, keepdim=True) + self.k_proj.eps)
            c_k_t = c_k_t * norm_k * self.k_proj.norm_weight
        if self.v_proj.norm_weight is not None:
            norm_v = torch.rsqrt(c_v_t.pow(2).mean(-1, keepdim=True) + self.v_proj.eps)
            c_v_t = c_v_t * norm_v * self.v_proj.norm_weight

        kv_cache["C_k"][:, slot : slot + 1, :].copy_(c_k_t)
        kv_cache["C_v"][:, slot : slot + 1, :].copy_(c_v_t)

        if "K_buf" not in kv_cache:
            kv_cache["K_buf"] = torch.zeros(B, max_len, self.kv_head_size, device=x_t.device, dtype=x_t.dtype)
            kv_cache["V_buf"] = torch.zeros(B, max_len, self.kv_head_size, device=x_t.device, dtype=x_t.dtype)
        kv_cache["K_buf"][:, slot : slot + 1, :] = self.k_proj(x_t)
        kv_cache["V_buf"][:, slot : slot + 1, :] = self.v_proj(x_t)

        kv_cache["step"] = step + 1

        curr_len = min(step + 1, max_len)
        if "K_all" in kv_cache and "V_all" in kv_cache and not self.is_decoder:
            k_all = kv_cache["K_all"]
            v_all = kv_cache["V_all"]
            curr_len = k_all.shape[-2]
        elif not self.is_decoder and kv_cache.get("full_seq_len"):
            tot = kv_cache["full_seq_len"]
            k_d = kv_cache["K_buf"][:, :tot, :]
            v_d = kv_cache["V_buf"][:, :tot, :]
            curr_len = tot
            k_all = k_d.view(B, curr_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
            v_all = v_d.view(B, curr_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
            if self.num_kv_heads < self.num_heads:
                k_all = repeat_kv(k_all, self.kv_repeat)
                v_all = repeat_kv(v_all, self.kv_repeat)
        else:
            if step + 1 <= max_len:
                active_c_k = kv_cache["C_k"][:, :curr_len, :]
                active_c_v = kv_cache["C_v"][:, :curr_len, :]
            else:
                idx = [(slot + 1 + i) % max_len for i in range(max_len)]
                active_c_k = kv_cache["C_k"][:, idx, :]
                active_c_v = kv_cache["C_v"][:, idx, :]

            k_decomp = F.linear(active_c_k, self.k_proj.U, self.k_proj.bias)
            v_decomp = F.linear(active_c_v, self.v_proj.U, self.v_proj.bias)

            if self.k_proj.lora_rank > 0 and self.k_proj.lora_A is not None and self.k_proj.lora_B is not None:
                lora_k_curr = F.linear(F.linear(x_t, self.k_proj.lora_A), self.k_proj.lora_B) * self.k_proj.lora_scaling
                k_decomp[:, -1:, :] = k_decomp[:, -1:, :] + lora_k_curr
            if self.v_proj.lora_rank > 0 and self.v_proj.lora_A is not None and self.v_proj.lora_B is not None:
                lora_v_curr = F.linear(F.linear(x_t, self.v_proj.lora_A), self.v_proj.lora_B) * self.v_proj.lora_scaling
                v_decomp[:, -1:, :] = v_decomp[:, -1:, :] + lora_v_curr

            k_all = k_decomp.view(B, curr_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
            v_all = v_decomp.view(B, curr_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

            if self.num_kv_heads < self.num_heads:
                k_all = repeat_kv(k_all, self.kv_repeat)
                v_all = repeat_kv(v_all, self.kv_repeat)

        if self.mode == "exact":
            scale = 1.0 / math.sqrt(max(int(self.head_dim), 1))
            scores = torch.matmul(q.float(), k_all.float().transpose(-1, -2).contiguous()) * scale
            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(q.dtype)
            context = torch.matmul(probs, v_all)
        else:
            phi_q = F.elu(q.float()) + 1.0
            phi_k = F.elu(k_all.float()) + 1.0
            v_f = v_all.float()
            kv_state = torch.matmul(phi_k.transpose(-1, -2).contiguous(), v_f)
            num = torch.matmul(phi_q, kv_state)
            den = torch.matmul(phi_q, phi_k.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()) + self.eps
            context = (num / den).to(q.dtype)

        context = context.transpose(1, 2).contiguous().view(B, 1, self.all_head_size)
        if self.is_nested_encoder:
            out_t = context
        else:
            out_t = self.out_proj(context)

        return out_t

    def save_compressed(self, path: str) -> None:
        """Save factorized SVD attention parameters in safetensors/pt format."""
        import json
        import os

        os.makedirs(path, exist_ok=True)
        state = self.state_dict()
        try:
            from safetensors.torch import save_file

            save_file({k: v.cpu() for k, v in state.items()}, os.path.join(path, "model.safetensors"))
        except ImportError:
            torch.save(state, os.path.join(path, "model.pt"))
        with open(os.path.join(path, "quantization_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "quant_method": "compressed-tensors",
                    "format": "svd_low_rank",
                    "hidden_size": self.hidden_size,
                    "rank": self.rank,
                    "num_heads": self.num_heads,
                },
                f,
                indent=2,
            )

    def load_compressed(self, path: str) -> None:
        """Load factorized SVD attention parameters from safetensors/pt directory."""
        import os

        safetensor_path = os.path.join(path, "model.safetensors")
        pt_path = os.path.join(path, "model.pt")
        if os.path.exists(safetensor_path):
            from safetensors.torch import load_file

            state = load_file(safetensor_path)
        elif os.path.exists(pt_path):
            state = torch.load(pt_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No compressed weights found at {path}")
        self.load_state_dict(state, strict=False)
