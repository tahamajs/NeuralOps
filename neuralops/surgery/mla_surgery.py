#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
r"""
DeepSeek Multi-Head Latent Attention (MLA) Surgery Bridge.

Implements dual-stream latent recurrence without uncompressing KV states,
preserving the 87.5% memory compression advantage of DeepSeek-V2 and V3 models
during linear attention transplants.

Mathematical Formulation:
    Content Stream (S_t^C):
        S_t^C = \gamma \cdot S_{t-1}^C + (c_t^{KV})^\top \phi(q_t^C) \in \mathbb{R}^{d_c \times d_c}
    Decoupled RoPE Stream (z_t^R):
        z_t^R = z_{t-1}^R + k_t^R (q_t^R)^\top \in \mathbb{R}^{d_R \times d_R}
    Output Reconstruction:
        O_t = \phi(q_t^C) S_t^C + q_t^R z_t^R
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("neuralops.surgery.mla")


class DualStreamLatentState(NamedTuple):
    """Container for dual-stream recurrent states in DeepSeek MLA.

    Attributes:
        state_c: Latent content state tensor S_t^C of shape [B, latent_kv_dim, latent_kv_dim].
        state_r: Decoupled RoPE state tensor z_t^R of shape [B, decoupled_rope_dim, decoupled_rope_dim].
    """

    state_c: torch.Tensor
    state_r: torch.Tensor


class DeepSeekMLASurgeryBridge(nn.Module):
    r"""DeepSeek Multi-Head Latent Attention (MLA) Surgery Bridge.

    Enables high-efficiency linear attention surgery on DeepSeek architectures
    by operating directly on compressed latent KV representations c^{KV} and decoupled
    rotary keys k^R, completely bypassing 16,384-dimensional uncompressed expansions.

    Args:
        latent_kv_dim: Dimension of compressed KV latent vector (default: 512).
        decoupled_rope_dim: Dimension of decoupled rotary key vector (default: 64).
        num_heads: Number of attention heads (default: 128).
        head_dim: Dimension per head (default: 128).
        decay: Recurrent state exponential decay factor \gamma \in (0, 1] (default: 0.99).
        hidden_dim: Optional model hidden dimension for projection wrapping.
    """

    def __init__(
        self,
        latent_kv_dim: int = 512,
        decoupled_rope_dim: int = 64,
        num_heads: int = 128,
        head_dim: int = 128,
        decay: float = 0.99,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.latent_kv_dim = latent_kv_dim
        self.decoupled_rope_dim = decoupled_rope_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.decay = decay
        self.hidden_dim = hidden_dim

        # Optional linear projections when operating in standalone mode from hidden states
        if hidden_dim is not None:
            self.q_content_proj = nn.Linear(hidden_dim, latent_kv_dim, bias=False)
            self.q_rope_proj = nn.Linear(hidden_dim, decoupled_rope_dim, bias=False)
            self.kv_latent_proj = nn.Linear(hidden_dim, latent_kv_dim, bias=False)
            self.k_rope_proj = nn.Linear(hidden_dim, decoupled_rope_dim, bias=False)
            self.out_proj = nn.Linear(latent_kv_dim + decoupled_rope_dim, hidden_dim, bias=False)
        else:
            self.q_content_proj = None
            self.q_rope_proj = None
            self.kv_latent_proj = None
            self.k_rope_proj = None
            self.out_proj = None

    def init_latent_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> DualStreamLatentState:
        """Initializes zero-filled recurrent latent states S_0^C and z_0^R.

        Args:
            batch_size: Batch size B.
            device: Target torch device.
            dtype: Target torch data type (default: torch.float32).

        Returns:
            DualStreamLatentState containing S_0^C and z_0^R.
        """
        s_c = torch.zeros(
            batch_size,
            self.latent_kv_dim,
            self.latent_kv_dim,
            device=device,
            dtype=dtype,
        )
        z_r = torch.zeros(
            batch_size,
            self.decoupled_rope_dim,
            self.decoupled_rope_dim,
            device=device,
            dtype=dtype,
        )
        return DualStreamLatentState(state_c=s_c, state_r=z_r)

    def forward_prefill(
        self,
        c_kv: torch.Tensor,
        k_rope: torch.Tensor,
        q_content: torch.Tensor,
        q_rope: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Parallel chunkwise associative scan on compressed latent space.

        Computes the sequence representation in parallel for prompt prefill:
            S^C = (c^{KV})^\top \phi(q^C)
            z^R = (k^R)^\top q^R
            O_{content} = \phi(q^C) S^C
            O_{rope} = q^R z^R
            O = O_{content} + O_{rope}

        Args:
            c_kv: Compressed KV latent states of shape [B, T, latent_kv_dim].
            k_rope: Decoupled rotary key states of shape [B, T, decoupled_rope_dim].
            q_content: Content query representations of shape [B, T, latent_kv_dim].
            q_rope: Decoupled rotary query representations of shape [B, T, decoupled_rope_dim].
            chunk_size: Optional chunk size for long-sequence chunked evaluation.

        Returns:
            Tuple of (output_tensor [B, T, latent_kv_dim], (state_c, state_r)).
        """
        q_phi = F.elu(q_content) + 1.0
        k_phi = F.elu(c_kv) + 1.0

        if chunk_size is not None and c_kv.shape[1] > chunk_size:
            # Parallel chunkwise associative scan
            batch_size, seq_len, _ = c_kv.shape
            outputs = []
            s_c = torch.zeros(
                batch_size,
                self.latent_kv_dim,
                self.latent_kv_dim,
                device=c_kv.device,
                dtype=c_kv.dtype,
            )
            z_r = torch.zeros(
                batch_size,
                self.decoupled_rope_dim,
                self.decoupled_rope_dim,
                device=k_rope.device,
                dtype=k_rope.dtype,
            )

            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                c_chunk = k_phi[:, start:end]
                q_chunk = q_phi[:, start:end]
                kr_chunk = k_rope[:, start:end]
                qr_chunk = q_rope[:, start:end]

                chunk_s_c = torch.matmul(c_chunk.transpose(-2, -1), q_chunk)
                chunk_z_r = torch.matmul(kr_chunk.transpose(-2, -1), qr_chunk)

                s_c = self.decay * s_c + chunk_s_c
                z_r = z_r + chunk_z_r

                chunk_out_c = torch.matmul(q_chunk, s_c)
                chunk_out_r = torch.matmul(qr_chunk, z_r)
                outputs.append(chunk_out_c + chunk_out_r)

            total_out = torch.cat(outputs, dim=1)
            return total_out, (s_c, z_r)

        # Latent content outer product: [B, latent_kv_dim, latent_kv_dim]
        s_c = torch.matmul(k_phi.transpose(-2, -1), q_phi)
        # Decoupled RoPE outer product: [B, decoupled_rope_dim, decoupled_rope_dim]
        z_r = torch.matmul(k_rope.transpose(-2, -1), q_rope)

        out_content = torch.matmul(q_phi, s_c)
        out_rope = torch.matmul(q_rope, z_r)
        return out_content + out_rope, (s_c, z_r)

    def forward_decode_step(
        self,
        c_kv_step: torch.Tensor,
        k_rope_step: torch.Tensor,
        q_content_step: torch.Tensor,
        q_rope_step: torch.Tensor,
        state_c: torch.Tensor,
        state_r: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""O(1) state update during autoregressive decoding.

        Recurrent state transitions:
            S_t^C = \gamma \cdot S_{t-1}^C + (c_t^{KV})^\top \phi(q_t^C)
            z_t^R = z_{t-1}^R + (k_t^R)^\top q_t^R
            O_t = \phi(q_t^C) S_t^C + q_t^R z_t^R

        Args:
            c_kv_step: Single-step compressed KV vector [B, 1, latent_kv_dim].
            k_rope_step: Single-step decoupled rotary key [B, 1, decoupled_rope_dim].
            q_content_step: Single-step content query [B, 1, latent_kv_dim].
            q_rope_step: Single-step decoupled rotary query [B, 1, decoupled_rope_dim].
            state_c: Accumulated latent content state S_{t-1}^C [B, latent_kv_dim, latent_kv_dim].
            state_r: Accumulated decoupled RoPE state z_{t-1}^R [B, decoupled_rope_dim, decoupled_rope_dim].

        Returns:
            Tuple of (step_output [B, 1, latent_kv_dim], (new_state_c, new_state_r)).
        """
        q_phi = F.elu(q_content_step) + 1.0
        k_phi = F.elu(c_kv_step) + 1.0

        new_state_c = self.decay * state_c + torch.matmul(k_phi.transpose(-2, -1), q_phi)
        new_state_r = state_r + torch.matmul(k_rope_step.transpose(-2, -1), q_rope_step)

        out_c = torch.matmul(q_phi, new_state_c)
        out_r = torch.matmul(q_rope_step, new_state_r)
        return out_c + out_r, (new_state_c, new_state_r)

    def forward(
        self,
        *args: torch.Tensor,
        past_state: Optional[Union[DualStreamLatentState, Tuple[torch.Tensor, torch.Tensor]]] = None,
        is_decode: bool = False,
        **kwargs: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Unified forward pass dispatching between prefill and decode regimes.

        Can accept:
            Option A (explicit 4 tensors): (c_kv, k_rope, q_content, q_rope)
            Option B (single hidden_states tensor): [B, T, hidden_dim] when projection weights exist.
        """
        if len(args) == 4:
            c_kv, k_rope, q_content, q_rope = args
        elif len(args) == 1:
            hidden_states = args[0]
            if self.q_content_proj is None:
                raise ValueError(
                    "DeepSeekMLASurgeryBridge received single hidden_states tensor "
                    "but was not initialized with hidden_dim and projection layers."
                )
            q_content = self.q_content_proj(hidden_states)
            q_rope = self.q_rope_proj(hidden_states)
            c_kv = self.kv_latent_proj(hidden_states)
            k_rope = self.k_rope_proj(hidden_states)
        elif "c_kv" in kwargs and "k_rope" in kwargs:
            c_kv = kwargs["c_kv"]
            k_rope = kwargs["k_rope"]
            q_content = kwargs.get("q_content", c_kv)
            q_rope = kwargs.get("q_rope", k_rope)
        else:
            raise ValueError(
                f"Invalid arguments to DeepSeekMLASurgeryBridge forward. Expected 4 tensors "
                f"(c_kv, k_rope, q_content, q_rope) or 1 tensor (hidden_states), got {len(args)} args."
            )

        seq_len = c_kv.shape[1]
        if (is_decode or seq_len == 1) and past_state is not None:
            if isinstance(past_state, DualStreamLatentState):
                state_c, state_r = past_state.state_c, past_state.state_r
            else:
                state_c, state_r = past_state
            out, new_state = self.forward_decode_step(
                c_kv_step=c_kv,
                k_rope_step=k_rope,
                q_content_step=q_content,
                q_rope_step=q_rope,
                state_c=state_c,
                state_r=state_r,
            )
        else:
            out, new_state = self.forward_prefill(
                c_kv=c_kv,
                k_rope=k_rope,
                q_content=q_content,
                q_rope=q_rope,
            )

        if self.out_proj is not None and len(args) == 1:
            # Map back to hidden dimension if standalone
            # out has shape [B, T, latent_kv_dim]
            # Concat with rope stream representation if dimension allows
            out = self.out_proj(torch.cat([out, q_rope], dim=-1))

        return out, new_state


__all__ = [
    "DualStreamLatentState",
    "DeepSeekMLASurgeryBridge",
]
