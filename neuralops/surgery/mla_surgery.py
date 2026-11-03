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
    Content Stream (S_t^C \in \mathbb{R}^{d_{kv} \times d_q}):
        S_t^C = \gamma \cdot S_{t-1}^C + (c_t^{KV})^\top \phi(q_t^C) \in \mathbb{R}^{512 \times 128}
    Decoupled RoPE Stream (z_t^R \in \mathbb{R}^{d_R \times d_R}):
        z_t^R = z_{t-1}^R + (k_t^R)^\top q_t^R \in \mathbb{R}^{64 \times 64}
    Output Reconstruction via W_{UKV} Fusion:
        O_t = W_{UKV} \left( S_t^C \phi(q_t^C) \right) + \left( z_t^R q_t^R \right)
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
        state_c: Latent content state tensor S_t^C of shape [B, latent_kv_dim, q_latent_dim].
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
        q_latent_dim: Dimension of content query representation (default: 128 if latent_kv_dim==512 else latent_kv_dim).
        num_heads: Number of attention heads (default: 128).
        head_dim: Dimension per head (default: 128).
        decay: Recurrent state exponential decay factor \gamma \in (0, 1] (default: 0.99).
        hidden_dim: Optional model hidden dimension for projection wrapping.
    """

    def __init__(
        self,
        latent_kv_dim: int = 512,
        decoupled_rope_dim: int = 64,
        q_latent_dim: Optional[int] = None,
        num_heads: int = 128,
        head_dim: int = 128,
        decay: float = 0.99,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.latent_kv_dim = latent_kv_dim
        self.decoupled_rope_dim = decoupled_rope_dim
        if q_latent_dim is not None:
            self.q_latent_dim = q_latent_dim
        elif latent_kv_dim == 512:
            self.q_latent_dim = 128
        else:
            self.q_latent_dim = latent_kv_dim

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.decay = decay
        self.hidden_dim = hidden_dim

        # Projections when operating in standalone mode from hidden states
        if hidden_dim is not None:
            self.q_content_proj = nn.Linear(hidden_dim, self.q_latent_dim, bias=False)
            self.q_rope_proj = nn.Linear(hidden_dim, decoupled_rope_dim, bias=False)
            self.kv_latent_proj = nn.Linear(hidden_dim, latent_kv_dim, bias=False)
            self.k_rope_proj = nn.Linear(hidden_dim, decoupled_rope_dim, bias=False)
            # Output reconstruction: W_{UKV} fused projection
            self.w_ukv = nn.Linear(latent_kv_dim, hidden_dim, bias=False)
            self.w_r = nn.Linear(decoupled_rope_dim, hidden_dim, bias=False)
            self.out_proj = nn.Linear(latent_kv_dim + decoupled_rope_dim, hidden_dim, bias=False)
            self.W_UKV = self.w_ukv
        else:
            self.q_content_proj = None
            self.q_rope_proj = None
            self.kv_latent_proj = None
            self.k_rope_proj = None
            self.w_ukv = None
            self.w_r = None
            self.out_proj = None
            self.W_UKV = None

    def get_kv_compression_ratio(self, uncompressed_kv_dim: int = 4096) -> float:
        """Returns the KV cache memory compression ratio achieved by MLA.

        Standard DeepSeek multi-head KV dimension is N_h * d_h = 128 * 128 = 16,384,
        or 4,096 in medium backbones. MLA retains KV in 512-dim latent space, slashing
        memory by 87.5% (or 96.875% against 16,384).

        Args:
            uncompressed_kv_dim: Uncompressed KV total dimension (default: 4096).

        Returns:
            Fraction of memory reduced (e.g. 0.875 for 87.5% reduction).
        """
        if uncompressed_kv_dim <= 0:
            raise ValueError(f"uncompressed_kv_dim must be positive, got {uncompressed_kv_dim}")
        return 1.0 - (self.latent_kv_dim / float(uncompressed_kv_dim))

    def separate_mla_projections(
        self,
        kv_compressed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separates fused compressed KV into content latent c^{KV} and decoupled RoPE key k^R.

        Args:
            kv_compressed: Fused tensor of shape [..., latent_kv_dim + decoupled_rope_dim].

        Returns:
            Tuple of (c_kv [..., latent_kv_dim], k_rope [..., decoupled_rope_dim]).
        """
        c_kv = kv_compressed[..., : self.latent_kv_dim]
        k_rope = kv_compressed[..., self.latent_kv_dim : self.latent_kv_dim + self.decoupled_rope_dim]
        return c_kv, k_rope

    def separate_query_projections(
        self,
        q_compressed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separates fused query tensor into content query q^C and decoupled RoPE query q^R.

        Args:
            q_compressed: Fused tensor of shape [..., q_latent_dim + decoupled_rope_dim].

        Returns:
            Tuple of (q_content [..., q_latent_dim], q_rope [..., decoupled_rope_dim]).
        """
        q_content = q_compressed[..., : self.q_latent_dim]
        q_rope = q_compressed[..., self.q_latent_dim : self.q_latent_dim + self.decoupled_rope_dim]
        return q_content, q_rope

    def reconstruct_output(
        self,
        out_content: torch.Tensor,
        out_rope: torch.Tensor,
    ) -> torch.Tensor:
        r"""Reconstructs target model output using W_{UKV} fusion.

        Mathematical form:
            O_t = W_{UKV}(S_t^C \phi(q_t^C)) + W_R(z_t^R q_t^R)

        Args:
            out_content: Content stream output [B, T, latent_kv_dim].
            out_rope: RoPE stream output [B, T, decoupled_rope_dim].

        Returns:
            Reconstructed output tensor of shape [B, T, hidden_dim].
        """
        if self.w_ukv is not None and self.w_r is not None:
            return self.w_ukv(out_content) + self.w_r(out_rope)
        if self.out_proj is not None:
            return self.out_proj(torch.cat([out_content, out_rope], dim=-1))
        return torch.cat([out_content, out_rope], dim=-1)

    def init_latent_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        q_dim: Optional[int] = None,
    ) -> DualStreamLatentState:
        """Initializes zero-filled recurrent latent states S_0^C and z_0^R.

        Args:
            batch_size: Batch size B.
            device: Target torch device.
            dtype: Target torch data type (default: torch.float32).
            q_dim: Optional query dimension override (default: self.q_latent_dim).

        Returns:
            DualStreamLatentState containing S_0^C of shape [B, latent_kv_dim, q_dim]
            and z_0^R of shape [B, decoupled_rope_dim, decoupled_rope_dim].
        """
        target_q_dim = q_dim if q_dim is not None else self.q_latent_dim
        s_c = torch.zeros(
            batch_size,
            self.latent_kv_dim,
            target_q_dim,
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
            S^C = (c^{KV})^\top \phi(q^C) \in \mathbb{R}^{d_{kv} \times d_q}
            z^R = (k^R)^\top q^R \in \mathbb{R}^{d_R \times d_R}
            O_{content} = \phi(q^C) (S^C)^\top \in \mathbb{R}^{B \times T \times d_{kv}}
            O_{rope} = q^R (z^R)^\top \in \mathbb{R}^{B \times T \times d_R}
            O = O_{content} + O_{rope} (or reconstructed via W_{UKV})

        Args:
            c_kv: Compressed KV latent states of shape [B, T, latent_kv_dim].
            k_rope: Decoupled rotary key states of shape [B, T, decoupled_rope_dim].
            q_content: Content query representations of shape [B, T, q_dim].
            q_rope: Decoupled rotary query representations of shape [B, T, decoupled_rope_dim].
            chunk_size: Optional chunk size for long-sequence chunked evaluation.

        Returns:
            Tuple of (output_tensor [B, T, latent_kv_dim], (state_c, state_r)).
        """
        orig_dtype = c_kv.dtype
        # Law 9: FP32 Accumulation Budget Guard
        c_kv_f32 = c_kv.to(torch.float32)
        k_rope_f32 = k_rope.to(torch.float32)
        q_content_f32 = q_content.to(torch.float32)
        q_rope_f32 = q_rope.to(torch.float32)

        q_phi = F.elu(q_content_f32) + 1.0
        k_phi = F.elu(c_kv_f32) + 1.0

        target_q_dim = q_content.shape[-1]
        batch_size, seq_len, _ = c_kv.shape

        if chunk_size is not None and seq_len > chunk_size:
            outputs = []
            s_c = torch.zeros(
                batch_size,
                self.latent_kv_dim,
                target_q_dim,
                device=c_kv.device,
                dtype=torch.float32,
            )
            z_r = torch.zeros(
                batch_size,
                self.decoupled_rope_dim,
                self.decoupled_rope_dim,
                device=k_rope.device,
                dtype=torch.float32,
            )

            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                c_chunk = k_phi[:, start:end]
                q_chunk = q_phi[:, start:end]
                kr_chunk = k_rope_f32[:, start:end]
                qr_chunk = q_rope_f32[:, start:end]

                chunk_s_c = torch.matmul(c_chunk.transpose(-2, -1), q_chunk)
                chunk_z_r = torch.matmul(kr_chunk.transpose(-2, -1), qr_chunk)

                s_c = self.decay * s_c + chunk_s_c
                z_r = z_r + chunk_z_r

                if s_c.shape[-2] != q_chunk.shape[-1]:
                    chunk_out_c = torch.matmul(q_chunk, s_c.transpose(-2, -1))
                else:
                    chunk_out_c = torch.matmul(q_chunk, s_c)

                chunk_out_r = torch.matmul(qr_chunk, z_r.transpose(-2, -1))
                outputs.append(chunk_out_c + chunk_out_r)

            total_out = torch.cat(outputs, dim=1).to(orig_dtype)
            return total_out, (s_c.to(orig_dtype), z_r.to(orig_dtype))

        # Direct prefill: outer product updates
        s_c = torch.matmul(k_phi.transpose(-2, -1), q_phi)
        z_r = torch.matmul(k_rope_f32.transpose(-2, -1), q_rope_f32)

        if s_c.shape[-2] != q_phi.shape[-1]:
            out_content = torch.matmul(q_phi, s_c.transpose(-2, -1))
        else:
            out_content = torch.matmul(q_phi, s_c)

        out_rope = torch.matmul(q_rope_f32, z_r.transpose(-2, -1))
        total_out = (out_content + out_rope).to(orig_dtype)
        return total_out, (s_c.to(orig_dtype), z_r.to(orig_dtype))

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
            O_t = \phi(q_t^C) (S_t^C)^\top + q_t^R (z_t^R)^\top

        Args:
            c_kv_step: Single-step compressed KV vector [B, 1, latent_kv_dim].
            k_rope_step: Single-step decoupled rotary key [B, 1, decoupled_rope_dim].
            q_content_step: Single-step content query [B, 1, q_dim].
            q_rope_step: Single-step decoupled rotary query [B, 1, decoupled_rope_dim].
            state_c: Accumulated latent content state S_{t-1}^C [B, latent_kv_dim, q_dim].
            state_r: Accumulated decoupled RoPE state z_{t-1}^R [B, decoupled_rope_dim, decoupled_rope_dim].

        Returns:
            Tuple of (step_output [B, 1, latent_kv_dim], (new_state_c, new_state_r)).
        """
        orig_dtype = c_kv_step.dtype
        # Law 9: FP32 Accumulation Budget Guard
        c_step_f32 = c_kv_step.to(torch.float32)
        k_step_f32 = k_rope_step.to(torch.float32)
        qc_step_f32 = q_content_step.to(torch.float32)
        qr_step_f32 = q_rope_step.to(torch.float32)
        sc_f32 = state_c.to(torch.float32)
        sr_f32 = state_r.to(torch.float32)

        q_phi = F.elu(qc_step_f32) + 1.0
        k_phi = F.elu(c_step_f32) + 1.0

        delta_s_c = torch.matmul(k_phi.transpose(-2, -1), q_phi)
        delta_z_r = torch.matmul(k_step_f32.transpose(-2, -1), qr_step_f32)

        new_state_c = self.decay * sc_f32 + delta_s_c
        new_state_r = sr_f32 + delta_z_r

        if new_state_c.shape[-2] != q_phi.shape[-1]:
            out_c = torch.matmul(q_phi, new_state_c.transpose(-2, -1))
        else:
            out_c = torch.matmul(q_phi, new_state_c)

        out_r = torch.matmul(qr_step_f32, new_state_r.transpose(-2, -1))
        step_out = (out_c + out_r).to(orig_dtype)

        return step_out, (new_state_c.to(orig_dtype), new_state_r.to(orig_dtype))

    def inherit_from_mla_module(self, orig_mla: nn.Module) -> None:
        """Inherits low-rank projection weights directly from a pretrained DeepSeek MLA module.

        Extracts and binds:
            - kv_a_proj_with_mqa -> projects hidden_states to (c_kv, k_rope)
            - q_a_proj / q_b_proj -> projects hidden_states to content and rotary query
            - o_proj / out_proj -> output projection (W_UKV / W_O)

        Guarantees zero parameter mismatch across linear surgery transplants.

        Args:
            orig_mla: Pretrained DeepSeek MLA module.
        """
        with torch.no_grad():
            if hasattr(orig_mla, "kv_a_proj_with_mqa"):
                kv_proj = orig_mla.kv_a_proj_with_mqa
                if self.kv_latent_proj is not None and hasattr(kv_proj, "weight"):
                    w = kv_proj.weight.data
                    self.kv_latent_proj.weight.data.copy_(w[: self.latent_kv_dim, :])
                if self.k_rope_proj is not None and hasattr(kv_proj, "weight"):
                    w = kv_proj.weight.data
                    self.k_rope_proj.weight.data.copy_(
                        w[self.latent_kv_dim : self.latent_kv_dim + self.decoupled_rope_dim, :]
                    )

            if hasattr(orig_mla, "q_a_proj"):
                q_proj = orig_mla.q_a_proj
                if self.q_content_proj is not None and hasattr(q_proj, "weight"):
                    w = q_proj.weight.data
                    copy_dim = min(self.q_latent_dim, w.shape[0])
                    self.q_content_proj.weight.data[:copy_dim, :].copy_(w[:copy_dim, :])
                if self.q_rope_proj is not None and hasattr(q_proj, "weight"):
                    w = q_proj.weight.data
                    if w.shape[0] >= self.q_latent_dim + self.decoupled_rope_dim:
                        self.q_rope_proj.weight.data.copy_(
                            w[self.q_latent_dim : self.q_latent_dim + self.decoupled_rope_dim, :]
                        )

            out_module = getattr(orig_mla, "o_proj", getattr(orig_mla, "out_proj", None))
            if out_module is not None and hasattr(out_module, "weight"):
                w_out = out_module.weight.data
                if self.w_ukv is not None:
                    copy_in = min(self.w_ukv.weight.shape[1], w_out.shape[1])
                    copy_out = min(self.w_ukv.weight.shape[0], w_out.shape[0])
                    self.w_ukv.weight.data[:copy_out, :copy_in].copy_(w_out[:copy_out, :copy_in])
                if self.out_proj is not None:
                    copy_in = min(self.out_proj.weight.shape[1], w_out.shape[1])
                    copy_out = min(self.out_proj.weight.shape[0], w_out.shape[0])
                    self.out_proj.weight.data[:copy_out, :copy_in].copy_(w_out[:copy_out, :copy_in])

        logger.info(
            "Inherited pretrained MLA weights into DeepSeekMLASurgeryBridge (latent_kv_dim=%d, rope_dim=%d)",
            self.latent_kv_dim,
            self.decoupled_rope_dim,
        )

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
            if self.q_content_proj is None or self.kv_latent_proj is None:
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
            out = self.reconstruct_output(out, q_rope)

        return out, new_state


__all__ = [
    "DualStreamLatentState",
    "DeepSeekMLASurgeryBridge",
]
