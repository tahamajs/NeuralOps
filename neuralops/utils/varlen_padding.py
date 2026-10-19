#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Variable-Length Padding Utilities

Handles padding and cumulative sequence length indexing for variable-length batches.
"""

from __future__ import annotations

from typing import Tuple
import torch


def construct_cu_seqlens_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Construct cumulative sequence lengths tensor from sequence lengths vector.

    Args:
        lengths: 1D Tensor of integer sequence lengths [B]

    Returns:
        cu_seqlens: 1D Tensor [B + 1] of cumulative sequence offsets (int32).
    """
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=lengths.device)
    cu_seqlens[1:] = torch.cumsum(lengths.to(torch.int32), dim=0)
    return cu_seqlens


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Unpad padded batch input tensor into a packed 1D token representation.

    Args:
        hidden_states: [B, S, D]
        attention_mask: [B, S] binary mask where 1 indicates valid token.

    Returns:
        packed_states: [total_tokens, D]
        indices: 1D indices for repadding
        cu_seqlens: [B + 1] cumulative sequence length offsets
        max_seqlen: scalar maximum sequence length in batch
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens_in_batch.max().item())
    cu_seqlens = construct_cu_seqlens_from_lengths(seqlens_in_batch)
    flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    packed_states = flat_states[indices]
    return packed_states, indices, cu_seqlens, max_seqlen


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Reconstruct dense padded tensor [B, S, D] from packed token representation.

    Args:
        hidden_states: [total_tokens, D]
        indices: 1D scatter indices
        batch: target batch size B
        seqlen: target sequence length S

    Returns:
        dense_tensor: [B, S, D]
    """
    dim = hidden_states.shape[-1]
    output = torch.zeros(
        (batch * seqlen, dim), dtype=hidden_states.dtype, device=hidden_states.device
    )
    output[indices] = hidden_states
    return output.view(batch, seqlen, dim)


class CuTeTiledVarlenPadding:
    """Namespace container for CuTe-tiled variable sequence length padding operations."""

    @staticmethod
    def pad(
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        batch: int,
        seqlen: int,
    ) -> torch.Tensor:
        return pad_input(hidden_states, indices, batch, seqlen)

    @staticmethod
    def unpad(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return unpad_input(hidden_states, attention_mask)
