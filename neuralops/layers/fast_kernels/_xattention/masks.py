#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Attention mask utilities for xAttention."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _extract_valid_token_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Return a boolean validity mask (B, seq_len) where True = valid token.

    Handles the many shapes HF emits: (B, L), (B, 1, 1, L), (B, 1, L, L).
    """
    if mask is None or not isinstance(mask, torch.Tensor):
        return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

    mask = mask.to(device=device)

    if mask.dim() == 2:
        if mask.dtype == torch.bool:
            return mask
        if mask.dtype.is_floating_point:
            if (mask < -1.0).any():
                return mask >= -1.0
            return mask > 0.5
        return mask > 0

    if mask.dim() == 4:
        if mask.size(2) == 1:
            m = mask[:, 0, 0, :]
            if m.dtype == torch.bool:
                return m
            if m.dtype.is_floating_point:
                if (m < -1.0).any():
                    return m >= -1.0
                if (m <= 0.0).all():
                    return torch.ones((batch_size, m.size(-1)), dtype=torch.bool, device=device)
                return m > 0.5
            return m > 0
        m4 = mask[:, 0]
        if m4.dtype == torch.bool:
            return m4.any(dim=1)
        if m4.dtype.is_floating_point:
            if (m4 < -1.0).any():
                return (m4 >= -1.0).any(dim=1)
            if (m4 <= 0.0).all():
                return torch.ones((batch_size, m4.size(-1)), dtype=torch.bool, device=device)
            return (m4 > 0.5).any(dim=1)
        return (m4 > 0).any(dim=1)

    return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)


def _extract_valid_query_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    q_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a boolean validity mask (B, q_len) for query positions."""
    if mask is None or not isinstance(mask, torch.Tensor):
        return torch.ones((batch_size, q_len), dtype=torch.bool, device=device)

    mask = mask.to(device=device)
    if mask.dim() == 2:
        vm = _extract_valid_token_mask(mask, batch_size, mask.size(-1), device)
        if vm.size(-1) > q_len:
            return vm[:, -q_len:]
        if vm.size(-1) < q_len:
            return F.pad(vm, (0, q_len - vm.size(-1)), value=True)
        return vm

    if mask.dim() == 4:
        if mask.size(2) == q_len:
            m4 = mask[:, 0]
            if m4.dtype == torch.bool:
                return m4.any(dim=-1)
            if m4.dtype.is_floating_point:
                if (m4 < -1.0).any():
                    return (m4 >= -1.0).any(dim=-1)
                if (m4 <= 0.0).all():
                    return torch.ones((batch_size, q_len), dtype=torch.bool, device=device)
                return (m4 > 0.5).any(dim=-1)
            return (m4 > 0).any(dim=-1)
        vm = _extract_valid_token_mask(mask, batch_size, mask.size(-1), device)
        if vm.size(-1) > q_len:
            return vm[:, -q_len:]
        if vm.size(-1) < q_len:
            return F.pad(vm, (0, q_len - vm.size(-1)), value=True)
        return vm

    return torch.ones((batch_size, q_len), dtype=torch.bool, device=device)

