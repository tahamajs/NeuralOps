#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Preference Alignment and On-Policy RL

Implements SimPO, ORPO, GRPO, and dynamic routing reinforcement learning for language models.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimPOLoss(nn.Module):
    """
    SimPO: Simple Preference Optimization with a Reference-Free Reward.
    Eliminates the reference model entirely, saving 50% VRAM on 6GB/8GB GPUs:
        loss = -log sigmoid( beta/|yw| * log_pi(yw) - beta/|yl| * log_pi(yl) - gamma )
    """

    def __init__(self, beta: float = 2.0, gamma: float = 0.5) -> None:
        super().__init__()
        self.beta = float(beta)
        self.gamma = float(gamma)

    def forward(
        self,
        chosen_logprobs: torch.Tensor,
        rejected_logprobs: torch.Tensor,
        chosen_lens: Optional[torch.Tensor] = None,
        rejected_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        len_w = chosen_lens if chosen_lens is not None else chosen_logprobs.size(-1)
        len_l = rejected_lens if rejected_lens is not None else rejected_logprobs.size(-1)

        r_w = (chosen_logprobs.sum(dim=-1) / len_w) * self.beta
        r_l = (rejected_logprobs.sum(dim=-1) / len_l) * self.beta

        logits = r_w - r_l - self.gamma
        return -F.logsigmoid(logits).mean()


class ORPOLoss(nn.Module):
    """Odds-Ratio Preference Optimization (ORPO) without reference model."""

    def __init__(self, lambda_orpo: float = 0.1, sft_weight: float = 1.0, odds_ratio_weight: Optional[float] = None) -> None:
        super().__init__()
        if odds_ratio_weight is not None:
            lambda_orpo = odds_ratio_weight
        self.lambda_orpo = float(lambda_orpo)
        self.sft_weight = float(sft_weight)

    def forward(
        self,
        sft_loss: Optional[torch.Tensor] = None,
        chosen_logprobs: Optional[torch.Tensor] = None,
        rejected_logprobs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if chosen_logprobs is None and "policy_chosen_logps" in kwargs:
            chosen_logprobs = kwargs["policy_chosen_logps"]
        if rejected_logprobs is None and "policy_rejected_logps" in kwargs:
            rejected_logprobs = kwargs["policy_rejected_logps"]
        if sft_loss is None and "sft_loss" in kwargs:
            sft_loss = kwargs["sft_loss"]

        if sft_loss is not None and chosen_logprobs is not None and rejected_logprobs is None and "sft_loss" in kwargs:
            rejected_logprobs = chosen_logprobs
            chosen_logprobs = sft_loss
            sft_loss = kwargs["sft_loss"]

        if sft_loss is None:
            sft_loss = torch.tensor(0.0, device=chosen_logprobs.device if chosen_logprobs is not None else "cpu")

        log_p_w = chosen_logprobs.sum(dim=-1)
        log_p_l = rejected_logprobs.sum(dim=-1)

        log_odds_w = log_p_w - torch.log1p(-torch.exp(log_p_w).clamp(max=0.9999))
        log_odds_l = log_p_l - torch.log1p(-torch.exp(log_p_l).clamp(max=0.9999))

        ratio = log_odds_w - log_odds_l
        orpo_penalty = -F.logsigmoid(ratio).mean()

        return self.sft_weight * sft_loss + self.lambda_orpo * orpo_penalty


class DynamicRoutingAndRankRL(nn.Module):
    """Differentiable Gumbel-Softmax STE + Dual Lagrangian FLOPs/Rank Controller."""

    def __init__(
        self,
        num_ranks_or_blocks: int = 4,
        target_budget_ratio: float = 0.5,
        dual_lambda: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_choices = num_ranks_or_blocks
        self.target_ratio = target_budget_ratio
        self.dual_lambda = dual_lambda

    def sample_routing(self, router_logits: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
        """Samples differentiable categorical decisions via Gumbel-Softmax."""
        return F.gumbel_softmax(router_logits, tau=tau, hard=hard, dim=-1)

    def compute_budget_penalty(self, decisions: torch.Tensor) -> torch.Tensor:
        """Quadratic penalty if active block ratio exceeds target budget ratio."""
        active_ratio = decisions.float().mean()
        violation = F.relu(active_ratio - self.target_ratio)
        return self.dual_lambda * (violation ** 2)


__all__ = [
    "SimPOLoss",
    "ORPOLoss",
    "DynamicRoutingAndRankRL",
]
