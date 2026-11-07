#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Homotopic Spectral SVD Annealing (HSA) Regime.

Linearly interpolates singular values of the original teacher attention projection
matrices into the transplanted student attention projection matrices:
    Sigma_{student}(t) = (1 - gamma(t)) * Sigma_{teacher} + gamma(t) * Sigma_{student}^{(0)}
Retains 100% of the spectral energy decay curve during the critical initial micro-steps.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Trainer.Regimes.SpectralAnnealing")


class SpectralAnnealer:
    """
    Homotopic Spectral SVD Annealer.

    Performs SVD decomposition on matching teacher and student weight matrices,
    interpolating their singular values along an annealing schedule gamma(t) in [0, 1].
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        total_steps: int = 150,
        target_substrings: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.total_steps = max(1, total_steps)
        self.target_substrings = target_substrings or [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "o_proj",
            "dense",
            "proj",
        ]
        self.device = device

        self.spectral_data: Dict[str, Dict[str, torch.Tensor]] = {}
        self._decompose_projections()

    def _decompose_projections(self) -> None:
        """Computes SVD on matching teacher and student projection matrices."""
        t_params = dict(self.teacher.named_parameters())
        s_params = dict(self.student.named_parameters())

        decomposed = 0
        for name, s_p in s_params.items():
            if not s_p.requires_grad or s_p.ndim != 2:
                continue
            if not any(sub in name for sub in self.target_substrings):
                continue
            if name not in t_params:
                continue

            t_p = t_params[name]
            if t_p.shape != s_p.shape:
                continue

            with torch.no_grad():
                # Teacher SVD
                u_t, s_t, vh_t = torch.linalg.svd(t_p.data.float(), full_matrices=False)
                # Student SVD
                u_s, s_s, vh_s = torch.linalg.svd(s_p.data.float(), full_matrices=False)

                self.spectral_data[name] = {
                    "u_s": u_s.to(s_p.device),
                    "vh_s": vh_s.to(s_p.device),
                    "sigma_teacher": s_t.to(s_p.device),
                    "sigma_student_init": s_s.to(s_p.device),
                    "param_ref": s_p,
                }
                decomposed += 1

        logger.info(
            "SpectralAnnealer decomposed %d 2D projection matrices for homotopic spectral annealing.",
            decomposed,
        )

    def get_gamma(self, step: int) -> float:
        """Returns annealing coefficient gamma(t): 0.0 (teacher spectrum) -> 1.0 (student spectrum)."""
        progress = min(1.0, max(0.0, step / self.total_steps))
        # Smooth cosine transition
        return 0.5 * (1.0 - math.cos(math.pi * progress))

    def step(self, step: int) -> None:
        """
        Updates student singular values according to current step:
            Sigma(t) = (1 - gamma) * Sigma_t + gamma * Sigma_s
            W = U_s @ diag(Sigma(t)) @ Vh_s
        """
        gamma = self.get_gamma(step)
        with torch.no_grad():
            for name, data in self.spectral_data.items():
                s_t = data["sigma_teacher"]
                s_s0 = data["sigma_student_init"]
                u_s = data["u_s"]
                vh_s = data["vh_s"]
                p = data["param_ref"]

                # Homotopic interpolation
                sigma_curr = (1.0 - gamma) * s_t + gamma * s_s0
                # Reconstruct weight
                w_recon = (u_s * sigma_curr.unsqueeze(0)) @ vh_s
                p.data.copy_(w_recon.to(p.dtype))


__all__ = ["SpectralAnnealer"]
