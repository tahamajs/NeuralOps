#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Pre-Benchmark Micro-Training & Surgical Healing Regimes.

Houses the 4 novel post-surgery micro-training paradigms:
- Fisher-Weighted Curvature Micro-Steps (F-Micro)
- DoRA Directional Healing (DoRA Fast Healing)
- Test-Time Training Online Adaptation (TTT-Online)
- Homotopic Spectral SVD Annealing (HSA)
"""

from __future__ import annotations

from neuralops.trainer.regimes.dora_healing import DoRAHealingHealer, DoRAProjectionWrapper
from neuralops.trainer.regimes.fisher_micro import FisherMicroHealer, FMicroOptimizer, compute_empirical_diagonal_fisher
from neuralops.trainer.regimes.spectral_annealing import SpectralAnnealer
from neuralops.trainer.regimes.ttt_online import TTTOnlineAdapter

__all__ = [
    "FMicroOptimizer",
    "FisherMicroHealer",
    "compute_empirical_diagonal_fisher",
    "DoRAProjectionWrapper",
    "DoRAHealingHealer",
    "TTTOnlineAdapter",
    "SpectralAnnealer",
]
