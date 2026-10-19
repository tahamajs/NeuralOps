#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Analysis Comparison Core Modules.
"""

import sys
from neuralops.analysis.comparison.statistical_testing import (
    bayesian_correlated_ttest,
    bayesian_signed_rank_test,
    finner_correction,
    hochberg_correction,
    holland_copenhaver_correction,
    holm_correction,
    hommel_correction,
    mcnemar_test,
    run_full_statistical_analysis,
    sign_test,
    wilcoxon_signed_rank_test,
)
from neuralops.analysis.comparison import statistical_testing
statistical_tests = statistical_testing
sys.modules["neuralops.analysis.comparison.statistical_tests"] = statistical_testing

__all__ = [
    "holm_correction",
    "hochberg_correction",
    "finner_correction",
    "hommel_correction",
    "holland_copenhaver_correction",
    "wilcoxon_signed_rank_test",
    "sign_test",
    "mcnemar_test",
    "bayesian_correlated_ttest",
    "bayesian_signed_rank_test",
    "run_full_statistical_analysis",
    "statistical_testing",
    "statistical_tests",
]
