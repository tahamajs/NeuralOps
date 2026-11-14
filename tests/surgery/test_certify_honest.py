#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for honest certification in UniversalSurgeryDirector.
Ensures laws 3, 5, 7, 9 return 'UNVERIFIED' and certified is only True if all are True.
"""

import torch.nn as nn

from neuralops.surgery.director import UniversalSurgeryDirector


def test_certify_model_returns_unverified_for_unprovable_laws():
    """Verify that Laws 3, 5, 7, 9 return 'UNVERIFIED' and certified is False."""
    model = nn.Sequential(nn.Linear(64, 64))
    director = UniversalSurgeryDirector(model)
    report = director.certify_model(strict=False)

    laws = report["laws_certified"]
    assert laws["Law_3_Zero_Bleed_Padding"] == "UNVERIFIED"
    assert laws["Law_5_RoPE_Pre_Projection"] == "UNVERIFIED"
    assert laws["Law_7_Causal_Monotonicity_State_Decay"] == "UNVERIFIED"
    assert laws["Law_9_FP32_Accumulation"] == "UNVERIFIED"

    # certified must be False because unverified laws cannot statically prove compliance
    assert report["certified"] is False


def test_certify_model_certified_only_when_all_true():
    """Verify certified is True only when all laws are True."""
    model = nn.Sequential(nn.Linear(64, 64))
    director = UniversalSurgeryDirector(model)
    report = director.certify_model(strict=False)
    assert report["certified"] is False
