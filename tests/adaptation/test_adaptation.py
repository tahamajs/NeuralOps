#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and functional tests for the neuralops.adaptation subpackage.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import neuralops.adaptation
from neuralops.adaptation import distillation, peft
from neuralops.adaptation.distillation import (
    BlockwiseOperatorCloningLoss,
    HedgehogFeatureMapKDLoss,
    LinearCKALoss,
    LogitDistillationLoss,
    MiniLMValueRelationKDLoss,
    SmoothMaxOverLayersCKASteeringLoss,
    SpikyFeatureMap,
)
from neuralops.adaptation.peft import (
    DoRALinear,
    LoRALinear,
    PiSSALinear,
    inject_pissa_adapters,
)


def test_adaptation_subpackage_exports():
    assert hasattr(neuralops.adaptation, "distillation")
    assert hasattr(neuralops.adaptation, "peft")


def test_logit_distillation_loss():
    loss_fn = LogitDistillationLoss(temperature=2.0)
    student_logits = torch.randn(2, 8, 100)
    teacher_logits = torch.randn(2, 8, 100)

    loss = loss_fn(student_logits, teacher_logits)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_minilm_relation_kd_loss():
    loss_fn = MiniLMValueRelationKDLoss()
    # (batch, heads, seq, dim)
    s_values = torch.randn(2, 4, 8, 16)
    t_values = torch.randn(2, 4, 8, 16)

    loss = loss_fn(s_values, t_values)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


def test_hedgehog_feature_map_kd_loss():
    loss_fn = HedgehogFeatureMapKDLoss(chunk_size=16)
    # (batch, heads, seq, dim)
    q_t = torch.randn(2, 4, 16, 16)
    k_t = torch.randn(2, 4, 16, 16)
    q_s = torch.randn(2, 4, 16, 16)
    k_s = torch.randn(2, 4, 16, 16)

    loss = loss_fn(q_t, k_t, q_s, k_s)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


def test_linear_cka_loss():
    loss_fn = LinearCKALoss()
    s_rep = torch.randn(2, 8, 32)
    t_rep = torch.randn(2, 8, 32)

    loss = loss_fn(s_rep, t_rep)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


def test_spiky_feature_map():
    feat_map = SpikyFeatureMap(head_dim=16)
    x = torch.randn(2, 8, 16)
    out = feat_map(x)
    assert out.shape == (2, 8, 32)


def test_peft_lora_linear():
    lora = LoRALinear(in_features=32, out_features=32, r=4)
    x = torch.randn(2, 8, 32)
    out = lora(x)
    assert out.shape == (2, 8, 32)


def test_peft_dora_linear():
    base_linear = nn.Linear(32, 32)
    dora = DoRALinear(base_layer=base_linear, rank=4)
    x = torch.randn(2, 8, 32)
    out = dora(x)
    assert out.shape == (2, 8, 32)


def test_peft_pissa_linear():
    base_linear = nn.Linear(32, 32)
    pissa = PiSSALinear(base_layer=base_linear, rank=4)
    x = torch.randn(2, 8, 32)
    out = pissa(x)
    assert out.shape == (2, 8, 32)
