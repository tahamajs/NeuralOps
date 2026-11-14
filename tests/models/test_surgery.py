#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for model surgery, backbone hooks, and micro-healing."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from neuralops.models.surgery import (
    SubspaceAnchorWrapper,
    apply_backbone_surgery_hooks,
    heal_replaced_model,
)


class DummyDecoderSelfAttn(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        attn = (q @ k.transpose(-1, -2)) * (1.0 / (hidden_states.shape[-1] ** 0.5))
        if attention_mask is not None:
            attn = attn + attention_mask
        out = self.o_proj(torch.softmax(attn, dim=-1) @ v)
        return out, None, None


class DummyDecoderModel(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, dim)
        self.self_attn = DummyDecoderSelfAttn(dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        attn_out = self.self_attn(h)[0]
        h = self.norm(h + attn_out)
        return self.mlp(h)


def test_apply_backbone_surgery_hooks_tags_metadata_honestly() -> None:
    """P2 Regression: apply_backbone_surgery_hooks honestly tags metadata without fake hooks."""
    model = DummyDecoderModel()
    assert not hasattr(model.self_attn, "supports_gqa")
    assert not hasattr(model.self_attn, "supports_rope")

    tagged = apply_backbone_surgery_hooks(model, model_type="llama")
    assert "self_attn" in tagged
    assert model.self_attn.supports_gqa is True
    assert model.self_attn.supports_rope is True
    # Without attach_forward_adapter, forward method is not altered
    assert not getattr(model.self_attn, "_has_surgery_adapter", False)

    # Unsupported architecture family returns empty list without error
    unsupported_tagged = apply_backbone_surgery_hooks(model, model_type="bert")
    assert unsupported_tagged == []


def test_apply_backbone_surgery_hooks_attaches_forward_adapter_when_requested() -> None:
    """P2 Regression: attach_forward_adapter=True wraps forward with adapt_decoder_attention_forward."""
    model = DummyDecoderModel()
    tagged = apply_backbone_surgery_hooks(model, model_type="llama", attach_forward_adapter=True)
    assert "self_attn" in tagged
    assert getattr(model.self_attn, "_has_surgery_adapter", False) is True

    # Test invoking forward with decoder backbone arguments (e.g. position_embeddings, position_ids)
    x = torch.randint(0, 32, (2, 4))
    h = model.embed(x)
    pos_ids = torch.arange(4).unsqueeze(0).expand(2, -1)
    res = model.self_attn(h, position_ids=pos_ids)
    assert isinstance(res, tuple)
    assert len(res) == 3
    assert res[0].shape == h.shape


def test_subspace_anchor_wrapper_mixed_dtype() -> None:
    """P1 Regression: SubspaceAnchorWrapper must handle mixed precision between base and surg layers."""
    base = nn.Linear(16, 16).half()
    surg = nn.Linear(16, 16).float()
    anchor = SubspaceAnchorWrapper(base, surg, alpha=0.3)

    # Test with float16 input
    x_half = torch.randn(2, 4, 16, dtype=torch.float16)
    out_half = anchor(x_half)
    assert out_half.dtype == torch.float16
    assert out_half.shape == (2, 4, 16)

    # Test with float32 input
    x_float = torch.randn(2, 4, 16, dtype=torch.float32)
    out_float = anchor(x_float)
    assert out_float.dtype == torch.float16  # Matches base layer dtype
    assert out_float.shape == (2, 4, 16)


def test_heal_replaced_model_mixed_dtype_teacher_student() -> None:
    """P1 Regression: heal_replaced_model executes without crashing when teacher and student have different dtypes."""
    teacher = DummyDecoderModel().float()
    student = DummyDecoderModel().half()

    # Pass integer token inputs
    calib_inputs = torch.randint(0, 32, (2, 8))

    stats = heal_replaced_model(
        teacher_model=teacher,
        student_model=student,
        calibration_inputs=calib_inputs,
        steps=3,
        also_ridge_init=False,
        log_to_ledger=False,
    )

    assert stats["num_healed_layers"] > 0
    assert "initial_mse" in stats
    assert "final_mse" in stats
    assert not torch.isnan(torch.tensor(stats["final_mse"]))


def test_heal_replaced_model_mixed_dtype_inside_student() -> None:
    """P1 Regression: heal_replaced_model executes when student model has mixed-dtype parameters."""
    teacher = DummyDecoderModel().float()
    student = DummyDecoderModel().half()
    # Injected replacement layer has float32 weights while base was half
    student.self_attn.o_proj = nn.Linear(16, 16).float()

    calib_inputs = torch.randint(0, 32, (2, 8))

    stats = heal_replaced_model(
        teacher_model=teacher,
        student_model=student,
        calibration_inputs=calib_inputs,
        steps=3,
        also_ridge_init=False,
        log_to_ledger=False,
    )

    assert stats["num_healed_layers"] > 0
    assert "initial_mse" in stats
    assert "final_mse" in stats
