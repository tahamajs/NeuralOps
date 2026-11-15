#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SSM and Recurrent Cross-Attention Contract & Bridge Verification Suite.

Validates:
1. VSpecAttention cross-attention bridge execution, shape preservation, and invalid input rejection.
2. SelectiveSSM cross-attention bridge execution, shape preservation, and invalid input rejection.
3. MambaBlock cross-attention bridge execution, shape preservation, and invalid input rejection.
4. SambaMoEBlock cross-attention bridge execution, shape preservation, and invalid input rejection.
5. IncompatibleArchitectureError raised upon contract violation (non-Tensor or non-3D conditioning states).
6. Backward pass compatibility and gradient propagation through cross-attention bridge.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.core.exceptions import IncompatibleArchitectureError
from neuralops.layers.ssm.mamba_block import MambaBlock
from neuralops.layers.ssm.samba_moe import SambaMoEBlock
from neuralops.layers.ssm.selective_ssm import SelectiveSSM
from neuralops.layers.ssm.vspec import VSpecAttention


@pytest.fixture
def tensor_fixtures():
    """Generates standard query and encoder conditioning tensors."""
    B, L_q, L_enc, D = 2, 4, 8, 64
    x = torch.randn(B, L_q, D, requires_grad=True)
    enc = torch.randn(B, L_enc, D, requires_grad=True)
    mask = torch.ones(B, L_q, dtype=torch.bool)
    enc_mask = torch.ones(B, L_enc, dtype=torch.bool)
    return {
        "x": x,
        "enc": enc,
        "mask": mask,
        "enc_mask": enc_mask,
        "B": B,
        "L_q": L_q,
        "L_enc": L_enc,
        "D": D,
    }


class TestVSpecCrossAttention:
    """Verifies cross-sequence conditioning contracts for VSpecAttention."""

    def test_vspec_cross_attention_forward(self, tensor_fixtures):
        f = tensor_fixtures
        layer = VSpecAttention(hidden_size=f["D"], num_heads=4, rank=16)
        out = layer(f["x"], encoder_hidden_states=f["enc"], encoder_attention_mask=f["enc_mask"])
        if isinstance(out, tuple):
            out_tensor = out[0]
        else:
            out_tensor = out
        assert out_tensor.shape == (f["B"], f["L_q"], f["D"])
        assert not torch.isnan(out_tensor).any()
        assert not torch.isinf(out_tensor).any()

    def test_vspec_cross_attention_backward(self, tensor_fixtures):
        f = tensor_fixtures
        layer = VSpecAttention(hidden_size=f["D"], num_heads=4, rank=16)
        out = layer(f["x"], encoder_hidden_states=f["enc"])
        out_tensor = out[0] if isinstance(out, tuple) else out
        loss = out_tensor.sum()
        loss.backward()
        assert f["x"].grad is not None
        assert f["enc"].grad is not None

    def test_vspec_cross_attention_invalid_inputs(self, tensor_fixtures):
        f = tensor_fixtures
        layer = VSpecAttention(hidden_size=f["D"], num_heads=4, rank=16)
        with pytest.raises(IncompatibleArchitectureError):
            layer(f["x"], encoder_hidden_states="invalid_string")
        with pytest.raises(IncompatibleArchitectureError):
            layer(f["x"], encoder_hidden_states=torch.randn(f["B"], f["D"]))


class TestSelectiveSSMCrossAttention:
    """Verifies cross-sequence conditioning contracts for SelectiveSSM."""

    def test_selective_ssm_cross_attention_forward(self, tensor_fixtures):
        f = tensor_fixtures
        layer = SelectiveSSM(d_model=f["D"], d_state=8, expand=2)
        out = layer(f["x"], encoder_hidden_states=f["enc"], encoder_attention_mask=f["enc_mask"])
        if isinstance(out, tuple):
            out_tensor = out[0]
        else:
            out_tensor = out
        assert out_tensor.shape == (f["B"], f["L_q"], f["D"])
        assert not torch.isnan(out_tensor).any()

    def test_selective_ssm_cross_attention_backward(self, tensor_fixtures):
        f = tensor_fixtures
        layer = SelectiveSSM(d_model=f["D"], d_state=8, expand=2)
        out = layer(f["x"], encoder_hidden_states=f["enc"])
        out_tensor = out[0] if isinstance(out, tuple) else out
        loss = out_tensor.sum()
        loss.backward()
        assert f["x"].grad is not None
        assert f["enc"].grad is not None

    def test_selective_ssm_cross_attention_invalid_inputs(self, tensor_fixtures):
        f = tensor_fixtures
        layer = SelectiveSSM(d_model=f["D"], d_state=8, expand=2)
        with pytest.raises(IncompatibleArchitectureError):
            layer(f["x"], encoder_hidden_states=[1, 2, 3])
        with pytest.raises(IncompatibleArchitectureError):
            layer(f["x"], encoder_hidden_states=torch.randn(f["B"], f["D"]))


class TestMambaBlockCrossAttention:
    """Verifies cross-sequence conditioning contracts for MambaBlock."""

    def test_mamba_block_cross_attention_forward(self, tensor_fixtures):
        f = tensor_fixtures
        layer = MambaBlock(d_model=f["D"], d_state=8, expand=2)
        out = layer(f["x"], encoder_hidden_states=f["enc"], encoder_attention_mask=f["enc_mask"])
        if isinstance(out, tuple):
            out_tensor = out[0]
        else:
            out_tensor = out
        assert out_tensor.shape == (f["B"], f["L_q"], f["D"])
        assert not torch.isnan(out_tensor).any()

    def test_mamba_block_cross_attention_backward(self, tensor_fixtures):
        f = tensor_fixtures
        layer = MambaBlock(d_model=f["D"], d_state=8, expand=2)
        out = layer(f["x"], encoder_hidden_states=f["enc"])
        out_tensor = out[0] if isinstance(out, tuple) else out
        loss = out_tensor.sum()
        loss.backward()
        assert f["x"].grad is not None
        assert f["enc"].grad is not None

    def test_mamba_block_cross_attention_invalid_inputs(self, tensor_fixtures):
        f = tensor_fixtures
        layer = MambaBlock(d_model=f["D"], d_state=8, expand=2)
        with pytest.raises(IncompatibleArchitectureError):
            layer(f["x"], encoder_hidden_states={"k": "v"})
        with pytest.raises(IncompatibleArchitectureError):
            layer(f["x"], encoder_hidden_states=torch.randn(f["B"], 1, 2, f["D"]))


class TestSambaMoECrossAttention:
    """Verifies cross-sequence conditioning contracts for SambaMoEBlock."""

    def test_samba_moe_cross_attention_forward(self, tensor_fixtures):
        f = tensor_fixtures
        layer = SambaMoEBlock(d_model=f["D"], n_heads=4, n_experts=2, window=4)
        out = layer(f["x"], encoder_hidden_states=f["enc"], encoder_attention_mask=f["enc_mask"])
        if isinstance(out, tuple):
            out_tensor = out[0]
        else:
            out_tensor = out
        assert out_tensor.shape == (f["B"], f["L_q"], f["D"])
        assert not torch.isnan(out_tensor).any()

    def test_samba_moe_cross_attention_backward(self, tensor_fixtures):
        f = tensor_fixtures
        layer = SambaMoEBlock(d_model=f["D"], n_heads=4, n_experts=2, window=4)
        out = layer(f["x"], encoder_hidden_states=f["enc"])
        out_tensor = out[0] if isinstance(out, tuple) else out
        loss = out_tensor.sum()
        loss.backward()
        assert f["x"].grad is not None
        assert f["enc"].grad is not None

    def test_samba_moe_cross_attention_invalid_inputs(self, tensor_fixtures):
        f = tensor_fixtures
        layer = SambaMoEBlock(d_model=f["D"], n_heads=4, n_experts=2, window=4)
        with pytest.raises(IncompatibleArchitectureError):
            layer(f["x"], encoder_hidden_states=12345)
        with pytest.raises(IncompatibleArchitectureError):
            layer(f["x"], encoder_hidden_states=torch.randn(f["B"], f["D"]))
