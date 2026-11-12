#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Regression test suite for Model Surgery Transactional Rollback, FP32 Softmax Headroom,
and Device/Dtype Preservation (DEF-025, DEF-026, DEF-031).
"""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from neuralops.core.exceptions import NumericalError, SurgeryTransactionError
from neuralops.models.surgery import (
    SurgeryTransactionError as SurgeryTransactionErrorExport,
    compute_attention_weights,
    transplant_model_attention,
)
from neuralops.pipeline_transform.snapshot import capture, restore


def test_surgery_transaction_error_hierarchy():
    """Verify SurgeryTransactionError derives from RuntimeError and NeuralOps base error."""
    from neuralops.core.exceptions import NeuralOpsError

    err = SurgeryTransactionError("Transaction failed")
    assert isinstance(err, RuntimeError)
    assert isinstance(err, NeuralOpsError)
    assert SurgeryTransactionError is SurgeryTransactionErrorExport


def test_compute_attention_weights_fp32_headroom_def026():
    """DEF-026: compute_attention_weights computes softmax in FP32 accumulation to avoid half-precision underflow."""
    # Construct half-precision logits with a large dynamic range
    logits_fp16 = torch.tensor([[-50.0, -10.0, 0.0, 5.0]], dtype=torch.float16)

    weights = compute_attention_weights(logits_fp16, dim=-1)
    assert weights.dtype == torch.float16
    assert weights.shape == logits_fp16.shape
    # Sum along last dim should be within fp16 tolerance of 1.0
    assert torch.allclose(weights.sum(dim=-1), torch.tensor([1.0], dtype=torch.float16), atol=1e-3)
    # Check that values are finite and non-negative
    assert not torch.isnan(weights).any()
    assert not torch.isinf(weights).any()
    assert (weights >= 0.0).all()

    # Repeat for bfloat16
    logits_bf16 = torch.tensor([[-100.0, -50.0, 0.0, 10.0]], dtype=torch.bfloat16)
    weights_bf16 = compute_attention_weights(logits_bf16, dim=-1)
    assert weights_bf16.dtype == torch.bfloat16
    assert torch.allclose(weights_bf16.sum(dim=-1), torch.tensor([1.0], dtype=torch.bfloat16), atol=1e-2)


def test_snapshot_device_and_dtype_preservation_def025(tmp_path):
    """DEF-025: Verify snapshot restore preserves exact parameter/buffer dtypes without unintended casts."""
    class MultiDtypeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.param_fp64 = nn.Parameter(torch.randn(4, 4, dtype=torch.float64))
            self.param_fp32 = nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
            self.param_fp16 = nn.Parameter(torch.randn(4, 4, dtype=torch.float16))
            self.register_buffer("buf_fp64", torch.randn(4, dtype=torch.float64))
            self.register_buffer("buf_fp16", torch.randn(4, dtype=torch.float16))

    mod = MultiDtypeModule()
    snap = capture(mod)

    # Mutate parameters and buffers in-place
    with torch.no_grad():
        mod.param_fp64.fill_(0.0)
        mod.param_fp32.fill_(0.0)
        mod.param_fp16.fill_(0.0)
        mod.buf_fp64.fill_(0.0)
        mod.buf_fp16.fill_(0.0)

    # Restore in-memory snapshot
    restore(mod, snap)

    assert mod.param_fp64.dtype == torch.float64
    assert mod.param_fp32.dtype == torch.float32
    assert mod.param_fp16.dtype == torch.float16
    assert mod.buf_fp64.dtype == torch.float64
    assert mod.buf_fp16.dtype == torch.float16
    assert not (mod.param_fp64 == 0.0).all()

    # Test SafeTensors disk-spill restore
    snap_disk = capture(mod, spill_to_disk=True, disk_dir=str(tmp_path))
    with torch.no_grad():
        mod.param_fp64.fill_(1.0)
        mod.param_fp32.fill_(1.0)
        mod.param_fp16.fill_(1.0)

    restore(mod, snap_disk)
    assert mod.param_fp64.dtype == torch.float64
    assert mod.param_fp32.dtype == torch.float32
    assert mod.param_fp16.dtype == torch.float16
    assert not (mod.param_fp64 == 1.0).all()
    snap_disk.cleanup()


def test_transplant_model_attention_atomic_rollback_on_gate_failure():
    """Verify that if a validation gate fails, transplant_model_attention rolls back to pre-surgery state."""
    cfg = BertConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=2, intermediate_size=128)
    model = BertModel(cfg)

    # Record pre-surgery parameter weights
    pre_surgery_state = {k: v.clone() for k, v in model.state_dict().items()}
    original_self_attn_cls = model.encoder.layer[0].attention.self.__class__

    def failing_gate(m: nn.Module):
        raise ValueError("Simulated validation gate rejection")

    with pytest.raises(SurgeryTransactionError) as exc_info:
        transplant_model_attention(
            model=model,
            attention="cosformer",
            gate_fn=failing_gate,
        )

    assert "Attention surgery transaction aborted" in str(exc_info.value)
    assert "Simulated validation gate rejection" in str(exc_info.value)

    # Verify that the model was completely restored to pre-surgery state
    post_rollback_state = model.state_dict()
    for k in pre_surgery_state:
        assert torch.equal(pre_surgery_state[k], post_rollback_state[k]), f"Mismatch in {k} after rollback"

    assert model.encoder.layer[0].attention.self.__class__ == original_self_attn_cls


def test_transplant_model_attention_atomic_rollback_on_invalid_attention():
    """Verify that an unexpected exception during replacement triggers rollback and raises SurgeryTransactionError."""
    cfg = BertConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=2, intermediate_size=128)
    model = BertModel(cfg)

    pre_surgery_state = {k: v.clone() for k, v in model.state_dict().items()}

    with pytest.raises(SurgeryTransactionError):
        transplant_model_attention(
            model=model,
            attention="non_existent_attention_mechanism_xyz",
        )

    # Verify state preserved
    post_state = model.state_dict()
    for k in pre_surgery_state:
        assert torch.equal(pre_surgery_state[k], post_state[k])


def test_transplant_model_attention_forward_backward_verification():
    """Verify forward verification gate pass with sample input."""
    cfg = BertConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=2, intermediate_size=128)
    model = BertModel(cfg)

    sample = {"input_ids": torch.randint(0, 100, (2, 8))}
    surgeried = transplant_model_attention(
        model=model,
        attention="cosformer",
        sample_input=sample,
        validate_gate=True,
    )
    assert surgeried is not None
    out = surgeried(**sample)
    assert out.last_hidden_state.shape == (2, 8, 64)


def test_legacy_attention_surgery_shim_delegation():
    """Verify neuralops.core._attention_surgery behaves as a thin backward-compat wrapper."""
    import importlib
    import neuralops.core._attention_surgery as legacy_surgery

    with pytest.deprecated_call():
        importlib.reload(legacy_surgery)

    # Verify key symbols resolve correctly to canonical implementations
    assert callable(legacy_surgery.calibrate_attention_subspace)
    assert callable(legacy_surgery.heal_replaced_model)
    assert callable(legacy_surgery.transplant_model_attention)
    assert callable(legacy_surgery.compute_attention_weights)
    assert hasattr(legacy_surgery, "FUSED_QKV_REGEX")
    assert hasattr(legacy_surgery, "PROJ_REGEX_PATTERNS")
    assert hasattr(legacy_surgery, "SubspaceAnchorWrapper")
    assert legacy_surgery.SurgeryTransactionError is SurgeryTransactionError
