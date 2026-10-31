#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Exhaustive integration and unit tests for the NeuralOps model surgery framework.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.surgery.attention.replace import replace_attention
from neuralops.surgery.layers.prune import prune_layers
from neuralops.surgery.mlp.shrink import shrink_mlp
from neuralops.surgery.pipeline import ApplyReport, SurgeryOp, SurgeryRecipe
from neuralops.surgery.weights.quantize import quantize_int4
from neuralops.surgery.recovery.distill import distill_from_teacher
from neuralops.surgery.registry import (
    SURGERY_REGISTRY,
    SurgeryMetadata,
    get_surgeries_for_model,
    get_surgery,
    list_surgeries,
    register_surgery,
)
from neuralops.surgery.validation import CheckResult, ValidationReport, validate_before_surgery


# ---------------------------------------------------------------------------
# Test Helpers and Mock Models
# ---------------------------------------------------------------------------

class MockConfig:
    def __init__(self, model_type: str = "llama", num_hidden_layers: int = 4):
        self.model_type = model_type
        self.num_hidden_layers = num_hidden_layers


class MockSwiGLUMLP(nn.Module):
    def __init__(self, dim: int = 64, inter_dim: int = 256):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class MockGELUMLP(nn.Module):
    def __init__(self, dim: int = 64, inter_dim: int = 256):
        super().__init__()
        self.c_fc = nn.Linear(dim, inter_dim, bias=True)
        self.c_proj = nn.Linear(inter_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(torch.relu(self.c_fc(x)))


class MockDecoderLayer(nn.Module):
    def __init__(self, dim: int = 64, mlp_type: str = "swiglu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        if mlp_type == "swiglu":
            self.mlp = MockSwiGLUMLP(dim=dim, inter_dim=dim * 4)
        else:
            self.mlp = MockGELUMLP(dim=dim, inter_dim=dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        return attn_out + self.mlp(x)


class MockTransformerModel(nn.Module):
    def __init__(self, model_type: str = "llama", num_layers: int = 4, dim: int = 64, mlp_type: str = "swiglu"):
        super().__init__()
        self.config = MockConfig(model_type=model_type, num_hidden_layers=num_layers)
        self.layers = nn.ModuleList([MockDecoderLayer(dim=dim, mlp_type=mlp_type) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MockFailingForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig(model_type="llama", num_hidden_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Forced forward failure")


class MockNaNOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig(model_type="llama", num_hidden_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[float("nan"), 1.0], [0.0, float("inf")]])


# ---------------------------------------------------------------------------
# Test Suite 1: Surgery Registry
# ---------------------------------------------------------------------------

def test_surgery_registry_canonical_operations():
    surgeries = list_surgeries()
    expected_five = {
        "replace_attention",
        "shrink_mlp",
        "prune_layers",
        "quantize_int4",
        "distill_from_teacher",
    }
    assert expected_five.issubset(set(surgeries))

    for op_name in expected_five:
        meta = get_surgery(op_name)
        assert isinstance(meta, SurgeryMetadata)
        assert meta.name == op_name
        assert meta.category in ("attention", "mlp", "layers", "weights", "recovery")
        assert len(meta.compat) > 0
        assert callable(meta.fn)


def test_surgery_registry_category_filtering():
    attn_surgeries = list_surgeries(category="attention")
    assert "replace_attention" in attn_surgeries
    assert "shrink_mlp" not in attn_surgeries

    mlp_surgeries = list_surgeries(category="mlp")
    assert "shrink_mlp" in mlp_surgeries

    empty = list_surgeries(category="non_existent")
    assert empty == []


def test_surgery_registry_model_compatibility():
    llama_ops = get_surgeries_for_model("llama")
    assert "replace_attention" in llama_ops
    assert "shrink_mlp" in llama_ops
    assert "prune_layers" in llama_ops

    unknown_ops = get_surgeries_for_model("nonexistent_arch_xyz")
    assert unknown_ops == []


def test_surgery_registry_override_behavior():
    try:
        @register_surgery(name="temp_surgery_override", category="attention")
        def op_v1(model):
            return "v1"

        assert get_surgery("temp_surgery_override").fn("dummy") == "v1"

        @register_surgery(name="temp_surgery_override", category="attention")
        def op_v2(model):
            return "v2"

        assert get_surgery("temp_surgery_override").fn("dummy") == "v2"
    finally:
        SURGERY_REGISTRY.pop("temp_surgery_override", None)


# ---------------------------------------------------------------------------
# Test Suite 2: Surgery Pipeline & Rollback Safety
# ---------------------------------------------------------------------------

def test_surgery_op_dataclass():
    op = SurgeryOp(name="test_op", params={"a": 1, "b": [1, 2]}, validate=False)
    assert op.name == "test_op"
    assert op.params == {"a": 1, "b": [1, 2]}
    assert not op.validate

    d = op.to_dict()
    assert d == {"name": "test_op", "params": {"a": 1, "b": [1, 2]}, "validate": False}

    reconstructed = SurgeryOp.from_dict(d)
    assert reconstructed.name == op.name
    assert reconstructed.params == op.params
    assert reconstructed.validate == op.validate


def test_surgery_recipe_rollback_on_nn_module_failure():
    model = MockTransformerModel(num_layers=3, dim=32)
    original_layer_count = len(model.layers)
    original_weight_norm = model.layers[0].mlp.gate_proj.weight.norm().item()

    try:
        @register_surgery(name="failing_mock_op", category="custom_fail")
        def failing_mock_op(m):
            raise RuntimeError("Fatal op failure")

        recipe = SurgeryRecipe(
            ops=[
                SurgeryOp(name="shrink_mlp", params={"ratio": 0.5}),
                SurgeryOp(name="failing_mock_op", params={}),
            ]
        )

        rolled_back_model, report = recipe.apply(model, rollback_on_failure=True)
        assert not report.success
        assert "Fatal op failure" in report.error_message
        assert len(rolled_back_model.layers) == original_layer_count
        # Verify gate_proj weight norm is restored exactly
        restored_weight_norm = rolled_back_model.layers[0].mlp.gate_proj.weight.norm().item()
        assert pytest.approx(original_weight_norm, rel=1e-5) == restored_weight_norm

    finally:
        SURGERY_REGISTRY.pop("failing_mock_op", None)


def test_surgery_recipe_raise_when_rollback_disabled():
    model = MockTransformerModel(num_layers=2, dim=32)
    try:
        @register_surgery(name="failing_mock_op2", category="custom_fail")
        def failing_mock_op2(m):
            raise RuntimeError("Crash without rollback")

        recipe = SurgeryRecipe(ops=[SurgeryOp(name="failing_mock_op2")])

        with pytest.raises(RuntimeError, match="Crash without rollback"):
            recipe.apply(model, rollback_on_failure=False)
    finally:
        SURGERY_REGISTRY.pop("failing_mock_op2", None)


def test_surgery_recipe_unregistered_operation():
    model = MockTransformerModel(num_layers=2, dim=32)
    recipe = SurgeryRecipe(ops=[SurgeryOp(name="completely_unregistered_surgery_xyz")])

    result_model, report = recipe.apply(model, rollback_on_failure=True)
    assert not report.success
    assert "not registered in SURGERY_REGISTRY" in report.error_message

    with pytest.raises(RuntimeError, match="not registered in SURGERY_REGISTRY"):
        recipe.apply(model, rollback_on_failure=False)


# ---------------------------------------------------------------------------
# Test Suite 3: Validation System
# ---------------------------------------------------------------------------

def test_validation_architecture_mismatch():
    incompatible_model = MockTransformerModel(model_type="t5_unsupported_family", num_layers=2)
    recipe = SurgeryRecipe(
        ops=[
            SurgeryOp(name="replace_attention", params={"attention": "chunk_gla"}),
        ]
    )
    report = validate_before_surgery(incompatible_model, recipe)
    assert not report.is_safe
    arch_check = next(c for c in report.checks if c.name == "architecture_compat")
    assert not arch_check.passed
    assert "t5_unsupported_family" in arch_check.details


def test_validation_recipe_category_conflict():
    model = MockTransformerModel(model_type="llama", num_layers=4)
    recipe = SurgeryRecipe(
        ops=[
            SurgeryOp(name="shrink_mlp", params={"ratio": 0.8}),
            SurgeryOp(name="shrink_mlp", params={"ratio": 0.5}),
        ]
    )
    report = validate_before_surgery(model, recipe)
    assert not report.is_safe
    compat_check = next(c for c in report.checks if c.name == "recipe_compat")
    assert not compat_check.passed
    assert "Multiple operations modify the same category: 'mlp'" in compat_check.details


def test_validation_forward_pass_failure():
    failing_model = MockFailingForwardModel()
    recipe = SurgeryRecipe(ops=[SurgeryOp(name="replace_attention")])
    report = validate_before_surgery(failing_model, recipe)
    assert not report.is_safe
    fwd_check = next(c for c in report.checks if c.name == "forward_pass")
    assert not fwd_check.passed
    assert "Forced forward failure" in fwd_check.details


def test_validation_output_nan_detection():
    nan_model = MockNaNOutputModel()
    recipe = SurgeryRecipe(ops=[SurgeryOp(name="replace_attention")])
    report = validate_before_surgery(nan_model, recipe)
    assert not report.is_safe
    sanity_check = next(c for c in report.checks if c.name == "output_sanity")
    assert not sanity_check.passed
    assert "NaNs" in sanity_check.details


def test_validation_with_calibration_data():
    model = MockTransformerModel(model_type="llama", num_layers=2, dim=32)
    calib_tensor = torch.randn(2, 8, 32)
    recipe = SurgeryRecipe(ops=[SurgeryOp(name="replace_attention")])

    report = validate_before_surgery(model, recipe, calibration_data=calib_tensor)
    assert report.is_safe
    fwd_check = next(c for c in report.checks if c.name == "forward_pass")
    assert fwd_check.passed


# ---------------------------------------------------------------------------
# Test Suite 4: Specific Surgery Operations
# ---------------------------------------------------------------------------

def test_shrink_mlp_ratio_validation():
    model = MockTransformerModel()
    with pytest.warns(UserWarning, match="experimental"):
        with pytest.raises(ValueError, match="Shrinkage ratio must be in"):
            shrink_mlp(model, ratio=0.0)

    with pytest.warns(UserWarning, match="experimental"):
        with pytest.raises(ValueError, match="Shrinkage ratio must be in"):
            shrink_mlp(model, ratio=1.5)


def test_shrink_mlp_swiglu():
    model = MockTransformerModel(num_layers=2, dim=32, mlp_type="swiglu")
    orig_inter = model.layers[0].mlp.gate_proj.out_features
    assert orig_inter == 128

    with pytest.warns(UserWarning, match="experimental"):
        shrunk = shrink_mlp(model, ratio=0.5)
    new_inter = shrunk.layers[0].mlp.gate_proj.out_features
    assert new_inter == 64
    assert shrunk.layers[0].mlp.up_proj.out_features == 64
    assert shrunk.layers[0].mlp.down_proj.in_features == 64

    # Test forward pass works post-shrink
    x = torch.randn(2, 4, 32)
    out = shrunk(x)
    assert out.shape == (2, 4, 32)


def test_shrink_mlp_gelu():
    model = MockTransformerModel(num_layers=2, dim=32, mlp_type="gelu")
    orig_inter = model.layers[0].mlp.c_fc.out_features
    assert orig_inter == 128

    with pytest.warns(UserWarning, match="experimental"):
        shrunk = shrink_mlp(model, ratio=0.5)
    assert shrunk.layers[0].mlp.c_fc.out_features == 64
    assert shrunk.layers[0].mlp.c_proj.in_features == 64

    x = torch.randn(2, 4, 32)
    out = shrunk(x)
    assert out.shape == (2, 4, 32)


def test_shrink_mlp_targeted_layers():
    model = MockTransformerModel(num_layers=3, dim=32, mlp_type="swiglu")
    # Only shrink layer 0
    with pytest.warns(UserWarning, match="experimental"):
        shrunk = shrink_mlp(model, ratio=0.5, layers=[0])
    assert shrunk.layers[0].mlp.gate_proj.out_features == 64
    assert shrunk.layers[1].mlp.gate_proj.out_features == 128
    assert shrunk.layers[2].mlp.gate_proj.out_features == 128


def test_prune_layers_keep_indices():
    model = MockTransformerModel(num_layers=4, dim=32)
    assert len(model.layers) == 4
    assert model.config.num_hidden_layers == 4

    with pytest.warns(UserWarning, match="experimental"):
        pruned = prune_layers(model, keep_layers=[0, 3])
    assert len(pruned.layers) == 2
    assert pruned.config.num_hidden_layers == 2


def test_prune_layers_drop_count():
    model = MockTransformerModel(num_layers=4, dim=32)
    with pytest.warns(UserWarning, match="experimental"):
        pruned = prune_layers(model, drop_count=2)
    assert len(pruned.layers) == 2
    assert pruned.config.num_hidden_layers == 2


def test_prune_layers_cannot_prune_all():
    model = MockTransformerModel(num_layers=2, dim=32)
    with pytest.warns(UserWarning, match="experimental"):
        with pytest.raises(ValueError, match="Cannot prune all layers"):
            prune_layers(model, keep_layers=[])


def test_quantize_int4_validation():
    model = MockTransformerModel()
    with pytest.warns(UserWarning, match="experimental"):
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            quantize_int4(model, method="unsupported_fmt")

    with pytest.warns(UserWarning, match="experimental"):
        with pytest.raises(IntegrationNotImplementedError, match="autoawq"):
            quantize_int4(model, method="awq")

    with pytest.warns(UserWarning, match="experimental"):
        with pytest.raises(IntegrationNotImplementedError, match="gptqmodel"):
            quantize_int4(model, method="gptq")


def test_distill_from_teacher_pillars():
    model = MockTransformerModel(num_layers=2, dim=32)
    # Check that calling distill_from_teacher with all 5 pillars constructs losses and returns model
    with pytest.warns(UserWarning, match="experimental"):
        out = distill_from_teacher(
            student=model,
            teacher=None,
            steps=10,
            pillars=["logit", "feature", "relation", "operator", "cka"],
            dim=32,
        )
    assert out is model
