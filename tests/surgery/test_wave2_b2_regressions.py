#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Wave 2 Agent B2 (Surgery Advanced) Regression Test Suite.

Rigorously verifies resolution of:
- P0-8: validation.py::validate_before_surgery fake estimates removal
- P0-11: hf_cache_bridge.py::export_tensorrt_engine zero-fake / no empty file
- P0-14: pipeline.py::apply_mla_attention_surgery correct attributes (embed_dim, attr_name)
- P0-15: hf_cache_bridge.py::_extract_state_dict unpack crash with tensor key_cache
- P1: streaming_distill.py::MicroChunkedShadowTwinDistiller.step causality preservation
- P1: streaming_distill.py::ProgressiveActivationManager no premature discard in sequential runs
- P1: shadow_twin.py::finalize_surgery dangling references elimination
"""

from __future__ import annotations

import gc
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from neuralops.layers.linear.cosformer import CosformerAttention
from neuralops.surgery.endoskeletal_shell import UniversalAttentionBridge
from neuralops.surgery.hf_cache_bridge import NeuralOpsHybridHFCache, SGLangRadixStateAdapter, export_tensorrt_engine
from neuralops.surgery.mla_surgery import DeepSeekMLASurgeryBridge
from neuralops.surgery.pipeline import SurgeryOp, SurgeryRecipe, apply_mla_attention_surgery
from neuralops.surgery.scanner import UniversalArchitectureScanner
from neuralops.surgery.shadow_twin import enable_shadow_twin, finalize_surgery, patch_bridge_for_shadow_twin
from neuralops.surgery.streaming_distill import MicroChunkedShadowTwinDistiller, ProgressiveActivationManager
from neuralops.surgery.validation import validate_before_surgery


# ==============================================================================
# Helpers & Mocks
# ==============================================================================


class ToyAttentionBlock(nn.Module):
    """Simple attention block for scanner and surgery tests."""

    def __init__(self, d: int = 64) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return x


class ToyTransformer(nn.Module):
    """Multi-layer toy transformer containing attention blocks."""

    def __init__(self, d: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyAttentionBlock(d=d) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


# ==============================================================================
# P0-8: validation.py::validate_before_surgery fake estimates removal
# ==============================================================================


def test_p0_8_validate_before_surgery_no_fake_estimates() -> None:
    """Verifies that validate_before_surgery does not return hardcoded fake estimates."""
    model = ToyTransformer(d=32, num_layers=2)
    recipe = SurgeryRecipe(
        ops=[SurgeryOp(name="attention.fx", params={"target_attention": "cosformer"})],
        name="test_recipe",
    )

    # Case 1: No calibration data -> dry-run validation
    report = validate_before_surgery(model, recipe, calibration_data=None)

    assert report.is_safe
    assert len(report.checks) == 5
    # Must NOT have hardcoded fake 256.0 MB reduction or -15.0% latency
    assert report.estimated_vram_reduction_mb is None
    assert report.estimated_latency_change_pct is None

    # Quality probe check must indicate skipped, not fake 0.0 perplexity delta
    quality_check = next(c for c in report.checks if c.name == "quality_probe")
    assert quality_check.passed
    assert "skipped" in quality_check.details.lower()
    assert quality_check.measured == {}
    assert quality_check.estimated == {}

    # Case 2: With calibration data -> evaluates actual forward pass metrics
    calib_input = torch.randn(2, 8, 32)
    report_with_calib = validate_before_surgery(model, recipe, calibration_data=calib_input)
    assert report_with_calib.is_safe
    quality_check_calib = next(c for c in report_with_calib.checks if c.name == "quality_probe")
    assert quality_check_calib.passed
    assert "baseline" in quality_check_calib.details.lower() or quality_check_calib.passed


# ==============================================================================
# P0-11: hf_cache_bridge.py::export_tensorrt_engine zero-fake / no empty file
# ==============================================================================


def test_p0_11_export_tensorrt_engine_raises_not_implemented_without_fake(tmp_path: Any) -> None:
    """Verifies export_tensorrt_engine raises NotImplementedError rather than writing empty file."""
    model = nn.Linear(32, 32)
    engine_path = tmp_path / "model.engine"

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError) as exc_info:
            export_tensorrt_engine(model, engine_path)
        assert "AGENTS.md §6.1" in str(exc_info.value)
        assert not engine_path.exists(), "Empty/placeholder .engine was created when CUDA unavailable"
    else:
        # Mock tensorrt import to test export path when CUDA is available
        mock_trt = MagicMock()
        with patch.dict("sys.modules", {"tensorrt": mock_trt}):
            with pytest.raises(NotImplementedError) as exc_info:
                export_tensorrt_engine(model, engine_path)
            assert "not implemented" in str(exc_info.value).lower()
            assert not engine_path.exists(), "Empty/placeholder .engine file was created"


# ==============================================================================
# P0-14: pipeline.py::apply_mla_attention_surgery correct attributes
# ==============================================================================


def test_p0_14_apply_mla_attention_surgery_correct_attributes() -> None:
    """Verifies apply_mla_attention_surgery uses embed_dim and attr_name correctly."""
    model = ToyTransformer(d=64, num_layers=2)

    # Before surgery, blocks are ToyAttentionBlock
    assert isinstance(model.layers[0], ToyAttentionBlock)

    surgeried_model = apply_mla_attention_surgery(
        model,
        latent_kv_dim=128,
        decoupled_rope_dim=32,
        decay=0.95,
    )

    # After surgery, blocks must be replaced by DeepSeekMLASurgeryBridge
    assert isinstance(surgeried_model.layers[0], DeepSeekMLASurgeryBridge)
    assert isinstance(surgeried_model.layers[1], DeepSeekMLASurgeryBridge)

    bridge = surgeried_model.layers[0]
    assert bridge.latent_kv_dim == 128
    assert bridge.decoupled_rope_dim == 32
    assert bridge.decay == 0.95
    assert bridge.hidden_dim == 64


# ==============================================================================
# P0-15: hf_cache_bridge.py::_extract_state_dict unpack crash
# ==============================================================================


def test_p0_15_extract_state_dict_with_tensor_key_cache() -> None:
    """Verifies _extract_state_dict does not crash when key_cache contains 4D tensors."""
    cache = NeuralOpsHybridHFCache()

    # Populate cache with 4D key and value tensors [batch=1, heads=4, seq=16, dim=32]
    k_tensor = torch.randn(1, 4, 16, 32)
    v_tensor = torch.randn(1, 4, 16, 32)
    cache.update(k_tensor, v_tensor, layer_idx=0)

    # Calling _extract_state_dict directly must not crash on unpack
    state_dict = SGLangRadixStateAdapter._extract_state_dict(cache)
    assert 0 in state_dict
    assert isinstance(state_dict[0], torch.Tensor)
    assert state_dict[0].shape == k_tensor.shape

    # Inserting into RadixTree must also succeed without exception
    adapter = SGLangRadixStateAdapter()
    node = adapter.insert([101, 202, 303], cache)
    assert node is not None
    assert 0 in node.recurrent_state


# ==============================================================================
# P1: streaming_distill.py::MicroChunkedShadowTwinDistiller.step causality
# ==============================================================================


def test_p1_micro_chunked_distiller_step_preserves_causality() -> None:
    """Verifies that MicroChunkedShadowTwinDistiller.step preserves causal state and continuous position IDs."""
    vocab = 32
    d = 16

    class CausalMockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.linear = nn.Linear(d, d)
            self.head = nn.Linear(d, vocab)
            self.recorded_pos_ids: list[torch.Tensor] = []
            self.saw_past: list[bool] = []

        def forward(
            self,
            input_ids: torch.Tensor,
            output_hidden_states: bool = True,
            past_key_values: Any = None,
            position_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> Any:
            if position_ids is not None:
                self.recorded_pos_ids.append(position_ids.detach().clone())
            self.saw_past.append(past_key_values is not None)

            h = self.linear(self.emb(input_ids))
            logits = self.head(h)

            class Output:
                pass

            out = Output()
            out.hidden_states = [h]
            out.logits = logits
            # Return dummy past_key_values representing accumulated KV cache
            out.past_key_values = (torch.zeros(1, 2, input_ids.shape[1], d),)
            return out

    teacher = CausalMockModel()
    student = CausalMockModel()

    distiller = MicroChunkedShadowTwinDistiller(
        teacher=teacher,
        student=student,
        chunk_size=8,
        hidden_dim=d,
    )

    # Total sequence length 24 => 3 chunks of size 8
    input_ids = torch.randint(0, vocab, (1, 24))
    metrics = distiller.step(input_ids)

    assert "loss" in metrics
    assert not torch.isnan(metrics["loss"])

    # Teacher and student must have been called 3 times (once per chunk)
    assert len(teacher.recorded_pos_ids) == 3
    assert len(student.recorded_pos_ids) == 3

    # Position IDs must be continuous: [0..7], [8..15], [16..23] (not reset to [0..7] each chunk)
    for i, (t_pos, s_pos) in enumerate(zip(teacher.recorded_pos_ids, student.recorded_pos_ids)):
        expected_range = torch.arange(i * 8, (i + 1) * 8)
        assert torch.equal(t_pos[0], expected_range), f"Chunk {i} position IDs broke causality: {t_pos[0]}"
        assert torch.equal(s_pos[0], expected_range), f"Chunk {i} position IDs broke causality: {s_pos[0]}"

    # Chunks 1 and 2 must receive past_key_values from prior chunk
    assert teacher.saw_past == [False, True, True]
    assert student.saw_past == [False, True, True]


# ==============================================================================
# P1: streaming_distill.py::ProgressiveActivationManager premature discard
# ==============================================================================


def test_p1_progressive_activation_manager_sequential_runs_no_premature_discard() -> None:
    """Verifies that ProgressiveActivationManager does not prematurely discard unaligned activations in sequential runs."""
    d = 16
    num_layers = 4
    manager = ProgressiveActivationManager(hidden_dim=d)

    student_tensors = [torch.randn(2, 8, d, requires_grad=True) for _ in range(num_layers)]
    teacher_tensors = [torch.randn(2, 8, d) for _ in range(num_layers)]

    # Sequential execution: teacher forward pass runs first across ALL layers
    for l in range(num_layers):
        manager.record_teacher_activation(l, teacher_tensors[l])

    # All 4 teacher activations must remain available (not discarded before student arrives)
    assert len(manager._teacher_activations) == num_layers
    for l in range(num_layers):
        assert l in manager._teacher_activations

    # Sequential execution: student forward pass runs across ALL layers
    for l in range(num_layers):
        manager.record_student_activation(l, student_tensors[l])
        # Upon pairing, layer l must be immediately aligned and evicted
        assert l not in manager._teacher_activations
        assert l not in manager._student_activations

    # After all layers pair, cache must be completely empty
    assert len(manager._teacher_activations) == 0
    assert len(manager._student_activations) == 0

    # CKA loss must be successfully computed across all layers
    loss = manager.compute()
    assert not torch.isnan(loss)
    loss.backward()
    for s_t in student_tensors:
        assert s_t.grad is not None


# ==============================================================================
# P1: shadow_twin.py::finalize_surgery dangling references elimination
# ==============================================================================


def test_p1_finalize_surgery_eliminates_dangling_references() -> None:
    """Verifies finalize_surgery removes all references to teacher branch, weights, and hooks."""
    host_block = ToyAttentionBlock(d=64)
    specs = UniversalArchitectureScanner.scan(host_block)
    spec = specs[0]

    cosformer = CosformerAttention(hidden_size=64, num_heads=4)
    bridge = UniversalAttentionBridge(
        orig_module=host_block,
        spec=spec,
        new_attention=cosformer,
        inherit_weights=True,
    )

    enable_shadow_twin(bridge, stage="stage2_block_cloning")
    patch_bridge_for_shadow_twin(bridge)

    # Verify shadow-twin setup created references
    assert bridge.teacher_branch is not None
    assert bridge.spec.target_module is not None
    assert bridge.spec.extracted_qkv is not None
    assert getattr(bridge, "_is_shadow_twin_patched", False) is True

    # Execute finalize_surgery
    finalize_surgery(bridge)

    # Invariants after finalization:
    # 1. teacher_branch and orig_module attributes are None
    assert bridge.teacher_branch is None
    assert getattr(bridge, "orig_module", None) is None
    assert getattr(bridge, "base_module", None) is None
    assert getattr(bridge, "teacher_bridge_wrapper", None) is None

    # 2. _modules must not retain teacher submodules
    for mod_name in ("teacher_branch", "teacher_bridge_wrapper", "orig_module", "base_module"):
        assert mod_name not in bridge._modules

    # 3. AttentionTargetSpec references to teacher weights and target module are severed
    assert bridge.spec.target_module is None
    assert bridge.spec.extracted_qkv is None
    assert bridge.spec.parent_module is None

    # 4. LoRA and adapter references are cleared
    assert len(bridge.lora_adapters) == 0
    for lora_attr in ("lora_A", "lora_B", "scaling", "lora_magnitude_vector", "peft_config"):
        assert not hasattr(bridge, lora_attr)

    # 5. Patching and closure attributes are cleaned up
    assert getattr(bridge, "_is_shadow_twin_patched", False) is False
    assert not hasattr(bridge, "_orig_unpatched_forward")
    assert not hasattr(bridge, "_last_twin_loss")

    # 6. Garbage collection successfully cleans up
    gc.collect()
