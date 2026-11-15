#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for model export pipelines: SafeTensors sharding, ONNX dynamic recurrent shapes, GGUF,
and unified export operation verbs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch
import torch.nn as nn

from neuralops.export.gguf import export_model_to_gguf
from neuralops.export.onnx import build_dynamic_axes_for_recurrent, export_to_onnx
from neuralops.export.safetensors import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SafeTensorsExporter,
    save_sharded_safetensors,
    shard_state_dict,
)
from neuralops.operations.export.model import ExportModelOperation


class RecurrentDummyModel(nn.Module):
    """Linear recurrent architecture with cumulative state integration across sequence steps."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, num_classes: int = 8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.decay = nn.Parameter(torch.tensor(0.95))
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.in_proj(x)
        y = torch.cumsum(h * self.decay, dim=1)
        return self.head(y)


class MultiLayerModel(nn.Module):
    """Feedforward network with multiple distinct weight matrices for sharding validation."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(64, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.layer4(x)


# ---------------------------------------------------------------------------
# SafeTensors Sharding Tests
# ---------------------------------------------------------------------------


def test_safetensors_shard_state_dict_single_shard():
    """Verify that small state dict fits in a single shard without index."""
    model = MultiLayerModel()
    shards, index_dict = shard_state_dict(model.state_dict(), max_shard_size="100MB")
    assert len(shards) == 1
    assert "weight_map" in index_dict
    assert all(filename == SAFE_WEIGHTS_NAME for filename in index_dict["weight_map"].values())


def test_safetensors_shard_state_dict_multiple_shards():
    """Verify state dict partitioning across multiple shards when exceeding threshold."""
    model = MultiLayerModel()
    state = model.state_dict()
    # Force tiny shard size (e.g., 2048 bytes) so each layer ends up in a separate shard
    shards, index_dict = shard_state_dict(state, max_shard_size=2048)
    assert len(shards) > 1
    weight_map = index_dict["weight_map"]
    assert len(weight_map) == len(state)
    # Check that filenames follow model-0000X-of-0000Y.safetensors pattern
    for shard_file in weight_map.values():
        assert "model-" in shard_file
        assert ".safetensors" in shard_file


def test_save_sharded_safetensors_writes_valid_files(tmp_path: Path):
    """Verify sharded export writes valid safetensors files and index JSON."""
    model = MultiLayerModel()
    out_dir = tmp_path / "sharded_st"

    res = save_sharded_safetensors(model, output_dir=out_dir, max_shard_size=4096)
    assert res["num_shards"] > 1
    assert Path(res["manifest_file"]).exists()
    assert Path(res["index_file"]).exists()

    with open(res["index_file"], "r", encoding="utf-8") as f:
        idx_data = json.load(f)
    assert "weight_map" in idx_data
    assert "metadata" in idx_data

    # Verify each shard can be loaded using safetensors
    from safetensors.torch import load_file

    total_loaded_keys = 0
    for shard_path in [p for p in res["files"] if p.endswith(".safetensors")]:
        loaded = load_file(shard_path)
        total_loaded_keys += len(loaded)
    assert total_loaded_keys == len(model.state_dict())


def test_safetensors_exporter_class_integration(tmp_path: Path):
    """Verify SafeTensorsExporter backward-compatible and sharded invocation."""
    model = MultiLayerModel()
    exporter = SafeTensorsExporter(output_dir=tmp_path / "exporter_class")

    # Single-file export (legacy contract)
    legacy_out = exporter.export_model(model, filename_prefix="model_legacy")
    assert Path(legacy_out["weights"]).exists()
    assert Path(legacy_out["manifest"]).exists()

    # Sharded export
    sharded_out = exporter.export_model(model, max_shard_size=2048)
    assert sharded_out["num_shards"] > 1
    assert Path(sharded_out["manifest_file"]).exists()


# ---------------------------------------------------------------------------
# ONNX Dynamic Shape & Recurrent Model Tests
# ---------------------------------------------------------------------------


def test_build_dynamic_axes_for_recurrent():
    """Verify dynamic axes configuration generation for recurrent architectures."""
    input_names = ["inputs", "hx"]
    output_names = ["logits", "hx_next"]
    axes = build_dynamic_axes_for_recurrent(
        input_names=input_names,
        output_names=output_names,
        has_recurrent_state=True,
        state_names=["hx", "hx_next"],
    )
    assert axes["inputs"] == {0: "batch_size", 1: "sequence_length"}
    assert axes["hx"] == {0: "batch_size"}
    assert axes["logits"] == {0: "batch_size", 1: "sequence_length"}
    assert axes["hx_next"] == {0: "batch_size"}


def test_onnx_export_recurrent_model(tmp_path: Path):
    """Verify ONNX export with dynamic sequence length and batch size on recurrent dummy model."""
    import onnx
    import onnxruntime as ort

    model = RecurrentDummyModel(input_dim=16, hidden_dim=32, num_classes=8)
    out_file = tmp_path / "recurrent.onnx"
    sample_input = torch.randn(2, 5, 16)

    exported_path = export_to_onnx(
        model=model,
        output_path=out_file,
        sample_input=sample_input,
        is_recurrent=True,
    )
    assert exported_path.exists()

    # Load with ONNX proto checker
    proto = onnx.load(str(exported_path))
    onnx.checker.check_model(proto)

    # Validate dynamic execution with onnxruntime across varying batch size and sequence length
    sess = ort.InferenceSession(str(exported_path))
    input_name = sess.get_inputs()[0].name

    for bsz, seq in [(1, 3), (4, 12), (2, 7)]:
        x_dyn = torch.randn(bsz, seq, 16).numpy()
        ort_outs = sess.run(None, {input_name: x_dyn})
        logits = ort_outs[0]
        assert logits.shape == (bsz, seq, 8)


# ---------------------------------------------------------------------------
# GGUF Export Tests
# ---------------------------------------------------------------------------


def test_gguf_export_model_to_gguf(tmp_path: Path):
    """Verify GGUF binary export creates valid container readable by GGUFWriter specifications."""
    model = MultiLayerModel()
    out_file = tmp_path / "model.gguf"

    exported = export_model_to_gguf(
        model=model,
        output_path=out_file,
        architecture="llama",
        model_name="test_dense",
        quantization="f16",
    )
    assert exported.exists()
    assert exported.stat().st_size > 0

    # Read binary header and check magic b'GGUF' and version
    with open(exported, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF"


# ---------------------------------------------------------------------------
# Operation export.model Integration Tests
# ---------------------------------------------------------------------------


def test_export_model_operation_all_formats(tmp_path: Path):
    """Verify ExportModelOperation executes across all supported formats without stubs."""
    op = ExportModelOperation()
    model = MultiLayerModel()

    # Format 1: safetensors
    st_dir = tmp_path / "op_st"
    _, meta_st = op.apply(model, {"format": "safetensors", "output_dir": str(st_dir), "max_shard_size": 2048})
    assert meta_st["format"] == "safetensors"
    assert meta_st["num_shards"] > 1

    # Format 2: onnx
    onnx_dir = tmp_path / "op_onnx"
    sample_input = torch.randn(1, 64)
    _, meta_onnx = op.apply(
        model,
        {"format": "onnx", "output_dir": str(onnx_dir), "sample_input": sample_input},
    )
    assert meta_onnx["format"] == "onnx"
    assert Path(meta_onnx["output_path"]).exists()

    # Format 3: gguf
    gguf_dir = tmp_path / "op_gguf"
    _, meta_gguf = op.apply(
        model,
        {"format": "gguf", "output_dir": str(gguf_dir), "model_name": "op_test_gguf"},
    )
    assert meta_gguf["format"] == "gguf"
    assert Path(meta_gguf["output_path"]).exists()

    # Format 4: vllm bundle
    vllm_dir = tmp_path / "op_vllm"
    _, meta_vllm = op.apply(
        model,
        {"format": "vllm", "output_dir": str(vllm_dir)},
    )
    assert meta_vllm["format"] == "vllm"
    assert Path(meta_vllm["output_path"]).exists()

    # Format 5: sglang bundle
    sglang_dir = tmp_path / "op_sglang"
    _, meta_sglang = op.apply(
        model,
        {"format": "sglang", "output_dir": str(sglang_dir)},
    )
    assert meta_sglang["format"] == "sglang"
    assert Path(meta_sglang["output_path"]).exists()
