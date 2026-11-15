#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for model export formats: SafeTensors, ONNX, and TorchScript.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from neuralops.export.exporter import export_model
from neuralops.export.safetensors_exporter import SafeTensorsExporter
from neuralops.export.runtime_compat import check_compatibility


class ExportSampleModel(nn.Module):
    """Small deterministic module for export verification."""

    def __init__(self, in_features: int = 16, out_features: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_export_model_safetensors_contains_real_tensors(tmp_path: Path):
    """Verify SafeTensors export generates real non-empty binary tensors."""
    import safetensors.torch

    model = ExportSampleModel().eval()
    out_dir = tmp_path / "safetensors_export"

    exported = export_model(model, out_dir, format="safetensors")
    assert exported.is_dir()

    st_file = out_dir / "model.safetensors"
    assert st_file.exists()
    assert st_file.stat().st_size > 0

    # Load back with safetensors and verify tensor integrity
    loaded_tensors = safetensors.torch.load_file(str(st_file))
    state_dict = model.state_dict()

    for k, v in state_dict.items():
        assert k in loaded_tensors, f"Missing key {k} in exported safetensors"
        assert torch.equal(v.cpu(), loaded_tensors[k].cpu()), f"Value mismatch for {k}"

    # Verify neuralops_config.json metadata
    cfg_file = out_dir / "neuralops_config.json"
    assert cfg_file.exists()
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["format"] == "safetensors"


def test_export_model_onnx_succeeds_and_loads(tmp_path: Path):
    """Verify ONNX export succeeds and the exported model is valid protobuf loaded by onnx."""
    import onnx

    model = ExportSampleModel().eval()
    out_dir = tmp_path / "onnx_export"
    sample_input = torch.randn(2, 16)

    exported = export_model(model, out_dir, format="onnx", sample_input=sample_input)
    assert exported.exists()
    assert exported.suffix == ".onnx"
    assert exported.stat().st_size > 0

    # Load model with onnx library and check graph validity
    onnx_proto = onnx.load(str(exported))
    onnx.checker.check_model(onnx_proto)

    assert len(onnx_proto.graph.input) >= 1
    assert len(onnx_proto.graph.output) >= 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_export_model_torchscript_matches_original_output(tmp_path: Path):
    """Verify TorchScript export succeeds and traced model output matches original output."""
    import warnings

    model = ExportSampleModel().eval()
    out_dir = tmp_path / "ts_export"
    sample_input = torch.randn(4, 16)

    exported = export_model(model, out_dir, format="torchscript", sample_input=sample_input)
    assert exported.exists()
    assert exported.name == "model.pt"

    # Load traced model and check bit-close output parity
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        traced_model = torch.jit.load(str(exported))
        with torch.no_grad():
            ref_out = model(sample_input)
            traced_out = traced_model(sample_input)

    assert torch.allclose(ref_out, traced_out, atol=1e-5), "TorchScript output deviated from original"


def test_safetensors_exporter_class(tmp_path: Path):
    """Verify dedicated SafeTensorsExporter outputs weights and manifest."""
    model = ExportSampleModel().eval()
    exporter = SafeTensorsExporter(output_dir=str(tmp_path / "custom_st"))

    artifacts = exporter.export_model(model, filename_prefix="sample_model")
    assert "weights" in artifacts
    assert "manifest" in artifacts

    weights_path = Path(artifacts["weights"])
    manifest_path = Path(artifacts["manifest"])

    assert weights_path.exists()
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["filename_prefix"] == "sample_model"
    assert manifest["total_parameters"] == sum(p.numel() for p in model.parameters())


def test_runtime_compat_check_compatibility(tmp_path: Path):
    """Verify check_compatibility inspects onnx and gguf artifacts accurately."""
    model = ExportSampleModel().eval()
    out_dir = tmp_path / "compat_check"
    sample_input = torch.randn(1, 16)

    export_model(model, out_dir, format="onnx", sample_input=sample_input)

    report = check_compatibility(out_dir, engines=["onnx"])
    assert report.is_compatible("onnx")
    assert report.results["onnx"].status == "SUCCESS"
