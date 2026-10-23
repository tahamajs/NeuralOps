#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for silent failure cleanup and explicit error handling (Wave 2, Agent C).

Covers:
- Zero bare 'except Exception: pass' across owned files.
- ONNX export verification raising RuntimeError on corrupted or failing sessions.
- ONNX export warning on missing onnxruntime.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from neuralops.export.exporter import export_model

OWNED_MODULES = [
    Path("neuralops/core/roofline.py"),
    Path("neuralops/core/composable.py"),
    Path("neuralops/core/inspector.py"),
    Path("neuralops/core/hardware.py"),
    Path("neuralops/core/snapshot_restore.py"),
    Path("neuralops/layers/hybrid/spectron_fp8_attention.py"),
    Path("neuralops/export/exporter.py"),
]


def test_zero_bare_except_in_owned_files():
    """Verify that no bare 'except Exception: pass' exists in owned files."""
    pattern = re.compile(r"except\s+Exception\s*:\s*(\n\s*)?pass")

    for path in OWNED_MODULES:
        assert path.exists(), f"File {path} does not exist"
        content = path.read_text(encoding="utf-8")
        matches = list(pattern.finditer(content))
        assert len(matches) == 0, (
            f"Found {len(matches)} bare except-pass blocks in {path}: "
            f"{[m.group(0) for m in matches]}"
        )


def test_grep_gate_zero_results():
    """Verify the exact gate command returns zero matching lines."""
    pattern = re.compile(r"except Exception:\s*$")
    pass_pattern = re.compile(r"^\s*pass")

    for path in OWNED_MODULES:
        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            if pattern.search(line):
                next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
                assert not pass_pattern.search(next_line), (
                    f"Found bare except-pass pattern at {path}:{idx+1}"
                )


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        return self.fc(x)


def test_onnx_export_raises_on_verification_failure(tmp_path: Path, monkeypatch):
    """ONNX export must raise RuntimeError if onnxruntime verification fails."""
    import builtins

    model = SimpleModel()
    dummy_input = torch.randn(1, 16)
    out_dir = tmp_path / "onnx_corrupt"

    def fake_export(wrapped_model, sample_input, path_str, **kwargs):
        Path(path_str).write_text("dummy onnx content")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            fake_ort = types.ModuleType("onnxruntime")

            def failing_session(*_args, **_kwargs):
                raise ValueError("Corrupted ONNX protobuf model: invalid node definition")

            fake_ort.InferenceSession = failing_session
            return fake_ort
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(RuntimeError, match="ONNX export verification failed"):
        export_model(
            model=model,
            output_dir=out_dir,
            format="onnx",
            sample_input=dummy_input,
        )


def test_onnx_export_warns_when_onnxruntime_missing(tmp_path: Path, monkeypatch, caplog):
    """ONNX export must log a warning if onnxruntime cannot be imported."""
    import builtins

    model = SimpleModel()
    dummy_input = torch.randn(1, 16)
    out_dir = tmp_path / "onnx_missing_ort"

    def fake_export(wrapped_model, sample_input, path_str, **kwargs):
        Path(path_str).write_text("dummy onnx content")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("No module named 'onnxruntime'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with caplog.at_level(logging.WARNING):
        exported_path = export_model(
            model=model,
            output_dir=out_dir,
            format="onnx",
            sample_input=dummy_input,
        )
        assert exported_path.exists()
        assert any(
            "onnxruntime is not installed; skipping ONNX runtime verification" in rec.message
            for rec in caplog.records
        )


def test_spectron_fp8_attention_shim():
    """Verify backward-compatibility shim redirects cleanly to layers/hybrid."""
    from neuralops.attentions.hybrid.spectron_fp8_attention import AegisSpectronFP8Attention as OldAttn
    from neuralops.layers.hybrid.spectron_fp8_attention import AegisSpectronFP8Attention as NewAttn
    assert OldAttn is NewAttn
