#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for GGUF v3 edge binary export and recipe pipeline integration."""

from pathlib import Path
import tempfile
import torch
import torch.nn as nn

from neuralops.export.exporter import export_model
from neuralops.export.gguf_exporter import GGUF_MAGIC, export_to_gguf
from neuralops.export.unified import export as unified_export
from neuralops.operations.export.model import ExportModelOperation


class ToyModelWithLowRank(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.attn_u = nn.Linear(16, 4, bias=False)
        self.attn_v = nn.Linear(4, 16, bias=False)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.attn_v(self.attn_u(h))
        return self.fc2(h)


def test_gguf_direct_export_f16():
    """Verify direct GGUF export with F16 precision."""
    model = ToyModelWithLowRank()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = export_to_gguf(model, output_path=Path(tmpdir) / "model.gguf", quantization="f16")
        assert Path(out_file).exists()
        with open(out_file, "rb") as f:
            magic = f.read(4)
            assert magic == GGUF_MAGIC


def test_gguf_unified_export():
    """Verify unified export interface supports gguf."""
    model = ToyModelWithLowRank()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = unified_export(model, format="gguf", output_path=Path(tmpdir) / "unified_model.gguf")
        assert Path(out_path).exists()
        with open(out_path, "rb") as f:
            assert f.read(4) == GGUF_MAGIC


def test_gguf_recipe_operation():
    """Verify ExportModelOperation applies gguf export without error."""
    model = ToyModelWithLowRank()
    op = ExportModelOperation()
    with tempfile.TemporaryDirectory() as tmpdir:
        _, res = op.apply(model, {"format": "gguf", "output_dir": tmpdir})
        assert res["format"] == "gguf"
        assert Path(res["output_path"]).exists()
