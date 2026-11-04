#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for GGUF v3 edge binary export and recipe pipeline integration."""

from pathlib import Path
import struct
import tempfile
import torch
import torch.nn as nn

from neuralops.export.exporter import export_model
from neuralops.export.gguf_exporter import GGUF_MAGIC, export_to_gguf
from neuralops.export.runtime_compat import check_compatibility
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


class TransformerBlockWithLowRank(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulates layers.0.attn_u and layers.0.attn_v
        self.attn_u = nn.Linear(64, 16, bias=False)
        self.attn_v = nn.Linear(16, 64, bias=False)
        self.mlp = nn.Linear(64, 64)

    def forward(self, x):
        h = self.attn_v(self.attn_u(x))
        return self.mlp(h)


def verify_gguf_binary(filepath: Path) -> dict:
    """Read and verify 32-byte alignment of header and tensor buffers in GGUF v3."""
    with open(filepath, "rb") as f:
        magic = f.read(4)
        assert magic == GGUF_MAGIC, f"Invalid GGUF magic: {magic}"
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        assert version == 3, f"Expected GGUF version 3, got {version}"

        # Skip KV metadata
        for _ in range(n_kv):
            k_len = struct.unpack("<Q", f.read(8))[0]
            _key = f.read(k_len).decode("utf-8")
            v_type = struct.unpack("<I", f.read(4))[0]
            if v_type in (0, 1, 7):
                f.seek(1, 1)
            elif v_type in (2, 3):
                f.seek(2, 1)
            elif v_type in (4, 5, 6):
                f.seek(4, 1)
            elif v_type in (10, 11, 12):
                f.seek(8, 1)
            elif v_type == 8:
                slen = struct.unpack("<Q", f.read(8))[0]
                f.seek(slen, 1)
            elif v_type == 9:
                elem_type = struct.unpack("<I", f.read(4))[0]
                arr_len = struct.unpack("<Q", f.read(8))[0]
                for _ in range(arr_len):
                    if elem_type in (0, 1, 7):
                        f.seek(1, 1)
                    elif elem_type in (2, 3):
                        f.seek(2, 1)
                    elif elem_type in (4, 5, 6):
                        f.seek(4, 1)
                    elif elem_type in (10, 11, 12):
                        f.seek(8, 1)
                    elif elem_type == 8:
                        slen = struct.unpack("<Q", f.read(8))[0]
                        f.seek(slen, 1)

        # Parse tensor info
        tensor_info = {}
        for _ in range(n_tensors):
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            t_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensor_info[name] = {"dims": dims, "type": t_type, "offset": offset}

        header_end = f.tell()
        rem = header_end % 32
        data_start = header_end + (32 - rem if rem != 0 else 0)
        assert data_start % 32 == 0

        # Check each tensor's absolute position in file
        for name, info in tensor_info.items():
            abs_pos = data_start + info["offset"]
            assert abs_pos % 32 == 0, f"Tensor {name} at byte {abs_pos} is not 32-byte aligned!"

        return {
            "version": version,
            "n_tensors": n_tensors,
            "n_kv": n_kv,
            "tensors": tensor_info,
            "data_start": data_start,
        }


def test_gguf_direct_export_f16():
    """Verify direct GGUF export with F16 precision."""
    model = ToyModelWithLowRank()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = export_to_gguf(model, output_path=Path(tmpdir) / "model.gguf", quantization="f16")
        assert Path(out_file).exists()
        parsed = verify_gguf_binary(Path(out_file))
        assert parsed["n_tensors"] == 6
        # Assert attn_u and attn_v are retained as separate low-rank factors
        assert any("attn_u" in t for t in parsed["tensors"])
        assert any("attn_v" in t for t in parsed["tensors"])


def test_gguf_direct_export_q8_0():
    """Verify direct GGUF export with Q8_0 block quantization."""
    model = TransformerBlockWithLowRank()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = export_to_gguf(model, output_path=Path(tmpdir) / "model_q8.gguf", quantization="q8_0")
        assert Path(out_file).exists()
        parsed = verify_gguf_binary(Path(out_file))
        # GGMLType 8 is Q8_0
        for name, t_info in parsed["tensors"].items():
            assert t_info["type"] == 8, f"Tensor {name} type is {t_info['type']}, expected 8 (Q8_0)"


def test_gguf_direct_export_fused_low_rank():
    """Verify low-rank factor pair fusion (attn_u @ attn_v)."""
    model = ToyModelWithLowRank()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = export_to_gguf(
            model,
            output_path=Path(tmpdir) / "fused_model.gguf",
            quantization="f32",
            fuse_low_rank=True,
        )
        assert Path(out_file).exists()
        parsed = verify_gguf_binary(Path(out_file))
        # With fusion, attn_u and attn_v should be collapsed into a single weight tensor
        assert not any("attn_u" in t for t in parsed["tensors"])
        assert not any("attn_v" in t for t in parsed["tensors"])
        assert any("weight" in t for t in parsed["tensors"])


def test_gguf_unified_export():
    """Verify unified export interface supports gguf format."""
    model = ToyModelWithLowRank()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = unified_export(model, format="gguf", output_path=Path(tmpdir) / "unified_model.gguf")
        assert Path(out_path).exists()
        parsed = verify_gguf_binary(Path(out_path))
        assert parsed["version"] == 3


def test_gguf_recipe_operation():
    """Verify ExportModelOperation applies gguf export without error."""
    model = ToyModelWithLowRank()
    op = ExportModelOperation()
    with tempfile.TemporaryDirectory() as tmpdir:
        _, res = op.apply(model, {"format": "gguf", "output_dir": tmpdir})
        assert res["format"] == "gguf"
        assert Path(res["output_path"]).exists()
        parsed = verify_gguf_binary(Path(res["output_path"]))
        assert parsed["version"] == 3


def test_gguf_exporter_model_roundtrip_compatibility():
    """Verify export_model with format='gguf' and runtime_compat inspection."""
    model = ToyModelWithLowRank()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = export_model(model, tmpdir, format="gguf")
        assert out_file.exists()
        parsed = verify_gguf_binary(out_file)
        assert parsed["n_tensors"] >= 1
        rep = check_compatibility(tmpdir, engines=["gguf"])
        assert rep.results["gguf"].status == "SUCCESS"
