#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Native GGUF v3 Binary Container Exporter.

Implements pure-Python GGUF v3 binary serialization conforming to the
official llama.cpp GGUF specification without external native library requirements.
Adheres strictly to AGENTS.md §6.1: Writes valid GGUF container format or raises.
"""

from __future__ import annotations

import enum
import io
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Tuple, Union

import torch
import torch.nn as nn

# GGUF Magic & Version Constants
GGUF_MAGIC = b"GGUF"  # 0x46554747 in LE
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32


class GGUFValueType(enum.IntEnum):
    """GGUF metadata value data types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(enum.IntEnum):
    """GGML tensor data types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    I8 = 16
    I16 = 17
    I32 = 18
    I64 = 19
    F64 = 20
    BF16 = 30


class GGUFWriter:
    """Pure-Python GGUF v3 binary container serializer."""

    def __init__(self, output_path: Union[str, Path]):
        self.output_path = Path(output_path)
        self.metadata: Dict[str, Tuple[GGUFValueType, Any]] = {}
        self.tensors: List[Tuple[str, torch.Tensor, GGMLType]] = []
        self.alignment = GGUF_DEFAULT_ALIGNMENT

    def add_metadata(self, key: str, value: Any, val_type: Optional[GGUFValueType] = None) -> None:
        """Add metadata key-value pair to GGUF header."""
        if val_type is None:
            if isinstance(value, bool):
                val_type = GGUFValueType.BOOL
            elif isinstance(value, int):
                val_type = GGUFValueType.UINT32 if value >= 0 else GGUFValueType.INT32
            elif isinstance(value, float):
                val_type = GGUFValueType.FLOAT32
            elif isinstance(value, str):
                val_type = GGUFValueType.STRING
            elif isinstance(value, (list, tuple)):
                val_type = GGUFValueType.ARRAY
            else:
                val_type = GGUFValueType.STRING
                value = str(value)
        self.metadata[key] = (val_type, value)

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Register a PyTorch tensor for binary GGUF export."""
        # Convert to CPU contiguous tensor
        t = tensor.detach().cpu().contiguous()
        if t.dtype == torch.float32:
            ggml_type = GGMLType.F32
        elif t.dtype == torch.float16:
            ggml_type = GGMLType.F16
        elif t.dtype == torch.bfloat16:
            ggml_type = GGMLType.BF16
        elif t.dtype == torch.int32:
            ggml_type = GGMLType.I32
        elif t.dtype == torch.int64:
            ggml_type = GGMLType.I64
        elif t.dtype == torch.int8:
            ggml_type = GGMLType.I8
        else:
            t = t.to(torch.float32)
            ggml_type = GGMLType.F32
        self.tensors.append((name, t, ggml_type))

    @staticmethod
    def _write_string(f: BinaryIO, s: str) -> None:
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def _write_value(self, f: BinaryIO, val_type: GGUFValueType, value: Any) -> None:
        if val_type == GGUFValueType.UINT8:
            f.write(struct.pack("<B", value))
        elif val_type == GGUFValueType.INT8:
            f.write(struct.pack("<b", value))
        elif val_type == GGUFValueType.UINT16:
            f.write(struct.pack("<H", value))
        elif val_type == GGUFValueType.INT16:
            f.write(struct.pack("<h", value))
        elif val_type == GGUFValueType.UINT32:
            f.write(struct.pack("<I", value))
        elif val_type == GGUFValueType.INT32:
            f.write(struct.pack("<i", value))
        elif val_type == GGUFValueType.FLOAT32:
            f.write(struct.pack("<f", value))
        elif val_type == GGUFValueType.BOOL:
            f.write(struct.pack("<B", 1 if value else 0))
        elif val_type == GGUFValueType.STRING:
            self._write_string(f, str(value))
        elif val_type == GGUFValueType.UINT64:
            f.write(struct.pack("<Q", value))
        elif val_type == GGUFValueType.INT64:
            f.write(struct.pack("<q", value))
        elif val_type == GGUFValueType.FLOAT64:
            f.write(struct.pack("<d", value))
        elif val_type == GGUFValueType.ARRAY:
            if not isinstance(value, (list, tuple)) or len(value) == 0:
                f.write(struct.pack("<I", GGUFValueType.STRING))
                f.write(struct.pack("<Q", 0))
                return
            elem_val = value[0]
            if isinstance(elem_val, str):
                elem_type = GGUFValueType.STRING
            elif isinstance(elem_val, int):
                elem_type = GGUFValueType.INT32
            elif isinstance(elem_val, float):
                elem_type = GGUFValueType.FLOAT32
            else:
                elem_type = GGUFValueType.STRING
            f.write(struct.pack("<I", elem_type))
            f.write(struct.pack("<Q", len(value)))
            for item in value:
                self._write_value(f, elem_type, item)
        else:
            raise ValueError(f"Unsupported GGUFValueType: {val_type}")

    def write(self) -> Path:
        """Write complete GGUF v3 file to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "wb") as f:
            # 1. Header Magic & Version
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            # 2. Metadata Key-Value pairs
            for key, (val_type, val) in self.metadata.items():
                self._write_string(f, key)
                f.write(struct.pack("<I", val_type))
                self._write_value(f, val_type, val)

            # 3. Calculate tensor body offsets
            # In GGUF v3:
            # Each tensor info entry:
            # - name: string (uint64 length + bytes)
            # - n_dims: uint32
            # - dims: uint64 * n_dims (reverse order / column-major)
            # - type: uint32 (GGMLType)
            # - offset: uint64 (offset relative to start of binary tensor data block)
            tensor_info_stream = io.BytesIO()
            tensor_data_offsets: List[int] = []
            current_data_offset = 0

            for name, tensor, ggml_type in self.tensors:
                # Align tensor data offset
                rem = current_data_offset % self.alignment
                if rem != 0:
                    current_data_offset += (self.alignment - rem)
                tensor_data_offsets.append(current_data_offset)

                # Reverse dimensions for GGUF/GGML convention
                dims = list(reversed(tensor.shape))
                self._write_string(tensor_info_stream, name)
                tensor_info_stream.write(struct.pack("<I", len(dims)))
                for d in dims:
                    tensor_info_stream.write(struct.pack("<Q", d))
                tensor_info_stream.write(struct.pack("<I", int(ggml_type)))
                tensor_info_stream.write(struct.pack("<Q", current_data_offset))

                # Increment byte size
                current_data_offset += tensor.numel() * tensor.element_size()

            f.write(tensor_info_stream.getvalue())

            # 4. Align start of tensor data block
            current_file_pos = f.tell()
            rem = current_file_pos % self.alignment
            if rem != 0:
                f.write(b"\x00" * (self.alignment - rem))

            # 5. Write raw tensor data buffers
            for idx, (name, tensor, ggml_type) in enumerate(self.tensors):
                target_offset = tensor_data_offsets[idx]
                pad_needed = target_offset - (f.tell() - (current_file_pos + (self.alignment - rem if rem != 0 else 0)))
                if pad_needed > 0:
                    f.write(b"\x00" * pad_needed)
                
                # Write tensor bytes directly
                raw_bytes = tensor.numpy().tobytes()
                f.write(raw_bytes)

        return self.output_path


def export_to_gguf(
    model: nn.Module,
    output_path: Union[str, Path],
    architecture: str = "llama",
    model_name: Optional[str] = None,
    context_length: int = 2048,
    embedding_length: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Export a PyTorch / NeuralOps model into a valid GGUF v3 binary artifact.

    Args:
        model: Target PyTorch neural network module.
        output_path: Destination .gguf file path or directory.
        architecture: GGUF architecture string (e.g., 'llama', 'bert', 'neuralops').
        model_name: Descriptive model identifier.
        context_length: Maximum sequence context length.
        embedding_length: Model hidden dimension.
        metadata: Optional dictionary of additional GGUF metadata tags.

    Returns:
        Path to the written .gguf file.
    """
    out = Path(output_path)
    if out.is_dir() or not out.suffix:
        out = out / f"{model_name or 'model'}.gguf"

    writer = GGUFWriter(out)

    # Standard GGUF General Metadata
    writer.add_metadata("general.architecture", architecture)
    writer.add_metadata("general.name", model_name or "neuralops-model")
    writer.add_metadata(f"{architecture}.context_length", context_length)
    if embedding_length:
        writer.add_metadata(f"{architecture}.embedding_length", embedding_length)

    if metadata:
        for k, v in metadata.items():
            writer.add_metadata(k, v)

    # Populate state dictionary tensors
    state = model.state_dict()
    for name, param in state.items():
        writer.add_tensor(name, param)

    written_path = writer.write()
    return written_path
