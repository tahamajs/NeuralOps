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
import re
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
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


def quantize_to_q8_0(tensor: torch.Tensor) -> bytes:
    """
    Quantize a tensor into GGML Q8_0 format.

    Q8_0 block format:
    Each block consists of 32 elements:
    - 2 bytes float16 scale factor d (in little endian)
    - 32 bytes int8 quantized values qs
    Total bytes per 32 elements: 34 bytes.
    """
    flat = tensor.detach().cpu().to(torch.float32).flatten()
    n = flat.numel()
    if n % 32 != 0:
        raise ValueError(
            f"Tensor element count ({n}) must be a multiple of 32 for Q8_0 quantization."
        )
    blocks = flat.view(-1, 32)
    # block scale factor d = amax / 127.0
    d = torch.max(torch.abs(blocks), dim=-1, keepdim=True).values / 127.0
    d = torch.where(d == 0, torch.tensor(1.0, dtype=torch.float32), d)
    qs = torch.clamp(torch.round(blocks / d), -128, 127).to(torch.int8)
    d_f16 = d.squeeze(-1).to(torch.float16)

    dt = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
    arr = np.empty(n // 32, dtype=dt)
    arr["d"] = np.frombuffer(d_f16.numpy().tobytes(), dtype="<f2")
    arr["qs"] = np.frombuffer(qs.numpy().tobytes(), dtype="i1").reshape(-1, 32)
    return arr.tobytes()


class GGUFWriter:
    """Pure-Python GGUF v3 binary container serializer."""

    def __init__(self, output_path: Union[str, Path]):
        self.output_path = Path(output_path)
        self.metadata: Dict[str, Tuple[GGUFValueType, Any]] = {}
        # List of (name, raw_bytes, shape, ggml_type)
        self.tensors: List[Tuple[str, bytes, Tuple[int, ...], GGMLType]] = []
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

    def add_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        quantization: Optional[str] = None,
    ) -> None:
        """Register a PyTorch tensor for binary GGUF export with optional quantization."""
        t = tensor.detach().cpu().contiguous()
        shape = tuple(t.shape)

        quant = (quantization or "").lower().strip()
        if quant == "q8_0" and t.numel() % 32 == 0 and t.dtype in (torch.float32, torch.float16, torch.bfloat16):
            raw_bytes = quantize_to_q8_0(t)
            ggml_type = GGMLType.Q8_0
        elif quant in ("f16", "fp16") or (not quant and t.dtype == torch.float16):
            t = t.to(torch.float16)
            raw_bytes = t.numpy().tobytes()
            ggml_type = GGMLType.F16
        elif quant in ("f32", "fp32") or (not quant and t.dtype == torch.float32):
            t = t.to(torch.float32)
            raw_bytes = t.numpy().tobytes()
            ggml_type = GGMLType.F32
        elif t.dtype == torch.bfloat16:
            if quant in ("f16", "fp16"):
                t = t.to(torch.float16)
                raw_bytes = t.numpy().tobytes()
                ggml_type = GGMLType.F16
            else:
                raw_bytes = t.view(torch.int16).numpy().tobytes()
                ggml_type = GGMLType.BF16
        elif t.dtype == torch.int32:
            raw_bytes = t.numpy().tobytes()
            ggml_type = GGMLType.I32
        elif t.dtype == torch.int64:
            raw_bytes = t.numpy().tobytes()
            ggml_type = GGMLType.I64
        elif t.dtype == torch.int8:
            raw_bytes = t.numpy().tobytes()
            ggml_type = GGMLType.I8
        else:
            t = t.to(torch.float32)
            raw_bytes = t.numpy().tobytes()
            ggml_type = GGMLType.F32

        self.tensors.append((name, raw_bytes, shape, ggml_type))

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
        """Write complete GGUF v3 file to disk with 32-byte alignment."""
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

            for name, raw_bytes, shape, ggml_type in self.tensors:
                # Align tensor data offset to self.alignment (32 bytes)
                rem = current_data_offset % self.alignment
                if rem != 0:
                    current_data_offset += (self.alignment - rem)
                tensor_data_offsets.append(current_data_offset)

                # Reverse dimensions for GGUF/GGML convention
                dims = list(reversed(shape))
                self._write_string(tensor_info_stream, name)
                tensor_info_stream.write(struct.pack("<I", len(dims)))
                for d in dims:
                    tensor_info_stream.write(struct.pack("<Q", d))
                tensor_info_stream.write(struct.pack("<I", int(ggml_type)))
                tensor_info_stream.write(struct.pack("<Q", current_data_offset))

                # Increment byte size
                current_data_offset += len(raw_bytes)

            f.write(tensor_info_stream.getvalue())

            # 4. Align start of tensor data block to 32 bytes
            current_file_pos = f.tell()
            rem = current_file_pos % self.alignment
            if rem != 0:
                f.write(b"\x00" * (self.alignment - rem))

            # Base position for relative tensor offsets
            tensor_block_base_pos = f.tell()

            # 5. Write raw tensor data buffers
            for idx, (name, raw_bytes, shape, ggml_type) in enumerate(self.tensors):
                target_offset = tensor_data_offsets[idx]
                target_file_pos = tensor_block_base_pos + target_offset
                pad_needed = target_file_pos - f.tell()
                if pad_needed > 0:
                    f.write(b"\x00" * pad_needed)

                # Write tensor bytes directly
                f.write(raw_bytes)

        return self.output_path


def format_llama_tensor_name(name: str) -> str:
    """
    Format standard PyTorch / NeuralOps parameter names into canonical llama.cpp tensor names.
    Examples:
        transformer.h.0.attn.attn_u.weight -> blk.0.attn_u.weight
        model.layers.3.attn_v.weight -> blk.3.attn_v.weight
        layer.2.U -> blk.2.attn_u.weight
        layer.2.V -> blk.2.attn_v.weight
    """
    # Replace transformer.h.X or model.layers.X or layers.X or layer.X with blk.X
    formatted = re.sub(r"^(?:model\.)?(?:transformer\.)?(?:layers|layer|h)\.(\d+)\.", r"blk.\1.", name)
    return formatted


def export_to_gguf(
    model: nn.Module,
    output_path: Union[str, Path],
    architecture: str = "llama",
    model_name: Optional[str] = None,
    context_length: int = 2048,
    embedding_length: Optional[int] = None,
    quantization: str = "f16",
    fuse_low_rank: bool = False,
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
        quantization: Target GGML quantization ('f32', 'f16', 'q8_0').
        fuse_low_rank: Whether to mathematically fuse low-rank linear pairs (attn_u @ attn_v).
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
    writer.add_metadata("general.file_type", 1 if quantization == "f16" else (8 if quantization == "q8_0" else 0))
    writer.add_metadata(f"{architecture}.context_length", context_length)
    if embedding_length:
        writer.add_metadata(f"{architecture}.embedding_length", embedding_length)

    if metadata:
        for k, v in metadata.items():
            writer.add_metadata(k, v)

    # Extract state dict and process low-rank pairs
    state = model.state_dict()
    consumed_keys = set()

    # Search for low-rank pairs (e.g. U/V or attn_u/attn_v)
    processed_tensors: Dict[str, torch.Tensor] = {}

    # 1. Identify pairs: prefix + ("attn_u" / "attn_v") or ("U" / "V")
    pairs: Dict[str, Dict[str, str]] = {}
    for key in state.keys():
        m_uv = re.match(r"^(.*?)(?:(?:\b|_)attn_([uv])(?:\b|_)|(?:\b|_)([UV])(?:\b|_))(.*)$", key)
        if m_uv:
            prefix = m_uv.group(1)
            kind = (m_uv.group(2) or m_uv.group(3)).lower()  # 'u' or 'v'
            suffix = m_uv.group(4)
            pair_key = f"{prefix}__PAIR__{suffix}"
            if pair_key not in pairs:
                pairs[pair_key] = {}
            pairs[pair_key][kind] = key

    for pair_key, pair_dict in pairs.items():
        if "u" in pair_dict and "v" in pair_dict:
            u_key = pair_dict["u"]
            v_key = pair_dict["v"]
            u_tensor = state[u_key].detach().cpu().to(torch.float32)
            v_tensor = state[v_key].detach().cpu().to(torch.float32)

            consumed_keys.add(u_key)
            consumed_keys.add(v_key)

            if fuse_low_rank:
                # Math: U @ V (or V @ U depending on shape: outer dimension matching)
                # Linear: y = x @ W^T or y = (x @ U^T) @ V^T
                # If attn_u is (r, in_features) and attn_v is (out_features, r):
                #   fused_weight = attn_v.weight @ attn_u.weight -> (out_features, in_features)
                # If SVDLinear: U is (out_features, r), V is (r, in_features):
                #   fused_weight = U @ V -> (out_features, in_features)
                if u_tensor.ndim == 2 and v_tensor.ndim == 2:
                    if u_tensor.shape[1] == v_tensor.shape[0]:
                        fused = torch.matmul(u_tensor, v_tensor)
                    elif v_tensor.shape[1] == u_tensor.shape[0]:
                        fused = torch.matmul(v_tensor, u_tensor)
                    else:
                        fused = torch.matmul(u_tensor.t(), v_tensor)
                    fused_name = format_llama_tensor_name(u_key.replace("attn_u", "weight").replace(".U", ".weight"))
                    processed_tensors[fused_name] = fused
                else:
                    # Fallback to separate tensors
                    processed_tensors[format_llama_tensor_name(u_key)] = u_tensor
                    processed_tensors[format_llama_tensor_name(v_key)] = v_tensor
            else:
                # Retain separate decomposed low-rank factors
                processed_tensors[format_llama_tensor_name(u_key)] = u_tensor
                processed_tensors[format_llama_tensor_name(v_key)] = v_tensor

    # Add remaining tensors
    for key, val in state.items():
        if key not in consumed_keys:
            formatted_name = format_llama_tensor_name(key)
            processed_tensors[formatted_name] = val

    # Register all processed tensors into writer
    for name, tensor in processed_tensors.items():
        writer.add_tensor(name, tensor, quantization=quantization)

    written_path = writer.write()
    return written_path
