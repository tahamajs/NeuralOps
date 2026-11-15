#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SafeTensors Serializer and Sharding Engine.

Provides unified single-file and sharded SafeTensors serialization for PyTorch models
and weight dictionaries with index generation and manifest tracking, compliant with HuggingFace
weight sharding specifications.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Export.SafeTensors")

# Default shard size: 5 GB in bytes
DEFAULT_MAX_SHARD_SIZE = 5 * 1024 * 1024 * 1024
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def _parse_size(size_str_or_int: Union[int, str]) -> int:
    """Parse integer bytes or human-readable size string (e.g. '500MB', '2GB').

    Args:
        size_str_or_int: Integer bytes or size string with units.

    Returns:
        Integer size in bytes.

    Raises:
        ValueError: If size cannot be parsed or is non-positive.
    """
    if isinstance(size_str_or_int, int):
        if size_str_or_int <= 0:
            raise ValueError(f"Shard size must be positive, got {size_str_or_int}")
        return size_str_or_int

    s = size_str_or_int.strip().upper()
    units = {
        "B": 1,
        "KB": 1024,
        "KIB": 1024,
        "MB": 1024 * 1024,
        "MIB": 1024 * 1024,
        "GB": 1024 * 1024 * 1024,
        "GIB": 1024 * 1024 * 1024,
        "TB": 1024 * 1024 * 1024 * 1024,
        "TIB": 1024 * 1024 * 1024 * 1024,
    }

    for unit, mult in sorted(units.items(), key=lambda x: -len(x[0])):
        if s.endswith(unit):
            num_part = s[: -len(unit)].strip()
            try:
                val = float(num_part)
                if val <= 0:
                    raise ValueError(f"Shard size must be positive, got {size_str_or_int}")
                return int(val * mult)
            except ValueError as exc:
                raise ValueError(f"Invalid size string {size_str_or_int!r}: {exc}") from exc

    try:
        val = int(s)
        if val <= 0:
            raise ValueError(f"Shard size must be positive, got {size_str_or_int}")
        return val
    except ValueError as exc:
        raise ValueError(f"Unrecognized size string {size_str_or_int!r}") from exc


def get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Return memory footprint of tensor in bytes.

    Args:
        tensor: PyTorch tensor.

    Returns:
        Total size in bytes.
    """
    return tensor.numel() * tensor.element_size()


def shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    max_shard_size: Union[int, str] = DEFAULT_MAX_SHARD_SIZE,
    filename_pattern: str = "model-{shard_idx:05d}-of-{num_shards:05d}.safetensors",
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    """Partition a state dict into shards each bounded by max_shard_size.

    Args:
        state_dict: Mapping of parameter/buffer names to PyTorch tensors.
        max_shard_size: Maximum size per shard in bytes or size string.
        filename_pattern: Format string for shard filenames.

    Returns:
        Tuple of (list of shard dicts, index metadata dictionary).
    """
    max_bytes = _parse_size(max_shard_size)
    shards: List[Dict[str, torch.Tensor]] = []
    current_shard: Dict[str, torch.Tensor] = {}
    current_size = 0

    weight_map: Dict[str, str] = {}
    total_size = 0

    # First pass: partition tensors
    for name, tensor in state_dict.items():
        t_bytes = get_tensor_size_bytes(tensor)
        total_size += t_bytes

        # If adding this tensor exceeds max_bytes and current shard is non-empty, start a new shard
        if current_shard and (current_size + t_bytes > max_bytes):
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[name] = tensor
        current_size += t_bytes

    if current_shard or not shards:
        shards.append(current_shard)

    num_shards = len(shards)
    for shard_idx, shard in enumerate(shards, start=1):
        if num_shards == 1:
            shard_filename = SAFE_WEIGHTS_NAME
        else:
            shard_filename = filename_pattern.format(shard_idx=shard_idx, num_shards=num_shards)
        for tensor_name in shard.keys():
            weight_map[tensor_name] = shard_filename

    index_dict = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    return shards, index_dict


def save_sharded_safetensors(
    model_or_state_dict: Union[nn.Module, Dict[str, torch.Tensor]],
    output_dir: Union[str, Path],
    max_shard_size: Union[int, str] = DEFAULT_MAX_SHARD_SIZE,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Save model or state dict as single or sharded SafeTensors with index and manifest.

    Args:
        model_or_state_dict: PyTorch Module or dictionary of tensors.
        output_dir: Target directory path for exported files.
        max_shard_size: Maximum shard size before splitting (default 5GB).
        metadata: Optional string metadata dictionary for SafeTensors header.

    Returns:
        Summary dictionary containing paths to written files and shard index.

    Raises:
        RuntimeError: If safetensors library is not installed.
    """
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "SafeTensors export requested but 'safetensors' package is not installed. "
            "Please install safetensors: pip install safetensors"
        ) from exc

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if isinstance(model_or_state_dict, nn.Module):
        state_dict = model_or_state_dict.state_dict()
    else:
        state_dict = model_or_state_dict

    # Ensure tensors are detached, cpu, contiguous
    clean_dict = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}

    shards, index_data = shard_state_dict(clean_dict, max_shard_size=max_shard_size)
    num_shards = len(shards)

    written_files: List[Path] = []
    meta = metadata or {"format": "pt"}

    if num_shards == 1:
        single_file = out_p / SAFE_WEIGHTS_NAME
        save_file(clean_dict, str(single_file), metadata=meta)
        written_files.append(single_file)
        index_file = None
    else:
        for shard_idx, shard in enumerate(shards, start=1):
            shard_name = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
            shard_path = out_p / shard_name
            save_file(shard, str(shard_path), metadata=meta)
            written_files.append(shard_path)

        index_file = out_p / SAFE_WEIGHTS_INDEX_NAME
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        written_files.append(index_file)

    manifest = {
        "format": "safetensors",
        "num_shards": num_shards,
        "total_tensors": len(clean_dict),
        "total_parameters": sum(t.numel() for t in clean_dict.values()),
        "total_size_bytes": index_data["metadata"]["total_size"],
        "files": [str(p.name) for p in written_files],
        "weight_map": index_data["weight_map"],
    }
    manifest_path = out_p / "safetensors_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    written_files.append(manifest_path)

    return {
        "output_dir": str(out_p),
        "num_shards": num_shards,
        "files": [str(p) for p in written_files],
        "index_file": str(index_file) if index_file else None,
        "manifest_file": str(manifest_path),
    }


class SafeTensorsExporter:
    """Unified SafeTensors and Surgery Manifest Exporter with Sharding Support."""

    def __init__(self, output_dir: Union[str, Path] = "results/artifacts") -> None:
        """Initialize SafeTensorsExporter with destination directory.

        Args:
            output_dir: Target directory path for exported files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_model(
        self,
        model: nn.Module,
        filename_prefix: str = "model",
        manifest_meta: Optional[Dict[str, Any]] = None,
        max_shard_size: Optional[Union[int, str]] = None,
    ) -> Dict[str, Any]:
        """Export PyTorch model weights to SafeTensors and JSON manifest.

        Args:
            model: PyTorch model module.
            filename_prefix: Filename prefix for single-file or sharded weights.
            manifest_meta: Optional metadata to embed in JSON manifest.
            max_shard_size: Maximum shard size in bytes or size string. If provided,
                sharded export is executed when total size exceeds this threshold.

        Returns:
            Dictionary with paths to exported weights and manifest.

        Raises:
            RuntimeError: If safetensors library is not installed.
        """
        if max_shard_size is not None:
            res = save_sharded_safetensors(
                model_or_state_dict=model,
                output_dir=self.output_dir,
                max_shard_size=max_shard_size,
            )
            return res

        state_dict = model.state_dict()
        tensor_dict = {k: v.detach().cpu().clone().contiguous() for k, v in state_dict.items()}

        st_path = self.output_dir / f"{filename_prefix}.safetensors"
        manifest_path = self.output_dir / f"{filename_prefix}_manifest.json"

        try:
            from safetensors.torch import save_file

            save_file(tensor_dict, str(st_path))
        except ImportError as exc:
            raise RuntimeError(
                f"SafeTensors export requested for '{filename_prefix}' but 'safetensors' package "
                f"is not installed. Underlying error: {exc}"
            ) from exc

        manifest = {
            "filename_prefix": filename_prefix,
            "num_tensors": len(tensor_dict),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "tensors": {k: list(v.shape) for k, v in tensor_dict.items()},
            "meta": manifest_meta or {},
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return {"weights": str(st_path), "manifest": str(manifest_path)}


__all__ = [
    "DEFAULT_MAX_SHARD_SIZE",
    "SAFE_WEIGHTS_INDEX_NAME",
    "SAFE_WEIGHTS_NAME",
    "SafeTensorsExporter",
    "get_tensor_size_bytes",
    "save_sharded_safetensors",
    "shard_state_dict",
]
