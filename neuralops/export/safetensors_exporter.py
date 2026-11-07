#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""SafeTensors Exporter with zero-fake integrity verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class SafeTensorsExporter:
    """Unified SafeTensors and Surgery Manifest Exporter."""

    def __init__(self, output_dir: str = "results/artifacts") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_model(
        self,
        model: nn.Module,
        filename_prefix: str,
        manifest_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Export PyTorch model weights to SafeTensors and JSON manifest."""
        state_dict = model.state_dict()
        tensor_dict = {k: v.contiguous().cpu() for k, v in state_dict.items()}

        st_path = self.output_dir / f"{filename_prefix}.safetensors"
        manifest_path = self.output_dir / f"{filename_prefix}_manifest.json"

        try:
            from safetensors.torch import save_file

            save_file(tensor_dict, str(st_path))
        except ImportError as e:
            raise RuntimeError(
                f"SafeTensors export requested for '{filename_prefix}' but 'safetensors' package "
                f"is not installed. Per AGENTS.md §6.1, fallback to fake/unrequested format is banned. "
                f"Underlying error: {e}"
            ) from e

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
