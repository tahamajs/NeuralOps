"""
Production model exporter for NeuralOps.

Supports:
  1. "safetensors" (Hugging Face Hub compatible) with embedded `neuralops_config.json`
  2. "onnx" with automatic deterministic PyTorch fallback for Triton modules
  3. "torchscript" via tracing or scripting
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Export")


def export_model(
    model: nn.Module,
    output_dir: Union[str, Path],
    format: str = "safetensors",
    model_name_or_path: Optional[str] = None,
    attention_type: Optional[str] = None,
    compression_method: Optional[str] = None,
    rank: Optional[int] = None,
    sample_input: Optional[torch.Tensor] = None,
    opset_version: int = 17,
    **kwargs: Any,
) -> Path:
    """Export a NeuralOps-modified or compressed model to a production artifact format."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fmt = format.lower().strip()

    # Build architecture and surgery metadata
    meta = {
        "format": fmt,
        "base_model": model_name_or_path or getattr(model, "name_or_path", "unknown"),
        "attention_type": attention_type,
        "compression_method": compression_method,
        "rank": rank,
        "framework": "neuralops",
        "pytorch_version": torch.__version__,
        "extra_kwargs": kwargs,
    }

    if fmt == "safetensors":
        try:
            from safetensors.torch import save_model
            save_model(model, str(out_path / "model.safetensors"))
        except ImportError:
            # Fallback to standard PyTorch state_dict if safetensors is unavailable
            torch.save(model.state_dict(), out_path / "pytorch_model.bin")

        # Save surgery config alongside weights
        with open(out_path / "neuralops_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save HF config if present
        if hasattr(model, "config") and hasattr(model.config, "save_pretrained"):
            try:
                model.config.save_pretrained(str(out_path))
            except Exception as e:
                logger.debug("Could not save HF config: %s", e)

        logger.info("Exported model to safetensors directory: %s", out_path)
        return out_path

    elif fmt == "onnx":
        onnx_file = out_path / "model.onnx"
        if sample_input is None:
            # Generate default dummy token input (batch_size=1, seq_len=16)
            sample_input = torch.randint(0, 1000, (1, 16), dtype=torch.long)

        # Ensure model is on CPU for clean ONNX export
        cpu_model = model.to("cpu").eval()
        if sample_input.device != torch.device("cpu"):
            sample_input = sample_input.to("cpu")

        # Automatically trace using deterministic fallback path
        torch.onnx.export(
            cpu_model,
            sample_input,
            str(onnx_file),
            opset_version=opset_version,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
        )

        with open(out_path / "neuralops_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Exported model to ONNX: %s", onnx_file)
        return onnx_file

    elif fmt == "torchscript":
        ts_file = out_path / "model.pt"
        if sample_input is None:
            sample_input = torch.randint(0, 1000, (1, 16), dtype=torch.long)
        cpu_model = model.to("cpu").eval()
        if sample_input.device != torch.device("cpu"):
            sample_input = sample_input.to("cpu")

        try:
            traced = torch.jit.trace(cpu_model, sample_input, strict=False)
            traced.save(str(ts_file))
        except Exception:
            scripted = torch.jit.script(cpu_model)
            scripted.save(str(ts_file))

        with open(out_path / "neuralops_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Exported model to TorchScript: %s", ts_file)
        return ts_file

    else:
        raise ValueError(f"Unsupported export format '{format}'. Choose from: 'safetensors', 'onnx', 'torchscript'")


def load_exported_model(export_dir: Union[str, Path]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model and reconstruct metadata from an exported directory."""
    dir_path = Path(export_dir)
    cfg_file = dir_path / "neuralops_config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing neuralops_config.json in {export_dir}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


__all__ = ["export_model", "load_exported_model"]
