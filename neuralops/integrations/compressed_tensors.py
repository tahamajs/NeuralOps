#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Serialize quantized tensors and explicit model/quantization metadata.

The exporter does not quantize weights or guarantee that a downstream engine
can load the emitted artifacts.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.CompressedTensorsExporter")

try:
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class CompressedTensorsExporter:
    """Exports quantized neuralops compressors / models into compressed-tensors format for vLLM."""

    def export(
        self,
        compressor_or_model: Any,
        path: str,
        quantization_config: Optional[dict[str, Any]] = None,
        model_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, str]:
        """
        Serialize model weights and quantization configs.
        Produces:
          - model.safetensors
          - quantization_config.json
          - config.json
        Engine compatibility depends on the supplied model and quantization schemas.
        """
        tensors: dict[str, torch.Tensor] = {}
        if hasattr(compressor_or_model, "compress_to_tensors"):
            tensors = compressor_or_model.compress_to_tensors()
        elif hasattr(compressor_or_model, "model") and isinstance(compressor_or_model.model, nn.Module):
            for name, param in compressor_or_model.model.named_parameters():
                tensors[name] = param.data
            for name, buf in compressor_or_model.model.named_buffers():
                tensors[name] = buf.data
        elif isinstance(compressor_or_model, nn.Module):
            for name, param in compressor_or_model.named_parameters():
                tensors[name] = param.data
            for name, buf in compressor_or_model.named_buffers():
                tensors[name] = buf.data
        elif isinstance(compressor_or_model, dict):
            tensors = compressor_or_model
        else:
            raise TypeError(f"Unsupported compressor_or_model type: {type(compressor_or_model)}")

        if not isinstance(tensors, dict) or not tensors:
            raise ValueError("The exporter requires a non-empty tensor state dictionary.")
        if any(not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in tensors.items()):
            raise TypeError("Compressed tensor state must map string names to torch.Tensor values.")

        quant_config = quantization_config
        if quant_config is None and hasattr(compressor_or_model, "get_quantization_config"):
            quant_config = compressor_or_model.get_quantization_config()
        if quant_config is None:
            quant_config = getattr(compressor_or_model, "quantization_config", None)
        if not isinstance(quant_config, dict) or not quant_config:
            raise ValueError(
                "A real non-empty quantization_config is required; the exporter will not label "
                "ordinary dense weights as compressed tensors."
            )

        model_obj = getattr(compressor_or_model, "model", compressor_or_model)
        hf_config = model_config
        if hf_config is None and hasattr(model_obj, "config") and model_obj.config is not None:
            config = model_obj.config
            if hasattr(config, "to_dict"):
                hf_config = config.to_dict()
            elif isinstance(config, dict):
                hf_config = dict(config)
        if not isinstance(hf_config, dict) or not hf_config:
            raise ValueError(
                "A Hugging Face model config is required; pass model_config explicitly for state-dict inputs."
            )
        hf_config = dict(hf_config)

        os.makedirs(path, exist_ok=True)
        artifacts: dict[str, str] = {}
        serialized_tensors = {k: v.detach().contiguous() for k, v in tensors.items()}

        safetensors_path = os.path.join(path, "model.safetensors")
        if HAS_SAFETENSORS:
            try:
                clean_tensors = {k: v.detach().cpu() for k, v in serialized_tensors.items() if isinstance(v, torch.Tensor)}
                safetensors.torch.save_file(clean_tensors, safetensors_path)
                artifacts["model.safetensors"] = safetensors_path
            except Exception as e:
                logger.warning("safetensors save failed, falling back to torch.save: %s", e)
                pt_path = os.path.join(path, "model.pt")
                torch.save(serialized_tensors, pt_path)
                artifacts["model.pt"] = pt_path
        else:
            pt_path = os.path.join(path, "model.pt")
            torch.save(serialized_tensors, pt_path)
            artifacts["model.pt"] = pt_path

        quant_config_path = os.path.join(path, "quantization_config.json")
        with open(quant_config_path, "w", encoding="utf-8") as f:
            json.dump(quant_config, f, indent=2)
        artifacts["quantization_config.json"] = quant_config_path

        hf_config["quantization_config"] = quant_config

        config_path = os.path.join(path, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(hf_config, f, indent=2)
        artifacts["config.json"] = config_path

        logger.info("Successfully exported compressed-tensors artifacts to %s", path)
        return artifacts
