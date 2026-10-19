# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Compressed Tensors Exporter

Exports compressed weights compatible with vLLM, HuggingFace, and downstream serving engines.
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

    def __init__(self) -> None:
        pass

    def export(self, compressor_or_model: Any, path: str) -> dict[str, str]:
        """
        Export model weights and quantization configs.
        Produces:
          - model.safetensors
          - quantization_config.json
          - config.json
        Compatible with `vllm.LLM(model=path)`
        """
        os.makedirs(path, exist_ok=True)
        artifacts: dict[str, str] = {}

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

        serialized_tensors = {k: v.contiguous() for k, v in tensors.items()}

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

        quant_config: dict[str, Any] = {}
        if hasattr(compressor_or_model, "get_quantization_config"):
            quant_config = compressor_or_model.get_quantization_config()
        elif hasattr(compressor_or_model, "quantization_config"):
            quant_config = getattr(compressor_or_model, "quantization_config")
        else:
            quant_config = {
                "quant_method": "compressed-tensors",
                "format": "W4A16",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "channel",
                        },
                        "targets": ["Linear"],
                    }
                },
            }

        quant_config_path = os.path.join(path, "quantization_config.json")
        with open(quant_config_path, "w", encoding="utf-8") as f:
            json.dump(quant_config, f, indent=2)
        artifacts["quantization_config.json"] = quant_config_path

        hf_config: dict[str, Any] = {}
        model_obj = getattr(compressor_or_model, "model", compressor_or_model)
        if hasattr(model_obj, "config") and model_obj.config is not None:
            c = model_obj.config
            if hasattr(c, "to_dict"):
                hf_config = c.to_dict()
            elif isinstance(c, dict):
                hf_config = dict(c)
            else:
                hf_config = {k: getattr(c, k) for k in dir(c) if not k.startswith("_") and not callable(getattr(c, k))}
        else:
            hf_config = {
                "architectures": ["NeuralOpsForCausalLM"],
                "model_type": "neuralops",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "vocab_size": 50257,
                "torch_dtype": "float16",
            }

        hf_config["quantization_config"] = quant_config

        config_path = os.path.join(path, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(hf_config, f, indent=2)
        artifacts["config.json"] = config_path

        logger.info("Successfully exported compressed-tensors artifacts to %s", path)
        return artifacts
