#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Serving Artifact Exporter

Prepares model artifacts, configurations, and tokenizer files for production inference servers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Serving")

try:
    import vllm
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    import text_generation_server
    HAS_TGI = True
except ImportError:
    HAS_TGI = False


class VLLMAdapter:
    """Wrapper for integrating NeuralOps custom attention layers with vLLM."""

    @staticmethod
    def is_available() -> bool:
        """Check if vLLM package is available."""
        return HAS_VLLM

    @staticmethod
    def register_attention_backend(name: str, backend_cls: type) -> bool:
        """Register custom attention backend into vLLM attention registry."""
        if not HAS_VLLM:
            raise RuntimeError(
                "vLLM is not installed. Please install neuralops[serving] or run: pip install vllm"
            )
        get_name = getattr(backend_cls, "get_name", None)
        if not callable(get_name) or get_name() != name:
            raise ValueError("backend_cls.get_name() must return the requested registry name")
        try:
            from vllm.attention.backends.abstract import AttentionBackend
            from vllm.attention.backends.registry import register_backend
        except ImportError as exc:
            raise RuntimeError(
                "This vLLM installation does not expose the 0.6.x attention backend registry API."
            ) from exc
        if not isinstance(backend_cls, type) or not issubclass(backend_cls, AttentionBackend):
            raise TypeError("backend_cls must subclass vLLM's AttentionBackend")
        register_backend(backend_cls)
        logger.info(
            "Registered vLLM attention backend %s with vLLM %s",
            name,
            getattr(vllm, "__version__", "unknown"),
        )
        return True

    @staticmethod
    def paged_attention_forward(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int = 16,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Execute paged attention forward pass over block_tables and KV-cache blocks.

        Args:
            query: Query tensor [num_seqs, num_heads, head_dim] or [num_seqs, 1, num_heads, head_dim].
            key_cache: Key block memory pool [num_blocks, block_size, num_heads, head_dim].
            value_cache: Value block memory pool [num_blocks, block_size, num_heads, head_dim].
            block_tables: Block table mapping logical blocks to physical blocks [num_seqs, max_blocks].
            seq_lens: Context sequence lengths [num_seqs].
            block_size: Paged memory block tile size (default: 16).
            scale: Softmax scale factor (default: 1 / sqrt(head_dim)).
        """
        if not HAS_VLLM:
            logger.debug("vLLM not installed; using NeuralOps native paged_attention_kernel fallback.")

        if query.dim() == 4 and query.shape[1] == 1:
            q = query.squeeze(1)
        elif query.dim() == 3:
            q = query
        else:
            raise ValueError("query must have shape [batch, heads, head_dim] or [batch, 1, heads, head_dim]")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")

        from neuralops.core.triton_kernels.paged_attention import paged_attention_kernel

        return paged_attention_kernel(
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
        )

    @staticmethod
    def wrap_forward_for_paged_kv(module: nn.Module) -> nn.Module:
        """Reject generic wrapping because hidden states are not projected Q/K/V tensors."""
        raise NotImplementedError(
            "Generic paged-KV wrapping is unsafe: this adapter cannot infer projected Q/K/V "
            "tensors or vLLM's cache-write lifecycle from an arbitrary module forward signature."
        )


class TGIAdapter:
    """Adapter for Text Generation Inference (TGI) serving pipelines."""

    @staticmethod
    def is_available() -> bool:
        """Check if TGI server environment is available."""
        return HAS_TGI

    @staticmethod
    def prepare_batch(batch_inputs: List[Any], max_batch_size: int = 32) -> List[Any]:
        """Truncates or batches inputs to match TGI runtime batch boundaries."""
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        return batch_inputs[:max_batch_size]

    @staticmethod
    def prepare_for_tgi(
        model: nn.Module,
        output_dir: Optional[Union[str, Path]] = None,
        max_batch_size: int = 32,
        max_sequence_length: int = 4096,
    ) -> nn.Module:
        """Set inference mode and optionally write TGI deployment metadata.

        This helper does not implement TGI runtime hooks or certify that the
        exported model is loadable by a particular TGI version.
        """
        for name, value in (("max_batch_size", max_batch_size), ("max_sequence_length", max_sequence_length)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            dtype_str = "float32"
            try:
                dtype_str = str(next(model.parameters()).dtype).replace("torch.", "")
            except StopIteration:
                pass

            tgi_config = {
                "deployment_target": "text-generation-inference",
                "max_batch_size": max_batch_size,
                "max_sequence_length": max_sequence_length,
                "dtype": dtype_str,
                "custom_attention": getattr(model, "_neuralops_metadata", []),
            }
            with open(out_p / "tgi_config.json", "w", encoding="utf-8") as f:
                json.dump(tgi_config, f, indent=2)
            logger.info("Saved TGI serving config to %s", out_p / "tgi_config.json")

        return model


def export_serving_bundle(
    model: nn.Module,
    output_dir: Union[str, Path],
    target_backend: str = "vllm",
    max_batch_size: int = 32,
    max_sequence_length: int = 4096,
    quantization: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """
    Export model weights and engine-specific configuration metadata.

    The generated directory is a serialization bundle, not proof that the target
    serving engine can load or execute the model.
    """
    backend = target_backend.lower().strip()
    backend_aliases = {
        "vllm": "vllm",
        "tgi": "tgi",
        "sglang": "sglang",
        "tensorrt_llm": "tensorrt_llm",
        "trt_llm": "tensorrt_llm",
        "trt": "tensorrt_llm",
    }
    if backend not in backend_aliases:
        supported = ", ".join(sorted(backend_aliases))
        raise ValueError(f"Unsupported serving backend {target_backend!r}. Choose one of: {supported}.")
    backend = backend_aliases[backend]
    for name, value in (("max_batch_size", max_batch_size), ("max_sequence_length", max_sequence_length)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if quantization is not None:
        raise NotImplementedError(
            "export_serving_bundle serializes weights and metadata but does not quantize them; "
            "quantize the model with a supported backend before exporting."
        )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file
        save_file(model.state_dict(), str(out_p / "model.safetensors"))
    except (ImportError, RuntimeError, OSError) as e:
        logger.debug("Safetensors export skipped, using pytorch bin: %s", e)
        torch.save(model.state_dict(), out_p / "pytorch_model.bin")

    param_count = sum(p.numel() for p in model.parameters())
    dtype_str = "float32"
    try:
        dtype_str = str(next(model.parameters()).dtype).replace("torch.", "")
    except StopIteration:
        pass

    manifest = {
        "format_version": "1.4.2",
        "target_backend": backend,
        "max_batch_size": max_batch_size,
        "max_sequence_length": max_sequence_length,
        "dtype": dtype_str,
        "num_parameters": param_count,
        "quantization": quantization,
        "custom_attention": getattr(model, "_neuralops_metadata", []),
        "extra_kwargs": kwargs,
    }
    with open(out_p / "serving_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if backend == "vllm":
        vllm_cfg = {
            "model": str(out_p),
            "max_num_seqs": max_batch_size,
            "max_model_len": max_sequence_length,
            "dtype": dtype_str,
            "kv_cache_dtype": "auto",
        }
        with open(out_p / "vllm_config.json", "w", encoding="utf-8") as f:
            json.dump(vllm_cfg, f, indent=2)

    elif backend == "tgi":
        TGIAdapter.prepare_for_tgi(
            model=model,
            output_dir=out_p,
            max_batch_size=max_batch_size,
            max_sequence_length=max_sequence_length,
        )

    elif backend == "sglang":
        sglang_cfg = {
            "model_path": str(out_p),
            "max_total_tokens": max_sequence_length * max_batch_size,
            "dtype": dtype_str,
            "enable_radix_cache": True,
        }
        with open(out_p / "sglang_config.json", "w", encoding="utf-8") as f:
            json.dump(sglang_cfg, f, indent=2)

    elif backend in ("tensorrt_llm", "trt_llm", "trt"):
        trt_cfg = {
            "backend": "tensorrt_llm",
            "max_batch_size": max_batch_size,
            "max_input_len": max_sequence_length,
            "max_output_len": max_sequence_length,
            "dtype": dtype_str,
        }
        with open(out_p / "tensorrt_llm_config.json", "w", encoding="utf-8") as f:
            json.dump(trt_cfg, f, indent=2)

    logger.info("Serving bundle exported for %s to %s", backend, out_p)
    return out_p


def validate_serving_metadata(bundle_dir: Union[str, Path]) -> tuple[bool, str]:
    """Check bundle structure without claiming target-engine compatibility."""
    p = Path(bundle_dir)
    if not p.is_dir():
        return False, f"Directory does not exist: {p}"

    manifest_file = p / "serving_manifest.json"
    if not manifest_file.exists():
        return False, f"Missing serving_manifest.json in {p}"

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, f"Invalid serving_manifest.json: {e}"

    if not manifest.get("target_backend"):
        return False, "serving_manifest.json missing 'target_backend'"

    has_weights = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
    if not has_weights:
        return False, "Missing weights (model.safetensors or pytorch_model.bin)"

    return True, "Structurally valid serialization bundle; target-engine loading is unverified"


__all__ = [
    "VLLMAdapter",
    "TGIAdapter",
    "HAS_VLLM",
    "HAS_TGI",
    "export_serving_bundle",
    "validate_serving_metadata",
]
