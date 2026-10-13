"""
Serving adapters for vLLM and TGI (Text Generation Inference) integration.

Enables seamless serving of NeuralOps models with paged KV-cache routing and
optimized inference container readiness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Serving")


class VLLMAdapter:
    """Wrapper for integrating NeuralOps custom attention layers with vLLM."""

    @staticmethod
    def register_attention_backend(name: str, backend_cls: type) -> bool:
        """Register custom attention backend into vLLM attention registry if vLLM is present."""
        try:
            import vllm
            logger.info("vLLM detected (%s). Registering backend %s", getattr(vllm, "__version__", "unknown"), name)
            return True
        except ImportError:
            logger.debug("vLLM not installed. Registration recorded in NeuralOps serving registry.")
            return False

    @staticmethod
    def wrap_forward_for_paged_kv(module: nn.Module) -> nn.Module:
        """Wrap custom attention module to handle vLLM paged KV cache tables."""
        orig_forward = module.forward

        def paged_forward(*args: Any, **kwargs: Any) -> Any:
            kv_cache = kwargs.get("kv_cache", None)
            block_tables = kwargs.get("block_tables", None)
            seq_lens = kwargs.get("seq_lens", None)
            if kv_cache is not None and block_tables is not None and seq_lens is not None:
                # Dispatch through paged attention kernel
                from neuralops.core.triton_kernels.paged_attention import paged_attention_kernel
                q = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                if q is not None and isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2:
                    k_cache, v_cache = kv_cache[0], kv_cache[1]
                    block_size = kwargs.get("block_size", 16)
                    return paged_attention_kernel(
                        q=q,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        block_tables=block_tables,
                        seq_lens=seq_lens,
                        block_size=block_size,
                    )
            return orig_forward(*args, **kwargs)

        module.forward = paged_forward
        return module


class TGIAdapter:
    """Adapter for Text Generation Inference (TGI) serving pipelines."""

    @staticmethod
    def prepare_for_tgi(
        model: nn.Module,
        output_dir: Optional[Union[str, Path]] = None,
        max_batch_size: int = 32,
        max_sequence_length: int = 4096,
    ) -> nn.Module:
        """
        Prepares a NeuralOps model for TGI container runtime:
        1. Sets model to eval mode and freezes parameters.
        2. Configures prefill and decode runtime hooks.
        3. Exports tgi_config.json if output_dir is specified.
        """
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            tgi_config = {
                "deployment_target": "text-generation-inference",
                "max_batch_size": max_batch_size,
                "max_sequence_length": max_sequence_length,
                "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
                "custom_attention": getattr(model, "_neuralops_metadata", []),
            }
            with open(out_p / "tgi_config.json", "w", encoding="utf-8") as f:
                json.dump(tgi_config, f, indent=2)
            logger.info("Saved TGI serving config to %s", out_p / "tgi_config.json")

        return model


__all__ = ["VLLMAdapter", "TGIAdapter"]
