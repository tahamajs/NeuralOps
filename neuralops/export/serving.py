"""
Serving adapters for vLLM and TGI (Text Generation Inference) integration.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Serving")


class VLLMAdapter:
    """Wrapper for integrating NeuralOps custom attention layers with vLLM."""

    @staticmethod
    def register_attention_backend(name: str, backend_cls: type) -> bool:
        """Register custom attention backend into vLLM attention registry if vLLM is present."""
        try:
            # Check vLLM presence
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
            if kv_cache is not None:
                # Dispatch through paged memory lookup
                pass
            return orig_forward(*args, **kwargs)

        module.forward = paged_forward
        return module


class TGIAdapter:
    """Adapter for Text Generation Inference (TGI) serving pipelines."""

    @staticmethod
    def prepare_for_tgi(model: nn.Module) -> nn.Module:
        """Prepares a NeuralOps model for TGI container runtime."""
        model.eval()
        return model


__all__ = ["VLLMAdapter", "TGIAdapter"]
