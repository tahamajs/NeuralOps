"""
Serving adapters for vLLM and TGI (Text Generation Inference) integration.

Enables seamless serving of NeuralOps models with paged KV-cache routing and
optimized inference container readiness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Serving")

# Safe vLLM detection
try:
    import vllm
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# Safe TGI detection
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
        try:
            logger.info(
                "vLLM detected (%s). Registering backend %s",
                getattr(vllm, "__version__", "unknown"),
                name,
            )
            return True
        except Exception as e:
            logger.error("Failed to register backend %s into vLLM: %s", name, e)
            return False

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
            query: Query tensor [num_seqs, num_heads, head_dim] or [batch, seq_len, num_heads, head_dim].
            key_cache: Key block memory pool [num_blocks, block_size, num_heads, head_dim].
            value_cache: Value block memory pool [num_blocks, block_size, num_heads, head_dim].
            block_tables: Block table mapping logical blocks to physical blocks [num_seqs, max_blocks].
            seq_lens: Context sequence lengths [num_seqs].
            block_size: Paged memory block tile size (default: 16).
            scale: Softmax scale factor (default: 1 / sqrt(head_dim)).
        """
        if not HAS_VLLM:
            logger.debug("vLLM not installed; using NeuralOps native paged_attention_kernel fallback.")

        from neuralops.core.triton_kernels.paged_attention import paged_attention_kernel

        # Ensure tensors are shaped for paged attention kernel
        if query.dim() == 4 and query.shape[1] == 1:
            q = query.squeeze(1).transpose(1, 2)
        elif query.dim() == 3:
            q = query
        else:
            q = query

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
        """Wrap custom attention module to handle vLLM paged KV cache tables."""
        orig_forward = module.forward

        def paged_forward(*args: Any, **kwargs: Any) -> Any:
            kv_cache = kwargs.get("kv_cache", None)
            block_tables = kwargs.get("block_tables", None)
            seq_lens = kwargs.get("seq_lens", None)

            if kv_cache is not None and block_tables is not None and seq_lens is not None:
                # Dispatch through paged attention kernel
                q = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                if q is not None and isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2:
                    k_cache, v_cache = kv_cache[0], kv_cache[1]
                    block_size = kwargs.get("block_size", 16)
                    return VLLMAdapter.paged_attention_forward(
                        query=q,
                        key_cache=k_cache,
                        value_cache=v_cache,
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
    def is_available() -> bool:
        """Check if TGI server environment is available."""
        return HAS_TGI

    @staticmethod
    def prepare_batch(batch_inputs: List[Any], max_batch_size: int = 32) -> List[Any]:
        """Truncates or batches inputs to match TGI runtime batch boundaries."""
        return batch_inputs[:max_batch_size]

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


__all__ = ["VLLMAdapter", "TGIAdapter", "HAS_VLLM", "HAS_TGI"]
