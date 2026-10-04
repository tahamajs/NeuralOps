"""
Base classes for attention modules and compressors.
"""

import logging
from abc import abstractmethod
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseAttention(nn.Module):
    """Every custom attention module must inherit from this."""

    def __init__(self, config: Any | None = None, rank: int | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.config = config
        self.rank = rank
        self.num_heads = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or (
                getattr(
                    config,
                    "num_attention_heads",
                    getattr(config, "num_heads", getattr(config, "n_heads", 12)),
                )
                if config is not None
                else 12
            )
        )
        self.hidden_size = (
            kwargs.get("hidden_size")
            or kwargs.get("dim")
            or kwargs.get("d_model")
            or (
                getattr(
                    config,
                    "hidden_size",
                    getattr(config, "d_model", getattr(config, "dim", 768)),
                )
                if config is not None
                else 768
            )
        )
        self.head_dim = getattr(
            config,
            "d_kv",
            getattr(
                config,
                "head_dim",
                self.hidden_size // self.num_heads if self.num_heads > 0 else 64,
            ),
        )

    @abstractmethod
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs,
    ) -> tuple[Any, Any | None]:
        """Must return (output, attention_weights_or_None)."""
        raise NotImplementedError("Subclasses of BaseAttention must implement forward()")

    def copy_weights(self, old_attn: Any) -> None:
        """Universal fallback: copy pretrained weights from old attention layer."""
        if old_attn is None:
            return
        import torch

        from neuralops.core.adapters import UniversalAdapter

        adapter = UniversalAdapter()

        try:
            w_q, b_q = adapter.get_weight_bias(old_attn, "q")
            w_k, b_k = adapter.get_weight_bias(old_attn, "k")
            w_v, b_v = adapter.get_weight_bias(old_attn, "v")
            w_o, b_o = adapter.get_weight_bias(old_attn, "o")

            with torch.no_grad():
                dim = self.hidden_size
                for q_name in ["W_q", "q_proj", "q_lin", "Wq", "query", "q"]:
                    if hasattr(self, q_name):
                        mod = getattr(self, q_name)
                        if hasattr(mod, "weight") and mod.weight.shape == w_q.shape:
                            mod.weight.copy_(w_q)
                            if b_q is not None and getattr(mod, "bias", None) is not None:
                                mod.bias.copy_(b_q)

                for k_name in ["W_k", "k_proj", "k_lin", "Wk", "key", "k"]:
                    if hasattr(self, k_name):
                        mod = getattr(self, k_name)
                        if hasattr(mod, "weight") and mod.weight.shape == w_k.shape:
                            mod.weight.copy_(w_k)
                            if b_k is not None and getattr(mod, "bias", None) is not None:
                                mod.bias.copy_(b_k)

                for v_name in ["W_v", "v_proj", "v_lin", "Wv", "value", "v"]:
                    if hasattr(self, v_name):
                        mod = getattr(self, v_name)
                        if hasattr(mod, "weight") and mod.weight.shape == w_v.shape:
                            mod.weight.copy_(w_v)
                            if b_v is not None and getattr(mod, "bias", None) is not None:
                                mod.bias.copy_(b_v)

                for o_name in ["out", "out_proj", "o_proj", "out_lin", "proj", "dense"]:
                    if hasattr(self, o_name):
                        mod = getattr(self, o_name)
                        if hasattr(mod, "weight") and mod.weight.shape == w_o.shape:
                            mod.weight.copy_(w_o)
                            if b_o is not None and getattr(mod, "bias", None) is not None:
                                mod.bias.copy_(b_o)

                # Handle fused in_proj
                if hasattr(self, "in_proj") and hasattr(self.in_proj, "weight"):
                    if self.in_proj.weight.shape == torch.Size([dim * 2, dim]):
                        self.in_proj.weight[:dim].copy_(w_v[:dim])
                        self.in_proj.weight[dim:].copy_(w_q[:dim])
                    elif self.in_proj.weight.shape == torch.Size([dim * 3, dim]):
                        self.in_proj.weight[:dim].copy_(w_q[:dim])
                        self.in_proj.weight[dim : 2 * dim].copy_(w_k[:dim])
                        self.in_proj.weight[2 * dim :].copy_(w_v[:dim])
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.debug("BaseAttention.copy_weights failed: %s", e)
        except Exception as e:
            logger.warning("Unexpected error in BaseAttention.copy_weights: %s", e)

    def extra_repr(self) -> str:
        """Extra repr."""
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, rank={self.rank}"


class BaseCompressor:
    """Base class for all compressors. Not a nn.Module to avoid parameter storage."""

    def __init__(self, model: nn.Module, name: str = "base", **kwargs) -> None:
        self.model = model
        self.name = name
        self.kwargs = kwargs
        self._original_params = self._count_params(model)

    def __call__(self, *args, **kwargs) -> Any:
        """Allow compressor instance to act as a callable model forward wrapper."""
        if not getattr(self, "_compressed_once", False):
            self.compress()
            self._compressed_once = True
        return self.model(*args, **kwargs)

    @abstractmethod
    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Return (compressed_model, statistics)."""
        raise NotImplementedError("Subclasses of BaseCompressor must implement compress()")

    def _count_params(self, model: nn.Module) -> int:
        """count params."""
        return sum(p.numel() for p in model.parameters())

    def get_compression_stats(self) -> dict[str, Any]:
        """Return the compression stats."""
        compressed = self._count_params(self.model)
        orig = self._original_params
        return {
            "compressor_name": self.name,
            "original_parameters": orig,
            "compressed_parameters": compressed,
            "saved_parameters": orig - compressed,
            "compression_ratio": compressed / orig if orig > 0 else 1.0,
            "reduction_percentage": ((orig - compressed) / orig * 100 if orig > 0 else 0.0),
        }

    def summary(self) -> str:
        """Summary."""
        s = self.get_compression_stats()
        return (
            f"\n========================================\n"
            f"Compression Summary: {self.name}\n"
            f"========================================\n"
            f"Original parameters:  {s['original_parameters']:,}\n"
            f"Compressed parameters: {s['compressed_parameters']:,}\n"
            f"Parameters saved:     {s['saved_parameters']:,}\n"
            f"Compression ratio:    {s['compression_ratio']:.3f}\n"
            f"Reduction:            {s['reduction_percentage']:.2f}%\n"
            f"========================================\n"
        )
