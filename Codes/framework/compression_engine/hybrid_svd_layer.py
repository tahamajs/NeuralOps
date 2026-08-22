"""
Hybrid SVD Layer: single-head SVD attention replacement with positional encoding,
plus full encoder layer and compression engine.
Registered as 'hybrid_svd' and 'direct_svd_encoder'.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class SVDSingleHead(nn.Module):
    """Single-head SVD attention approximation with sinusoidal positional encoding."""

    def __init__(self, hidden_size: int, svd_rank: int = 20) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.svd_rank = svd_rank
        self.register_buffer("pos_encoding", None, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with SVD spectral projection."""
        batch_size, seq_len, _ = hidden_states.shape

        # Sinusoidal positional encoding
        if self.pos_encoding is None or self.pos_encoding.shape[1] != seq_len:
            pe = torch.zeros(seq_len, self.hidden_size, device=hidden_states.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=hidden_states.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.hidden_size, 2, dtype=torch.float, device=hidden_states.device)
                * (-math.log(10000.0) / self.hidden_size)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pos_encoding = pe.unsqueeze(0)

        x = hidden_states + self.pos_encoding[:, :seq_len, :]
        x_flat = F.normalize(x.reshape(-1, self.hidden_size).float(), p=2, dim=-1)

        try:
            U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)
            k = min(self.svd_rank, len(S))
            top_s = S[:k]
            top_Vh = Vh[:k, :]
            # Softmax-weighted singular energy distribution
            s_max = top_s.max()
            exp_s = torch.exp(top_s - s_max)
            softmax_s = exp_s / (exp_s.sum() + 1e-12)
            weighted_U = U[:, :k] * softmax_s.view(1, -1)
            output_flat = torch.matmul(weighted_U, top_Vh)
            output = output_flat.reshape(batch_size, seq_len, self.hidden_size).to(dtype=hidden_states.dtype)
            return output
        except Exception:
            return x


class DirectSVDEncoderLayer(nn.Module):
    """Encoder layer that replaces multi-head attention with SVDSingleHead + standard FFN."""

    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        svd_rank: int = 20,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = (
            hidden_size
            or getattr(config, "hidden_size", None)
            or 768
        )
        self.intermediate_size = (
            intermediate_size
            or getattr(config, "intermediate_size", None)
            or (4 * self.hidden_size)
        )
        self.num_attention_heads = (
            num_attention_heads
            or getattr(config, "num_attention_heads", 12)
        )
        self.svd_rank = (
            getattr(config, "svd_rank", None)
            or svd_rank
        )

        eps = getattr(config, "layer_norm_eps", layer_norm_eps)
        drop = getattr(config, "hidden_dropout_prob", hidden_dropout_prob)

        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=eps)
        self.svd_layer = SVDSingleHead(hidden_size=self.hidden_size, svd_rank=self.svd_rank)

        # FFN
        self.intermediate = nn.Linear(self.hidden_size, self.intermediate_size)
        self.output_dense = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(drop)
        self.output_layernorm = nn.LayerNorm(self.hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        svd_out = self.svd_layer(normed)
        hidden_states = residual + svd_out

        residual = hidden_states
        normed_ffn = self.output_layernorm(hidden_states)
        intermediate_out = self.intermediate(normed_ffn)
        intermediate_out = F.gelu(intermediate_out)
        ff_out = self.output_dense(intermediate_out)
        ff_out = self.dropout(ff_out)
        hidden_states = residual + ff_out

        if return_dict:
            return (hidden_states,)
        return hidden_states


@COMPRESSOR_REGISTRY.register("hybrid_svd")
@COMPRESSOR_REGISTRY.register("direct_svd_encoder")
class HybridSVDCompressor(BaseCompressor):
    """
    Hybrid SVD Compressor.

    Replaces transformer multi-head self-attention mechanisms with SVDSingleHead
    spectral projection modules and DirectSVDEncoderLayer blocks.

    Args:
        model: PyTorch model.
        svd_rank: SVD rank for single-head projection.
        rank: Optional alias for svd_rank.
        target_modules: Optional list of layer name filters.
    """

    def __init__(
        self,
        model: nn.Module,
        svd_rank: int = 20,
        rank: Optional[int] = None,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="hybrid_svd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            svd_rank = getattr(
                cfg, "hybrid_svd_rank", getattr(cfg, "svd_rank", getattr(cfg, "rank", svd_rank))
            )

        if rank is not None:
            svd_rank = rank
        if "rank" in kwargs and kwargs["rank"] is not None:
            svd_rank = kwargs["rank"]

        self.svd_rank = int(svd_rank)
        self.target_modules = target_modules

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Find and replace transformer layers dynamically."""
        target_class_names = [
            "BertLayer",
            "DistilBertLayer",
            "RobertaLayer",
            "AlbertLayer",
            "TransformerEncoderLayer",
        ]

        parent_map = {}
        for name, module in self.model.named_modules():
            for child_name, child in module.named_children():
                parent_map[child] = (module, child_name, name)

        replaced_count = 0
        for module, (parent, child_name, parent_path) in list(parent_map.items()):
            full_name = f"{parent_path}.{child_name}" if parent_path else child_name
            if self.target_modules is not None and not any(t in full_name for t in self.target_modules):
                continue

            if module.__class__.__name__ in target_class_names or (
                hasattr(module, "attention") and hasattr(module, "intermediate")
            ):
                cfg = getattr(module, "config", None)
                new_layer = DirectSVDEncoderLayer(config=cfg, svd_rank=self.svd_rank)
                setattr(parent, child_name, new_layer)
                replaced_count += 1

        if replaced_count == 0:
            # Fallback: if no full encoder layer was found, compress linear layers
            for name, module in list(self.model.named_modules()):
                if isinstance(module, nn.Linear) and not isinstance(module, SVDLinear):
                    out_f, in_f = module.out_features, module.in_features
                    r = min(self.svd_rank, out_f, in_f)
                    W = module.weight.data.float()
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                    U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=module.weight.dtype, device=module.weight.device)
                    V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=module.weight.dtype, device=module.weight.device)
                    svd_layer = SVDLinear(in_f, out_f, r, bias=module.bias is not None, dtype=module.weight.dtype, device=module.weight.device)
                    svd_layer.set_low_rank_factors(U_r, V_r)
                    if module.bias is not None:
                        svd_layer.bias.data.copy_(module.bias.data)
                    replace_module(self.model, name, svd_layer)
                    replaced_count += 1

        stats = self.get_compression_stats()
        stats["replaced_layers_count"] = replaced_count
        return self.model, stats


def hybrid_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Hybrid SVD factory function."""
    compressor = HybridSVDCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


direct_svd_encoder_factory = hybrid_svd_factory
hybrid_svd_compressor_factory = hybrid_svd_factory

__all__ = [
    "SVDSingleHead",
    "DirectSVDEncoderLayer",
    "HybridSVDCompressor",
    "hybrid_svd_factory",
    "direct_svd_encoder_factory",
    "hybrid_svd_compressor_factory",
]
