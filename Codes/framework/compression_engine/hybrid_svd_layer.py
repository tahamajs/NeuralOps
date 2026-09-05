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
            weighted_U = U[:, :k] * S[:k].view(1, -1)
            output_flat = torch.matmul(weighted_U, Vh[:k, :])
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
        lora_r: int = 0,
        lora_alpha: float = 1.0,
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
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (lora_alpha / lora_r) if lora_r > 0 else 1.0

        eps = getattr(config, "layer_norm_eps", layer_norm_eps)
        drop = getattr(config, "hidden_dropout_prob", hidden_dropout_prob)

        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=eps)
        self.svd_layer = SVDSingleHead(hidden_size=self.hidden_size, svd_rank=self.svd_rank)

        # FFN
        self.intermediate = nn.Linear(self.hidden_size, self.intermediate_size)
        self.output_dense = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(drop)
        self.output_layernorm = nn.LayerNorm(self.hidden_size, eps=eps)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty((self.lora_r, self.hidden_size), dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros((self.hidden_size, self.lora_r), dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            for p in [self.intermediate.weight, self.intermediate.bias, self.output_dense.weight, self.output_dense.bias]:
                if p is not None:
                    p.requires_grad = False
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
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

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            hidden_states = hidden_states + self.scaling * F.linear(
                F.linear(normed_ffn.float(), self.lora_A), self.lora_B
            ).to(hidden_states.dtype)

        if output_attentions:
            return (hidden_states, None)
        return hidden_states


@COMPRESSOR_REGISTRY.register("hybrid_svd")
@COMPRESSOR_REGISTRY.register("direct_svd_encoder")
class HybridSVDCompressor(BaseCompressor):
    """
    Hybrid SVD Compressor.

    Replaces transformer multi-head self-attention mechanisms with SVDSingleHead
    spectral projection modules and DirectSVDEncoderLayer blocks, or decomposes
    individual projection layers with SVDLinear.

    Args:
        model: PyTorch model.
        svd_rank: SVD rank for single-head projection / linear decomposition.
        rank: Optional alias for svd_rank.
        target_modules: Optional list of layer name filters.
        lora_r: LoRA rank for trainable adapters.
        lora_alpha: LoRA scaling alpha.
        skip_attention_qkv: Skip asymmetric GQA projections in fallback mode.
        max_error: Maximum allowed reconstruction error before fallback.
        fallback_to_original: If True, keep original layer on excess error.
    """

    def __init__(
        self,
        model: nn.Module,
        svd_rank: int = 20,
        rank: Optional[int] = None,
        target_modules: Optional[list[str]] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        max_error: float = 0.30,
        fallback_to_original: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="hybrid_svd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            svd_rank = getattr(
                cfg, "hybrid_svd_rank", getattr(cfg, "svd_rank", getattr(cfg, "rank", svd_rank))
            )
            lora_r = getattr(cfg, "lora_r", getattr(cfg, "lora_rank", lora_r))
            lora_alpha = getattr(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = getattr(cfg, "skip_attention_qkv", skip_attention_qkv)
            max_error = getattr(cfg, "max_error", max_error)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        if rank is not None:
            svd_rank = rank
        if "rank" in kwargs and kwargs["rank"] is not None:
            svd_rank = kwargs["rank"]

        self.svd_rank = int(svd_rank)
        self.target_modules = target_modules
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

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
        compressed_layers: list[str] = []
        fallback_layers: list[str] = []
        skipped_layers: list[str] = []
        layer_errors: dict[str, float] = {}

        for module, (parent, child_name, parent_path) in list(parent_map.items()):
            full_name = f"{parent_path}.{child_name}" if parent_path else child_name
            if self.target_modules is not None and not any(t in full_name for t in self.target_modules):
                continue

            if module.__class__.__name__ in target_class_names or (
                hasattr(module, "attention") and hasattr(module, "intermediate")
            ):
                cfg = getattr(module, "config", None) or getattr(self.model, "config", None)
                h_size = getattr(cfg, "hidden_size", None)
                if h_size is None and hasattr(module, "intermediate") and hasattr(module.intermediate, "dense"):
                    h_size = module.intermediate.dense.in_features
                new_layer = DirectSVDEncoderLayer(
                    config=cfg,
                    hidden_size=h_size,
                    svd_rank=self.svd_rank,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                orig_param = next(module.parameters(), None)
                if orig_param is not None:
                    new_layer.to(device=orig_param.device, dtype=orig_param.dtype)
                setattr(parent, child_name, new_layer)
                replaced_count += 1
                compressed_layers.append(full_name)

        if replaced_count == 0:
            # Fallback: if no full encoder layer was found, compress linear layers
            seen_weights: set[int] = set()
            from framework.utils.compat import should_compress_linear

            for name, module in list(self.model.named_modules()):
                cls_name = module.__class__.__name__.lower()
                is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
                if not is_linear_like or isinstance(module, SVDLinear):
                    continue

                if not should_compress_linear(self.model, name):
                    continue

                if self._is_gqa_asymmetric(name, module):
                    continue

                if not hasattr(module, "weight") or module.weight is None:
                    continue

                w_id = id(module.weight)
                if w_id in seen_weights:
                    skipped_layers.append(name)
                    continue
                seen_weights.add(w_id)

                raw_w = module.weight.data
                is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
                W = raw_w.t().float() if is_conv1d else raw_w.float()
                out_f, in_f = W.shape[0], W.shape[1]
                r = min(self.svd_rank, out_f, in_f)

                try:
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                    U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=raw_w.dtype, device=raw_w.device)
                    V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=raw_w.dtype, device=raw_w.device)

                    W_recon = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vt[:r, :])
                    err = (torch.norm(W - W_recon, p="fro") / torch.norm(W, p="fro").clamp_min(1e-8)).item()
                    layer_errors[name] = float(err)

                    if err > self.max_error:
                        logger.warning("HybridSVD: Layer %s error %.4f exceeded max_error %.4f", name, err, self.max_error)
                        if self.fallback_to_original:
                            fallback_layers.append(name)
                            continue

                    svd_layer = SVDLinear(
                        in_f,
                        out_f,
                        r,
                        bias=getattr(module, "bias", None) is not None and module.bias is not None,
                        dtype=raw_w.dtype,
                        device=raw_w.device,
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                    )
                    svd_layer.set_low_rank_factors(U_r, V_r)
                    if getattr(module, "bias", None) is not None and module.bias is not None:
                        svd_layer.bias.data.copy_(module.bias.data)
                    replace_module(self.model, name, svd_layer)
                    replaced_count += 1
                    compressed_layers.append(name)
                except Exception as e:
                    logger.warning("HybridSVD compression exception on %s: %s", name, e)
                    fallback_layers.append(name)

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    first_linear = next(
                        (m for m in self.model.modules() if isinstance(m, (nn.Linear, SVDLinear))),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception as e:
            logger.warning("HybridSVD post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "replaced_layers_count": replaced_count,
                "compressed_layers": compressed_layers,
                "fallback_layers": fallback_layers,
                "skipped_layers": skipped_layers,
                "layer_errors": layer_errors,
                "svd_rank": self.svd_rank,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "forward_check_success": forward_check_success,
            }
        )
        return self.model, stats


def hybrid_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Hybrid SVD factory function."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    svd_rank = kwargs.pop("svd_rank", getattr(comp_config, "hybrid_svd_rank", getattr(comp_config, "svd_rank", 20)) if comp_config else 20)
    lora_r = kwargs.pop("lora_r", getattr(comp_config, "lora_r", 0) if comp_config else 0)
    lora_alpha = kwargs.pop("lora_alpha", getattr(comp_config, "lora_alpha", 1.0) if comp_config else 1.0)
    max_error = kwargs.pop("max_error", getattr(comp_config, "max_error", 0.30) if comp_config else 0.30)
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "fallback_to_original", True) if comp_config else True)

    compressor = HybridSVDCompressor(
        model,
        svd_rank=svd_rank,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
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
