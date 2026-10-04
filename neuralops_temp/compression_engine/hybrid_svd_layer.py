"""
Hybrid SVD Layer: weight-based SVD compression for attention/linear layers.

Registered under:
  - 'hybrid_svd'
  - 'hybrid_svd_layer'
  - 'direct_svd_encoder'

The deprecated SVDSingleHead (which performed SVD on live activations) is
retained only for backward compatibility and emits a warning on use.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.compression_engine.aa_svd_compressor import SVDLinear
from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "c_attn",
    "c_proj",
    "query",
    "key",
    "value",
    "q_lin",
    "k_lin",
    "v_lin",
}


# ===========================================================================
# Deprecated single-head SVD attention (backward compat only)
# ===========================================================================


class SVDSingleHead(nn.Module):
    """
    DEPRECATED single-head SVD attention.

    Performing SVD on live activations is O(N^2 d), numerically fragile, and
    theoretically incorrect as an attention replacement. Use the weight-based
    HybridSVDCompressor instead.

    Kept strictly for backwards compatibility. First use emits a warning.
    """

    is_custom_attention: bool = True
    _warned: bool = False

    def __init__(self, hidden_size: int, svd_rank: int = 20) -> None:
        super().__init__()
        if not SVDSingleHead._warned:
            warnings.warn(
                "SVDSingleHead is deprecated (SVD on live activations is "
                "computationally prohibitive and incorrect). Use "
                "HybridSVDCompressor for weight-based SVD instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            SVDSingleHead._warned = True
        self.hidden_size = hidden_size
        self.svd_rank = int(svd_rank)
        self._pos_encoding: torch.Tensor | None = None

    def _get_pos_encoding(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if (
            self._pos_encoding is None
            or self._pos_encoding.shape[1] != seq_len
            or self._pos_encoding.device != device
        ):
            pe = torch.zeros(seq_len, self.hidden_size, device=device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.hidden_size, 2, dtype=torch.float, device=device)
                * (-math.log(10000.0) / self.hidden_size)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self._pos_encoding = pe.unsqueeze(0).to(dtype)
        return self._pos_encoding

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        pe = self._get_pos_encoding(L, hidden_states.device, hidden_states.dtype)
        x = hidden_states + pe[:, :L, :]

        x_flat = F.normalize(x.reshape(-1, self.hidden_size).float(), p=2, dim=-1)
        try:
            U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)
            k = min(self.svd_rank, len(S))
            weighted_U = U[:, :k] * S[:k].view(1, -1)
            output_flat = torch.matmul(weighted_U, Vh[:k, :])
            return output_flat.reshape(B, L, self.hidden_size).to(hidden_states.dtype)
        except Exception:
            return x


# ===========================================================================
# DirectSVDEncoderLayer (unchanged; kept for backward compat)
# ===========================================================================


class DirectSVDEncoderLayer(nn.Module):
    """Encoder layer that replaces MHA with SVDSingleHead + standard FFN."""

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        num_attention_heads: int | None = None,
        svd_rank: int = 20,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(hidden_size or getattr(config, "hidden_size", 768))
        self.intermediate_size = int(
            intermediate_size
            or getattr(config, "intermediate_size", None)
            or (4 * self.hidden_size)
        )
        self.num_attention_heads = int(
            num_attention_heads or getattr(config, "num_attention_heads", 12)
        )
        self.svd_rank = int(getattr(config, "svd_rank", None) or svd_rank)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)

        self.svd_attention = SVDSingleHead(self.hidden_size, svd_rank=self.svd_rank)
        self.dense_attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm_attn = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.dropout_attn = nn.Dropout(hidden_dropout_prob)

        self.dense_intermediate = nn.Linear(self.hidden_size, self.intermediate_size)
        self.intermediate_act = nn.GELU()
        self.dense_output = nn.Linear(self.intermediate_size, self.hidden_size)
        self.LayerNorm_output = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.dropout_output = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        attn_out = self.svd_attention(hidden_states)
        attn_proj = self.dropout_attn(self.dense_attn(attn_out))
        h = self.LayerNorm_attn(hidden_states + attn_proj)

        inter = self.intermediate_act(self.dense_intermediate(h))
        inter = self.dropout_output(self.dense_output(inter))
        output = self.LayerNorm_output(h + inter)

        if output_attentions:
            return (output, None)
        return output


# ===========================================================================
# Main Compressor
# ===========================================================================


@COMPRESSOR_REGISTRY.register("hybrid_svd")
@COMPRESSOR_REGISTRY.register("hybrid_svd_layer")
@COMPRESSOR_REGISTRY.register("direct_svd_encoder")
class HybridSVDCompressor(BaseCompressor):
    """
    Weight-based SVD compressor for Linear / Conv1D layers.
    """

    def __init__(
        self,
        model: nn.Module,
        svd_rank: int = 128,
        use_full_layer_replacement: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        max_error: float = 0.85,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        cfg: Any | None = None,
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="hybrid_svd", **kwargs)
        cfg = cfg or kwargs.get("comp_config")
        self.cfg = cfg

        self.svd_rank = int(
            _cfg_get(
                cfg,
                "hybrid_svd_rank",
                _cfg_get(cfg, "svd_rank", _cfg_get(cfg, "rank", svd_rank)),
            )
        )
        self.use_full_layer_replacement = bool(
            _cfg_get(cfg, "use_full_layer_replacement", use_full_layer_replacement)
        )
        self.lora_r = int(_cfg_get(cfg, "lora_r", _cfg_get(cfg, "lora_rank", lora_r)))
        self.lora_alpha = float(_cfg_get(cfg, "lora_alpha", lora_alpha))
        self.max_error = float(
            _cfg_get(cfg, "hybrid_svd_max_error", _cfg_get(cfg, "max_error", max_error))
        )
        self.target_attention_only = bool(
            _cfg_get(
                cfg,
                "hybrid_svd_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        self.fallback_to_original = bool(
            _cfg_get(
                cfg,
                "hybrid_svd_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
        )
        self.target_modules = target_modules or _cfg_get(cfg, "target_modules", None)

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(
            skip in name_lower
            for skip in [
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embedding",
                "router",
                "gate",
            ]
        ):
            return False

        if self.target_attention_only:
            if not any(attn_kw in name_lower for attn_kw in _ATTN_NAMES):
                return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d) or isinstance(target_mod, SVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    # ------------------------------------------------------------------
    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Decompose linear layers via SVD and replace with SVDLinear."""
        replaced_count = 0
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set = set()

        total_orig_params = 0
        total_comp_params = 0

        # Freeze non-target layers if attention-only.
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
                    continue
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if (is_linear or is_conv1d) and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        # Fallback: linear-layer SVD.
        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            target_mod = getattr(module, "base_layer", module)
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = target_mod.weight.data
            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            W = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()
            out_f, in_f = W.shape
            orig_params = in_f * out_f
            total_orig_params += orig_params

            max_rank = min(in_f, out_f)
            bound_r = (in_f * out_f) // (in_f + out_f)
            if max_rank < 16 or bound_r < 4:
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_params
                continue

            r = min(self.svd_rank, bound_r, max_rank)

            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                for escalation_step in range(4):
                    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                    U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(
                        dtype=raw_w.dtype, device=raw_w.device
                    )
                    V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(
                        dtype=raw_w.dtype, device=raw_w.device
                    )

                    W_recon = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vt[:r, :])
                    err = (
                        torch.norm(W - W_recon, p="fro") / torch.norm(W, p="fro").clamp_min(1e-8)
                    ).item()
                    self.layer_errors[name] = float(err)

                    if err > self.max_error and r < min(bound_r, max_rank):
                        r_new = min(int(r * 1.5), bound_r, max_rank)
                        if r_new > r:
                            logger.info(
                                "HybridSVD: %s err=%.4f at rank=%d, escalating to rank=%d",
                                name,
                                err,
                                r,
                                r_new,
                            )
                            r = r_new
                            continue

                    if err > self.max_error:
                        logger.warning(
                            "HybridSVD: layer %s reconstruction error %.4f > max_error %.4f (final rank=%d)",
                            name,
                            err,
                            self.max_error,
                            r,
                        )
                        if self.fallback_to_original:
                            self.fallback_layers.append(name)
                            for p in module.parameters():
                                p.requires_grad = False
                            total_comp_params += orig_params
                            break
                    break

                if name in self.fallback_layers:
                    continue

                # Detect existing LoRA rank.
                existing_lora_A = getattr(module, "lora_A", None)
                existing_lora_B = getattr(module, "lora_B", None)
                effective_lora_r = self.lora_r
                if (
                    existing_lora_A is not None
                    and hasattr(existing_lora_A, "weight")
                    and hasattr(existing_lora_A.weight, "shape")
                ):
                    try:
                        effective_lora_r = int(existing_lora_A.weight.shape[0])
                    except (AttributeError, IndexError, TypeError):
                        pass

                svd_layer = SVDLinear(
                    in_f,
                    out_f,
                    r,
                    bias=(getattr(target_mod, "bias", None) is not None),
                    dtype=raw_w.dtype,
                    device=raw_w.device,
                    lora_r=effective_lora_r,
                    lora_alpha=self.lora_alpha,
                )
                svd_layer.set_low_rank_factors(U_r, V_r)
                if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None:
                    svd_layer.bias.data.copy_(target_mod.bias.data)

                if (
                    existing_lora_A is not None
                    and existing_lora_B is not None
                    and getattr(svd_layer, "lora_A", None) is not None
                ):
                    w_A = getattr(existing_lora_A, "weight", existing_lora_A)
                    w_B = getattr(existing_lora_B, "weight", existing_lora_B)
                    if hasattr(w_A, "data") and hasattr(w_B, "data"):
                        try:
                            svd_layer.lora_A.data.copy_(w_A.data)
                            svd_layer.lora_B.data.copy_(w_B.data)
                            svd_layer.lora_A.requires_grad = True
                            svd_layer.lora_B.requires_grad = True
                        except Exception as e:
                            logger.debug("LoRA copy failed for %s: %s", name, e)

                replace_module(self.model, name, svd_layer)
                replaced_count += 1
                self.compressed_layers.append(name)

                # FIX: use actual svd_layer.rank for parameter accounting.
                actual_rank = int(getattr(svd_layer, "rank", r))
                comp_params = actual_rank * (in_f + out_f)
                if getattr(svd_layer, "lora_r", 0) > 0:
                    comp_params += int(svd_layer.lora_r) * (in_f + out_f)
                total_comp_params += comp_params

            except Exception as e:
                logger.warning("HybridSVD compression exception on %s: %s", name, e)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_params

        # Post-compression forward sanity check (single attempt).
        forward_check_success = True
        try:
            self.model.eval()
            with torch.no_grad():
                first_p = next(self.model.parameters(), None)
                dev = first_p.device if first_p is not None else torch.device("cpu")
                cfg = getattr(self.model, "config", None)
                vocab_size = getattr(cfg, "vocab_size", 1000)
                max_tok = min(100, vocab_size - 1) if vocab_size > 10 else 2
                min_tok = min(2, max_tok)
                dummy = torch.randint(min_tok, max_tok + 1, (1, 8), device=dev, dtype=torch.long)
                attn_mask = torch.ones_like(dummy)
                try:
                    self.model(input_ids=dummy, attention_mask=attn_mask)
                except Exception:
                    # Try hidden-states form.
                    first_linear = next(
                        (m for m in self.model.modules() if isinstance(m, (nn.Linear, SVDLinear))),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", getattr(cfg, "hidden_size", 8))
                    dummy_float = torch.randn(
                        1,
                        8,
                        in_dim,
                        device=dev,
                        dtype=first_p.dtype if first_p is not None else torch.float32,
                    )
                    self.model(dummy_float)
        except Exception as e:
            logger.warning("HybridSVD post-compression forward check warning: %s", e)
            forward_check_success = False

        effective_ratio = (total_comp_params / total_orig_params) if total_orig_params > 0 else 1.0

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        stats = self.get_compression_stats()
        avg_err = (
            (sum(self.layer_errors.values()) / max(1, len(self.layer_errors)))
            if self.layer_errors
            else 0.0
        )
        stats.update(
            {
                "replaced_layers_count": replaced_count,
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "skipped_layers": self.skipped_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "svd_rank": self.svd_rank,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "max_error": self.max_error,
                "target_attention_only": self.target_attention_only,
                "fallback_to_original": self.fallback_to_original,
                "avg_reconstruction_error": avg_err,
                "effective_parameter_ratio": round(effective_ratio, 4),
                "forward_check_success": forward_check_success,
                "method": "hybrid_svd",
            }
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================


@COMPRESSOR_REGISTRY.register("hybrid_svd_factory")
@COMPRESSOR_REGISTRY.register("hybrid_svd_layer_factory")
@COMPRESSOR_REGISTRY.register("direct_svd_encoder_factory")
@COMPRESSOR_REGISTRY.register("hybrid_svd_compressor_factory")
def hybrid_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Hybrid SVD factory function."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    svd_rank = kwargs.pop(
        "svd_rank",
        _cfg_get(
            comp_config,
            "hybrid_svd_rank",
            _cfg_get(comp_config, "svd_rank", _cfg_get(comp_config, "rank", 128)),
        ),
    )
    use_full = kwargs.pop(
        "use_full_layer_replacement",
        _cfg_get(comp_config, "use_full_layer_replacement", False),
    )
    lora_r = kwargs.pop(
        "lora_r",
        _cfg_get(comp_config, "lora_r", _cfg_get(comp_config, "lora_rank", 16)),
    )
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config,
            "hybrid_svd_max_error",
            _cfg_get(comp_config, "max_error", 0.85),
        ),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "hybrid_svd_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", False),
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "hybrid_svd_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )

    compressor = HybridSVDCompressor(
        model,
        svd_rank=svd_rank,
        use_full_layer_replacement=use_full,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        cfg=comp_config,
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
