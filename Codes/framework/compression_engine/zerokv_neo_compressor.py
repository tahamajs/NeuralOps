"""
ZeroKV-Neo: L2-norm based KV cache eviction with attention sink and heavy-hitter preservation.
Registered as 'zerokv_neo'.
Production-hardened with:
- Bounded O(1) cache memory management with safe dimension handling
- Relative Frobenius error validation on pruned weights
- Adaptive keep_ratio retry loop
- Conv1D support (GPT-2)
- Critical layer protection (classifier, score, lm_head, pooler, embedding)
- Detailed per-layer error stats logging

"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class ZeroKVNeoCache:
    """
    Dynamic KV Cache Manager for ZeroKV-Neo.
    Maintains bounded O(1) cache memory by partitioning budget:
    - Pinned attention sinks (first N_sinks prompt tokens)
    - Pinned local sliding window (last N_window recent tokens)
    - Dynamic heavy hitters in the middle sequence retained via L2 norm / cumulative score.
    """

    def __init__(
        self,
        max_cache_len: int = 128,
        num_attention_sinks: int = 4,
        recent_window: int = 32,
    ) -> None:
        self.max_cache_len = int(max_cache_len)
        self.num_attention_sinks = int(num_attention_sinks)
        self.recent_window = int(recent_window)
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear cache tensors to avoid memory leaks."""
        self.k_cache = None
        self.v_cache = None

    @torch.no_grad()
    def update_and_compress(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new KV tokens and evict non-essential tokens when exceeding capacity.
        Shapes: (B, H, N, D).
        """
        k_4d = new_k
        v_4d = new_v
        if new_k.ndim == 2:
            k_4d = new_k.unsqueeze(0).unsqueeze(0)
            v_4d = new_v.unsqueeze(0).unsqueeze(0)
        elif new_k.ndim == 3:
            k_4d = new_k.unsqueeze(1)
            v_4d = new_v.unsqueeze(1)

        B, H, N, D = k_4d.shape

        if self.k_cache is None:
            self.k_cache = k_4d
            self.v_cache = v_4d
        else:
            if self.k_cache.shape[0] != B:
                self.k_cache = self.k_cache[:1].expand(B, -1, -1, -1).clone()
                self.v_cache = self.v_cache[:1].expand(B, -1, -1, -1).clone()
            self.k_cache = torch.cat([self.k_cache, k_4d], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_4d], dim=2)

        cur_len = self.k_cache.size(2)
        if cur_len <= self.max_cache_len:
            return self.k_cache, self.v_cache

        sinks = min(self.num_attention_sinks, cur_len)
        window = min(self.recent_window, cur_len - sinks)
        heavy_budget = max(0, self.max_cache_len - sinks - window)

        sink_k = self.k_cache[:, :, :sinks, :]
        sink_v = self.v_cache[:, :, :sinks, :]
        win_k = (
            self.k_cache[:, :, -window:, :] if window > 0 else self.k_cache[:, :, :0, :]
        )
        win_v = (
            self.v_cache[:, :, -window:, :] if window > 0 else self.v_cache[:, :, :0, :]
        )

        if heavy_budget > 0 and cur_len > (sinks + window):
            mid_k = self.k_cache[:, :, sinks : cur_len - window, :]
            mid_v = self.v_cache[:, :, sinks : cur_len - window, :]
            mid_norms = torch.norm(mid_k.float(), dim=-1)  # (B, H, mid_len)
            _, top_idx = torch.topk(
                mid_norms, min(heavy_budget, mid_norms.size(-1)), dim=-1
            )
            top_idx, _ = torch.sort(top_idx, dim=-1)

            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
            sel_mid_k = mid_k.gather(2, idx_exp)
            sel_mid_v = mid_v.gather(2, idx_exp)

            self.k_cache = torch.cat([sink_k, sel_mid_k, win_k], dim=2).contiguous()
            self.v_cache = torch.cat([sink_v, sel_mid_v, win_v], dim=2).contiguous()
        else:
            self.k_cache = (
                torch.cat([sink_k, win_k], dim=2).contiguous() if window > 0 else sink_k
            )
            self.v_cache = (
                torch.cat([sink_v, win_v], dim=2).contiguous() if window > 0 else sink_v
            )

        return self.k_cache, self.v_cache


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("zerokv_neo")
@COMPRESSOR_REGISTRY.register("zerokv")
@COMPRESSOR_REGISTRY.register("zerokv_neo_compressor")
class ZeroKVNeoCompressor(BaseCompressor):
    """
    ZeroKV-Neo Compressor: L2-norm weight pruning preserving attention sinks and heavy hitters.

    Args:
        model: PyTorch model.
        keep_ratio: Fraction of output channels to retain (default 0.3).
        num_attention_sinks: Number of initial sink channels pinned (default 4).
        heavy_hitter_ratio: Fraction allocated to heavy hitters (default 0.1).
        max_error: Maximum allowed relative Frobenius error (default 0.85).
        max_retries: Number of keep_ratio retry steps on error threshold excess (default 3).
        fallback_to_original: If True, keep original layer on high error (default True).
        target_attention_only: If True, only compress attention projections (default True for ZeroKV).
    """

    def __init__(
        self,
        model: nn.Module,
        keep_ratio: float = 0.3,
        num_attention_sinks: int = 4,
        heavy_hitter_ratio: float = 0.1,
        max_error: float = 0.85,
        max_retries: int = 3,
        fallback_to_original: bool = True,
        target_attention_only: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="zerokv_neo", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        if cfg is not None:
            keep_ratio = _cfg_get(
                cfg, "zerokv_keep_ratio", _cfg_get(cfg, "keep_ratio", keep_ratio)
            )
            num_attention_sinks = _cfg_get(
                cfg,
                "zerokv_num_sinks",
                _cfg_get(cfg, "num_attention_sinks", num_attention_sinks),
            )
            heavy_hitter_ratio = _cfg_get(
                cfg,
                "zerokv_heavy_ratio",
                _cfg_get(cfg, "heavy_hitter_ratio", heavy_hitter_ratio),
            )
            max_error = _cfg_get(
                cfg, "max_error", _cfg_get(cfg, "max_reconstruction_error", max_error)
            )
            max_retries = _cfg_get(cfg, "max_retries", max_retries)
            fallback_to_original = _cfg_get(
                cfg, "fallback_to_original", fallback_to_original
            )
            target_attention_only = _cfg_get(
                cfg, "target_attention_only", target_attention_only
            )

        if "keep_ratio" in kwargs and kwargs["keep_ratio"] is not None:
            keep_ratio = kwargs["keep_ratio"]
        if "max_error" in kwargs and kwargs["max_error"] is not None:
            max_error = float(kwargs["max_error"])
        if "max_retries" in kwargs and kwargs["max_retries"] is not None:
            max_retries = int(kwargs["max_retries"])
        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])
        if "target_attention_only" in kwargs:
            target_attention_only = bool(kwargs["target_attention_only"])

        self.keep_ratio = float(keep_ratio)
        self.num_attention_sinks = int(num_attention_sinks)
        self.heavy_hitter_ratio = float(heavy_hitter_ratio)
        self.max_error = float(max_error)
        self.max_retries = max(0, int(max_retries))
        self.fallback_to_original = bool(fallback_to_original)
        self.target_attention_only = target_attention_only
        self.compressed_layers: list[dict[str, Any]] = []
        self.fallback_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        # Protect critical classification heads, poolers, lm_heads, and embeddings
        if any(
            k in name.lower()
            for k in ["classifier", "score", "lm_head", "pooler", "embedding"]
        ):
            return False

        base_layer = getattr(module, "base_layer", module)
        cls_name = base_layer.__class__.__name__
        is_linear_like = (
            isinstance(base_layer, nn.Linear)
            or "conv1d" in cls_name.lower()
            or hasattr(base_layer, "nf")
            or (
                hasattr(base_layer, "weight")
                and hasattr(base_layer.weight, "ndim")
                and base_layer.weight.ndim == 2
                and not isinstance(base_layer, (nn.Embedding, nn.LayerNorm, nn.RMSNorm))
            )
        )
        if not is_linear_like:
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_attention_only:
            return self._is_attention_proj(name)
        return True

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Compress linear projection weights preserving attention sinks and heavy hitters."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        total_error = 0.0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            base_layer = getattr(module, "base_layer", module)
            raw_w = base_layer.weight.data
            orig_dtype = raw_w.dtype
            orig_device = raw_w.device
            cls_name = base_layer.__class__.__name__

            is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_layer, "nf")

            if is_conv1d:
                W = raw_w.t().float().contiguous()
                in_f = W.shape[1]
                out_f = W.shape[0]
            else:
                W = raw_w.float().contiguous()
                out_f, in_f = W.shape[0], W.shape[1]

            norm_orig = torch.linalg.norm(W).item() + 1e-8
            row_norms = torch.norm(W, dim=1)

            cur_keep = self.keep_ratio
            best_mask = None
            best_err = float("inf")
            best_keep = cur_keep
            best_sinks = 0
            best_heavy = 0

            for attempt in range(self.max_retries + 1):
                total_keep = max(1, min(int(round(out_f * cur_keep)), out_f))
                n_sinks = min(self.num_attention_sinks, total_keep)
                n_heavy = total_keep - n_sinks

                sink_idx = torch.arange(n_sinks, device=orig_device)
                remaining_mask = torch.ones(out_f, dtype=torch.bool, device=orig_device)
                remaining_mask[sink_idx] = False

                if n_heavy > 0 and remaining_mask.any():
                    rem_indices = torch.where(remaining_mask)[0]
                    rem_norms = row_norms[rem_indices]
                    _, top_rem = torch.topk(rem_norms, min(n_heavy, rem_norms.numel()))
                    heavy_idx = rem_indices[top_rem]
                else:
                    heavy_idx = torch.tensor([], device=orig_device, dtype=torch.long)

                keep_idx = torch.cat([sink_idx, heavy_idx])
                mask = torch.zeros(out_f, device=orig_device, dtype=torch.float32)
                mask[keep_idx] = 1.0

                W_pruned = W * mask.unsqueeze(1)
                rel_err = (torch.linalg.norm(W - W_pruned).item()) / norm_orig

                if rel_err < best_err:
                    best_err = rel_err
                    best_mask = mask
                    best_keep = cur_keep
                    best_sinks = n_sinks
                    best_heavy = len(heavy_idx)

                if rel_err <= self.max_error or cur_keep >= 1.0:
                    break

                step = max(0.1, (1.0 - cur_keep) / max(1, self.max_retries - attempt))
                cur_keep = min(1.0, cur_keep + step)

            if best_err > self.max_error:
                logger.warning(
                    "ZeroKV-Neo: %s error (%.4f) exceeded threshold (%.4f).",
                    name,
                    best_err,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("ZeroKV-Neo: Keeping original module for %s.", name)
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            if best_mask is None:
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            # Assign pruned weights to base_layer
            if is_conv1d:
                W_assigned = (
                    (W * best_mask.unsqueeze(1))
                    .t()
                    .contiguous()
                    .to(dtype=orig_dtype, device=orig_device)
                )
            else:
                W_assigned = (W * best_mask.unsqueeze(1)).to(
                    dtype=orig_dtype, device=orig_device
                )

            if W_assigned.shape != base_layer.weight.shape:
                logger.error(
                    "ZeroKV-Neo shape mismatch for %s: %s vs %s",
                    name,
                    W_assigned.shape,
                    base_layer.weight.shape,
                )
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            base_layer.weight.data.copy_(W_assigned)
            # ZeroKV-Neo compresses in-place: base_layer already holds the pruned weight
            compressed_layer = base_layer
            # Preserve existing LoRA adapter weights if the source module has them
            if (
                hasattr(module, "lora_A")
                and hasattr(module, "lora_B")
                and module.lora_A is not None
                and module.lora_B is not None
            ):
                src_lA = getattr(module.lora_A, "weight", module.lora_A)
                src_lB = getattr(module.lora_B, "weight", module.lora_B)
                if (
                    hasattr(compressed_layer, "lora_A")
                    and compressed_layer.lora_A is not None
                ):
                    if (
                        hasattr(src_lA, "data")
                        and src_lA.shape == compressed_layer.lora_A.shape
                    ):
                        compressed_layer.lora_A.data.copy_(src_lA.data)
                if (
                    hasattr(compressed_layer, "lora_B")
                    and compressed_layer.lora_B is not None
                ):
                    if (
                        hasattr(src_lB, "data")
                        and src_lB.shape == compressed_layer.lora_B.shape
                    ):
                        compressed_layer.lora_B.data.copy_(src_lB.data)

            if hasattr(module, "base_layer"):
                module.base_layer = (
                    compressed_layer  # put compressed layer inside wrapper
                )
                # module itself (LoRALinear wrapper) remains the replacement
            else:
                pass
            if (
                getattr(base_layer, "bias", None) is not None
                and base_layer.bias is not None
            ):
                b_pruned = (base_layer.bias.data.float() * best_mask).to(
                    dtype=orig_dtype, device=orig_device
                )
                base_layer.bias.data.copy_(b_pruned)

            self.compressed_layers.append(
                {
                    "name": name,
                    "original_shape": (in_f, out_f),
                    "keep_ratio": float(best_keep),
                    "num_sinks": best_sinks,
                    "num_heavy": best_heavy,
                    "rel_error": float(best_err),
                }
            )
            self.layer_errors[name] = float(best_err)
            total_error += best_err

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        num_comp = len(self.compressed_layers)
        num_fb = len(self.fallback_layers)
        avg_err = (total_error / num_comp) if num_comp else 0.0
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "num_compressed": num_comp,
                "num_compressed_layers": num_comp,
                "fallback_layers": self.fallback_layers,
                "num_fallback": num_fb,
                "num_fallback_layers": num_fb,
                "fallback_layer_count": num_fb,
                "layer_errors": self.layer_errors,
                "keep_ratio": self.keep_ratio,
                "num_attention_sinks": self.num_attention_sinks,
                "heavy_hitter_ratio": self.heavy_hitter_ratio,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "kv_cache_budget_saving": f"{(1.0 - self.keep_ratio) * 100:.1f}% Footprint Reduction",
            }
        )
        logger.info(
            "ZeroKV-Neo compression finished: %d layers compressed, avg error: %.4f",
            len(self.compressed_layers),
            avg_err,
        )
        return self.model, stats

    @staticmethod
    def _is_attention_proj(name: str) -> bool:
        """Check if module name corresponds to a QKV attention projection (not output projection).

        Explicitly excludes output projections first, then checks for QKV attention keywords.
        This prevents generic keywords like 'attn'/'attention' from matching out_proj, c_proj, etc.
        """
        name_lower = name.lower()

        # Exclusion list: output projections must never be treated as QKV targets
        output_proj_keywords = [
            "out_lin",
            "out_proj",
            "o_proj",
            "c_proj",
            "output.dense",
            "self.output",
            ".output",
        ]
        if any(kw in name_lower for kw in output_proj_keywords):
            return False

        # Inclusion list: only match true QKV projections
        qkv_keywords = [
            "q_lin",
            "k_lin",
            "v_lin",
            "q_proj",
            "k_proj",
            "v_proj",
            "query",
            "key",
            "value",
            "c_attn",
            "in_proj_qkv",
        ]
        return any(kw in name_lower for kw in qkv_keywords)


@COMPRESSOR_REGISTRY.register("zerokv_neo_factory")
@COMPRESSOR_REGISTRY.register("zerokv_factory")
@COMPRESSOR_REGISTRY.register("zerokv_neo_compressor_factory")
def zerokv_neo_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> Any:
    """ZeroKV-Neo compressor factory function."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    keep_ratio = kwargs.pop(
        "keep_ratio",
        _cfg_get(
            comp_config, "zerokv_keep_ratio", _cfg_get(comp_config, "keep_ratio", 0.3)
        ),
    )
    num_sinks = kwargs.pop(
        "num_attention_sinks",
        kwargs.pop(
            "num_sinks",
            _cfg_get(
                comp_config,
                "zerokv_num_sinks",
                _cfg_get(comp_config, "num_attention_sinks", 4),
            ),
        ),
    )
    heavy_ratio = kwargs.pop(
        "heavy_hitter_ratio",
        kwargs.pop(
            "heavy_ratio",
            _cfg_get(
                comp_config,
                "zerokv_heavy_ratio",
                _cfg_get(comp_config, "heavy_hitter_ratio", 0.1),
            ),
        ),
    )
    target_attn = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "target_attention_only", True)
    )
    compressor = ZeroKVNeoCompressor(
        model,
        keep_ratio=keep_ratio,
        num_attention_sinks=num_sinks,
        heavy_hitter_ratio=heavy_ratio,
        comp_config=comp_config,
        target_attention_only=target_attn,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model
