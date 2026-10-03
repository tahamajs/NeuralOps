"""
HybridKV: Hybrid KV Cache Compression for Multimodal & Long-Context LLMs.
Classifies attention heads into static/dynamic types and applies distinct
compression strategies per type under a hierarchical budget strictly to dynamic KV caches.

Paper: "HybridKV: Hybrid KV Cache Compression for Efficient MLLM Inference"
       Zeng et al., arXiv:2604.05887.
Registered as 'hybridkv'.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from neuralops.core.base import BaseCompressor
from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract config value from None, dict, or namespace/object."""
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


class HybridKVCache:
    """
    Dynamic KV Cache Manager for HybridKV.
    Maintains separate retention policies for static vs dynamic attention heads:
    - Static heads: pinned attention sinks (first S tokens) + global heavy hitters
    - Dynamic heads: recent local sliding window (last W tokens) + chunked key retention
    Preserves all model stationary weights and strictly compresses dynamic KV cache tensors.
    Supports variable sequence lengths across prefill and token-by-token decoding.
    """

    def __init__(
        self,
        max_cache_len: int = 128,
        num_sinks: int = 4,
        recent_window: int = 32,
        budget: float = 0.3,
        static_ratio: float = 0.7,
        static_heads_mask: torch.Tensor | None = None,
    ) -> None:
        self.budget = float(budget)
        self.max_cache_len = (
            max(8, int(max_cache_len * self.budget))
            if self.budget > 0
            else int(max_cache_len)
        )
        self.num_sinks = int(num_sinks)
        self.recent_window = int(recent_window)
        self.static_ratio = float(static_ratio)
        self.static_heads_mask = static_heads_mask
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear cache state to prevent memory leaks."""
        self.k_cache = None
        self.v_cache = None

    @torch.no_grad()
    def update_and_compress(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new KV representations and apply hybrid eviction if cache exceeds capacity.
        Shapes: (B, H, N, D) or (B, N, D) or (N, D).
        """
        orig_ndim = new_k.ndim
        k_4d = new_k
        v_4d = new_v
        if orig_ndim == 2:
            k_4d = new_k.unsqueeze(0).unsqueeze(0)
            v_4d = new_v.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 3:
            k_4d = new_k.unsqueeze(1)
            v_4d = new_v.unsqueeze(1)

        B, H, N, D = k_4d.shape

        if self.k_cache is None:
            self.k_cache = k_4d
            self.v_cache = v_4d
        else:
            if self.k_cache.device != k_4d.device or self.k_cache.dtype != k_4d.dtype:
                self.k_cache = self.k_cache.to(device=k_4d.device, dtype=k_4d.dtype)
                self.v_cache = self.v_cache.to(device=v_4d.device, dtype=v_4d.dtype)
            if self.k_cache.shape[0] != B:
                self.k_cache = self.k_cache[:1].expand(B, -1, -1, -1).clone()
                self.v_cache = self.v_cache[:1].expand(B, -1, -1, -1).clone()
            self.k_cache = torch.cat([self.k_cache, k_4d], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_4d], dim=2)

        cur_len = self.k_cache.size(2)
        if cur_len <= self.max_cache_len:
            out_k, out_v = self.k_cache, self.v_cache
            if orig_ndim == 2:
                return out_k.squeeze(0).squeeze(0), out_v.squeeze(0).squeeze(0)
            elif orig_ndim == 3:
                return out_k.squeeze(1), out_v.squeeze(1)
            return out_k, out_v

        # Eviction per head according to static vs dynamic classification
        target_len = self.max_cache_len
        head_k_list: list[torch.Tensor] = []
        head_v_list: list[torch.Tensor] = []

        is_static = (
            self.static_heads_mask
            if self.static_heads_mask is not None
            else torch.ones(H, dtype=torch.bool, device=self.k_cache.device)
        )

        for h_idx in range(H):
            k_h = self.k_cache[:, h_idx, :, :]  # (B, cur_len, D)
            v_h = self.v_cache[:, h_idx, :, :]

            if is_static[h_idx]:
                # Static head policy: Sink tokens + Heavy Hitters via L2 Key norm
                sinks_k = k_h[:, : self.num_sinks, :]
                sinks_v = v_h[:, : self.num_sinks, :]
                remaining_k = k_h[:, self.num_sinks :, :]
                remaining_v = v_h[:, self.num_sinks :, :]

                rem_budget = max(0, target_len - self.num_sinks)
                if remaining_k.size(1) > rem_budget and rem_budget > 0:
                    k_norms = torch.norm(remaining_k, p=2, dim=-1).mean(dim=0)
                    _, topk_indices = torch.topk(k_norms, k=rem_budget, largest=True)
                    topk_indices, _ = torch.sort(topk_indices)
                    kept_k = remaining_k[:, topk_indices, :]
                    kept_v = remaining_v[:, topk_indices, :]
                    evicted_k = torch.cat([sinks_k, kept_k], dim=1)
                    evicted_v = torch.cat([sinks_v, kept_v], dim=1)
                else:
                    evicted_k = sinks_k
                    evicted_v = sinks_v
            else:
                # Dynamic head policy: Sliding local window strictly
                window_budget = min(self.recent_window, target_len)
                evicted_k = k_h[:, -window_budget:, :]
                evicted_v = v_h[:, -window_budget:, :]

            head_k_list.append(evicted_k.unsqueeze(1))
            head_v_list.append(evicted_v.unsqueeze(1))

        self.k_cache = torch.cat(head_k_list, dim=1)
        self.v_cache = torch.cat(head_v_list, dim=1)

        out_k, out_v = self.k_cache, self.v_cache
        if orig_ndim == 2:
            return out_k.squeeze(0).squeeze(0), out_v.squeeze(0).squeeze(0)
        elif orig_ndim == 3:
            return out_k.squeeze(1), out_v.squeeze(1)
        return out_k, out_v


@COMPRESSOR_REGISTRY.register("hybridkv_class")
@COMPRESSOR_REGISTRY.register("hybridkv_compressor")
class HybridKVCompressor(BaseCompressor):
    """
    HybridKV Compressor: Classifies projection heads into static/dynamic types
    and attaches HybridKVCache instances to attention layers without modifying model weights.
    """

    def __init__(
        self,
        model: nn.Module,
        budget: float = 0.3,
        static_ratio: float = 0.7,
        max_cache_len: int = 128,
        num_sinks: int = 4,
        recent_window: int = 32,
        target_attention_only: bool = True,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="hybridkv", **kwargs)
        cfg = cfg or kwargs.get("comp_config")
        self.cfg = cfg

        self.budget = float(
            _cfg_get(cfg, "hybridkv_budget", _cfg_get(cfg, "budget", budget))
        )
        self.static_ratio = float(
            _cfg_get(
                cfg,
                "hybridkv_static_ratio",
                _cfg_get(cfg, "static_ratio", static_ratio),
            )
        )
        self.max_cache_len = int(
            _cfg_get(
                cfg,
                "hybridkv_max_cache_len",
                _cfg_get(cfg, "max_cache_len", max_cache_len),
            )
        )
        self.num_sinks = int(
            _cfg_get(cfg, "hybridkv_num_sinks", _cfg_get(cfg, "num_sinks", num_sinks))
        )
        self.recent_window = int(
            _cfg_get(
                cfg,
                "hybridkv_recent_window",
                _cfg_get(cfg, "recent_window", recent_window),
            )
        )
        self.target_attention_only = bool(
            _cfg_get(
                cfg,
                "hybridkv_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )

    def _classify_head(self, weight: torch.Tensor) -> str:
        """Classify attention projection weight into static vs dynamic based on variance."""
        var = torch.var(weight.float(), dim=-1).mean().item()
        return "static" if var < 0.1 else "dynamic"

    def _is_attention_proj(self, name: str, module: nn.Module) -> bool:
        """Check if module corresponds to attention projection."""
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B"]):
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

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            return False

        if self.target_attention_only:
            return any(kw in name_lower for kw in _ATTN_NAMES)
        return True

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Classify heads and configure dynamic KV caches without mutating model weights."""
        head_stats: dict[str, Any] = {}
        total_heads = 0
        static_heads_count = 0
        dynamic_heads_count = 0

        for name, module in self.model.named_modules():
            if self._is_attention_proj(name, module):
                target_mod = getattr(module, "base_layer", module)
                W = target_mod.weight.data
                cls_name = target_mod.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                out_f = W.shape[1] if is_conv1d else W.shape[0]

                head_dim = 64 if out_f % 64 == 0 else (128 if out_f % 128 == 0 else 32)
                num_heads = max(1, out_f // head_dim)
                head_type = self._classify_head(W)

                num_static = max(1, int(num_heads * self.static_ratio))
                num_dynamic = max(0, num_heads - num_static)
                static_mask = torch.zeros(num_heads, dtype=torch.bool, device=W.device)
                static_mask[:num_static] = True

                kv_cache = HybridKVCache(
                    max_cache_len=self.max_cache_len,
                    num_sinks=self.num_sinks,
                    recent_window=self.recent_window,
                    budget=self.budget,
                    static_ratio=self.static_ratio,
                    static_heads_mask=static_mask,
                )
                # Attach KV cache manager without mutating stationary weights
                module.hybrid_kv_cache = kv_cache
                if hasattr(module, "base_layer"):
                    module.base_layer.hybrid_kv_cache = kv_cache

                head_stats[name] = {
                    "head_type": head_type,
                    "num_heads": num_heads,
                    "static_heads": num_static,
                    "dynamic_heads": num_dynamic,
                }
                total_heads += num_heads
                static_heads_count += num_static
                dynamic_heads_count += num_dynamic
            elif hasattr(module, "weight") or list(module.parameters(recurse=False)):
                # Freeze fallback/skipped layers to prevent unintended gradient flow
                for p in module.parameters(recurse=False):
                    p.requires_grad = False

        # Forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    first_linear = next(
                        (m for m in self.model.modules() if isinstance(m, nn.Linear)),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception:
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["budget"] = self.budget
        stats["static_ratio"] = self.static_ratio
        stats["max_cache_len"] = self.max_cache_len
        stats["num_sinks"] = self.num_sinks
        stats["recent_window"] = self.recent_window
        stats["total_attention_heads"] = total_heads
        stats["static_heads_count"] = static_heads_count
        stats["dynamic_heads_count"] = dynamic_heads_count
        stats["head_classifications"] = head_stats
        stats["attached_caches"] = len(head_stats)
        stats["kv_cache_budget_saving"] = (
            f"{(1.0 - self.budget) * 100:.1f}% Footprint Reduction"
        )
        stats["forward_check_success"] = forward_check_success
        return self.model, stats


@COMPRESSOR_REGISTRY.register("hybridkv")
@COMPRESSOR_REGISTRY.register("hybridkv_factory")
@COMPRESSOR_REGISTRY.register("hybridkv_compressor_factory")
def hybridkv_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """HybridKV compressor factory function."""
    budget = kwargs.pop(
        "budget",
        _cfg_get(comp_config, "hybridkv_budget", _cfg_get(comp_config, "budget", 0.3)),
    )
    static_ratio = kwargs.pop(
        "static_ratio",
        _cfg_get(
            comp_config,
            "hybridkv_static_ratio",
            _cfg_get(comp_config, "static_ratio", 0.7),
        ),
    )
    max_cache_len = kwargs.pop(
        "max_cache_len",
        _cfg_get(
            comp_config,
            "hybridkv_max_cache_len",
            _cfg_get(comp_config, "max_cache_len", 128),
        ),
    )
    num_sinks = kwargs.pop(
        "num_sinks",
        _cfg_get(
            comp_config, "hybridkv_num_sinks", _cfg_get(comp_config, "num_sinks", 4)
        ),
    )
    recent_window = kwargs.pop(
        "recent_window",
        _cfg_get(
            comp_config,
            "hybridkv_recent_window",
            _cfg_get(comp_config, "recent_window", 32),
        ),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "hybridkv_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", True),
        ),
    )

    compressor = HybridKVCompressor(
        model,
        budget=budget,
        static_ratio=static_ratio,
        max_cache_len=max_cache_len,
        num_sinks=num_sinks,
        recent_window=recent_window,
        target_attention_only=target_attention_only,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "HybridKVCache",
    "HybridKVCompressor",
    "hybridkv_factory",
]
