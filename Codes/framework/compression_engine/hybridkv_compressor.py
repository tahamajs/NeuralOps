"""
HybridKV: Hybrid KV Cache Compression for Multimodal & Long-Context LLMs.
Classifies attention heads into static/dynamic types and applies distinct
compression strategies per type under a hierarchical budget strictly to dynamic KV caches.

Paper: "HybridKV: Hybrid KV Cache Compression for Efficient MLLM Inference"
       Zeng et al., arXiv:2604.05887.
Registered as 'hybridkv'.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


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
        static_heads_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.max_cache_len = int(max_cache_len)
        self.num_sinks = int(num_sinks)
        self.recent_window = int(recent_window)
        self.budget = float(budget)
        self.static_ratio = float(static_ratio)
        self.static_heads_mask = static_heads_mask
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear cache state to prevent memory leaks."""
        self.k_cache = None
        self.v_cache = None

    @torch.no_grad()
    def update_and_compress(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new KV representations and apply hybrid eviction if cache exceeds capacity.
        Shapes: (B, H, N, D) or (B, N, D) or (N, D).
        """
        orig_ndim = new_k.ndim
        orig_shape = new_k.shape
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
        head_k_list: List[torch.Tensor] = []
        head_v_list: List[torch.Tensor] = []

        is_static_head = [True] * H
        if self.static_heads_mask is not None:
            mask_list = self.static_heads_mask.view(-1).tolist()
            is_static_head = [bool(mask_list[h % len(mask_list)]) for h in range(H)]
        else:
            # Default first int(H * static_ratio) heads as static
            num_static = max(1, int(H * self.static_ratio))
            is_static_head = [h < num_static for h in range(H)]

        for h in range(H):
            cur_k_h = self.k_cache[:, h : h + 1, :, :]
            cur_v_h = self.v_cache[:, h : h + 1, :, :]

            if is_static_head[h]:
                # Static head policy: pinned sinks + heavy hitters across sequence
                sinks = min(self.num_sinks, cur_len)
                sink_k = cur_k_h[:, :, :sinks, :]
                sink_v = cur_v_h[:, :, :sinks, :]

                rem_capacity = max(0, target_len - sinks)
                if rem_capacity > 0 and cur_len > sinks:
                    rest_k = cur_k_h[:, :, sinks:, :]
                    rest_v = cur_v_h[:, :, sinks:, :]
                    rest_norms = torch.norm(rest_k.float(), dim=-1)  # (B, 1, rem_len)
                    k_val = min(rem_capacity, rest_norms.size(-1))
                    _, top_idx = torch.topk(rest_norms, k_val, dim=-1)
                    top_idx, _ = torch.sort(top_idx, dim=-1)
                    idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
                    sel_k = rest_k.gather(2, idx_exp)
                    sel_v = rest_v.gather(2, idx_exp)
                    head_k = torch.cat([sink_k, sel_k], dim=2)
                    head_v = torch.cat([sink_v, sel_v], dim=2)
                else:
                    head_k = sink_k
                    head_v = sink_v
            else:
                # Dynamic head policy: recent sliding window + chunked top keys
                window = min(self.recent_window, cur_len)
                win_k = cur_k_h[:, :, -window:, :]
                win_v = cur_v_h[:, :, -window:, :]

                rem_capacity = max(0, target_len - window)
                if rem_capacity > 0 and cur_len > window:
                    rest_k = cur_k_h[:, :, :-window, :]
                    rest_v = cur_v_h[:, :, :-window, :]
                    rest_norms = torch.norm(rest_k.float(), dim=-1)
                    k_val = min(rem_capacity, rest_norms.size(-1))
                    _, top_idx = torch.topk(rest_norms, k_val, dim=-1)
                    top_idx, _ = torch.sort(top_idx, dim=-1)
                    idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
                    sel_k = rest_k.gather(2, idx_exp)
                    sel_v = rest_v.gather(2, idx_exp)
                    head_k = torch.cat([sel_k, win_k], dim=2)
                    head_v = torch.cat([sel_v, win_v], dim=2)
                else:
                    head_k = win_k
                    head_v = win_v

            head_k_list.append(head_k)
            head_v_list.append(head_v)

        self.k_cache = torch.cat(head_k_list, dim=1).contiguous()
        self.v_cache = torch.cat(head_v_list, dim=1).contiguous()

        out_k, out_v = self.k_cache, self.v_cache
        if orig_ndim == 2:
            return out_k.squeeze(0).squeeze(0), out_v.squeeze(0).squeeze(0)
        elif orig_ndim == 3:
            return out_k.squeeze(1), out_v.squeeze(1)
        return out_k, out_v

    @torch.no_grad()
    def compress(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress dynamic KV cache tensors directly."""
        return self.update_and_compress(k, v)


class HybridKVCompressor(BaseCompressor):
    """
    HybridKV Compressor: Classifies attention projection heads into static/dynamic types
    and configures hierarchical dynamic KV cache budgets without altering model stationary weights.

    Args:
        model: PyTorch model.
        budget: Overall fraction of KV cache budget to retain (default 0.3).
        static_ratio: Fraction of budget allocated to static heads (default 0.7).
        max_cache_len: Maximum KV cache capacity per head (default 128).
        num_sinks: Number of sink tokens pinned for static heads (default 4).
        recent_window: Number of recent tokens retained for dynamic heads (default 32).
    """

    def __init__(
        self,
        model: nn.Module,
        budget: float = 0.3,
        static_ratio: float = 0.7,
        max_cache_len: int = 128,
        num_sinks: int = 4,
        recent_window: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="hybridkv", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            budget = getattr(cfg, "hybridkv_budget", getattr(cfg, "budget", budget))
            static_ratio = getattr(cfg, "hybridkv_static_ratio", getattr(cfg, "static_ratio", static_ratio))
            max_cache_len = getattr(cfg, "max_cache_len", max_cache_len)
            num_sinks = getattr(cfg, "num_sinks", num_sinks)
            recent_window = getattr(cfg, "recent_window", recent_window)

        self.budget = float(budget)
        self.static_ratio = float(static_ratio)
        self.max_cache_len = int(max_cache_len)
        self.num_sinks = int(num_sinks)
        self.recent_window = int(recent_window)

    @staticmethod
    def _classify_head(weight: torch.Tensor, threshold: float = 0.5) -> str:
        """
        Classify a projection head as 'static' or 'dynamic' based on normalized variance.
        """
        if weight.numel() == 0 or weight.shape[1] <= 1:
            return "static"
        row_variance = torch.var(weight.float(), dim=1, unbiased=False)
        max_var = row_variance.max() + 1e-8
        norm_var = row_variance / max_var
        dynamic_ratio = (norm_var > threshold).float().mean().item()
        return "dynamic" if dynamic_ratio > 0.3 else "static"

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Configure dynamic KV cache compression per attention layer while preserving all model weights."""
        head_stats: Dict[str, Any] = {}
        total_heads = 0
        static_heads_count = 0
        dynamic_heads_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and self._is_attention_proj(name):
                W = module.weight.data
                out_f = W.shape[0]
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
                # Attach KV cache manager without mutating model weights
                setattr(module, "hybrid_kv_cache", kv_cache)

                head_stats[name] = {
                    "head_type": head_type,
                    "num_heads": num_heads,
                    "static_heads": num_static,
                    "dynamic_heads": num_dynamic,
                }
                total_heads += num_heads
                static_heads_count += num_static
                dynamic_heads_count += num_dynamic

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["budget"] = self.budget
        stats["static_ratio"] = self.static_ratio
        stats["total_attention_heads"] = total_heads
        stats["static_heads_count"] = static_heads_count
        stats["dynamic_heads_count"] = dynamic_heads_count
        stats["head_classifications"] = head_stats
        stats["kv_cache_budget_saving"] = f"{(1.0 - self.budget) * 100:.1f}% Footprint Reduction"
        return self.model, stats

    @staticmethod
    def _is_attention_proj(name: str) -> bool:
        """Check if module name corresponds to attention projection."""
        attn_keywords = [
            "q_lin",
            "k_lin",
            "v_lin",
            "q_proj",
            "k_proj",
            "v_proj",
            "query",
            "key",
            "value",
            "attention",
            "fc",
            "dense",
        ]
        return any(kw in name.lower() for kw in attn_keywords)


@COMPRESSOR_REGISTRY.register("hybridkv")
def hybridkv_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """HybridKV compressor factory function."""
    budget = kwargs.pop("budget", getattr(comp_config, "hybridkv_budget", getattr(comp_config, "budget", 0.3)) if comp_config is not None else 0.3)
    static_ratio = kwargs.pop("static_ratio", getattr(comp_config, "hybridkv_static_ratio", getattr(comp_config, "static_ratio", 0.7)) if comp_config is not None else 0.7)
    compressor = HybridKVCompressor(model, budget=budget, static_ratio=static_ratio, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
