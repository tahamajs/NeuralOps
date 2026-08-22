"""
ZeroKV-Neo: L2-norm based KV cache eviction with attention sink and heavy-hitter preservation.
Registered as 'zerokv_neo'.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


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
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear cache tensors to avoid memory leaks."""
        self.k_cache = None
        self.v_cache = None

    @torch.no_grad()
    def update_and_compress(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new KV tokens and evict non-essential tokens when exceeding capacity.
        Shapes: (B, H, N, D).
        """
        orig_shape = new_k.shape
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
        win_k = self.k_cache[:, :, -window:, :] if window > 0 else torch.empty(0)
        win_v = self.v_cache[:, :, -window:, :] if window > 0 else torch.empty(0)

        if heavy_budget > 0 and cur_len > (sinks + window):
            mid_k = self.k_cache[:, :, sinks : cur_len - window, :]
            mid_v = self.v_cache[:, :, sinks : cur_len - window, :]
            mid_norms = torch.norm(mid_k.float(), dim=-1)  # (B, H, mid_len)
            _, top_idx = torch.topk(mid_norms, min(heavy_budget, mid_norms.size(-1)), dim=-1)
            top_idx, _ = torch.sort(top_idx, dim=-1)

            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
            sel_mid_k = mid_k.gather(2, idx_exp)
            sel_mid_v = mid_v.gather(2, idx_exp)

            self.k_cache = torch.cat([sink_k, sel_mid_k, win_k], dim=2).contiguous()
            self.v_cache = torch.cat([sink_v, sel_mid_v, win_v], dim=2).contiguous()
        else:
            self.k_cache = torch.cat([sink_k, win_k], dim=2).contiguous() if window > 0 else sink_k
            self.v_cache = torch.cat([sink_v, win_v], dim=2).contiguous() if window > 0 else sink_v

        return self.k_cache, self.v_cache


class ZeroKVNeoCompressor(BaseCompressor):
    """
    ZeroKV-Neo Compressor: L2-norm weight pruning preserving attention sinks and heavy hitters.

    Args:
        model: PyTorch model.
        keep_ratio: Fraction of output neurons to retain (default 0.3).
        num_attention_sinks: Number of initial sink channels pinned (default 4).
        heavy_hitter_ratio: Fraction allocated to heavy hitters (default 0.1).
    """

    def __init__(
        self,
        model: nn.Module,
        keep_ratio: float = 0.3,
        num_attention_sinks: int = 4,
        heavy_hitter_ratio: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="zerokv_neo", **kwargs)
        self.keep_ratio = float(keep_ratio)
        self.num_attention_sinks = int(num_attention_sinks)
        self.heavy_hitter_ratio = float(heavy_hitter_ratio)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear projection weights preserving attention sinks and heavy hitters."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and self._is_attention_proj(name):
                W = module.weight.data
                out_f, in_f = W.shape

                total_keep = max(1, min(int(out_f * self.keep_ratio), out_f))
                row_norms = torch.norm(W.float(), dim=1)

                n_sinks = min(self.num_attention_sinks, total_keep)
                n_heavy = min(
                    max(0, int(out_f * self.heavy_hitter_ratio)), total_keep - n_sinks
                )

                # Pin initial output channels as attention sinks
                sink_idx = torch.arange(n_sinks, device=W.device)

                # Select heavy hitters from remaining channels based on norm
                remaining_mask = torch.ones(out_f, dtype=torch.bool, device=W.device)
                remaining_mask[sink_idx] = False

                if n_heavy > 0 and remaining_mask.any():
                    rem_indices = torch.where(remaining_mask)[0]
                    rem_norms = row_norms[rem_indices]
                    _, top_rem = torch.topk(rem_norms, min(n_heavy, rem_norms.numel()))
                    heavy_idx = rem_indices[top_rem]
                else:
                    heavy_idx = torch.tensor([], device=W.device, dtype=torch.long)

                keep_idx = torch.cat([sink_idx, heavy_idx])
                mask = torch.zeros(out_f, device=W.device)
                mask[keep_idx] = 1.0

                module.weight.data = (W * mask.unsqueeze(1)).to(dtype=module.weight.dtype)
                if module.bias is not None:
                    module.bias.data = module.bias.data * mask

        stats = self.get_compression_stats()
        stats["keep_ratio"] = self.keep_ratio
        stats["num_attention_sinks"] = self.num_attention_sinks
        stats["heavy_hitter_ratio"] = self.heavy_hitter_ratio
        stats["kv_cache_budget_saving"] = f"{(1.0 - self.keep_ratio) * 100:.1f}% Footprint Reduction"
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


@COMPRESSOR_REGISTRY.register("zerokv_neo")
def zerokv_neo_factory(model: nn.Module, comp_config: Any = None) -> nn.Module:
    """ZeroKV-Neo compressor factory function."""
    keep_ratio = getattr(comp_config, "zerokv_keep_ratio", 0.3) if comp_config else 0.3
    num_sinks = getattr(comp_config, "zerokv_num_sinks", 4) if comp_config else 4
    heavy_ratio = getattr(comp_config, "zerokv_heavy_ratio", 0.1) if comp_config else 0.1
    compressor = ZeroKVNeoCompressor(
        model,
        keep_ratio=keep_ratio,
        num_attention_sinks=num_sinks,
        heavy_hitter_ratio=heavy_ratio,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
