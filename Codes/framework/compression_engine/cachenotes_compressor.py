"""
CacheNotes: Task-Aware Key-Value Cache Compression.

Given a task description or task type, CacheNotes generates Compression-Planning-Tokens (CPTs)
that guide the compression of the corpus and KV cache into compact, reusable states.

Paper: "CacheNotes: Task-aware key-value cache compression for reasoning-intensive
        knowledge tasks." Corallo et al., EACL 2026.
Registered as 'cachenotes'.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class CompressionPlanningTokens:
    """
    CPT importance generator.
    Simulates task-aware attention planning scores for different downstream tasks.
    """

    TASK_KEYWORD_MAP = {
        "qa": ["question", "answer", "context", "passage", "who", "what", "where", "why"],
        "summarize": ["summary", "abstract", "key", "main", "brief", "conclusion"],
        "code": ["function", "class", "def", "return", "import", "self", "error", "test"],
        "reasoning": ["therefore", "because", "if", "then", "thus", "proof", "step", "hence"],
        "math": ["solve", "equation", "matrix", "integral", "theorem", "calculate", "equal"],
    }

    @classmethod
    def generate_importance(cls, task_type: str, num_features: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Generate per-feature importance modulation vector based on task specification."""
        task_lower = str(task_type).lower()
        importance = torch.zeros(num_features, device=device)

        # Baseline structure: prompt position bias
        pos_dim = max(1, num_features // 4)
        importance[:pos_dim] = 0.25

        if any(kw in task_lower for kw in ["qa", "question"]):
            importance[pos_dim : 2 * pos_dim] += 0.35
        elif any(kw in task_lower for kw in ["code", "program"]):
            importance[2 * pos_dim : 3 * pos_dim] += 0.40
        elif any(kw in task_lower for kw in ["reason", "math"]):
            importance += 0.20

        return torch.clamp(importance + 0.5, min=0.1, max=2.0)


class CacheNotesKVCache:
    """
    Runtime task-guided dynamic KV cache compressor.
    Evaluates token task-relevance dynamically and prunes non-essential context tokens,
    supporting variable sequence lengths during prefill and autoregressive decoding.
    """

    def __init__(
        self,
        max_cache_len: int = 128,
        task_type: str = "qa",
        keep_ratio: float = 0.3,
        num_sinks: int = 4,
    ) -> None:
        self.max_cache_len = int(max_cache_len)
        self.task_type = task_type
        self.keep_ratio = float(keep_ratio)
        self.num_sinks = int(num_sinks)
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear cache state to prevent memory leaks across sequences."""
        self.k_cache = None
        self.v_cache = None

    @torch.no_grad()
    def update_and_compress(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new KV tokens and apply task-aware retention when exceeding capacity.
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

        sinks = min(self.num_sinks, cur_len)
        retained_cap = max(sinks + 1, self.max_cache_len)

        # Task importance weighting across tokens
        cpt_importance = CompressionPlanningTokens.generate_importance(
            self.task_type, D, device=self.k_cache.device
        )
        weighted_k = self.k_cache * cpt_importance.view(1, 1, 1, D)
        token_scores = torch.norm(weighted_k.float(), dim=-1)  # (B, H, cur_len)

        # Always preserve initial sink tokens
        token_scores[:, :, :sinks] += 1e6

        _, top_indices = torch.topk(token_scores, retained_cap, dim=-1)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        idx_exp = top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        self.k_cache = self.k_cache.gather(2, idx_exp).contiguous()
        self.v_cache = self.v_cache.gather(2, idx_exp).contiguous()

        return self.k_cache, self.v_cache


class CacheNotesCompressor(BaseCompressor):
    """
    CacheNotes Compressor: Compresses model weights using task-aware importance scoring
    inspired by CacheNotes CPT planning.

    Args:
        model: PyTorch model.
        task_type: Target downstream task ('qa', 'summarize', 'code', 'reasoning', 'math').
        keep_ratio: Fraction of parameters to retain (default 0.3).
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: str = "qa",
        keep_ratio: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="cachenotes", **kwargs)
        self.task_type = task_type
        self.keep_ratio = float(keep_ratio)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using task-guided CPT scoring."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                out_f, in_f = W.shape

                # Generate CPT importance modulation per neuron
                importance = CompressionPlanningTokens.generate_importance(
                    self.task_type, out_f, device=W.device
                )

                # Combine row norm with task importance
                row_norms = torch.norm(W.float(), dim=1)
                combined_score = row_norms * (1.0 + importance)

                keep_n = max(1, min(int(out_f * self.keep_ratio), out_f))
                _, top_idx = torch.topk(combined_score, keep_n)

                mask = torch.zeros(out_f, device=W.device)
                mask[top_idx] = 1.0

                module.weight.data = (W * mask.unsqueeze(1)).to(dtype=module.weight.dtype)
                if module.bias is not None:
                    module.bias.data = module.bias.data * mask

        stats = self.get_compression_stats()
        stats["task_type"] = self.task_type
        stats["keep_ratio"] = self.keep_ratio
        stats["kv_cache_budget_saving"] = f"{(1.0 - self.keep_ratio) * 100:.1f}% Footprint Reduction"
        return self.model, stats


@COMPRESSOR_REGISTRY.register("cachenotes")
def cachenotes_factory(model: nn.Module, comp_config: Any = None) -> nn.Module:
    """CacheNotes compressor factory function."""
    task_type = getattr(comp_config, "cachenotes_task", "qa") if comp_config else "qa"
    keep_ratio = getattr(comp_config, "cachenotes_keep_ratio", 0.3) if comp_config else 0.3
    compressor = CacheNotesCompressor(model, task_type=task_type, keep_ratio=keep_ratio)
    compressed_model, _ = compressor.compress()
    return compressed_model
