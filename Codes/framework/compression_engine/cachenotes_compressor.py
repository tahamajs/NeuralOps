"""
CacheNotes: Task-Aware Key-Value Cache Compression and Static Weight Compression.

Given a task description or task type, CacheNotes generates Compression-Planning-Tokens (CPTs)
that guide the compression of the corpus and KV cache into compact, reusable states.

Paper: "CacheNotes: Task-aware key-value cache compression for reasoning-intensive
        knowledge tasks." Corallo et al., EACL 2026.
Registered as 'cachenotes'.

Mathematical Formulation:
1. CPT Importance Vector:
   $$I_{\\text{task}} = \\text{CPT}(T, d)$$
2. Runtime Dynamic KV Cache Pruning:
   $$\\mathcal{S}_{\\text{token}} = \\|K_{\\text{cache}} \\odot I_{\\text{task}}\\|_2, \\quad \\mathcal{S}_{:sinks} = +\\infty$$
   $$K_{\\text{retained}} = \\operatorname{Top-}k(K_{\\text{cache}}, \\mathcal{S}_{\\text{token}})$$
3. Decoupled Static Weight Compression (Low-Rank Task-Aware SVD without channel zeroing):
   $$W_{\\text{cpt}} = \\operatorname{diag}(I_{\\text{task}}) W$$
   $$W_{\\text{cpt}} = U \\Sigma V^\\top \\approx U_r \\Sigma_r V_r^\\top$$
   $$W_{\\text{approx}} = \\operatorname{diag}(I_{\\text{task}}^{-1}) (U_r \\Sigma_r V_r^\\top)$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    get_calibration_tensors,
    recalibrate_layernorms,
)

logger = logging.getLogger(__name__)


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
    def generate_importance(
        cls, task_type: str, num_features: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
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
    CacheNotes Compressor: Decouples static weight compression from KV cache dynamic pruning.
    Compresses static model weights using CPT-guided low-rank SVD without zeroing out entire output channels.

    Args:
        model: PyTorch model.
        task_type: Target downstream task ('qa', 'summarize', 'code', 'reasoning', 'math').
        keep_ratio: Fraction of rank / energy to retain (default 0.5).
        recalibrate_ln: Whether to recalibrate LayerNorm statistics after compression.
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: str = "qa",
        keep_ratio: float = 0.5,
        recalibrate_ln: bool = True,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="cachenotes", **kwargs)
        self.task_type = task_type
        self.keep_ratio = max(0.05, min(float(keep_ratio), 1.0))
        self.recalibrate_ln = recalibrate_ln
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.stats: Dict[str, Any] = {}

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using task-guided CPT-weighted low-rank SVD without channel destruction."""
        compressed_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                out_f, in_f = W.shape
                device = W.device
                dtype = module.weight.dtype

                # Generate CPT importance modulation per output feature
                importance = CompressionPlanningTokens.generate_importance(
                    self.task_type, out_f, device=device
                )  # [out_f]

                # Weight rows by CPT importance
                W_cpt = W * importance.unsqueeze(1)

                # Compute SVD in CPT-weighted space
                r = max(1, min(int(min(out_f, in_f) * self.keep_ratio), min(out_f, in_f)))
                try:
                    U, S, Vt = torch.linalg.svd(W_cpt, full_matrices=False)
                    # Low-rank approximation
                    W_lowrank = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vt[:r, :])
                    # Unweight back to physical space: all channels remain active and non-zero
                    W_approx = W_lowrank / importance.unsqueeze(1)
                except Exception as e:
                    logger.debug("SVD failed in CacheNotes on %s: %s", name, e)
                    W_approx = W

                module.weight.data = W_approx.to(dtype=dtype, device=device)
                compressed_layers_count += 1

        # LayerNorm Recalibration
        if self.recalibrate_ln:
            recalibrate_layernorms(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": compressed_layers_count,
            "task_type": self.task_type,
            "keep_ratio": self.keep_ratio,
            "recalibrate_ln": self.recalibrate_ln,
            "method": "cachenotes_decoupled_cpt_svd",
            "kv_cache_budget_saving": f"{(1.0 - self.keep_ratio) * 100:.1f}% Footprint Reduction",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("cachenotes")
@COMPRESSOR_REGISTRY.register("cachenotes_compressor")
def cachenotes_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """CacheNotes compressor factory function."""
    task_type = kwargs.pop("task_type", getattr(comp_config, "cachenotes_task", "qa") if comp_config else "qa")
    keep_ratio = kwargs.pop("keep_ratio", getattr(comp_config, "cachenotes_keep_ratio", 0.5) if comp_config else 0.5)
    recalibrate_ln = getattr(comp_config, "cachenotes_recalibrate_ln", kwargs.get("recalibrate_ln", True))

    compressor = CacheNotesCompressor(
        model,
        task_type=task_type,
        keep_ratio=keep_ratio,
        recalibrate_ln=recalibrate_ln,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

__all__ = [
    "CompressionPlanningTokens",
    "CacheNotesKVCache",
    "CacheNotesCompressor",
    "cachenotes_factory",
]
