"""
DEPRECATED — Legacy ensemble modules.

These classes are kept for backward compatibility only and are NOT part of
the compression pipeline. They are not related to attention or low-rank
compression and should not be used in production.

If you need a multi-model ensemble, consider:
  - Simple voting
  - Stacking with a small meta-model
  - Hugging Face's built-in ensemble utilities

Bug-fix map (2026 revision):
  LE-01  Embedding indexing with explicit long dtype.
  LE-02  sim_mat bias computed via gather (not advanced indexing).
  LE-03  to() override removed; use standard nn.Module.to().
  LE-04  NaN-safe weight normalization.
  LE-05  performance update under no_grad + explicit clamp.
  LE-06  DeprecationWarning emitted only on class instantiation.
  LE-07  Dynamic weight extraction respects trainable_dynamic.
  LE-08  model forward wrapped with fallback for missing logits.
  LE-09  model_task_ids validated for non-empty.
  LE-10  Each model's logits moved to ensemble device.
  LE-11  combine_logits validates shapes.
  LE-12  TaskAwareAttention.copy_weights silently no-op (framework hook).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DEPRECATION_MSG = (
    "framework.attentions.legacy_ensemble is deprecated and will be removed "
    "in a future version. It is not part of the compression pipeline."
)


def _warn_deprecated() -> None:
    """LE-06: emit deprecation warning only when the class is instantiated."""
    import warnings

    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)


# ===========================================================================
# Task-aware attention
# ===========================================================================


class TaskAwareAttention(nn.Module):
    """
    Learns to weight model predictions based on task embeddings and a
    pairwise task similarity matrix.
    """

    def __init__(self, num_tasks: Any = 1, hidden_dim: int = 64) -> None:
        super().__init__()

        # Accept either int or config object
        if isinstance(num_tasks, int):
            self.num_tasks = max(1, int(num_tasks))
        else:
            extracted = getattr(
                num_tasks, "num_labels", getattr(num_tasks, "num_tasks", 2)
            )
            if not isinstance(extracted, int) or extracted <= 0:
                extracted = 2
            self.num_tasks = max(1, extracted)

        self.hidden_dim = int(hidden_dim)
        self.task_emb = nn.Embedding(self.num_tasks, self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # Similarity matrix: logits added as bias to attention scores
        self.sim_mat = nn.Parameter(torch.eye(self.num_tasks))

    def copy_weights(self, old_attn: Any) -> None:
        """LE-12: framework compatibility hook. This module has no pretrained
        weights to transfer; silently no-op."""
        return None

    def forward(
        self,
        task_id: int,
        all_logits: torch.Tensor,
        model_task_ids: list[int],
    ) -> torch.Tensor:
        device = all_logits.device
        num_models = all_logits.shape[0]

        if len(model_task_ids) != num_models:
            raise ValueError(
                f"model_task_ids length ({len(model_task_ids)}) must match "
                f"all_logits.shape[0] ({num_models})"
            )

        task_id = max(0, min(self.num_tasks - 1, int(task_id)))

        # LE-01: explicit long dtype for Embedding
        cur_id = torch.tensor([task_id], dtype=torch.long, device=device)
        cur_emb = self.q_proj(self.task_emb(cur_id))  # (1, H)

        valid_ids = [
            max(0, min(self.num_tasks - 1, int(tid))) for tid in model_task_ids
        ]
        all_ids = torch.tensor(valid_ids, dtype=torch.long, device=device)
        all_embs = self.k_proj(self.task_emb(all_ids))  # (M, H)

        scale = 1.0 / math.sqrt(self.hidden_dim)
        scores = torch.matmul(cur_emb, all_embs.t().contiguous()) * scale  # (1, M)

        # LE-02: bias via gather — cleaner than advanced indexing
        # sim_mat[t, task_id] → (M,)
        bias = self.sim_mat.index_select(0, all_ids)[:, task_id]  # (M,)
        scores = scores + bias.unsqueeze(0)

        weights = F.softmax(scores, dim=-1)  # (1, M)
        weighted = (weights.view(num_models, 1, 1) * all_logits).sum(dim=0)
        return weighted


# ===========================================================================
# Ensemble
# ===========================================================================


class SmartWeightedEnsemble(nn.Module):
    """
    Ensemble that combines multiple model predictions using various strategies:
      - simple_average
      - weighted by performance history
      - task-aware attention
      - hybrid (attention + dynamic confidence)
    """

    def __init__(
        self,
        models: list[nn.Module],
        num_tasks: int = 1,
        device: torch.device | None = None,
        trainable_dynamic: bool = False,
    ) -> None:
        _warn_deprecated()  # LE-06

        super().__init__()
        if not models:
            raise ValueError("SmartWeightedEnsemble: 'models' list cannot be empty.")

        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_tasks = max(1, int(num_tasks))
        self.trainable_dynamic = bool(trainable_dynamic)

        # Determine device
        if device is None:
            try:
                device = next(models[0].parameters()).device
            except (StopIteration, AttributeError):
                device = torch.device("cpu")
        self._device_hint = device

        self.register_buffer(
            "performance",
            torch.full((self.num_models, self.num_tasks), 0.5),
        )
        self.attention = TaskAwareAttention(self.num_tasks)

        # LE-03: use standard nn.Module.to() — no override.
        self.to(device)

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """Compute device from the first parameter or fallback hint."""
        try:
            p = next(self.parameters())
            return p.device
        except StopIteration:
            return getattr(self, "_device_hint", torch.device("cpu"))

    # ------------------------------------------------------------------
    def update_performance(
        self,
        task_id: int,
        model_outputs: list[torch.Tensor],
        target: torch.Tensor,
        metric: str = "accuracy",
    ) -> None:
        task_id = max(0, min(self.num_tasks - 1, int(task_id)))
        device = self.device
        target = target.to(device)

        # LE-05: explicit no_grad + clamp
        with torch.no_grad():
            for i, logits in enumerate(model_outputs):
                if i >= self.num_models:
                    break
                logits = logits.to(device)
                if metric == "accuracy":
                    preds = logits.argmax(dim=-1)
                    acc = float((preds == target).float().mean().item())
                    new_val = (
                        0.9 * float(self.performance[i, task_id].item()) + 0.1 * acc
                    )
                    # Clamp to [0, 1] to prevent drift
                    new_val = max(0.0, min(1.0, new_val))
                    self.performance[i, task_id] = new_val

    # ------------------------------------------------------------------
    def _forward_single(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """LE-08: robust forward with logits fallback."""
        try:
            if attention_mask is not None:
                out = model(
                    input_ids=input_ids, attention_mask=attention_mask, **kwargs
                )
            else:
                out = model(input_ids=input_ids, **kwargs)
        except TypeError:
            # Some models don't accept input_ids kwarg
            out = (
                model(input_ids, attention_mask=attention_mask, **kwargs)
                if attention_mask is not None
                else model(input_ids, **kwargs)
            )

        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return out[0]
        if isinstance(out, torch.Tensor):
            return out
        raise RuntimeError(
            f"Model {model.__class__.__name__} returned unexpected output: {type(out)}"
        )

    # ------------------------------------------------------------------
    def _get_dynamic_weights(
        self,
        task_id: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        device = self.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        def _run() -> torch.Tensor:
            weights = []
            for model in self.models:
                logits = self._forward_single(
                    model, input_ids, attention_mask, **kwargs
                ).to(
                    device
                )  # LE-10
                probs = F.softmax(logits.float(), dim=-1)
                conf = float(probs.max(dim=-1)[0].mean().item())
                weights.append(conf)
            return torch.tensor(weights, device=device, dtype=torch.float32)

        # LE-07: respect trainable_dynamic
        if self.trainable_dynamic:
            raw = _run()
        else:
            with torch.no_grad():
                raw = _run()

        # LE-04: NaN-safe normalization
        if raw.numel() == 0:
            return torch.ones(self.num_models, device=device) / max(1, self.num_models)

        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(raw.sum().item())
        if total <= 1e-8:
            return torch.ones(self.num_models, device=device) / max(1, self.num_models)
        return raw / raw.sum()

    # ------------------------------------------------------------------
    def combine_logits(
        self,
        task_id: int,
        all_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        method: str = "hybrid",
        model_task_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # LE-11: shape validation
        if all_logits.dim() < 2:
            raise ValueError(
                f"combine_logits expects all_logits with at least 2 dims, "
                f"got {all_logits.shape}"
            )

        device = self.device
        all_logits = all_logits.to(device)
        num_models = all_logits.shape[0]
        task_id = max(0, min(self.num_tasks - 1, int(task_id)))

        if num_models != self.num_models:
            raise ValueError(
                f"num_models in logits ({num_models}) != ensemble.num_models "
                f"({self.num_models})"
            )

        # LE-09: validate model_task_ids
        if model_task_ids is not None and len(model_task_ids) == 0:
            model_task_ids = None

        if model_task_ids is not None and len(model_task_ids) != num_models:
            raise ValueError(
                f"model_task_ids length ({len(model_task_ids)}) must match "
                f"num_models ({num_models})"
            )

        t_ids = (
            list(model_task_ids)
            if model_task_ids is not None
            else [i % self.num_tasks for i in range(num_models)]
        )

        if method == "simple_average":
            return all_logits.mean(dim=0)

        if method == "weighted":
            perf_weights = self.performance[:, task_id].to(all_logits.dtype)
            sum_w = float(perf_weights.sum().item())
            if sum_w > 1e-8:
                perf_weights = perf_weights / perf_weights.sum()
            else:
                perf_weights = torch.ones_like(perf_weights) / num_models
            return (all_logits * perf_weights.view(-1, 1, 1)).sum(dim=0)

        if method == "task_aware_attention":
            return self.attention(task_id, all_logits, t_ids)

        if method == "hybrid":
            attn_weighted = self.attention(task_id, all_logits, t_ids)
            if input_ids is not None:
                dyn_weights = self._get_dynamic_weights(
                    task_id, input_ids, attention_mask=attention_mask, **kwargs
                ).to(all_logits.dtype)
                dyn_weighted = (all_logits * dyn_weights.view(-1, 1, 1)).sum(dim=0)
                return 0.6 * attn_weighted + 0.4 * dyn_weighted
            return attn_weighted

        raise ValueError(f"Unknown ensemble method: {method}")

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        task_id: int = 0,
        attention_mask: torch.Tensor | None = None,
        method: str = "hybrid",
        model_task_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        device = self.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if model_task_ids is not None and len(model_task_ids) != self.num_models:
            raise ValueError(
                f"model_task_ids length ({len(model_task_ids)}) must match "
                f"num_models ({self.num_models})"
            )

        all_logits = []
        for model in self.models:
            logits = self._forward_single(model, input_ids, attention_mask, **kwargs)
            all_logits.append(logits.to(device))

        stacked = torch.stack(all_logits, dim=0)  # (M, B, C)

        return self.combine_logits(
            task_id=task_id,
            all_logits=stacked,
            input_ids=input_ids,
            attention_mask=attention_mask,
            method=method,
            model_task_ids=model_task_ids,
            **kwargs,
        )

    # ------------------------------------------------------------------
    def forward_simple_average(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.forward(
            input_ids,
            task_id=0,
            attention_mask=attention_mask,
            method="simple_average",
            **kwargs,
        )

    def copy_weights(self, old_attn: Any) -> None:
        """LE-12: framework compatibility hook. No-op."""
        return None


# ===========================================================================
def run_smart_ensemble(config: dict, dataset_path: str) -> None:
    """Deprecated stub. Use SmartWeightedEnsemble directly in your script."""
    print(
        "Use the class 'SmartWeightedEnsemble' directly in your "
        "training/evaluation script."
    )


# LE-06: no module-level warning — only emitted on class instantiation.
__all__ = [
    "TaskAwareAttention",
    "SmartWeightedEnsemble",
    "run_smart_ensemble",
]
