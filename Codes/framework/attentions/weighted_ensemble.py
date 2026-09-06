r"""
Smart Weighted Ensemble with Task-Aware Attention and Hybrid Combination.

Implements multi-model ensemble fusion strategies combining:
- Simple average pooling of per-model output logits.
- Performance-history weighted voting with exponential moving average tracking.
- Task-aware scaled dot-product attention over learned task embeddings.
- Hybrid fusion combining task attention and per-input dynamic confidence.

Hardened with:
- Subclassing nn.Module with nn.ModuleList for full PyTorch compatibility.
- Automatic device placement and input tensor movement to self.device.
- Complete attention_mask propagation to all sub-models.
- trainable_dynamic flag for optional end-to-end gradient flow.
- Strict model_task_ids length validation.
- Unified 60/40 hybrid combination.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TaskAwareAttention", "SmartWeightedEnsemble", "run_smart_ensemble"]


class TaskAwareAttention(nn.Module):
    """
    Learns to weight model predictions based on task embeddings and pairwise
    task similarity matrix.
    """

    def __init__(self, num_tasks: int, hidden_dim: int = 64):
        super().__init__()
        self.num_tasks = max(1, num_tasks)
        self.hidden_dim = hidden_dim
        self.task_emb = nn.Embedding(self.num_tasks, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.sim_mat = nn.Parameter(torch.eye(self.num_tasks))  # (num_tasks, num_tasks)

    def forward(
        self, task_id: int, all_logits: torch.Tensor, model_task_ids: List[int]
    ) -> torch.Tensor:
        """
        Args:
            task_id: target task id (integer)
            all_logits: (num_models, batch_size, num_labels) logit outputs
            model_task_ids: list of length num_models, each entry is the task
                            that model was trained on (or a proxy)
        Returns:
            weighted_logits: (batch_size, num_labels)
        """
        device = all_logits.device
        num_models = all_logits.shape[0]

        if len(model_task_ids) != num_models:
            raise ValueError(
                f"model_task_ids length ({len(model_task_ids)}) must match all_logits.shape[0] ({num_models})"
            )

        # Clamp task_id within bounds
        task_id = max(0, min(self.num_tasks - 1, task_id))
        cur_id = torch.tensor([task_id], device=device)
        cur_emb = self.q_proj(self.task_emb(cur_id))  # (1, H)

        # Clamped model task embeddings
        valid_model_tasks = [max(0, min(self.num_tasks - 1, tid)) for tid in model_task_ids]
        all_ids = torch.tensor(valid_model_tasks, device=device)  # (M,)
        all_embs = self.k_proj(self.task_emb(all_ids))  # (M, H)

        # Attention scores (dot product)
        scale = 1.0 / math.sqrt(self.hidden_dim)
        scores = torch.matmul(cur_emb, all_embs.t()) * scale  # (1, M)

        # Add task similarity bias
        bias = self.sim_mat[valid_model_tasks, task_id].unsqueeze(0)  # (1, M)
        scores = scores + bias
        weights = F.softmax(scores, dim=-1)  # (1, M)

        # Correctly weight the models' logits: (M, 1, 1) * (M, B, C) -> sum -> (B, C)
        weighted = (weights.view(num_models, 1, 1) * all_logits).sum(dim=0)
        return weighted


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
        models: List[nn.Module],
        num_tasks: int = 1,
        device: Optional[torch.device] = None,
        trainable_dynamic: bool = False,
    ):
        super().__init__()
        if not models:
            raise ValueError("SmartWeightedEnsemble: 'models' list cannot be empty.")

        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_tasks = max(1, num_tasks)
        self.trainable_dynamic = trainable_dynamic

        if device is None:
            try:
                self.device = next(models[0].parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Performance tracking buffer: (num_models, num_tasks) matrix
        self.register_buffer(
            "performance",
            torch.ones(self.num_models, self.num_tasks, device=self.device) * 0.5,
        )

        self.attention = TaskAwareAttention(self.num_tasks).to(self.device)
        self.to(self.device)

    def to(self, *args: Any, **kwargs: Any) -> SmartWeightedEnsemble:
        res = super().to(*args, **kwargs)
        # Update self.device to match
        try:
            p = next(self.parameters())
            self.device = p.device
        except StopIteration:
            pass
        return res

    def update_performance(
        self,
        task_id: int,
        model_outputs: List[torch.Tensor],
        target: torch.Tensor,
        metric: str = "accuracy",
    ) -> None:
        """Update performance scores based on validation results."""
        task_id = max(0, min(self.num_tasks - 1, task_id))
        target = target.to(self.device)
        with torch.no_grad():
            for i, logits in enumerate(model_outputs):
                if i >= self.num_models:
                    break
                logits = logits.to(self.device)
                if metric == "accuracy":
                    preds = logits.argmax(dim=-1)
                    acc = (preds == target).float().mean().item()
                    self.performance[i, task_id] = (
                        0.9 * self.performance[i, task_id] + 0.1 * acc
                    )

    def _get_dynamic_weights(
        self,
        task_id: int,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute dynamic weights based on input-specific confidence."""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        def _forward_all():
            weights_list = []
            for i, model in enumerate(self.models):
                if not self.trainable_dynamic:
                    model.eval()
                if attention_mask is not None:
                    out = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                else:
                    out = model(input_ids=input_ids, **kwargs)
                if hasattr(out, "logits"):
                    logits = out.logits
                elif isinstance(out, (tuple, list)):
                    logits = out[0]
                else:
                    logits = out

                probs = F.softmax(logits.float(), dim=-1)
                conf = probs.max(dim=-1)[0].mean()
                weights_list.append(conf)
            return torch.stack(weights_list)

        if self.trainable_dynamic:
            raw_weights = _forward_all()
        else:
            with torch.no_grad():
                raw_weights = _forward_all()

        sum_w = raw_weights.sum()
        if sum_w > 0:
            return raw_weights / sum_w
        return torch.ones(self.num_models, device=self.device) / self.num_models

    def combine_logits(
        self,
        task_id: int,
        all_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        method: str = "hybrid",
        model_task_ids: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Combine logits from all models.

        Args:
            task_id: target task (integer)
            all_logits: (num_models, batch_size, num_labels) – already computed.
            input_ids, attention_mask: used for dynamic weighting.
            method: 'simple_average', 'weighted', 'task_aware_attention', 'hybrid'
            model_task_ids: task assignments for each sub-model.
        """
        all_logits = all_logits.to(self.device)
        num_models = all_logits.shape[0]
        task_id = max(0, min(self.num_tasks - 1, task_id))

        if model_task_ids is not None and len(model_task_ids) != num_models:
            raise ValueError(
                f"model_task_ids length ({len(model_task_ids)}) must match num_models ({num_models})"
            )

        t_ids = (
            model_task_ids
            if model_task_ids is not None
            else [i % self.num_tasks for i in range(num_models)]
        )

        if method == "simple_average":
            return all_logits.mean(dim=0)

        elif method == "weighted":
            perf_weights = self.performance[:, task_id]  # (M,)
            sum_w = perf_weights.sum()
            if sum_w > 0:
                perf_weights = perf_weights / sum_w
            else:
                perf_weights = torch.ones_like(perf_weights) / max(1, len(perf_weights))
            return (all_logits * perf_weights.view(-1, 1, 1)).sum(dim=0)

        elif method == "task_aware_attention":
            return self.attention(task_id, all_logits, t_ids)

        elif method == "hybrid":
            attn_weighted = self.attention(task_id, all_logits, t_ids)
            if input_ids is not None:
                dyn_weights = self._get_dynamic_weights(
                    task_id, input_ids, attention_mask=attention_mask, **kwargs
                )  # (M,)
                dyn_weighted = (all_logits * dyn_weights.view(-1, 1, 1)).sum(dim=0)
                return 0.6 * attn_weighted + 0.4 * dyn_weighted
            return attn_weighted

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def forward(
        self,
        input_ids: torch.Tensor,
        task_id: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        method: str = "hybrid",
        model_task_ids: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass querying each model and combining logits with attention_mask propagation."""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if model_task_ids is not None and len(model_task_ids) != self.num_models:
            raise ValueError(
                f"model_task_ids length ({len(model_task_ids)}) must match num_models ({self.num_models})"
            )

        all_logits = []
        for model in self.models:
            if attention_mask is not None:
                out = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            else:
                out = model(input_ids=input_ids, **kwargs)

            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            all_logits.append(logits)

        stacked = torch.stack(all_logits, dim=0)

        return self.combine_logits(
            task_id=task_id,
            all_logits=stacked,
            input_ids=input_ids,
            attention_mask=attention_mask,
            method=method,
            model_task_ids=model_task_ids,
            **kwargs,
        )

    def forward_simple_average(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.forward(
            input_ids,
            task_id=0,
            attention_mask=attention_mask,
            method="simple_average",
            **kwargs,
        )

    def forward_task_attention(
        self,
        task_id: int,
        model_task_ids: List[int],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.forward(
            input_ids,
            task_id=task_id,
            attention_mask=attention_mask,
            method="task_aware_attention",
            model_task_ids=model_task_ids,
            **kwargs,
        )

    def forward_hybrid(
        self,
        task_id: int,
        model_task_ids: List[int],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.forward(
            input_ids,
            task_id=task_id,
            attention_mask=attention_mask,
            method="hybrid",
            model_task_ids=model_task_ids,
            **kwargs,
        )


def run_smart_ensemble(config: dict, dataset_path: str) -> None:
    """Command-line entry point for running smart ensemble."""
    print("Use the class 'SmartWeightedEnsemble' directly in your training/evaluation script.")
