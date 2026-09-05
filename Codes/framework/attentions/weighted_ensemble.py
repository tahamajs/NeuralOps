r"""
Smart Weighted Ensemble with Task-Aware Attention and Hybrid Combination.

Implements multi-model ensemble fusion strategies combining:
- Simple average pooling of per-model output logits.
- Performance-history weighted voting with exponential moving average tracking.
- Task-aware scaled dot-product attention over learned task embeddings.
- Hybrid fusion combining task attention and per-input dynamic confidence.
"""

from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAwareAttention(nn.Module):
    """
    Learns to weight model predictions based on task embeddings and pairwise
    task similarity matrix.
    """

    def __init__(self, num_tasks: int, hidden_dim: int = 64):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.task_emb = nn.Embedding(num_tasks, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.sim_mat = nn.Parameter(torch.eye(num_tasks))  # (num_tasks, num_tasks)

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

        # Clamp task_id within bounds
        task_id = max(0, min(self.num_tasks - 1, task_id))
        cur_id = torch.tensor([task_id], device=device)
        cur_emb = self.q_proj(self.task_emb(cur_id))  # (1, H)

        # Clamped model task embeddings
        valid_model_tasks = [max(0, min(self.num_tasks - 1, tid)) for tid in model_task_ids]
        all_ids = torch.tensor(valid_model_tasks, device=device)  # (M,)
        all_embs = self.k_proj(self.task_emb(all_ids))  # (M, H)

        # Attention scores (dot product)
        scale = 1.0 / np.sqrt(self.hidden_dim)
        scores = torch.matmul(cur_emb, all_embs.t()) * scale  # (1, M)

        # Add task similarity bias
        bias = self.sim_mat[valid_model_tasks, task_id].unsqueeze(0)  # (1, M)
        scores = scores + bias
        weights = F.softmax(scores, dim=-1)  # (1, M)

        # Correctly weight the models' logits: (M, 1, 1) * (M, B, C) -> sum -> (B, C)
        weighted = (weights.view(num_models, 1, 1) * all_logits).sum(dim=0)
        return weighted


class SmartWeightedEnsemble:
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
        num_tasks: int,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            models: list of PyTorch models (each should have a forward method returning logits)
            num_tasks: total number of tasks (for task embeddings)
        """
        if not models:
            raise ValueError("SmartWeightedEnsemble: 'models' list cannot be empty.")

        self.models = models
        self.num_models = len(models)
        self.num_tasks = max(1, num_tasks)

        if device is None:
            try:
                self.device = next(models[0].parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Performance tracking: (num_models, num_tasks) matrix
        self.performance = (
            torch.ones(self.num_models, self.num_tasks, device=self.device) * 0.5
        )

        self.attention = TaskAwareAttention(self.num_tasks).to(self.device)

    def update_performance(
        self,
        task_id: int,
        model_outputs: List[torch.Tensor],
        target: torch.Tensor,
        metric: str = "accuracy",
    ) -> None:
        """Update performance scores based on validation results."""
        task_id = max(0, min(self.num_tasks - 1, task_id))
        with torch.no_grad():
            for i, logits in enumerate(model_outputs):
                if i >= self.num_models:
                    break
                if metric == "accuracy":
                    preds = logits.argmax(dim=-1)
                    acc = (preds == target).float().mean().item()
                    self.performance[i, task_id] = (
                        0.9 * self.performance[i, task_id] + 0.1 * acc
                    )

    def _get_dynamic_weights(
        self, task_id: int, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute dynamic weights based on input-specific confidence."""
        weights = torch.zeros(self.num_models, device=self.device)
        with torch.no_grad():
            for i, model in enumerate(self.models):
                model.eval()
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(out, "logits"):
                    logits = out.logits
                elif isinstance(out, (tuple, list)):
                    logits = out[0]
                else:
                    logits = out

                probs = F.softmax(logits.float(), dim=-1)
                # Confidence = max probability across classes, averaged over batch
                conf = probs.max(dim=-1)[0].mean()
                weights[i] = conf.to(self.device)

        sum_w = weights.sum()
        if sum_w > 0:
            return weights / sum_w
        return torch.ones(self.num_models, device=self.device) / self.num_models

    def combine_logits(
        self,
        task_id: int,
        all_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        method: str = "hybrid",
    ) -> torch.Tensor:
        """
        Combine logits from all models.

        Args:
            task_id: target task (integer)
            all_logits: (num_models, batch_size, num_labels) – already computed.
            input_ids, attention_mask: used for dynamic weighting.
            method: 'simple_average', 'weighted', 'task_aware_attention', 'hybrid'
        """
        task_id = max(0, min(self.num_tasks - 1, task_id))

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
            model_task_ids = [i % self.num_tasks for i in range(self.num_models)]
            return self.attention(task_id, all_logits, model_task_ids)

        elif method == "hybrid":
            model_task_ids = [i % self.num_tasks for i in range(self.num_models)]
            attn_weighted = self.attention(task_id, all_logits, model_task_ids)

            if input_ids is not None and attention_mask is not None:
                dyn_weights = self._get_dynamic_weights(
                    task_id, input_ids, attention_mask
                )  # (M,)
                dyn_weighted = (all_logits * dyn_weights.view(-1, 1, 1)).sum(dim=0)
                return 0.6 * attn_weighted + 0.4 * dyn_weighted
            return attn_weighted

        else:
            raise ValueError(f"Unknown ensemble method: {method}")


def run_smart_ensemble(config: dict, dataset_path: str):
    """Command-line entry point for running smart ensemble."""
    print("Use the class 'SmartWeightedEnsemble' directly in your training/evaluation script.")
