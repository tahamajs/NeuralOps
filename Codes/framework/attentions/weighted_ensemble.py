"""
Weighted ensemble with task‑aware attention and hybrid combination.
Supports simple average, performance‑based weighting, and attention‑based fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


class TaskAwareAttention(nn.Module):
    """
    Learns to weight model predictions based on task embeddings and pairwise
    task similarity matrix.
    """
    def __init__(self, num_tasks: int, hidden_dim: int = 64):
        super().__init__()
        self.task_emb = nn.Embedding(num_tasks, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.sim_mat = nn.Parameter(torch.eye(num_tasks))  # (num_tasks, num_tasks)

    def forward(self, task_id: int, all_logits: torch.Tensor,
                model_task_ids: List[int]) -> torch.Tensor:
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

        # Current task embedding
        cur_id = torch.tensor([task_id], device=device)
        cur_emb = self.q_proj(self.task_emb(cur_id))                # (1, H)
        # All model task embeddings
        all_ids = torch.tensor(model_task_ids, device=device)       # (M,)
        all_embs = self.k_proj(self.task_emb(all_ids))              # (M, H)

        # Attention scores (dot product)
        scores = torch.matmul(cur_emb, all_embs.T) / np.sqrt(self.task_emb.embedding_dim)  # (1, M)
        # Add task similarity bias
        bias = self.sim_mat[model_task_ids, task_id].unsqueeze(0)    # (1, M)
        scores = scores + bias
        weights = F.softmax(scores, dim=-1)                          # (1, M)

        # Weighted sum
        weighted = (all_logits * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
        return weighted


class SmartWeightedEnsemble:
    """
    Ensemble that combines multiple model predictions using various strategies:
        - simple_average
        - weighted by performance history
        - task‑aware attention
        - hybrid (attention + dynamic confidence)
    """
    def __init__(self, models: List[nn.Module], num_tasks: int, device: torch.device = None):
        """
        Args:
            models: list of PyTorch models (each should have a forward method returning logits)
            num_tasks: total number of tasks (for task embeddings)
        """
        self.models = models
        self.num_models = len(models)
        self.device = device or next(models[0].parameters()).device

        # Performance tracking: (num_models, num_tasks) matrix
        self.performance = torch.ones(self.num_models, num_tasks, device=self.device) * 0.5

        self.attention = TaskAwareAttention(num_tasks).to(self.device)

    def update_performance(self, task_id: int, model_outputs: List[torch.Tensor],
                           target: torch.Tensor, metric: str = 'accuracy'):
        """
        Update performance scores based on validation results.
        Not implemented in detail – placeholder.
        """
        with torch.no_grad():
            for i, logits in enumerate(model_outputs):
                if metric == 'accuracy':
                    preds = logits.argmax(dim=-1)
                    acc = (preds == target).float().mean().item()
                    self.performance[i, task_id] = 0.9 * self.performance[i, task_id] + 0.1 * acc
                # other metrics can be added

    def _get_dynamic_weights(self, task_id: int, input_ids: torch.Tensor,
                             attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic weights based on input‑specific confidence.
        Simplified version: uses model softmax probabilities.
        """
        weights = torch.zeros(self.num_models, device=self.device)
        with torch.no_grad():
            for i, model in enumerate(self.models):
                model.eval()
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = F.softmax(logits, dim=-1)
                # Confidence = max probability across classes, averaged over batch
                conf = probs.max(dim=-1)[0].mean()
                weights[i] = conf
        return weights / (weights.sum() + 1e-8)

    def combine_logits(self, task_id: int, all_logits: torch.Tensor,
                       input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       method: str = "hybrid") -> torch.Tensor:
        """
        Combine logits from all models.

        Args:
            task_id: target task (integer)
            all_logits: (num_models, batch_size, num_labels) – already computed.
            input_ids, attention_mask: used for dynamic weighting.
            method: 'simple_average', 'weighted', 'task_aware_attention', 'hybrid'
        """
        if method == "simple_average":
            return all_logits.mean(dim=0)

        elif method == "weighted":
            # Use performance matrix as fixed weights
            perf_weights = self.performance[:, task_id]               # (M,)
            perf_weights = perf_weights / (perf_weights.sum() + 1e-8)
            return (all_logits * perf_weights.view(-1, 1, 1)).sum(dim=0)

        elif method == "task_aware_attention":
            # Each model is associated with a prototype task (e.g., index of the task it was trained on)
            # For demonstration, assign models to tasks in round‑robin:
            num_tasks = self.performance.shape[1]
            model_task_ids = [i % num_tasks for i in range(self.num_models)]
            return self.attention(task_id, all_logits, model_task_ids)

        elif method == "hybrid":
            # Combine task_aware attention with input‑adaptive confidence
            num_tasks = self.performance.shape[1]
            model_task_ids = [i % num_tasks for i in range(self.num_models)]
            attn_weighted = self.attention(task_id, all_logits, model_task_ids)
            dyn_weights = self._get_dynamic_weights(task_id, input_ids, attention_mask)  # (M,)
            dyn_weighted = (all_logits * dyn_weights.view(-1, 1, 1)).sum(dim=0)
            # Blend: 60% attention, 40% dynamic
            return 0.6 * attn_weighted + 0.4 * dyn_weighted

        else:
            raise ValueError(f"Unknown ensemble method: {method}")


def run_smart_ensemble(config: dict, dataset_path: str):
    """
    Command‑line entry point for running smart ensemble.
    (Placeholder – real implementation would load config, models, data, etc.)
    """
    print("Smart ensemble not fully implemented in this example.")
    print("Use the class 'SmartWeightedEnsemble' directly in your training/evaluation script.")
