r"""
Smart Weighted Ensemble with Task-Aware Attention and Hybrid Combination.

Implements multi-model ensemble fusion strategies combining:
- Simple average pooling of per-model output logits.
- Performance-history weighted voting with exponential moving average tracking.
- Task-aware scaled dot-product attention over learned task embeddings.
- Hybrid fusion combining task attention and per-input dynamic confidence.

Mathematical Mechanism:
    Given $M$ model logit outputs $\{L_m\}_{m=1}^M \in \mathbb{R}^{B \times C}$ for task $t$:
    1. **Task-Aware Attention Weighting:**
       Let task embeddings $e_t = W_Q E[t]$ and model embeddings $e_m = W_K E[\text{task}_m]$:
       $$w_m^{\text{attn}} = \text{Softmax}\left(\frac{e_t e_m^\top}{\sqrt{D}} + S_{m, t}\right)$$
       $$L^{\text{attn}} = \sum_{m=1}^M w_m^{\text{attn}} L_m$$
    2. **Dynamic Confidence Weighting:**
       $$c_m = \mathbb{E}_B\left[\max_c p_{m, c}\right], \quad w_m^{\text{dyn}} = \frac{c_m}{\sum_{m'} c_{m'}}$$
       $$L^{\text{dyn}} = \sum_{m=1}^M w_m^{\text{dyn}} L_m$$
    3. **Hybrid Fusion:**
       $$L^{\text{hybrid}} = 0.6 \cdot L^{\text{attn}} + 0.4 \cdot L^{\text{dyn}}$$

Computational Complexity:
    - Task attention: $\mathcal{O}(M^2 \cdot D)$.
    - Dynamic confidence: $\mathcal{O}(M \cdot B \cdot C)$.
    - Ensemble combination: $\mathcal{O}(M \cdot B \cdot C)$.
"""

from typing import List

import numpy as np
import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F


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
        all_logits.shape[0]

        # Current task embedding
        cur_id = torch.tensor([task_id], device=device)
        cur_emb = self.q_proj(self.task_emb(cur_id))  # (1, H)
        # All model task embeddings
        all_ids = torch.tensor(model_task_ids, device=device)  # (M,)
        all_embs = self.k_proj(self.task_emb(all_ids))  # (M, H)

        # Attention scores (dot product)
        scores = torch.matmul(cur_emb, all_embs.T) / np.sqrt(
            self.task_emb.embedding_dim
        )  # (1, M)
        # Add task similarity bias
        bias = self.sim_mat[model_task_ids, task_id].unsqueeze(0)  # (1, M)
        scores = scores + bias
        weights = F.softmax(scores, dim=-1)  # (1, M)

        # Weighted sum
        weighted = (all_logits * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
        return weighted, getattr(self, '_attn_weights', None)


class SmartWeightedEnsemble:
    """
    Ensemble that combines multiple model predictions using various strategies:
        - simple_average
        - weighted by performance history
        - task‑aware attention
        - hybrid (attention + dynamic confidence)
    """

    def __init__(
        self, models: List[nn.Module], num_tasks: int, device: torch.device = None
    ):
        """
        Args:
            models: list of PyTorch models (each should have a forward method returning logits)
            num_tasks: total number of tasks (for task embeddings)
        """
        self.models = models
        self.num_models = len(models)
        self.device = device or next(models[0].parameters()).device

        # Performance tracking: (num_models, num_tasks) matrix
        self.performance = (
            torch.ones(self.num_models, num_tasks, device=self.device) * 0.5
        )

        self.attention = TaskAwareAttention(num_tasks).to(self.device)

    def update_performance(
        self,
        task_id: int,
        model_outputs: List[torch.Tensor],
        target: torch.Tensor,
        metric: str = "accuracy",
    ):
        """
        Update performance scores based on validation results.
        Not implemented in detail – placeholder.
        """
        with torch.no_grad():
            for i, logits in enumerate(model_outputs):
                if metric == "accuracy":
                    preds = logits.argmax(dim=-1)
                    acc = (preds == target).float().mean().item()
                    self.performance[i, task_id] = (
                        0.9 * self.performance[i, task_id] + 0.1 * acc
                    )
                # other metrics can be added

    def _get_dynamic_weights(
        self, task_id: int, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamic weights based on input‑specific confidence.
        Simplified version: uses model softmax probabilities.
        """
        weights = torch.zeros(self.num_models, device=self.device)
        with torch.no_grad():
            for i, model in enumerate(self.models):
                model.eval()
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                probs = F.softmax(logits, dim=-1)
                # Confidence = max probability across classes, averaged over batch
                conf = probs.max(dim=-1)[0].mean()
                weights[i] = conf
        return weights / (weights.sum() + 1e-5)

    def combine_logits(
        self,
        task_id: int,
        all_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
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
        if method == "simple_average":
            return all_logits.mean(dim=0)

        elif method == "weighted":
            # Use performance matrix as fixed weights
            perf_weights = self.performance[:, task_id]  # (M,)
            perf_weights = perf_weights / (perf_weights.sum() + 1e-5)
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
            dyn_weights = self._get_dynamic_weights(
                task_id, input_ids, attention_mask
            )  # (M,)
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
    print(
        "Use the class 'SmartWeightedEnsemble' directly in your training/evaluation script."
    )
