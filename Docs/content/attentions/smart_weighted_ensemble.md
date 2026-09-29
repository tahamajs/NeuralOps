# Smart Weighted Ensemble: Task-Aware Attention & Dynamic Confidence Fusion

## 1. Abstract & Theoretical Motivation

Combining multi-model predictions across heterogeneous downstream tasks typically relies on static averaging or scalar voting weights. However, static ensemble strategies fail to account for task-domain alignment or sample-specific predictive uncertainty.

**Smart Weighted Ensemble (`SmartWeightedEnsemble`)** provides a multi-model fusion framework powered by Task-Aware Attention (`TaskAwareAttention`) and dynamic instance-level confidence weighting. By embedding task IDs into a shared metric subspace and incorporating a learnable pairwise task similarity matrix $\Sigma \in \mathbb{R}^{T \times T}$, the architecture computes task-conditioned attention weights $w^{\text{attn}}$ over model logit streams $\{L_m\}_{m=1}^M$. Combined with real-time per-sample softmax confidence scores $w^{\text{dyn}}$, the hybrid ensemble dynamically adapts model weights based on task relevance and sample certainty.

*(Note: `SmartWeightedEnsemble` is an ensemble fusion utility, not a standard attention layer registered under `ATTENTION_REGISTRY`).*

---

## 2. Architecture & Data Flow

```
Inputs: input_ids [B, N], attention_mask [B, N], task_id (int), model_task_ids (List[int])
   │
   ├── Forward each sub-model in self.models:
   │     ├── Model 1 (input_ids, attention_mask) ──> Logits_1 [B, C]
   │     ├── Model 2 (input_ids, attention_mask) ──> Logits_2 [B, C]
   │     └── ... Model M (input_ids, attention_mask) ──> Logits_M [B, C]
   │
   ├── Stack Logits: all_logits = stack([Logits_1, ..., Logits_M]) ──> [M, B, C]
   │
   ├── Logit Combination Strategy (method):
   │     ├── 'simple_average':
   │     │      Output = all_logits.mean(dim=0) ──> [B, C]
   │     │
   │     ├── 'weighted':
   │     │      Output = sum_m (perf_weights[m] * all_logits[m]) ──> [B, C]
   │     │
   │     ├── 'task_aware_attention':
   │     │      Compute w_attn via TaskAwareAttention(task_id, all_logits, model_task_ids)
   │     │      Output = sum_m (w_attn[m] * all_logits[m]) ──> [B, C]
   │     │
   │     └── 'hybrid' (Default):
   │            w_attn = TaskAwareAttention(task_id, all_logits, model_task_ids)
   │            w_dyn  = DynamicConfidence(all_logits)
   │            Output = 0.6 * Logits_attn + 0.4 * Logits_dyn ──> [B, C]
   │
   └── Return Fused Logits: [B, C]
```

### Execution Trace & Tensor Shapes
1. **Model Forward Passes**: Each model $m \in \{1, \dots, M\}$ consumes tokens `[B, N]` and produces logits `[B, C]`.
2. **Logit Stacking**: Logits are stacked into `all_logits: [M, B, C]` on `self.device`.
3. **Weight Computation**:
   - Task Attention: Evaluates dot product between query task embedding $e_t$ `[1, d_emb]` and model prototype task embeddings $e_m$ `[M, d_emb]`, added to similarity bias $\Sigma_{m, t}$, yielding `[1, M]`.
   - Dynamic Confidence: Computes average peak Softmax confidence `[M]` across the batch.
4. **Logit Blending**: Convex combination yields fused output logits `[B, C]`.

---

## 3. Mathematical Derivations & Formulations

Let $M$ be the number of models, $T$ the total number of tasks, $B$ the batch size, and $C$ the number of output classes. Let $\{L_m\}_{m=1}^M \in \mathbb{R}^{M \times B \times C}$ denote per-model logits.

### 1. Task-Aware Attention Weighting
Let $E \in \mathbb{R}^{T \times d_{\text{emb}}}$ be the task embedding matrix. For target task $t \in \{0, \dots, T-1\}$ and model prototype assignments $\{t_1, \dots, t_M\}$:
\[
e_t = W_Q E[t] \in \mathbb{R}^{1 \times d_{\text{emb}}}, \quad e_m = W_K E[t_m] \in \mathbb{R}^{M \times d_{\text{emb}}}
\]
Raw attention scores combine scaled dot-product affinity with the task similarity bias parameter $\Sigma \in \mathbb{R}^{T \times T}$:
\[
S_m = \frac{e_t \cdot e_m^\top}{\sqrt{d_{\text{emb}}}} + \Sigma_{t_m, t} \in \mathbb{R}^{1 \times M}
\]
\[
w_m^{\text{attn}} = \operatorname{Softmax}_m (S_m) = \frac{\exp(S_m)}{\sum_{j=1}^M \exp(S_j)}
\]
\[
L^{\text{attn}} = \sum_{m=1}^M w_m^{\text{attn}} \cdot L_m \in \mathbb{R}^{B \times C}
\]

### 2. Dynamic Instance Confidence Weighting
For each model $m$, instance-level confidence $c_m$ is determined by the mean maximum class probability:
\[
p_{m, b, c} = \operatorname{Softmax}_c(L_{m, b, c}) = \frac{\exp(L_{m, b, c})}{\sum_{c'=1}^C \exp(L_{m, b, c'})}
\]
\[
c_m = \frac{1}{B} \sum_{b=1}^B \max_{c \in \{1, \dots, C\}} p_{m, b, c}
\]
\[
w_m^{\text{dyn}} = \frac{c_m}{\sum_{j=1}^M c_j + \epsilon}
\]
\[
L^{\text{dyn}} = \sum_{m=1}^M w_m^{\text{dyn}} \cdot L_m \in \mathbb{R}^{B \times C}
\]

### 3. Hybrid Logit Fusion
\[
L^{\text{hybrid}} = 0.6 \cdot L^{\text{attn}} + 0.4 \cdot L^{\text{dyn}} \in \mathbb{R}^{B \times C}
\]

---

## 4. Configuration & Hyperparameters

### `SmartWeightedEnsemble` Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `models` | `List[nn.Module]` | **Required** | List of constituent PyTorch models to ensemble. |
| `num_tasks` | `int` | `1` | Total number of distinct downstream task IDs ($T$). |
| `device` | `Optional[torch.device]` | `None` | Target device for ensemble buffers and execution. |
| `trainable_dynamic` | `bool` | `False` | Whether to maintain gradient tracking during dynamic confidence forward passes. |

### `TaskAwareAttention` Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_tasks` | `int` | **Required** | Total number of task embedding indices ($T$). |
| `hidden_dim` | `int` | `64` | Embedding dimensionality for task projections ($d_{\text{emb}}$). |

*(Note: `SmartWeightedEnsemble` is not registered under `ATTENTION_REGISTRY`; it is an ensemble wrapper operating over model logits).*

---

## 5. Usage & Integration Examples

```python
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from framework.attentions.weighted_ensemble import SmartWeightedEnsemble

# 1. Initialize constituent models
model1 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model2 = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 2. Wrap into SmartWeightedEnsemble
ensemble = SmartWeightedEnsemble(
    models=[model1, model2],
    num_tasks=2,
    trainable_dynamic=False,
)

# 3. Forward pass with hybrid task-aware fusion
input_ids = torch.randint(0, 1000, (4, 128))
attention_mask = torch.ones((4, 128), dtype=torch.long)
logits = ensemble(
    input_ids=input_ids,
    attention_mask=attention_mask,
    task_id=0,
    model_task_ids=[0, 1],
    method="hybrid",
)
print("Ensemble fused logits:", logits.shape)
```

---

## 6. Empirical Benchmark Results & Verification

### Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert** | sst2 | `PASS` | N/A | **90.48%** | N/A | 0.9077 | 1.00x | 0 | [svd-linear-attention-distilbert-smart_weighted_ensemble](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-smart_weighted_ensemble) | 2026-09-29 |
| **roberta** | sst2 | `PASS` | N/A | **75.11%** | N/A | 0.7586 | 1.00x | 0 | [svd-linear-attention-roberta-smart_weighted_ensemble](https://huggingface.co/Hooshaai/svd-linear-attention-roberta-smart_weighted_ensemble) | 2026-09-29 |
| **gpt2** | sst2 | `PASS` | N/A | **60.78%** | N/A | 0.6354 | 1.00x | 0 | [svd-linear-attention-gpt2-smart_weighted_ensemble](https://huggingface.co/Hooshaai/svd-linear-attention-gpt2-smart_weighted_ensemble) | 2026-09-29 |


| Evaluation Suite | Ensemble Composition | Aggregation Method | Benchmark Metric | Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GLUE Suite | BERT + RoBERTa | Simple Average | Accuracy | Pending actual benchmark results | Pending Sweep |
| GLUE Suite | BERT + RoBERTa | Task-Aware Attention | Accuracy | Pending actual benchmark results | Pending Sweep |
| GLUE Suite | BERT + RoBERTa | Hybrid Fusion (60/40) | Accuracy | Pending actual benchmark results | Pending Sweep |

---

## 7. Architectural Bottlenecks & Failure Modes

1. **Sequential Model Forward Overhead**: In eager execution mode, iterating across $M$ separate transformer backbones multiplies latency by $M\times$. Deploying model concurrency across separate CUDA streams or asynchronous worker pools is recommended for large ensembles.
2. **Uncalibrated Softmax Overconfidence**: Deep neural networks can assign near-1.0 confidence to misclassified samples. Uncalibrated models can skew dynamic confidence weights $w_m^{\text{dyn}}$, dominating better-calibrated models. Temperature scaling per model is advised when calibration varies significantly.
3. **Task Prototype ID Assignment Mismatch**: If `model_task_ids` is omitted or incorrectly set to arbitrary round-robin indices rather than the true task domain of each model, task similarity matrix $\Sigma$ cannot establish meaningful domain-expert alignments, degrading attention weights toward uniform distribution.

---

## 8. Proposed Future Optimizations

- **Asynchronous Concurrent CUDA Streams**: Execute constituent model forward passes across non-blocking CUDA streams to parallelize GPU execution.
- **Batched Fused Logit Reduction**: A lightweight GPU kernel to compute class-wise Softmax confidence and convex logit weighting directly in SRAM scratchpads.
- **Temperature Calibration Tuning**: Calibrate temperature scalar parameters $\tau_m$ per model on a validation split prior to dynamic confidence weighting.

---

## 9. Complete Source Code Implementation (`weighted_ensemble.py`)

```python
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
        valid_model_tasks_tensor = torch.tensor(valid_model_tasks, device=device)
        bias = self.sim_mat[valid_model_tasks_tensor, task_id].unsqueeze(0)  # (1, M)
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

        if raw_weights.numel() == 0 or raw_weights.sum() <= 0:
            return torch.ones(self.num_models, device=self.device) / self.num_models
        return raw_weights / raw_weights.sum()

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
        stacked = stacked.to(self.device)

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
```