# Weighted Ensemble

## Overview

**Weighted Ensemble** combines multi-model predictions across heterogeneous downstream tasks using dynamic weighting strategies based on task embeddings, pairwise task similarity metrics, and prediction confidence:

- `simple_average`: Uniform average of prediction logits across models.
- `performance_history`: Weights models using exponentially moving historical performance logs.
- `task_aware_attention`: Evaluates scaled dot-product attention over task embeddings with a learned pairwise task similarity matrix.
- `hybrid`: Convex combination ($60/40$) of task-aware attention and instance-level dynamic softmax confidence.

---

## Configuration & Hyperparameters

### `SmartWeightedEnsemble` Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `models` | `List[nn.Module]` | **Required** | Constituent PyTorch models to ensemble |
| `num_tasks` | `int` | `1` | Total number of distinct downstream task domains |
| `device` | `Optional[torch.device]` | `None` | Target compute device for ensemble evaluation |
| `trainable_dynamic` | `bool` | `False` | Enables gradient flow through dynamic confidence forward passes |

### `TaskAwareAttention` Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_tasks` | `int` | **Required** | Dimension of the task similarity matrix $\Sigma \in \mathbb{R}^{T \times T}$ |
| `hidden_dim` | `int` | `64` | Embedding dimensionality for projected task vectors |

---

## Usage Examples

```python
import torch
from transformers import AutoModelForSequenceClassification
from framework.attentions.weighted_ensemble import SmartWeightedEnsemble

# 1. Initialize constituent models
model1 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model2 = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 2. Instantiate ensemble
ensemble = SmartWeightedEnsemble(
    models=[model1, model2],
    num_tasks=2,
    trainable_dynamic=False,
)

# 3. Forward pass with task-aware attention
input_ids = torch.randint(0, 1000, (4, 128))
attention_mask = torch.ones((4, 128), dtype=torch.long)
logits = ensemble(
    input_ids=input_ids,
    attention_mask=attention_mask,
    task_id=0,
    model_task_ids=[0, 1],
    method="hybrid",
)
print("Ensemble output logits shape:", logits.shape)
```

---

## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **90.02%** | -1.04% | 0.9045 | 1.00x | 6 | [svd-linear-attention-distilbert-sst2-weighted-ensemble](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-weighted-ensemble) | 2026-09-19 |
