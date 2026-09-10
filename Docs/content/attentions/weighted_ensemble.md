# Weighted Ensemble

## Overview
Learns to weight model predictions based on task embeddings and pairwise
task similarity matrix.

Ensemble that combines multiple model predictions using various strategies:
    - simple_average
    - weighted by performance history
    - task-aware attention
    - hybrid (attention + dynamic confidence)

## Configuration & Hyperparameters

| Parameter | Description |
|-----------|-------------|
| `num_tasks` | TODO: Description for num_tasks |
| `hidden_dim` | TODO: Description for hidden_dim |
| `models` | TODO: Description for models |
| `num_tasks` | TODO: Description for num_tasks |
| `device` | TODO: Description for device |
| `trainable_dynamic` | TODO: Description for trainable_dynamic |

## Usage Examples

```python
from Codes.framework.attentions import TaskAwareAttention
```

## Benchmark Results

| Status | Accuracy | Val Loss | Peak VRAM | Inference Latency |
|--------|----------|----------|-----------|-------------------|
| Pending| Pending  | Pending  | Pending   | Pending           |

## Limitations & Future Work

TODO: Identify specific limitations and edge cases for this module.
