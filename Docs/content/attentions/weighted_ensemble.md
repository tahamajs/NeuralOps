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

## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **90.02%** | -1.04% | 0.9045 | 1.00x | 6 | [svd-linear-attention-distilbert-sst2-weighted-ensemble](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-weighted-ensemble) | 2026-09-19 |
