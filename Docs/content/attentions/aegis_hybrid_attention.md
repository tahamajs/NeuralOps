# Aegis Hybrid Attention

## Overview
## Configuration & Hyperparameters

| Parameter | Description |
|-----------|-------------|

## Usage Examples

```python
from Codes.framework.attentions import Module
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
| **distilbert-sst2** | ag_news | `PASS` | 51.00% | **59.50%** | +8.50% | 0.5023 | 1.18x | 6 | N/A | 2026-09-18 |
| **roberta-base** | ag_news | `PASS` | 74.50% | **35.00%** | -39.50% | 0.2763 | 1.12x | 12 | N/A | 2026-09-18 |
| **gpt2** | ag_news | `PASS` | 79.70% | **59.50%** | -20.20% | 0.5597 | 1.20x | 12 | N/A | 2026-09-18 |
| **distilbert-sst2** | sst2 | `PASS` | 90.60% | **81.88%** | -8.72% | 0.8229 | 1.21x | 6 | [svd-linear-attention-distilbert-sst2-aegis-hybrid-attention](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-aegis-hybrid-attention) | 2026-09-23 |