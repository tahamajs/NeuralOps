# Projection Factorized Attn

## Overview
Projection Factorized SVD Linear Attention.

## Configuration & Hyperparameters

| Parameter | Description |
|-----------|-------------|

## Usage Examples

```python
from Codes.framework.attentions import ProjectionFactorizedAttention
```
## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **84.52%** | -6.54% | 0.8525 | 1.20x | 6 | [svd-linear-attention-distilbert-sst2-projection-factorized-attn](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-projection-factorized-attn) | 2026-09-19 |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **53.90%** | -27.90% | 0.5017 | 1.20x | 6 | N/A | 2026-09-19 |