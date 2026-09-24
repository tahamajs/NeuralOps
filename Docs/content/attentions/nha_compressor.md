# Nha Compressor

## Overview
## Configuration & Hyperparameters

| Parameter | Description |
|-----------|-------------|

## Usage Examples

```python
from Codes.framework.attentions import Module
```
## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **90.48%** | -0.58% | 0.9083 | 1.00x | 6 | [svd-linear-attention-distilbert-sst2-nha](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-nha) | 2026-09-19 |
