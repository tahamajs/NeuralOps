# Svd Sequence Attention

## Overview
## Configuration & Hyperparameters

| Parameter | Description |
|-----------|-------------|

## Usage Examples

```python
from Codes.framework.attentions import Module
```

## Benchmark Results

| Model | Dataset | Status | Baseline Acc | Injected Acc | F1 Score | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** | sst2 | `PASS_LOW` | 90.94% | **51.26%** | 0.5737 | 329.4 MB |
| **RoBERTa** | sst2 | `PASS_LOW` | 75.46% | **50.00%** | 0.6572 | 655.8 MB |
| **GPT-2** | sst2 | `PASS_LOW` | 57.91% | **53.10%** | 0.6832 | 569.8 MB |

## Limitations & Future Work

TODO: Identify specific limitations and edge cases for this module.
