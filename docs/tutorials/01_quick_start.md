# Quick Start Guide

Install NeuralOps:
```bash
pip install neuralops
```

## Basic Usage

```python
import torch
import torch.nn as nn
from neuralops.core.attention_utils import inject_attention

# Replace standard attention with high-throughput sliding window or linear attention
model = nn.Sequential(nn.Linear(64, 64))
injected_model = inject_attention(model, attention_type="sliding_window_attention")
```
