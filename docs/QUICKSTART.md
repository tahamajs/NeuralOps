# Quickstart Guide

## Installation

```bash
pip install neuralops
```

## 1. Drop-In Attention Surgery

```python
from transformers import AutoModelForSequenceClassification
import neuralops

model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2")
# Replace standard quadratic attention with subquadratic SVD Linear Attention
model = neuralops.replace_attention(model, attention="svd_linear_attention", rank=64)
```

## 2. High-Performance Model Compression

```python
import neuralops

# Compress model weights using Fisher-Weighted SVD
compressed_model = neuralops.compress(model, method="fwsvd", rank=32)
```

## 3. Continuous Batching Serving

```python
from neuralops.integrations.tgi import TGIServingAdapter

serving_engine = TGIServingAdapter(model=compressed_model, max_batch_size=32)
completions = serving_engine.continuous_batching_generate(
    ["Explain subquadratic attention:", "What is low-rank factorization?"],
    max_new_tokens=25,
)
```
