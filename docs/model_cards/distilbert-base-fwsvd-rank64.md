---
language:
- en
license: mit
library_name: transformers
tags:
- neuralops
- linear-attention
- compression
- fwsvd
- distilbert
datasets:
- glue
metrics:
- accuracy
---

# DistilBERT Base compressed with FWSVD (Rank 64) via NeuralOps

## Model Details
- **Base Architecture**: DistilBERT Base (66M parameters, 6 layers, 12 heads, hidden size 768)
- **Compression Method**: Fisher-Weighted SVD (`fwsvd`) from `neuralops`
- **Compressed Rank ($r$)**: 64
- **Parameters**: ~48M
- **Framework**: `neuralops` v1.3.7

## Evaluation Results
- **Dataset**: GLUE SST-2 (Sentiment Analysis)
- **Dense Baseline Accuracy**: 91.06%
- **Post-Compression Accuracy**: 83.20%
- **Post-KD Recovery Accuracy**: 86.40%
- **Inference Latency**: 0.42x of baseline latency (2.38x speedup)

## Usage with NeuralOps

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("neuralops/distilbert-base-fwsvd-rank64")

inputs = tokenizer("Outstanding execution and great performance.", return_tensors="pt")
logits = model(**inputs).logits
print("Predicted class:", logits.argmax(-1).item())
```
