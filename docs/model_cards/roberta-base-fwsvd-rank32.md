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
- roberta
datasets:
- glue
metrics:
- accuracy
---

# RoBERTa Base compressed with FWSVD (Rank 32) via NeuralOps

## Model Details
- **Base Architecture**: RoBERTa Base (125M parameters, 12 layers, 12 heads, hidden size 768)
- **Compression Method**: Fisher-Weighted SVD (`fwsvd`) from `neuralops`
- **Compressed Rank ($r$)**: 32 (24x compression ratio on Q/K/V/O projection matrices)
- **Parameters**: ~82M
- **Framework**: `neuralops` v1.3.7

## Evaluation Results
- **Dataset**: GLUE SST-2 (Binary Sentiment Classification)
- **Dense Baseline Accuracy**: 93.80%
- **Post-Compression Accuracy (Zero-Shot)**: 79.10%
- **Post-KD Recovery Accuracy**: 80.10%
- **Throughput Speedup**: 1.52x on NVIDIA RTX 3090 / 4070

## Usage with NeuralOps

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("neuralops/roberta-base-fwsvd-rank32")

inputs = tokenizer("This movie was remarkably thrilling!", return_tensors="pt")
logits = model(**inputs).logits
prediction = logits.argmax(-1).item()
print("Prediction:", prediction)
```
