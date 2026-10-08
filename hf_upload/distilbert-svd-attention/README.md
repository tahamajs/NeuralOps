---
language: en
license: mit
library_name: transformers
tags:
  - neuralops
  - svd
  - attention
  - compression
  - distilbert
  - sst2
datasets:
  - glue
metrics:
  - accuracy
base_model: distilbert-base-uncased
---

# DistilBERT with SVD Attention

This model is a [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased) 
with the attention layers replaced by **SVD-factorized attention** from 
[NeuralOps](https://github.com/tahamajs/SVD_linear_Attention).

## Model Details

- **Base model:** distilbert-base-uncased
- **Attention type:** `svd_linear_attention`
- **Compression rank:** 64
- **Parameters:** 56M (vs 67M original, 1.20× compression)
- **Task:** Binary classification (SST-2)

## Performance

| Metric | Baseline | This Model |
|:---|:---:|:---:|
| Accuracy (SST-2) | 91.20% | 86.93% |
| Parameters | 67M | 56M |
| Compression | 1.00× | 1.20× |

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "tahamajs/distilbert-svd-attention"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("This movie was great!", return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
print(f"Predicted: {predicted_class}")  # 1 = positive
```

## How it was made

```python
from transformers import AutoModelForSequenceClassification
from neuralops.compression_engine.svd import ASVDCompressor

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
compressor = ASVDCompressor(model)
compressed = compressor.compress(target_rank=64)
```

## Citation

```bibtex
@software{neuralops2026,
  title = {NeuralOps: Unified Framework for Subquadratic Attention and Deep Model Compression},
  author = {Majlesi, Taha},
  year = {2026},
  url = {https://github.com/tahamajs/SVD_linear_Attention}
}
```
