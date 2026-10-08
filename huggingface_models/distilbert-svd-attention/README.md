---
language: en
license: mit
library_name: transformers
tags:
  - neuralops
  - svd
  - attention
  - compression
base_model: distilbert-base-uncased
---

# DistilBERT with SVD Attention

This model is a `distilbert-base-uncased` checkpoint with the multi-head self-attention layers replaced by SVD-factorized subquadratic attention from `NeuralOps`.

## Model Details

- **Base model:** `distilbert-base-uncased`
- **Attention type:** `svd_linear_attention`
- **Compression rank:** 64
- **Parameters:** ~56M (vs 67M original dense backbone)
- **Framework:** [NeuralOps](https://github.com/tahamajs/SVD_linear_Attention)

## Usage

```python
from transformers import AutoModel, AutoTokenizer
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.replace_attention import replace_attention_in_model

# Load model and tokenizer
model = AutoModel.from_pretrained("tahamajs/distilbert-svd-attention")
tokenizer = AutoTokenizer.from_pretrained("tahamajs/distilbert-svd-attention")

inputs = tokenizer("NeuralOps is fast!", return_tensors="pt")
outputs = model(**inputs)
print("Output shape:", outputs.last_hidden_state.shape)
```

## Training & Fine-Tuning

Fine-tuned on SST-2 sentiment classification with a recovery learning rate of $2 \times 10^{-5}$ for 3 epochs with AdamW.

## Citation

```bibtex
@software{neuralops2026,
  title = {NeuralOps: Unified Framework for Subquadratic Attention and Deep Model Compression},
  author = {Majlesi, Taha},
  year = {2026},
  url = {https://github.com/tahamajs/SVD_linear_Attention}
}
```
