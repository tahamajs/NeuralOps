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
- gpt2
datasets:
- wikitext
metrics:
- perplexity
---

# GPT-2 Medium compressed with FWSVD (Rank 64) via NeuralOps

## Model Details
- **Base Architecture**: GPT-2 Medium (355M parameters, 24 layers, 16 heads, embedding dimension 1024)
- **Compression Method**: Fisher-Weighted SVD (`fwsvd`) from the canonical 72 compression engines in `neuralops`
- **Compressed Rank ($r$)**: 64
- **Parameters**: ~214M (39.7% parameter reduction on projection matrices)
- **Framework**: `neuralops` v1.3.7

## Evaluation Results
- **Dataset**: WikiText-2 (Language Modeling)
- **Perplexity (Dense Baseline)**: 19.82
- **Perplexity (FWSVD Post-KD Recovery)**: 21.45
- **Inference Latency Speedup**: 1.38x on NVIDIA RTX 3090 / 4070

## Usage with NeuralOps

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuralops import replace_attention

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("neuralops/gpt2-medium-fwsvd-rank64")

prompt = "Deep learning efficiency requires"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```
