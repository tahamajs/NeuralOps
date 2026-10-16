# Tutorial 1: End-to-End Quickstart

This tutorial walks through loading a model, applying attention surgery, compressing with low-rank SVD, and serving with continuous batching.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import neuralops
from neuralops.integrations.tgi import TGIServingAdapter

# 1. Load HuggingFace checkpoint
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Inject Subquadratic Attention
model = neuralops.replace_attention(model, attention="cosformer")

# 3. Compress Linear Projections
compressed = neuralops.compress(model, method="fwsvd", rank=32)

# 4. Continuous Batching Serving
adapter = TGIServingAdapter(model=compressed, tokenizer=tokenizer, max_batch_size=8)
for token, text in adapter.streaming_generate("NeuralOps achieves", max_new_tokens=15):
    print(text, end="", flush=True)
print()
```
