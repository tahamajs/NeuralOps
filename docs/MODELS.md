# NeuralOps HuggingFace Transformers Architecture Guide

NeuralOps provides full HuggingFace Transformers compatibility, enabling seamless use of 159 attention mechanisms and 73 model compression engines via standard `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM` classes.

## Features
- **Auto Registration**: `neuralops` is registered as a recognized `model_type` at import time.
- **Dynamic Hybrid Spec**: Configure custom attention mechanisms and compression techniques per layer via `config.attn` and `config.compression`.
- **KV Cache & Generation**: Supports autoregressive generation (`model.generate()`), greedy decoding, sampling, beam search, and DynamicCache.
- **Safe Hub Operations**: Push and load models to/from HuggingFace Hub with automated secret leakage detection and model card generation.

## 3-Line Quickstart

```python
from transformers import AutoModelForCausalLM
import neuralops.models

# 1. Build a Causal LM using any registered subquadratic attention mechanism
config = neuralops.models.NeuralOpsConfig(
    vocab_size=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    attention_type="cosformer",  # Or 'rpc_attention', 'galerkin_attention', etc.
)
model = AutoModelForCausalLM.from_config(config)

# 2. Text Generation
output_ids = model.generate(input_ids, max_new_tokens=50)

# 3. Serialization
model.save_pretrained("./my_neuralops_model")
reloaded_model = AutoModelForCausalLM.from_pretrained("./my_neuralops_model")
```

## Hybrid Layer Configurations
Specify different attention types across transformer depth:

```python
config = neuralops.models.NeuralOpsConfig(
    num_hidden_layers=6,
    attention_type="cosformer",  # Default for layers
    attn=[
        {"layers": [0, 2], "type": "rpc_attention", "svd_rank": 32},
        {"layers": [1, 3], "type": "galerkin_attention"},
        {"layers": [4, 5], "type": "cosformer"},
    ]
)
model = AutoModelForCausalLM.from_config(config)
```
EOF
