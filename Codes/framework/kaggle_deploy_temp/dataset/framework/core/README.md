
## Overview

The **Efficient Transformer Toolkit Core** provides a model‑agnostic foundation for compressing and injecting custom attention modules into **any HuggingFace model**—from DistilBERT to Llama, Gemma, Mistral, and beyond.  
It detects architectures automatically, handles weight copying between different naming conventions, manages grouped‑query attention (GQA) and RoPE, and offers a sequential compression pipeline.

---

## Directory Structure

```
framework/core/
├── __init__.py                  # Re‑exports all public symbols
├── registry.py                 # Global component registries (attention, compressors, etc.)
├── base.py                     # BaseAttention and BaseCompressor abstract classes
├── architecture.py             # Model‑agnostic layer/attention/FFN/embedding access
├── model_utils.py              # Calibration, weight copying, activation hooks
├── adapters.py                 # Model‑family specific helpers (weight names, GQA, RoPE)
├── attention_utils.py          # Unified attention injection (uses adapters)
├── compression_engine.py       # Multi‑step compression pipeline
├── model_wrapper.py            # High‑level UnifiedTransformerModel wrapper
├── utils.py                    # Device, seeding, formatting utilities
└── config.py                   # YAML config loader
```

---

## Key Concepts

### 1. Architecture Detection (`architecture.py`)
`detect_architecture(model)` returns a canonical string (`'distilbert'`, `'llama'`, `'bert'`, …).  
Other functions provide access to transformer layers, attention modules, FFN blocks, embeddings, and classifier heads **without hard‑coded paths**.

### 2. Model‑Family Adapters (`adapters.py`)
Each adapter knows:
- how to get the original Q/K/V/O weight tensors (and biases) from the old attention module,
- whether the model uses rotary embeddings (RoPE),
- the number of key‑value heads for grouped‑query attention (GQA).

Supported architectures: DistilBERT, BERT/RoBERTa, Llama, Mistral, Gemma, Phi.

### 3. Unified Attention Injection (`attention_utils.py`)
`inject_attention(model, "attention_type", rank=…)` replaces **every** attention module in the model with a variant registered in `ATTENTION_REGISTRY`.  
It automatically copies pretrained weights using the appropriate adapter.

### 4. Compression Pipeline (`compression_engine.py`)
`CompressionPipeline` applies a sequence of compressors (from `COMPRESSOR_REGISTRY`) one by one.  
Each compressor receives the model and a configuration dict.

### 5. Base Classes (`base.py`)
- `BaseAttention` – every custom attention must implement `forward(hidden_states, attention_mask, …)` and return `(output, attn_weights_or_None)`.
- `BaseCompressor` – defines `compress()` → `(compressed_model, stats)`.

### 6. Model Utilities (`model_utils.py`)
- `run_calibration_forward` – runs a forward pass on calibration texts.
- `copy_attention_weights` – generic weight‑copying across naming conventions.
- `replace_module` / `get_parent_and_child_name` – safe module replacement.
- `activation_hooks` – context manager to collect activations for activation‑aware compressors.

---

## Quick Start

### 1. Load any HuggingFace model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

### 2. Replace attention with a custom variant

```python
from framework.core import inject_attention
inject_attention(model, "galerkin", rank=32)
# or use the wrapper
from framework.core import UnifiedTransformerModel
wrapper = UnifiedTransformerModel(model)
wrapper.replace_attention("linformer", rank=64, copy_weights=True)
```

### 3. Compress the model sequentially

```python
from framework.core import compress_model

config = {
    "compressors": [
        {"method": "aasvd", "aasvd_rank": 64},
        {"method": "kmeans", "kmeans_clusters": 256},
    ],
    "seed": 42,
}
compressed_model = compress_model(model, config, tokenizer=tokenizer)
```

---

## Creating a Custom Attention Module

1. Inherit from `BaseAttention`.
2. Implement `forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs)` → `(output, None)`.
3. Accept `config` (and optionally `rank`, `pretrained_attn`, `rotary_emb`, etc.).
4. Register it with `@ATTENTION_REGISTRY.register("my_attention")`.

```python
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("my_attention")
class MyAttention(BaseAttention):
    def __init__(self, config, rank=None, pretrained_attn=None):
        super().__init__(config, rank)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # … initialise from pretrained_attn if given …
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # … compute …
        return output, None
```

Now `inject_attention(model, "my_attention")` will find and replace all attention layers.

---

## Creating a Custom Compressor

1. Inherit from `BaseCompressor`.
2. Implement `compress(self)` → `(compressed_model, stats_dict)`.
3. Register with `@COMPRESSOR_REGISTRY.register("my_compressor")`.

```python
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("my_compressor")
class MyCompressor(BaseCompressor):
    def __init__(self, model, **kwargs):
        super().__init__(model, name="my_compressor", **kwargs)
    def compress(self):
        # modify self.model …
        return self.model, self.get_compression_stats()
```

Then use it in the pipeline:

```python
compress_model(model, {"compressors": [{"method": "my_compressor", "param": value}]})
```

---

## Available Registries

| Registry | Purpose |
|----------|---------|
| `ATTENTION_REGISTRY` | Attention variants (Galerkin, Linformer, Performer, …) |
| `COMPRESSOR_REGISTRY` | Compression strategies (A3, AASVD, KMeans, FRanDI, …) |
| `METHOD_REGISTRY` | Training methods (LoRA, R‑Drop, …) |
| `AUX_REGISTRY` | Auxiliary modules (RoPE, SSM, token pruning, …) |
| `MODEL_REGISTRY` | Base model classes |
| `DATASET_REGISTRY` | Datasets (GLUE, SuperGLUE, …) |
| `LOSS_REGISTRY` | Custom loss functions |
| `TRAINER_REGISTRY` | Trainer classes |
| `VISUALIZER_REGISTRY` | Visualization utilities |

Use `ATTENTION_REGISTRY.keys()` or `list_all_components()` to see all registered items.

---

## Handling Masking Convention

All attention modules in this framework expect a **2D binary mask** of shape `(batch, length)` where `1` = keep, `0` = pad.  
The modules internally convert it to additive form `(1 - mask) * -1e9`.  
If your model already passes a 4D additive mask, the modules will still work because they check `mask.dim() == 2` before converting.

---

## Model‑Agnostic Helpers

- `detect_architecture(model)` – returns `'distilbert'`, `'llama'`, etc.
- `get_transformer_layers(model)` – list of transformer blocks (works for Llama, BERT, …).
- `get_attention_modules(model)` – list of `(name, module)` for all attention sub‑layers.
- `get_ffn_modules(model)` – same for feed‑forward networks.
- `get_embedding_module(model)` – returns the input word embedding layer.
- `get_classifier_head(model)` – returns the final classification layer (if present).

---

## Dependencies

- `torch`
- `transformers` (HuggingFace)
- `numpy`
- `pyyaml`
- `scipy` (for some compressors like SVD)

---

## Compatibility

Tested with:
- **DistilBERT**, **BERT**, **RoBERTa**
- **GPT‑2**
- **Llama‑2**, **Llama‑3**
- **Mistral**
- **Gemma**
- **Phi‑1.5**, **Phi‑3**

The framework automatically adapts to the naming differences among these architectures.

