# 🧩 Models – Loading, Compression & Attention Injection

**The central hub for loading, compressing, and running Transformer models with any registered attention.**

This package provides a complete, production‑ready pipeline that:
- Loads **any HuggingFace model** (DistilBERT, BERT, RoBERTa, GPT‑2, Llama, Mistral, etc.)
- Injects **any attention module** from the `ATTENTION_REGISTRY` – from minimal SVD to full AURORA
- Includes pre‑built **fully compressed models** (Hybrid‑SVD BERT, Compressed BERT)
- Offers **Unsloth integration** for 4‑bit loading and fast inference
- Is easily extensible to **new architectures**

All components work together with a single line of code, making it easy to experiment with different attention methods on different base models.

---

## 📁 Package Structure

```
framework/models/
├── __init__.py                     # Public API
├── model_loader.py                 # main entry: load model + inject attention
├── base_model_wrapper.py           # core logic to replace attention layers
├── hybrid_bert_svd.py              # BERT where every layer uses a trainable SVD
├── compressed_bert.py              # extreme compression: SVD embeddings + TransformerEncoder
├── unsloth_integration.py          # Unsloth (4‑bit) loading + attention swap
└── README.md
```

---

## 🚀 Quick Start

### Load a DistilBERT with AURORA attention

```python
from framework.models import load_model

model, tokenizer = load_model(
    model_name="distilbert-base-uncased",
    num_labels=2,
    attention_type="aurora",      # replaces full attention with AURORA
    rank=64,
    device="cuda"
)
```

That’s it – the model is immediately ready for fine‑tuning or inference with drastically reduced parameters and memory.

### Load a Llama‑3.2 with Unsloth + MinimalSVD attention

```python
model, tokenizer = load_model(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    num_labels=2,
    attention_type="minimal_nova",
    use_unsloth=True,
    device="cuda"
)
```

---

## ⚙️ Main Components

### 1. `model_loader.py` – The Easy Button

```python
load_model(model_name, num_labels, attention_type, rank, device, cache_dir, use_unsloth)
```

**Parameters:**
- `model_name` – any HuggingFace identifier
- `num_labels` – number of output classes (for classification)
- `attention_type` – registry key of the attention to inject (`"full"`, `"aurora"`, `"nova"`, etc.)
- `rank` – compression rank (passed to the injection function, default 64)
- `device` – torch device
- `cache_dir` – optional cache directory
- `use_unsloth` – if `True`, loads via Unsloth with 4‑bit quantization

This function automatically:
1. Downloads the model and tokenizer.
2. Replaces **all** self‑attention layers with the specified attention variant.
3. Moves the model to the desired device.

### 2. `base_model_wrapper.py` – The Engine

The function `replace_attention_in_model` is the core logic that inspects a model, iterates over its Transformer layers, and calls the appropriate injection function from `ATTENTION_REGISTRY`.

It currently supports:
- DistilBERT
- BERT
- RoBERTa
- GPT‑2
- GPT‑NeoX (used by Unsloth)
- And others by adding the appropriate logic to `_get_transformer_layers()` and `_get_attention_module()`.

**How to add a new architecture:**
- Extend `_get_transformer_layers` to return the list of encoder/decoder layers.
- Extend `_get_attention_module` to extract the self‑attention sub‑module from a layer.

### 3. `hybrid_bert_svd.py` – BERT with Trainable Low‑Rank Projections

A from‑scratch BERT variant where **every encoder layer** is replaced by a `DirectSVDEncoderLayer`. This layer:
- Replaces Q, K, V projections with **trainable low‑rank matrices** (`U`, `S`, `V`) that are learned from scratch (no pretrained weights).
- Uses standard scaled dot‑product attention on the projected tensors.
- Includes layer norm, residual connections, and FFN.
- Can be instantiated with a configurable rank (`config.svd_rank`).

```python
from framework.models.hybrid_bert_svd import HybridBertForSequenceClassification
from transformers import BertConfig

config = BertConfig.from_pretrained("bert-base-uncased", num_labels=3)
config.svd_rank = 64
model = HybridBertForSequenceClassification(config)
```

### 4. `compressed_bert.py` – Extreme Compression

Compresses a pretrained BERT model by:
- Applying **SVD truncation** to word, position, and token‑type embeddings.
- Replacing the encoder with simple `nn.TransformerEncoderLayer`s using the reduced embedding dimension.
- Adding a classifier head.

This gives a **much smaller model** that can be trained from scratch or fine‑tuned (though the original weights are used only for initialisation, not for exact reconstruction).

```python
from framework.models.compressed_bert import CompressedBertModel

model = CompressedBertModel(model_path="bert-base-uncased", rank=64, num_labels=2)
out = model(input_ids)
```

### 5. `unsloth_integration.py` – 4‑bit + Custom Attention

Loads any supported HuggingFace model using **Unsloth** (requires `pip install unsloth`), applies 4‑bit quantization, and then swaps attention.

```python
from framework.models.unsloth_integration import load_unsloth_model_with_attention

model, tokenizer = load_unsloth_model_with_attention(
    model_name="mistralai/Mistral-7B-v0.1",
    num_labels=2,
    attention_type="prism_full",
    rank=64
)
```

---

## 🔌 Supported Attention Types

All attention variants registered in `ATTENTION_REGISTRY` are supported. Here is a non‑exhaustive list:

| Key | Description |
|-----|-------------|
| `"full"` | Original BERT / DistilBERT attention |
| `"aurora"` | Full AURORA (10 techniques) |
| `"minimal_aurora"` | SVD‑only, zero‑shot |
| `"nova"` | NOVA (8 techniques) |
| `"minimal_nova"` | SVD‑only |
| `"prism_full"` | PRISM (6 techniques) |
| `"minimal_prism"` | SVD‑only |
| `"orion"` | SVD + head‑split kernel/full |
| `"phantom"` | CUR + LSH attention |
| `"oha"` | Optimized Hybrid Attention |
| `"flash3"` | FlashAttention‑3 kernel |
| `"lsh"` | Reformer LSH |
| `"kda"` | Kernelised Dynamic Decay |
| … | (and many more) |

To get a full list, call `ATTENTION_REGISTRY.keys()`.

---

## 📊 Benchmarks & Compression

Typical results when swapping attention in `distilbert-base-uncased`:

| Attention | Params (MP) | Memory | Accuracy (MRPC) |
|-----------|-------------|--------|-----------------|
| full       | 66.9M        | 100%   | 84.5%           |
| minimal_aurora | ~28M    | ~60%   | 83.9%           |
| nova       | ~15M        | ~45%   | 83.2%           |
| prism_full | ~14M        | ~40%   | 83.0%           |

*Figures are indicative – performance depends on fine‑tuning and dataset.*

---

## 🛠️ Customisation & Extension

### Using a Different Registry

The `base_model_wrapper` imports `ATTENTION_REGISTRY` from `framework.attentions.core.registry`. If you have your own registry, simply adjust the import.

### Adding a New Model Architecture

1. Open `base_model_wrapper.py`.
2. In `_get_transformer_layers`, add a branch that returns the layers of your model:

```python
elif hasattr(model, 'transformer'):               # GPT‑2
    return model.transformer.h
elif hasattr(model, 'model'):                      # LLaMA
    return model.model.layers
```

3. In `_get_attention_module`, return the attention sub‑module:

```python
elif hasattr(layer, 'self_attn'):
    return layer.self_attn
```

### Building a Completely Custom Model

Use the `hybrid_bert_svd` as a template:

- Define your own encoder layer that implements a specific attention pattern.
- Stack them in a `nn.ModuleList`.
- Add embeddings and a classifier.

---

## 📚 Example Pipeline

```python
from framework.models import load_model
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# 1. Load model with NOVA attention
model, tokenizer = load_model(
    "distilbert-base-uncased", num_labels=2,
    attention_type="nova", rank=64, device="cuda"
)

# 2. Tokenize dataset
dataset = load_dataset("glue", "mrpc")
def tokenize(batch):
    return tokenizer(batch["sentence1"], batch["sentence2"], truncation=True)
dataset = dataset.map(tokenize, batched=True)

# 3. Train
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./results", per_device_train_batch_size=16, num_train_epochs=3),
    train_dataset=dataset["train"],
)
trainer.train()
```

---

## 📜 References & Credits

This module is part of the **Efficient Transformer Toolkit**, which builds upon ideas from:
- HuggingFace Transformers
- Unsloth for fast inference
- SVD, low‑rank, and hybrid attention works (AURORA, NOVA, PRISM, etc.)

All code is designed for research and production.
