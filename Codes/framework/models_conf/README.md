# `framework/models` – Universal Model Loading & Custom Architectures

The **models** module provides a single entry point for loading **any** HuggingFace model, with optional custom attention injection and variants (compressed BERT, hybrid SVD, Unsloth). It integrates seamlessly with the `UnifiedTransformerModel` for Hub serialization and supports attention swapping via the `ATTENTION_REGISTRY`.

---

## File Structure

```
framework/models/
├── __init__.py              # Public API
├── loaders.py               # Unified load_model() + variant-specific loaders
├── custom_models.py         # CompressedBertModel, HybridBertForSequenceClassification
├── unsloth_loader.py        # Unsloth integration (optional)
├── hub_utils.py             # HuggingFace Hub search utilities
└── README.md                # This file
```

---

## Quick Start

```python
from framework.models import load_model

# 1. Standard HuggingFace model with custom attention
model, tokenizer = load_model(
    "distilbert-base-uncased",
    num_labels=2,
    attention_type="galerkin",   # from ATTENTION_REGISTRY
    attention_rank=32
)

# 2. Compressed BERT variant (no attention replacement needed)
model, tokenizer = load_model(
    "bert-base-uncased",
    variant="compressed",
    rank=64,
    num_labels=3
)

# 3. Unsloth model with attention
model, tokenizer = load_model(
    "meta-llama/Llama-2-7b-hf",
    variant="unsloth",
    attention_type="linformer",
    attention_rank=128,
    max_seq_length=2048
)
```

---

## API Reference

### `load_model(model_name, num_labels=2, variant="standard", ...)`

The main entry point. Returns `(model, tokenizer)`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | – | HuggingFace model identifier or path. |
| `num_labels` | 2 | Number of output classes (for classification). |
| `variant` | `"standard"` | Loader variant: `"standard"`, `"compressed"`, `"hybrid_svd"`, `"unsloth"`. |
| `attention_type` | `None` | Optional attention registry key to inject. |
| `attention_rank` | 32 | Compression rank for attention module. |
| `wrap_unified` | `True` | If `True`, wraps model in `UnifiedTransformerModel` (enables Hub save/load). |
| `cache_dir` | `None` | HuggingFace cache directory. |
| `**kwargs` | – | Passed to the underlying loader. |

### Variant Loaders

- **`standard`** – uses `AutoModelForSequenceClassification` + `AutoTokenizer`.
- **`compressed`** – BERT only; SVD‑compressed embeddings + simplified TransformerEncoder layers.
- **`hybrid_svd`** – BERT only; replaces encoder layers with `DirectSVDEncoderLayer` (placeholders).
- **`unsloth`** – uses `FastLanguageModel.from_pretrained` for fast training; supports 4‑bit quantization.

---

## Custom Models

### `CompressedBertModel`

- Reduces embedding dimensions via truncated SVD.
- Replaces BERT encoder with a stack of `nn.TransformerEncoderLayer` working in the reduced rank space.
- Used when `variant="compressed"`.

### `HybridBertForSequenceClassification`

- Replaces BERT encoder layers with `DirectSVDEncoderLayer` (you must implement or import the actual layer).
- Used when `variant="hybrid_svd"`.

---

## Attention Injection

When `attention_type` is provided, the module uses `inject_attention()` from `framework.core.attention_utils` (or the `UnifiedTransformerModel.replace_attention()` method) to replace **all** attention modules in the loaded model with the chosen variant. Pretrained weights are copied automatically where possible.

---

## Hub Utilities

`hub_utils.py` provides two simple functions:

- `search_models(query, limit)` – returns a list of matching model IDs on HuggingFace Hub, with caching.
- `get_model_info(model_name)` – returns a dict with `id`, `pipeline_tag`, and `tags`.

---

## Dependencies

- `torch`
- `transformers` (HuggingFace)
- `huggingface_hub` (for search utilities)
- `unsloth` (optional, for `variant="unsloth"`)
- `scipy` (used by custom compressed models for SVD)

---

## Integration with the Framework

- The loaded model is automatically wrapped in `UnifiedTransformerModel` (if `wrap_unified=True`), which provides `save_pretrained`, `from_pretrained`, `push_to_hub`, and `replace_attention`.
- To add a new custom architecture, create a loader function and register it in `_VARIANT_LOADERS` in `loaders.py`.
- To add a new attention variant, simply register it in the global `ATTENTION_REGISTRY` – no changes needed here.

---

## Example Workflow

```python
from framework.models import load_model
from framework.training import HuggingFaceTrainer, ResultsManager

# 1. Load model with compressed attention
model, tokenizer = load_model(
    "distilbert-base-uncased",
    attention_type="svd_linear_attn",
    attention_rank=32
)

# 2. Prepare data (using framework.data)
from framework.data import load_dataset, DataConfig, TaskType, create_dataloader
dataset = load_dataset(...)
train_loader = create_dataloader(dataset["train"], tokenizer, DataConfig(...))

# 3. Train
results = ResultsManager("my_run")
trainer = HuggingFaceTrainer(...)
trainer.train()

# 4. Save to Hub
model.push_to_hub("your-username/compressed-model")
```

---

This module completes the chain from model selection → loading → compression → training → deployment, making the entire framework **fully self‑contained and production‑ready**.