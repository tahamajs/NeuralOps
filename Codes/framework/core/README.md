# Core Subsystem (`Codes/framework/core`)

The `core` subsystem provides the foundational infrastructure, model wrapping abstractions, dynamic attention injection engine, compression runtime, and extensible registry mechanisms powering the framework.

---

## 🏗 Architecture Overview

The core subsystem orchestrates model lifecycle management and layer replacement:
- **Registry Engine (`registry.py`)**: Type-safe component registries (`ATTENTION_REGISTRY`, `COMPRESSION_REGISTRY`, `ADAPTER_REGISTRY`) enabling decoupled module registration and dynamic discovery.
- **Attention Injection Engine (`attention_utils.py`)**: Universal layer replacement utility with automatic projection discovery (`q`, `k`, `v`, `o`), fused QKV tensor slicing (MHA/GQA/MQA), and symmetric SVD weight initialization ( = U \sqrt{S}, V_r = \sqrt{S} V^T$).
- **Architecture Detection & Adapters (`architecture.py`, `adapters.py`)**: Automatic identification of transformer families (BERT, RoBERTa, DistilBERT, GPT-2, LLaMA, Mistral, Falcon) and normalized weight/bias extraction.
- **Model Wrappers & Base Classes (`base.py`, `model_wrapper.py`)**: Standardized interfaces (`BaseModelWrapper`, `BaseAttentionModule`, `BaseCompressor`) ensuring unified forward signatures and metric hooks.
- **Hardware & Memory Optimizations (`memory.py`)**: Multi-device memory management, cache flushing (CUDA, MPS, CPU), and gradient checkpointing helpers.

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `registry.py` | Central `Registry` class and global decorator-based registries. |
| `attention_utils.py` | `inject_attention()`, fused QKV slicing, and RoPE coordinate preservation. |
| `architecture.py` | Architecture detection (`detect_architecture`) and layer extraction (`get_attention_modules`). |
| `adapters.py` | Model-specific adapters extracting and mapping projection layers. |
| `model_wrapper.py` | `UniversalModelWrapper` managing wrapped models, attention hooks, and compression passes. |
| `compression_engine.py` | Pipeline orchestrator applying spectral and low-rank compression across layers. |
| `memory.py` | Device memory tracking, VRAM allocation profiling, and cache cleanup. |

---

## 🚀 Quick Start & Usage

### 1. Dynamic Attention Injection
```python
from transformers import AutoModelForSequenceClassification
from framework.core.attention_utils import inject_attention

# Load any HuggingFace transformer model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Inject SVD Linear Attention across all attention layers
inject_attention(model, attention_type="svd_linear_attention", rank=32, copy_weights=True)
```

### 2. Registering Custom Attention Mechanisms
```python
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("custom_linear_attn")
class CustomLinearAttention(nn.Module):
    def __init__(self, config, rank: int = 32, **kwargs):
        super().__init__()
        self.dim = config.hidden_size
        self.rank = rank

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Attention forward computation
        return hidden_states, None
```
