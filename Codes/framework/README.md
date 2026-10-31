# ⚙️ SVD Linear Attention Framework Core

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Registries](https://img.shields.io/badge/Registries-106%20Attn%20%7C%2084%20Comp-blueviolet.svg)](./core/registry.py)

The core PyTorch library for the SVD Linear Attention framework, providing **106 custom attention mechanisms**, **84 post-training compression engines**, model wrapping and attention injection, distributed trainers, dataset loaders, and benchmark visualization tools.

---

## 📑 Table of Contents
1. [Core Subsystems Overview](#-core-subsystems-overview)
2. [Base Classes & Registry Design](#-base-classes--registry-design)
3. [Zero-Shot Attention Injection (`inject_attention`)](#-zero-shot-attention-injection-inject_attention)
4. [Compression Pipelines (`compress_model`)](#-compression-pipelines-compress_model)
5. [Trainers & Distributed Execution](#-trainers--distributed-execution)
6. [Benchmark Datasets & Synthetic Generation](#-benchmark-datasets--synthetic-generation)
7. [Visualization & Chart Generation](#-visualization--chart-generation)
8. [Testing & Validation Suites](#-testing--validation-suites)

---

## 🏗️ Core Subsystems Overview

```
Codes/framework/
├── attentions/             # 106 Custom Attention modules (SVD, Mamba, MLA, GQA, Flash3, etc.)
├── compression_engine/     # 84 Compression algorithms (AASVD, SliceGPT INT4, DeltaKV, AWQ, etc.)
├── core/                   # Core abstractions, registries, and model wrappers
│   ├── base.py             # BaseAttention, BaseCompressor abstract base classes
│   ├── registry.py         # ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
│   ├── attention_utils.py  # inject_attention() for zero-shot layer replacement
│   ├── architecture.py     # Multi-architecture transformer layer detector
│   ├── compression_engine.py # Chained CompressionPipeline executor
│   └── model_wrapper.py    # UnifiedTransformerModel wrapper
├── trainer/                # HuggingFaceTrainer, CustomLoopTrainer, FSDP/DDP hooks
├── data/                   # 22 benchmark datasets (GLUE, WikiText-2, C4, Synthetic)
├── visualization/          # Heatmaps, Pareto frontiers, and HTML report generators
├── analysis/               # Spectral energy decay, Hessian, and outlier analyzers
└── results_saver/          # Structured SQLite, Parquet, and JSON exporters
```

---

## 🧩 Base Classes & Registry Design

### 1. `BaseAttention` (`core/base.py`)
All custom attention mechanisms inherit from `BaseAttention`:
```python
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("svd_linear")
class SVDLinearAttention(BaseAttention):
    def __init__(self, config, rank: int = 32):
        super().__init__(config, rank=rank)
        self.U = nn.Parameter(torch.empty(self.hidden_size, rank))
        self.V = nn.Parameter(torch.empty(rank, self.hidden_size))
        self.reset_parameters()

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        latent = F.linear(hidden_states, self.V)
        output = F.linear(latent, self.U)
        return output, None
```

### 2. `BaseCompressor` (`core/base.py`)
All model compressors inherit from `BaseCompressor`:
```python
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("aasvd")
class AASVDCompressor(BaseCompressor):
    def __init__(self, model, rank: int = 64, **kwargs):
        super().__init__(model, name="aasvd", **kwargs)
        self.rank = rank

    def compress(self):
        # Activation-aware SVD decomposition logic
        # Returns (compressed_model, statistics_dict)
        return self.model, self.get_compression_stats()
```

---

## 💉 Zero-Shot Attention Injection (`inject_attention`)

The `inject_attention` utility in `core/attention_utils.py` automatically detects the underlying model architecture (Llama, Mistral, BERT, DistilBERT, GPT-2, Gemma, T5) and replaces its multi-head attention layers with the specified registered module:

```python
from transformers import AutoModelForSequenceClassification
from framework.core.attention_utils import inject_attention

# 1. Load any standard HuggingFace transformer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# 2. Swap standard attention with SVD Linear Attention across all layers
model = inject_attention(model, attn_type="svd_linear", rank=64)
```

---

## 🗜️ Compression Pipelines (`compress_model`)

The `CompressionPipeline` in `core/compression_engine.py` orchestrates multiple compression methods sequentially:

```python
from framework.core.compression_engine import compress_model

pipeline_config = {
    "seed": 42,
    "compressors": [
        {"method": "aasvd", "rank": 64, "activation_smoothing": True},
        {"method": "rotational_slicegpt_int4", "bits": 4, "group_size": 128}
    ]
}

compressed_model = compress_model(model, pipeline_config, tokenizer=tokenizer)
```

---

## 🏋️ Trainers & Distributed Execution

The `framework.trainer` module provides unified training wrappers:
- **`HuggingFaceTrainer`**: Integrates with standard HuggingFace `TrainingArguments` and metrics.
- **`CustomLoopTrainer`**: Low-level, high-performance training loop supporting mixed-precision (FP16/BF16), gradient accumulation, and custom learning rate schedulers.
- **`OPTIMIZER_REGISTRY` & `SCHEDULER_REGISTRY`**: Pre-configured AdamW, Lion, Sophia, CosineAnnealing, and LinearWarmup schedulers.

---

## 📊 Benchmark Datasets & Synthetic Generation

The `framework.data` module provides unified loaders with offline synthetic fallbacks:
```python
from framework.data.benchmark_datasets import load_benchmark_dataset

# Loads tokenized DataLoader for target benchmark with automatic caching
dataloader, info = load_benchmark_dataset(
    name="wikitext2",
    tokenizer=tokenizer,
    batch_size=8,
    seq_len=512
)
```
Supported Benchmarks: `wikitext2`, `c4`, `gsm8k`, `humaneval`, `mmlu`, `sst2`, `mrpc`, `qnli`, `rte`, `qqp`, `mnli`.

---

## 📈 Visualization & Chart Generation

The `framework.visualization` package provides publication-ready Matplotlib/Seaborn routines:
- `plot_attention_heatmap`: Multi-head attention weight distributions.
- `plot_singular_value_decay`: Spectral energy decay across transformer layers.
- `plot_kv_cache_scaling`: VRAM footprint as a function of sequence length ($512 \to 128\text{k}$).
- `plot_pareto_frontier`: Trade-off curves between model throughput and perplexity.
- `generate_dashboard_html`: Comprehensive HTML report summarizing benchmark execution.

---

## 🧪 Testing & Validation Suites

Run the exhaustive test suites to verify all 106 attention mechanisms and 84 compressors:

```bash
# 1. Run unit tests
pytest Codes/framework/tests/

# 2. Run deep integration validation suite
python Codes/framework/deep_validation_suite.py

# 3. Audit all attention and compression modules
python Codes/framework/verify_all_framework_modules.py
```

---

*Maintained by the NeuralOps Core Research Team.*
