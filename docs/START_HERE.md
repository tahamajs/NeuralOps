# 🚀 Getting Started with NeuralOps v2.1.0

Welcome to **NeuralOps** (`SVD_linear_Attention`), the production-grade deep learning systems framework and neural model surgery ecosystem providing:
- **115 Canonical Physical Attention Mechanisms** (spanning 131 registry variants across 14 architectural families)
- **82 Model Surgery Compression Engines** (unifying 328 multi-rank models across GLUE SST-2 and WikiText-2)
- **58 Native Triton JIT GPU Acceleration Kernels**
- **140 Native CUDA Kernel Packages**
- **94 Canonical Interactive Jupyter Notebooks**
- **432 Surgeried Checkpoints** published on Hugging Face Hub under [`Hooshaai`](https://huggingface.co/Hooshaai)

---

## 📦 Official Production Installation

Install NeuralOps directly from PyPI:

```bash
# Core framework and torch-free utilities
pip install neuralops==2.1.0

# With full CUDA GPU acceleration dependencies
pip install "neuralops[cuda]==2.1.0"

# With CPU-only baseline support
pip install "neuralops[cpu]==2.1.0"
```

To install from source for development:
```bash
git clone https://github.com/tahamajs/NeuralOps.git
cd NeuralOps
pip install -e ".[all]"
```

---

## ⚡ Core 5-Minute Quickstart Workflows

### 1. In-Place Attention Surgery with Universal Hub Loader
Download and run any pre-surgeried attention checkpoint from the official `Hooshaai` organization in seconds:

```python
import torch
from neuralops.models import load_surgeried_model

# Load FlashAttention-3 surgeried BERT directly from Hooshaai hub
model, tokenizer = load_surgeried_model("Hooshaai/bert-flash_attention-sst2")

inputs = tokenizer("NeuralOps accelerates transformer sequence modeling.", return_tensors="pt")
outputs = model(**inputs)
print("Classification logits shape:", outputs.logits.shape)
```

### 2. Live Attention Replacement on Dense Backbones
Perform in-place topological attention surgery on any standard Hugging Face model:

```python
import torch
from transformers import AutoModelForSequenceClassification
from neuralops.surgery import UniversalAttentionTransplanter

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Transplant standard SDPA attention with Subquadratic Gated DeltaNet
transplanter = UniversalAttentionTransplanter(target_attention="gated_delta_net")
surgeried_model, manifest = transplanter.transplant(model)

print(f"Surgery successfully applied across {len(manifest.modified_layers)} layers!")
```

### 3. Low-Rank Model Compression via 82-Compressor Zoo
Compress transformer projection layers using hardware-aware low-rank SVD factorization:

```python
from neuralops.compression import UniversalModelCompressor

# Apply Direct SVD compression with rank=32
compressor = UniversalModelCompressor(algorithm="direct_svd", rank=32)
compressed_model, metrics = compressor.compress(model)

print(f"Parameter reduction ratio: {metrics.compression_ratio:.2f}x")
print(f"Memory footprint reduction: {metrics.memory_reduction_pct:.1f}%")
```

---

## 💻 Canonical Command Line Interface (CLI)

NeuralOps provides an industrial-grade CLI suite:

```bash
# List all 115 attention mechanisms and 82 compressors
neuralops benchmark list

# Run cold-cache benchmark on a specific attention kernel
neuralops benchmark run --benchmark flash_attention3 --device cuda --flush-l2

# Execute parameter sweep across sequence lengths (128 -> 2048) and batch sizes
neuralops matrix --modules flash_attention3,gla,gated_delta_net --seq-lens 128,512,1024,2048

# Inspect hardware health, CUDA capability, and compiler toolchains
neuralops doctor

# Export benchmark telemetry simultaneously across 5 canonical formats
neuralops export --output-dir results/benchmarks/ --format all
```

---

## 📚 Architectural Documentation Roadmap

- [🏛️ Full Framework Architecture (7 Structural Layers)](ARCHITECTURE.md)
- [📦 Modular Surgical Delta Protocol (97% Storage Savings)](MODULAR_SURGICAL_DELTA_SPEC.md)
- [⚡ Offline Teacher Representation Caching Guide](OFFLINE_TEACHER_CACHING_GUIDE.md)
- [🦁 Hooshaai Model Zoo Catalog (432 Models)](model_zoo.md)
- [🔬 82-Compressor Multi-Rank Ledger Catalog](COMPRESSOR_ZOO_CATALOG.md)
- [📑 115 Attention Mechanisms Taxonomy](ATTENTION_TAXONOMY_CATALOG.md)
- [💻 Production CLI Reference Guide](CLI_REFERENCE.md)
- [🛡️ Security Policies & Fail-Closed Protocols](security.md)
