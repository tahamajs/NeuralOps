# Frequently Asked Questions (FAQ)

## General

### What is NeuralOps?
NeuralOps is a unified, production-grade library and benchmarking suite for subquadratic attention mechanisms, post-training model compression (SVD, pruning, quantization), and hardware-aware transformer optimization.

### Who is it for?
Researchers, ML engineers, and production teams training, fine-tuning, or serving transformer architectures who need lower latency, reduced memory footprint, and high-throughput execution.

### What Python versions are supported?
Python 3.10, 3.11, and 3.12 (as well as 3.14 preview compatibility).

### What PyTorch versions are supported?
PyTorch 2.2.0 and above.

---

## Installation & Hardware

### How do I install NeuralOps?
```bash
pip install neuralops
```

### Do I need a GPU?
No. NeuralOps is fully multi-platform and works out of the box on CPU, Apple Silicon (Metal Performance Shaders / MPS), and NVIDIA/AMD CUDA GPUs.

### Do I need Triton?
No. Triton kernels are optional. On systems without Triton or CUDA (such as macOS, Windows, or CPU-only Linux), pure PyTorch fallbacks are engaged automatically with zero code modification.

### How do I install with Triton?
```bash
pip install neuralops[triton]  # Linux with NVIDIA or AMD GPU
```

---

## Usage & API

### How do I inject a subquadratic attention module?
```python
import torch
from transformers import AutoModelForSequenceClassification
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.replace_attention import replace_attention_in_model

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
attention_cls = ATTENTION_REGISTRY.get("cosformer")
model = replace_attention_in_model(model, attention_cls)
```

### How do I apply model compression?
```python
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.core.adapters import apply_module_to_model

compressor_cls = COMPRESSOR_REGISTRY.get("svd_linear")
model, stats = apply_module_to_model(model, "svd_linear", module_type="compressor", rank=64)
```

### How do I benchmark modules via CLI?
```bash
python -m neuralops.benchmarks run --module cosformer --model distilbert --dataset sst2
```

---

## Registry & Architecture

### How many modules are available?
- **159 Attention Mechanisms**: Linear attention, kernel feature maps, state-space models (SSMs), gated deltanets, grouped-query attention, rotary position embeddings.
- **73 Model Compressors**: SVD low-rank factorizations, structured pruning, activation-aware quantization, KV cache compression.
- **Total: 232 active modules.**

### What are backward compatibility aliases?
Historical module names (e.g. legacy research names) are automatically mapped to canonical implementations via the registry and import shims.

---

## Performance & Memory

### Why is NeuralOps fast?
- **SRAM-resident Kernels**: Fused flash decoding, fused SVD linear, and selective scan avoid DRAM memory round-trips.
- **Lazy Module Resolution**: Fast import time (< 5ms) ensuring zero startup overhead.
- **Linear Complexity**: Replaces $O(L^2)$ attention matrices with $O(L \cdot d)$ recurrent and kernel formulations.

---

## Troubleshooting

### `ImportError: No module named 'triton'`
Triton is optional. The engine will transparently use vectorized PyTorch operations. To enable Triton kernels on Linux with CUDA, run `pip install neuralops[triton]`.

### CUDA out of memory (OOM)
Reduce sequence length or micro-batch size (`--batch-size 4`), or pass `low_rank=32` to factorized layers.

### Deprecation Warnings
Migrate from deprecated aliases to canonical names using [`docs/MIGRATION_GUIDE.md`](file:///Users/tahamajs/Documents/uni/Project/docs/MIGRATION_GUIDE.md).

---

## Contributing & Issues
- Open an issue or discussion on GitHub: [https://github.com/tahamajs/SVD_linear_Attention](https://github.com/tahamajs/SVD_linear_Attention)
- Review contribution guidelines in [`CONTRIBUTING.md`](file:///Users/tahamajs/Documents/uni/Project/CONTRIBUTING.md).
