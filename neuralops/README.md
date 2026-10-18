# 🧬 NeuralOps Core Package (`neuralops`)

`neuralops` is the core Python library of the NeuralOps platform, providing a unified, high-performance, and hardware-certified ecosystem for subquadratic attention mechanisms and deep model compression.

---

## 🏛️ Architecture & Subsystem Map

```
neuralops/
├── attentions/             # 131 Canonical Attention Mechanisms (12 families)
│   ├── linear/             # Linear & kernelized attention (cosformer, performer, etc.)
│   ├── recurrent/          # Gated recurrent scans (GLA, HGRN2, Gated DeltaNet, RWKV)
│   ├── ssm/                # State-space models (Mamba, selective scan, S4)
│   ├── svd/                # SVD low-rank projections (WGQA, SVD-Linear, AdaSVD)
│   ├── sparse/             # Sparse & windowed attention (BigBird, Longformer, NSA)
│   └── fast_kernels/       # FlashAttention-2/3, FP8 Flash, and SDPA wrappers
├── compression_engine/     # 72 Compression Engines
│   ├── svd/                # SVD factorizations (ASVD, AdaSVD, FWSVD, SVD-LLM v2, HNC-SVD)
│   ├── tensor/             # High-order tensor decompositions (Tensor-Train, Tucker, Kronecker)
│   ├── quantization/       # Low-bit quantization (INT4, FP8, K-Means VQ, SmoothQuant)
│   └── pruning/            # Structured pruning (FLAP, Bonsai, Wanda, ADMM)
├── core/                   # Hardware-Aware Acceleration & Dispatch
│   ├── cuda_kernels/       # 114 Native CUDA C++ NVCC implementations across 110 modules
│   ├── triton_kernels/     # Complete suite of hardware-autotuned OpenAI Triton JIT kernels
│   ├── autograd/           # Analytical double-precision backward functions
│   ├── registry.py         # Canonical registries (ATTENTION_REGISTRY, COMPRESSOR_REGISTRY)
│   ├── fallback.py         # Strict 3-level fallback dispatch router
│   ├── surgery.py          # In-place zero-shot attention surgical transplant
│   └── serving/            # PagedKVCache, Copy-on-Write & Self-Speculative Decoding
├── training/               # DRY Distillation & Training Engine
│   ├── loss.py             # Vectorized 6-Pillar Distillation Loss
│   ├── optimizers.py       # Lion, SophiaG, AdaFactor, and 8-bit optimizers
│   └── dataset_loader.py   # Streaming Hugging Face dataset pipeline
└── cli/                    # Fast terminal CLI tool suite
```

---

## 🔢 Canonical Invariant Counts

NeuralOps enforces strict mathematical and architectural invariants certified across the entire codebase:
- **131 canonical attention mechanisms** (`neuralops.core.registry.ATTENTION_REGISTRY`)
- **72 compression engines** (`neuralops.core.registry.COMPRESSOR_REGISTRY`)
- **62 Triton hardware kernels** (`neuralops.list_triton_kernels()`)
- **114 native CUDA C++ kernels** (`neuralops.list_cuda_kernels()`)

All numbers are verified continuously via `scripts/utils/verify_readme_counts.py`.

---

## ⚡ 3-Level Fallback Dispatch Hierarchy

To ensure zero-crash robustness across diverse compute platforms, `neuralops` dispatches operations through a 3-tier hierarchy:

$$\text{Level 1 (CUDA C++ / ROCm HIP)} \longrightarrow \text{Level 2 (OpenAI Triton JIT)} \longrightarrow \text{Level 3 (Vectorized PyTorch / CPU / MPS)}$$

1. **Level 1 (Native CUDA C++ / HIP)**:
   Custom NVCC compiled binaries leveraging warp-level shuffles, register staging, and tensor cores (SM 7.0–12.0).
2. **Level 2 (OpenAI Triton JIT)**:
   Dynamic hardware-autotuned JIT kernels with automatic SRAM block sizing, delivering up to **179.1x speedup**.
3. **Level 3 (Vectorized PyTorch / SDPA Reference)**:
   Pure PyTorch vectorized implementation utilizing `F.scaled_dot_product_attention` or chunked matrix scans. Always active on CPU, Apple Silicon Metal (MPS), and non-CUDA environments.

See [`docs/fallback_list.md`](../docs/fallback_list.md) for individual kernel bottleneck tracking and dispatch rules.

---

## 🚀 Quick Code Examples

### 1. In-Place Attention Surgery
Transplant any standard transformer (Hugging Face DistilBERT, RoBERTa, GPT-2, LLaMA) to a subquadratic attention module in one line:

```python
import neuralops
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Replace quadratic attention with Gated DeltaNet or SVD Linear Attention
neuralops.replace_attention(model, "gated_deltanet")
print("Transplanted model ready for linear-time inference!")
```

### 2. Fisher-Weighted SVD Model Compression
Compress dense weight matrices using second-order activation gradients:

```python
from neuralops.compression_engine import get_compressor

compressor = get_compressor("fwsvd")(rank=32)
compressed_model = compressor.compress(model, calibration_dataloader=train_loader)
```

### 3. Paged KV Cache Serving with Copy-on-Write
```python
from neuralops.core.serving import PagedKVCache, SelfSpeculativeDecoder

cache = PagedKVCache(num_blocks=1024, block_size=16, num_heads=12, head_dim=64)
# Allocates non-contiguous physical memory blocks with zero-copy page forking
```

---

## 🛠️ Compilation & Development

```bash
# Editable developer installation
pip install -e .

# Verify registry invariants
python scripts/utils/verify_readme_counts.py
```
