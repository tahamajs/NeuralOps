# 🧬 NeuralOps Core Package (`neuralops`)

`neuralops` is the core Python library of the NeuralOps platform, providing a unified, high-performance, and correctness-gated ecosystem for subquadratic attention mechanisms, model compression, and hardware operators.

---

## 🏛️ Architecture & Subsystem Map (v1.6.0 Tier Structure)

```
neuralops/
├── layers/                 # 131 Canonical Attention Mechanisms across 12 families
│   ├── linear/             # Linear & kernelized attention (cosformer, performer, etc.)
│   ├── recurrent/          # Gated recurrent scans (GLA, HGRN2, Gated DeltaNet, RWKV)
│   ├── ssm/                # State-space models (Mamba, selective scan, S4)
│   ├── svd/                # SVD low-rank projections (WGQA, SVD-Linear, AdaSVD)
│   ├── sparse/             # Sparse & windowed attention (BigBird, Longformer, NSA)
│   └── hybrid/             # Multi-scale, hierarchical, and hybrid attention layers
├── compression/            # 72 Model Compression Engines across 7 paradigms
│   ├── svd/                # SVD factorizations (ASVD, AdaSVD, FWSVD, SVD-LLM v2, HNC-SVD)
│   ├── low_rank/           # Canonical factorized representations (FactorizedLinear, SVDLinear)
│   ├── quantization/       # Low-bit quantization (INT4, FP8, K-Means VQ, SmoothQuant)
│   └── pruning/            # Structured pruning (FLAP, Bonsai, Wanda, ADMM)
├── ops/                    # Hardware-Aware Acceleration & Dispatch
│   ├── cuda/               # 112 Native CUDA C++ NVCC packages (114 implementations)
│   ├── triton/             # 26 Triton kernel modules housing 62 unified operators across 4 tiers
│   └── pytorch/            # Vectorized PyTorch reference implementations for CPU and Apple Silicon
├── trainer/                # Unified Distillation & Training Engine
│   ├── loss.py             # Vectorized 6-Pillar Distillation Loss
│   ├── optimizers/         # Lion, Muon, SophiaG, Stiefel, and Riemannian optimizers
│   └── schedulers.py       # Warmup-Stable-Decay (WSD) and cosine schedulers
├── analysis/               # Decoupled Analysis, Inspector & Telemetry Tools
├── benchmarks/             # Benchmark Execution Suites
├── cli/                    # Production CLI Suite (train, compress, info, doctor, list)
└── _lazy.py                # PEP 562 Lazy Loading Implementation (<30ms bare import)
```

---

## 🔢 Canonical Invariant Counts

NeuralOps enforces strict mathematical and architectural invariants certified across the entire codebase:
- **131 canonical attention mechanisms** (`neuralops.layers.ATTENTION_REGISTRY`)
- **72 compression engines** (`neuralops.compression.COMPRESSOR_REGISTRY`)
- **26 Triton hardware kernel modules** housing **62 unified operators**
- **114 native CUDA C++ kernels** across **112 packages** (`neuralops.ops.cuda`)

---

## ⚡ 3-Level Fallback Dispatch Hierarchy

To ensure zero-crash robustness across diverse compute platforms, `neuralops` dispatches operations through a 3-tier hierarchy:

$$\text{Level 1 (CUDA C++ / ROCm HIP)} \longrightarrow \text{Level 2 (OpenAI Triton JIT)} \longrightarrow \text{Level 3 (Vectorized PyTorch / CPU / MPS)}$$

1. **Level 1 (Native CUDA C++ / HIP)**:
   Custom NVCC compiled binaries leveraging warp-level shuffles, register staging, and tensor cores (SM 7.0–12.0).
2. **Level 2 (OpenAI Triton JIT)**:
   Dynamic hardware-autotuned JIT kernels with automatic SRAM block sizing and fused elementwise operations.
3. **Level 3 (Vectorized PyTorch / SDPA Reference)**:
   Pure PyTorch vectorized implementation utilizing `F.scaled_dot_product_attention` or chunked matrix scans. Always active on CPU, Apple Silicon Metal (MPS), and non-CUDA environments.

---

## 🚀 Quick Code Examples

### 1. In-Place Attention Surgery
Transplant any standard transformer to a subquadratic attention module in one line:

```python
import neuralops
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("gpt2")

# Replace quadratic attention with Gated DeltaNet or Cosformer Linear Attention
neuralops.replace_attention(model, "cosformer")
print("Transplanted model ready for linear-time inference!")
```

### 2. Fisher-Weighted SVD Model Compression
Compress dense weight matrices using second-order activation gradients:

```python
from neuralops import compress

compressed_model = compress(model, method="fwsvd", rank=32)
```

### 3. Paged KV Cache Serving with Copy-on-Write
```python
from neuralops.models.paged_state import PagedStateTable

cache = PagedStateTable(max_pages=1024, page_size=16, num_heads=12, head_dim=64)
# Allocates non-contiguous physical memory blocks with zero-copy page forking
```

---

## 🛠️ Compilation & Development

```bash
# Editable developer installation
pip install -e ".[dev]"
```
