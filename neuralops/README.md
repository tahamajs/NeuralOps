# 🧬 NeuralOps Core Package (`neuralops`)

`neuralops` is the high-performance core Python engine of the NeuralOps ecosystem, purpose-built for subquadratic attention research, in-place neural architecture surgery, mathematical model compression, and hardware-accelerated operator execution across NVIDIA GPUs, Apple Silicon Metal (MPS), and modern multicore CPUs.

---

## 🏛️ Comprehensive Architecture & Subsystem Map (v1.6.0 Tier Structure)

```
neuralops/
├── layers/                 # 131 Canonical Attention Mechanisms across 12 families
│   ├── linear/             # Linear & kernelized attention (cosformer, performer, rala, etc.)
│   ├── recurrent/          # Gated recurrent associative scans (GLA, HGRN2, Gated DeltaNet, RWKV)
│   ├── ssm/                # State-space models & selective scan blocks (Mamba, S4, Bi-SSM)
│   ├── svd/                # Factorized & low-rank projections (WGQA, SVD-Linear, AdaSVD)
│   ├── sparse/             # Windowed, block-sparse & routing attention (BigBird, Longformer, NSA)
│   └── hybrid/             # Multi-scale, hierarchical, and hybrid attention layers
├── compression/            # 72 Model Compression Engines across 7 paradigms
│   ├── svd/                # SVD factorizations (ASVD, AdaSVD, FWSVD, SVD-LLM v2, HNC-SVD)
│   ├── low_rank/           # Canonical factorized representations (FactorizedLinear, SVDLinear)
│   ├── quantization/       # Microscaled & low-bit quantization (INT4, FP8, SmoothQuant, K-Means VQ)
│   └── pruning/            # Structured channel & head pruning (FLAP, Bonsai, Wanda, ADMM)
├── ops/                    # Hardware-Aware Acceleration & Dispatch Hierarchy
│   ├── cuda/               # 110 canonical CUDA modules (114 native implementations)
│   ├── triton/             # 26 Triton kernel modules housing 62 unified operators across 4 tiers
│   └── pytorch/            # Vectorized PyTorch reference implementations for CPU and Apple Silicon
├── trainer/                # Unified Distillation, Surgery Recovery & Training Engine
│   ├── distillation/       # 6-Pillar Knowledge Distillation (Logit, Feature, Relation, Operator, CKA, Steering)
│   ├── optimizers/         # Lion, Muon, SophiaG, Stiefel, and Riemannian Adam optimizers
│   └── schedulers.py       # Warmup-Stable-Decay (WSD) and adaptive cosine decay schedulers
├── surgery/                # Zero-Downtime Attention Replacement & Surgical Healing
├── models/                 # Reference Architectures, Paged KV-Cache Tables & Speculative Engines
├── evaluation/             # Standardized PPL, LM-Eval Harness, Needle-in-a-Haystack (NIAH), & Perplexity
├── analysis/               # Decoupled Analysis, Curvature Inspection & Hardware Telemetry Tools
├── benchmarks/             # Benchmark Execution Suites & Master Ledger Telemetry
├── cli/                    # Production CLI Suite (train, compress, info, doctor, list)
└── _lazy.py                # PEP 562 Lazy Loading Implementation (<30ms bare import footprint)
```

---

## 🔢 Certified Canonical Repository Invariants

NeuralOps enforces strict mathematical and architectural invariants certified continuously across the codebase:
- **131 canonical attention mechanisms** (`neuralops.layers.ATTENTION_REGISTRY`)
- **72 compression engines** (`neuralops.compression.COMPRESSOR_REGISTRY`)
- **26 Triton hardware kernel modules** housing **62 unified operators**
- **110 canonical CUDA modules** housing **114 native kernel implementations**. The raw
  `neuralops/ops/cuda` tree also contains compatibility and support directories; those
  filesystem entries are not canonical registry modules.
- **29 master demonstration notebooks** verified across compute platforms

---

## ⚡ The 3-Level Fallback Dispatch Hierarchy

Every custom operator in `neuralops` adheres to a strict 3-tier fallback contract, guaranteeing maximum hardware acceleration without crashing in non-CUDA environments:

$$	ext{Level 1 (Native CUDA C++ / HIP)} \longrightarrow 	ext{Level 2 (OpenAI Triton JIT)} \longrightarrow 	ext{Level 3 (Vectorized PyTorch / CPU / MPS)}$$

1. **Level 1 — Native CUDA C++ / HIP (Peak Throughput)**:
   Custom NVCC compiled binaries leveraging warp-level shuffles (`__shfl_down_sync`), 128-bit vectorized vector loads (`float4`), register staging, and tensor cores (SM 7.0–12.0).
2. **Level 2 — OpenAI Triton JIT (Hardware Autotuning)**:
   Dynamic 2D-tiled JIT kernels with automatic SRAM block sizing, fused elementwise activations, and zero-allocation dispatch tables.
3. **Level 3 — Vectorized PyTorch Reference (Universal Fallback)**:
   Pure PyTorch vectorized implementation utilizing `F.scaled_dot_product_attention` or chunked parallel associative scans. Always active on CPU, Apple Silicon Metal (MPS), and non-CUDA environments.

---

## 🧬 Architectural Deep Dive: 12 Attention Families

`neuralops` houses 131 attention keys organized into 12 distinct mathematical paradigms:

1. **Linear Attention & Kernel Feature Maps**:
   - Replaces $O(N^2)$ quadratic softmax matrix multiplication with associative feature map kernelization $\phi(Q)(\phi(K)^T V)$.
   - Key Implementations: `cosformer`, `performer`, `linear_attention`, `fast_attention`, `rala`.
2. **Gated Recurrent Associative Scans**:
   - Implements parallel associative Blelloch scans with input/forget gating for linear state updates.
   - Key Implementations: `gated_linear_attn` (GLA), `hgrn2`, `gated_delta_net` (GDN), `rwkv6`.
3. **State Space Models (SSM)**:
   - Structured continuous-time state representations discretized via Bilinear/ZOH transforms.
   - Key Implementations: `mamba_block`, `selective_scan`, `s4`, `spectral_mamba`.
4. **SVD & Low-Rank Attentions**:
   - Spectral decomposition of key-value projection matrices with adaptive rank budgets.
   - Key Implementations: `asvd_attention`, `svd_linear_attention`, `wgqa`, `svd_projection_attention`.
5. **Sparse, Windowed & Multi-Scale Attention**:
   - Local sliding window paired with random/global anchor tokens or native block-sparse masks.
   - Key Implementations: `adaptive_sparse_attention`, `sliding_window_attention`, `bigbird`, `longformer`.
6. **Adaptive Span & Token Pruning Attentions**:
   - Dynamically learns per-head context horizons and sheds redundant sequence tokens.
   - Key Implementations: `adaptive_span_attention`, `token_pruning_attention`, `streaming_llm`.
7. **Hybrid, Mixture-of-Experts & Speculative Layers**:
   - Blends dense attention anchors with linear recurrent layers or sparse routing experts.
   - Key Implementations: `aegis_hybrid`, `granite_hybrid`, `bam_umoe`, `samba_m3`.

---

## 🗜️ Model Compression Paradigms (72 Engines)

`neuralops.compression` provides cutting-edge factorization, quantization, and pruning pipelines:

- **Fisher-Weighted SVD (FWSVD)**: Leverages empirical Fisher Information Matrix diagonal curvatures to preserve salient semantic subspaces during low-rank truncation.
- **Hierarchical Nullspace Clamping SVD (HNC-SVD)**: Preserves model activation nullspaces to prevent catastrophic forgetting after weight truncation.
- **Microscaling & Low-Bit Quantization (FP8 / INT4 / MX)**: Outlier-aware channel scaling (SmoothQuant) and micro-scaled tensor floating point formats.
- **ADMM Structured Channel & Head Pruning**: Alternating Direction Method of Multipliers for exact $2:4$ hardware sparsity and structural head pruning.

---

## 🔬 Multi-Pillar Post-Surgery Knowledge Distillation

When performing in-place attention replacement or structural compression, `neuralops` restores lost model perplexity using a unified 6-pillar distillation objective:

$$\mathcal{L}_{	ext{total}} = lpha \mathcal{L}_{	ext{CE}} + eta \mathcal{L}_{	ext{Logit}} + \gamma \mathcal{L}_{	ext{Feature}} + \delta \mathcal{L}_{	ext{Relation}} + \epsilon \mathcal{L}_{	ext{Operator}} + \zeta \mathcal{L}_{	ext{CKA}}$$

- **Logit Distillation ($\mathcal{L}_{	ext{Logit}}$)**: Temperature-scaled forward KL divergence against teacher output distributions.
- **Feature Map Matching ($\mathcal{L}_{	ext{Feature}}$)**: Hedgehog chunked attention-map kernel alignment.
- **Value-Relation Alignment ($\mathcal{L}_{	ext{Relation}}$)**: MiniLM Gramian matrix value-relation distance matching.
- **Blockwise Operator Cloning ($\mathcal{L}_{	ext{Operator}}$)**: Hidden state residual representation cloning across transformer layers.
- **Linear CKA Alignment ($\mathcal{L}_{	ext{CKA}}$)**: Centered Kernel Alignment matching representational geometry.
- **Layer Steering ($\mathcal{L}_{	ext{Steering}}$)**: SmoothMax layerwise representational steering to eliminate gradient deadlocks.

---

## 🚀 Quickstart & Practical Usage

### 1. In-Place Attention Surgery
Transplant any standard transformer to a subquadratic attention mechanism in a single line of code:

```python
import neuralops
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# Replace quadratic softmax attention with linear-time Gated DeltaNet
neuralops.replace_attention(model, "gated_delta_net")
print("Model successfully transplanted to linear attention!")
```

### 2. SVD Matrix Compression
Compress transformer weights via Fisher-weighted singular value decomposition:

```python
from neuralops import compress

compressed_model = compress(model, method="asvd", rank=32)
```

### 3. Paged KV-Cache Management
High-throughput serving cache with copy-on-write page table support:

```python
from neuralops.models.paged_state import PagedStateTable

cache = PagedStateTable(max_pages=2048, page_size=16, num_heads=16, head_dim=64)
# Allocates non-contiguous physical memory blocks with zero-copy page forking
```

### 4. Custom Backend Routing
Enforce specific execution backends for strict benchmarking or debugging:

```python
import torch
from neuralops.ops.dispatch import force_kernel_backend

# Force Level 2 (Triton JIT) or Level 3 (Vectorized PyTorch)
with force_kernel_backend("triton"):
    # Executes Triton kernel path
    pass
```

---

## 🛠️ Developer Installation & Quality Gates

```bash
# Clone the repository
git clone https://github.com/tahamajs/NeuralOps.git
cd NeuralOps

# Editable installation with development dependencies
pip install -e ".[dev]"

# Run static quality audits
python3 scripts/utils/audit_dead_code.py
python3 scripts/audit_readme_consistency.py
```

---

## 📜 License & Attribution

NeuralOps is licensed under the **MIT License**. See [`LICENSE`](file:///Users/tahamajs/Documents/uni/Project/LICENSE) for details.
