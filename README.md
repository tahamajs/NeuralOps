# NeuralOps

### A Production-Grade Framework for Subquadratic Attention, Model Compression, and Hardware-Aware Transformer Optimization

<p align="center">
  <img
    src="https://raw.githubusercontent.com/tahamajs/SVD_linear_Attention/main/assets/banner.png"
    alt="NeuralOps"
    width="100%"
  />
</p>

<p align="center">
  <strong>
    Empirical Evaluation · Zero-Shot Attention Replacement · Activation-Calibrated Compression · Hardware-Aware Execution
  </strong>
</p>

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1--12.7-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-Fused%20Kernels-FF6B6B?style=for-the-badge)](https://github.com/openai/triton)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</p>

<p align="center">

[Documentation](#quick-start) ·
[Quick Start](#quick-start) ·
[Architecture](#architecture) ·
[Attention Zoo](#attention-module-taxonomy) ·
[Compression](#compression-engine-taxonomy) ·
[Benchmarks](#benchmarking-and-reproduction) ·
[Testing](#testing) ·
[FAQ](#faq)

</p>

> **114 attention mechanisms · 73 compression engines · 16 Triton kernels · 8 transformer architectures · 617 Python files · 249,271 LOC**

---

## Overview

**NeuralOps** is a unified framework for experimenting with, evaluating, and deploying alternatives to conventional quadratic self-attention together with structured model-compression techniques.

The framework is designed around a simple principle:

> **Changing the computational structure of a Transformer should not require rewriting the model, rebuilding the training stack, or abandoning the Hugging Face ecosystem.**

NeuralOps therefore provides a common infrastructure for:

- subquadratic and linear attention mechanisms,
- SVD and low-rank attention,
- state-space models,
- recurrent and gated attention,
- sparse and MoE attention,
- hybrid attention architectures,
- activation-aware model compression,
- pruning and quantization,
- KV-cache compression,
- fused Triton kernels,
- zero-shot attention replacement,
- reproducible benchmarking,
- hardware-aware profiling.

The result is a single framework in which an existing Transformer can be inspected, modified, compressed, benchmarked, and compared against alternative computational mechanisms without changing the surrounding application code.

---

# Why NeuralOps?

Conventional Multi-Head Attention (MHA) has quadratic complexity with respect to sequence length:

\[
\mathcal{O}(N^2)
\]

for the attention interaction matrix.

As context length increases, the cost of explicitly materializing or processing token-to-token interactions becomes a major computational and memory bottleneck.

NeuralOps approaches this problem from several complementary directions.

| Challenge | Conventional approach | NeuralOps approach |
|---|---|---|
| Long-context inference | Quadratic attention | Linear, kernel, sparse, recurrent, and SSM alternatives |
| Model size | Full-rank parameters | SVD, low-rank, pruning, and quantization |
| Memory traffic | Multiple intermediate tensors | Fused Triton execution |
| Attention experimentation | Architecture-specific rewrites | Universal attention replacement |
| Reproducibility | Ad-hoc scripts | Standardized benchmark/result structure |
| Deployment constraints | Hardware-specific implementations | Automatic hardware-aware fallbacks |

---

## What Makes NeuralOps Different?

### Zero-Shot Attention Injection

NeuralOps can replace attention modules in supported Transformer architectures without requiring a complete model rewrite.

Supported model families include:

- BERT
- RoBERTa
- DistilBERT
- GPT-2
- LLaMA
- Qwen
- T5
- DeBERTa

The replacement mechanism is designed to preserve architecture-specific details such as:

- RoPE,
- GQA,
- fused QKV projections,
- architecture-specific weight layouts,
- output projections,
- nested attention modules.

---

### Structured Model Compression

NeuralOps provides multiple families of compression techniques, including:

- truncated SVD,
- activation-aware SVD,
- low-rank factorization,
- Fisher-weighted methods,
- pruning,
- quantization,
- KV-cache compression,
- tensor decompositions.

The objective is not simply to reduce parameter count, but to expose the trade-off between:

\[
\text{model quality}
\leftrightarrow
\text{parameter count}
\leftrightarrow
\text{latency}
\leftrightarrow
\text{memory}
\]

---

### Activation-Calibrated Decomposition

Weight-only compression treats all directions in parameter space equally.

NeuralOps additionally supports activation-aware decomposition using empirical activation statistics:

\[
W_{\text{effective}}
=
W\cdot
\mathbb{E}[X^T X]^{1/2}
\]

where \(X\) represents real input activations collected from a calibration dataset.

This allows the decomposition to account for the directions that are actually exercised by the model.

The original project reports, for RoBERTa SST-2 at rank 64:

| Method | Accuracy |
|---|---:|
| Weight-only SVD | 84.2% |
| Activation-calibrated SVD | **88.7%** |
| Improvement | **+4.5%** |

---

### Hardware-Aware Execution

Algorithmic complexity is only one part of real-world performance.

NeuralOps therefore includes hand-written Triton kernels designed to reduce:

- kernel-launch overhead,
- intermediate tensor materialization,
- HBM traffic,
- unnecessary memory round-trips.

Where Triton is unavailable, the framework can transparently fall back to PyTorch implementations.

---

# Design Principles

NeuralOps is built around six core engineering principles.

## 1. Composability

Attention mechanisms, compression engines, model adapters, and kernels are independent components.

For example:

```python
model = replace_attention(
    model,
    ATTENTION_REGISTRY.get("svd_linear_attention"),
)

model = replace_attention(
    model,
    ATTENTION_REGISTRY.get("cosformer"),
)

model = replace_attention(
    model,
    ATTENTION_REGISTRY.get("gated_delta_net"),
)
```

The surrounding model code does not need to be rewritten for every experiment.

---

## 2. Backward Compatibility

NeuralOps maintains compatibility with historical registry names and import paths.

The project reports:

- **159 canonical attention registry keys**
- **73 canonical compression keys**
- **641 deprecated aliases**

Deprecated names are redirected to canonical implementations while emitting `DeprecationWarning`.

```python
attn = ATTENTION_REGISTRY.get(
    "svd_linear_attention"
)

# Legacy alias — preserved for compatibility
attn = ATTENTION_REGISTRY.get(
    "aegis_v2"
)
```

---

## 3. Empirical Evaluation

Performance claims should be backed by measurements rather than anecdotal observations.

NeuralOps stores benchmark outputs under `results/` and organizes them into reproducible experiment categories.

Example:

```text
results/
└── 02_rank_sweeps/
    └── dobi_svd_mnli_r128.json
```

Example record:

```json
{
  "accuracy": 0.8647,
  "params": 0.861,
  "latency_ms": 12.3
}
```

---

## 4. Hardware Awareness

The same high-level API can execute through different implementations depending on the available hardware.

```python
output = fused_attention(q, k, v)
```

Possible execution paths include:

```text
CUDA + Triton  →  fused GPU kernel
CPU / MPS      →  PyTorch fallback
```

---

## 5. Test-Driven Correctness

Attention implementations are validated against reference implementations using numerical tolerances.

The project reports:

- **1,358 passing tests**
- **0 collection errors**

Some baseline failures are attributed to CPU/macOS-specific limitations.

---

## 6. Self-Documenting Components

Public modules are expected to provide:

- documentation strings,
- type hints,
- reference papers where applicable,
- examples,
- tests.

---

# Attention Module Taxonomy

NeuralOps organizes its attention implementations into multiple computational families.

| Family | Modules | Examples |
|---|---:|---|
| SVD & Low-Rank | 21 | SVD Linear, ASVD, LoRA Attention |
| Linear & Kernel | 16 | CosFormer, Performer, Linformer |
| State-Space Models | 17 | Mamba, Selective SSM, Jamba |
| Hybrid | 15 | Split-Head Hybrid, Tri-Branch Hybrid |
| Sparse & MoE | 12 | NSA, MoBA, Routing Attention |
| Gated Recurrent | 9 | DeltaNet, GLA, HGRN2 |
| Adaptive / Compression-Aware | 9 | Adaptive Span, token-compressed variants |
| Modern LLM | 7 | MLA, GQA, RoPE-based variants |
| Fast Kernels | 5 | FlashAttention-3 and related kernels |
| Dense | 2 | Full Attention, Zero Attention |
| Experimental | 1 | Template Attention |

Total:

> **114 attention modules**

---

## SVD & Low-Rank Attention

Representative modules include:

| Module | Description |
|---|---|
| `svd_linear_attention` | SVD-factorized linear attention |
| `asvd_attention` | Activation-aware SVD attention |
| `adaptive_svd_attention` | Rank-adaptive SVD |
| `minimal_svd` | Rank-constrained SVD |
| `naive_svd` | Weight-only SVD baseline |
| `svd_attention` | Standard SVD attention |
| `svd_compressed` | Compressed attention |
| `svd_linear_decay` | SVD with exponential decay |
| `svd_llm_v2_attention` | Improved SVD attention |
| `svd_projection_attention` | Projection-only SVD |
| `triton_svd_linear_attention` | Triton implementation |
| `compressed_attention` | Truncated-SVD attention |
| `compressed_mha_attention` | Joint Q/K/V compression |
| `factorised_matrix_state` | Low-rank matrix state |
| `projection_factorized_attn` | Factorized projections |
| `residual_svd_attention` | Residual SVD formulation |
| `salt_attention` | Singular-value adaptation |
| `lora_attention` | LoRA + attention |

Relevant references include SVD-LLM and LoRA.

---

## Linear & Kernel Attention

Representative implementations include:

- `cosformer`
- `performer_attention`
- `linformer_attention`
- `nystrom_attention`
- `tlinformer_attention`
- `galerkin_attention`
- `chela_slay`
- `nasc_attention`
- `rpc_attention`
- `spectra_v2`
- `spectraformer_attention`
- `spectron`
- `primal_attention`
- `rala_attention`
- `tpa_attention`
- `triton_cosformer`

These implementations cover kernelization, random-feature approximations, landmark methods, low-rank projections, and spectral formulations.

---

## State-Space Models

NeuralOps includes SSM-oriented components such as:

- `mamba_block`
- `mamba_svd_flash`
- `selective_ssm`
- `simple_ssm`
- `spectral_mamba`
- `upi_stuffed_mamba`
- `jamba`
- `jamba_info`
- `samba_m3`
- `samba_moe`
- `griffin_ssd`
- `hymba_transx`
- `goldfinch_ahn`
- `nexus`
- `orion`
- `phantom`
- `vspec`

---

## Hybrid Architectures

Hybrid modules combine multiple computational mechanisms, including local attention, kernel attention, recurrence, SVD, and dense attention.

Representative modules:

- `split_head_hybrid_attention`
- `tri_branch_hybrid_attention`
- `gated_hybrid_kernel_attention`
- `hybrid_local_kernel_attention`
- `hybrid_attention`
- `fusion_attn`
- `granite_hybrid`
- `halo_regla`
- `regla_attention`
- `mixattention_moska`
- `optimized_hybrid_attention`
- `positive_kernel_linear_attention`
- `refined_landmark_attention`
- `spectron_fp8_attention`

---

## Sparse & MoE Attention

Representative implementations:

- `adaptive_sparse_attention`
- `cluster_attention`
- `bam_umoe`
- `expert_choice_moe`
- `reap_moe`
- `moba_attention`
- `moe_deltakv_paged`
- `nsa_attention`
- `padre_attention`
- `routing_attention`
- `hamming_attention`
- `lsh_attention`

---

## Gated Recurrent Attention

NeuralOps includes recurrent and gated attention mechanisms such as:

- `gated_delta_net`
- `gated_deltanet_block`
- `gated_linear_attention`
- `decay_linear_attention`
- `hgrn2_gated_deltanet`
- `hgrn2_gated_layer`
- `hgrn2_gated_outer`
- `hla_attention`
- `kda_attention`

---

# Compression Engine Taxonomy

NeuralOps provides **73 compression engines** distributed across several families.

| Family | Engines | Examples |
|---|---:|---|
| Low-Rank | 24 | CUR, NMF, RPCA |
| SVD | 18 | Direct SVD, AdaSVD |
| Pruning | 9 | SparseGPT, ADMM |
| Quantization | 8 | FAQ, K-means |
| KV Cache | 6 | ZeroKV |
| Tensor Decomposition | 3 | Kronecker, Tensor Train |
| Experimental | 5 | Flat-LLM, HashedNet |

---

# Quick Start

## Requirements

Recommended environment:

- Python 3.10, 3.11, or 3.12
- PyTorch 2.2+
- CUDA 12.1–12.7 for GPU execution
- Triton for fused GPU kernels
- Linux, macOS, or Windows/WSL2

GPU acceleration is recommended for production benchmarking.

---

## Installation

```bash
git clone https://github.com/tahamajs/SVD_linear_Attention.git
cd SVD_linear_Attention

python3 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip setuptools wheel
```

### CUDA 12.4

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
```

### CUDA 12.1

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

### CPU-only

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu
```

Install NeuralOps:

```bash
pip install -e .
```

Optional Triton installation:

```bash
pip install triton
```

Verify:

```bash
python -c "import neuralops; print('NeuralOps installed successfully')"

python -c "
from neuralops.core.registry import ATTENTION_REGISTRY
print(f'Attention registry keys: {len(ATTENTION_REGISTRY.keys())}')
"
```

---

# Hardware Support

| Tier | Hardware | VRAM | Features | Recommended Batch |
|---|---|---:|---|---:|
| Tier 1 | H100 / A100 | 40–80 GB | Full Triton, FP8, long context | 64–128 |
| Tier 2 | RTX 4090 / 3090 | 24 GB | Most attention/compression paths | 16–32 |
| Tier 3 | RTX 4070 / L4 / T4 | 8–24 GB | Reduced batch / selected kernels | 4–16 |
| CPU / MPS | Apple / CPU | — | PyTorch fallback | Small |

---

# Attention Replacement

One of the primary NeuralOps workflows is replacing an existing attention implementation without rewriting the surrounding model.

```python
from transformers import AutoModel

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.replace_attention import replace_attention


# 1. Load a pretrained model
model = AutoModel.from_pretrained(
    "distilbert-base-uncased"
)

# 2. Replace attention
model = replace_attention(
    model,
    ATTENTION_REGISTRY.get("svd_linear_attention")
)

# 3. Run inference
outputs = model(input_ids)
```

---

# Model Compression

Activation-aware SVD compression can be applied using a calibration dataset.

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from neuralops.data import (
    get_calibration_dataloader,
)

from neuralops.compression_engine.svd import (
    ASVDCompressor,
)


model = AutoModelForSequenceClassification.from_pretrained(
    "models/roberta_sst2"
)

tokenizer = AutoTokenizer.from_pretrained(
    "models/roberta_sst2"
)

calib_loader = get_calibration_dataloader(
    task_name="sst2",
    tokenizer=tokenizer,
    batch_size=8,
    num_samples=128,
    seq_len=128,
)

compressor = ASVDCompressor(
    model,
    calibration_loader=calib_loader,
)

compressed_model = compressor.compress(
    target_rank=64
)

compressed_model.save_pretrained(
    "models/roberta_sst2_svd_r64"
)
```

---

# Registry

NeuralOps uses centralized registries for attention and compression components.

```python
from neuralops.core.registry import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
)

print(
    f"Attention keys: "
    f"{len(ATTENTION_REGISTRY.keys())}"
)

print(
    f"Compressor keys: "
    f"{len(COMPRESSOR_REGISTRY.keys())}"
)
```

Canonical lookup:

```python
attention_cls = ATTENTION_REGISTRY.get(
    "svd_linear_attention"
)
```

Deprecated aliases remain available for backward compatibility:

```python
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    old_cls = ATTENTION_REGISTRY.get(
        "aegis_v2"
    )

assert old_cls is ATTENTION_REGISTRY.get(
    "hybrid_local_kernel"
)
```

---

# Architecture

## Universal Zero-Shot Injection

The universal replacement mechanism is designed to abstract away architecture-specific attention implementations.

The process can be viewed as:

```text
Pretrained Transformer
        │
        │ replace_attention()
        ▼
UniversalAdapter
        │
        ├── Weight extraction
        ├── Architecture detection
        ├── Q/K/V mapping
        ├── RoPE preservation
        ├── GQA preservation
        └── Output projection mapping
        │
        ▼
Alternative Attention
```

The framework handles architecture-specific conventions including:

- nested encoder structures,
- fused QKV projections,
- Conv1D-based GPT-2 projections,
- GQA,
- RoPE,
- relative attention bias,
- disentangled attention.

---

# Activation-Calibrated Compression

Conventional SVD approximates a weight matrix as:

\[
W \approx U\Sigma V^T
\]

but this approximation does not explicitly account for the input distribution encountered during inference.

NeuralOps instead supports an activation-aware formulation:

\[
W_{\mathrm{effective}}
=
W\mathbb{E}[X^TX]^{1/2}
\]

where \(X\) is collected from representative model inputs.

The conceptual pipeline is:

```text
Pretrained Model
       │
       ▼
Calibration Data
       │
       ▼
Activation Collection
       │
       ▼
Empirical XᵀX
       │
       ▼
Activation-Aware Transformation
       │
       ▼
SVD / Low-Rank Factorization
       │
       ▼
Compressed Model
```

The purpose is to allocate approximation capacity toward directions that matter for actual model computation.

---

# Fused Triton Kernels

NeuralOps includes **16 hand-written Triton kernels**.

The primary motivation is reducing memory traffic.

A simplified GPU memory hierarchy is:

```text
GPU HBM / VRAM
      │
      ▼
   L2 Cache
      │
      ▼
     SRAM
      │
      ▼
  Registers
```

A conventional implementation may materialize several intermediate tensors between operations.

A fused kernel instead attempts to keep intermediate state close to the compute units.

Conceptually:

```text
Q ─────┐
K ─────┼──► fused kernel ──► Output
V ─────┘
```

rather than:

```text
Q → kernel → HBM
K → kernel → HBM
V → kernel → HBM
   ↓
intermediate tensors
   ↓
multiple additional kernels
```

The project reports substantial reductions in HBM round-trips for its fused implementations.

---

## Example: Fused Linear Attention

```python
@triton.jit
def fused_linear_attn_kernel(
    Q,
    K,
    V,
    Out,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    ...,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q = tl.load(Q + ...)
    k = tl.load(K + ...)
    v = tl.load(V + ...)

    q_feat = tl.maximum(q, 0.0) + 1.0
    k_feat = tl.maximum(k, 0.0) + 1.0

    # Fused state construction and query-state multiplication.
    # Intermediate tensors are not materialized in HBM.

    ...

    tl.store(Out + ..., out)
```

---

# Triton Kernel Catalog

| Category | Kernels |
|---|---|
| Recurrence | `linear_attn`, `selective_scan`, `fms_scan`, `gdn_scan`, `gla_scan`, `hgrn2_scan`, `state_update` |
| Projection | `low_rank`, `triton_svd_linear_attention`, `triton_linear_attn`, `triton_cosformer` |
| Attention | `sparse_attn`, `triton_sparse_attn` |
| Utility | `_common`, `gqa`, `softplus` |

---

# Automatic Fallback

Triton is optional.

```python
from neuralops.core.triton_kernels import HAS_TRITON

print(f"Triton available: {HAS_TRITON}")

output = fused_attention(q, k, v)
```

Execution is selected automatically:

```text
CUDA + Triton
     │
     └──► Triton implementation

CPU / MPS / unavailable Triton
     │
     └──► PyTorch fallback
```

---

# Triton Performance

The project reports the following benchmark examples:

| Kernel | Hardware | Triton | PyTorch | Reported Speedup |
|---|---|---:|---:|---:|
| `linear_attn` | H100 | 0.42 ms | 3.15 ms | **7.5×** |
| `selective_scan` | H100 | 0.68 ms | 4.82 ms | **7.1×** |
| `gdn_scan` | A100 | 1.27 ms | 6.35 ms | **5.0×** |
| `low_rank` | RTX 4090 | 0.21 ms | 1.12 ms | **5.3×** |
| `sparse_attn` | H100 | 0.35 ms | 2.41 ms | **6.9×** |

Reported configuration:

```text
batch = 8
sequence length = 4096
heads = 12
head dimension = 64
```

---

# Attention API Examples

## SVD Linear Attention

```python
from neuralops.attentions.svd import (
    SVDLinearAttention,
)

attn = SVDLinearAttention(
    hidden_size=768,
    num_heads=12,
    head_dim=64,
    svd_rank=192,
    lora_rank=16,
    lora_alpha=32.0,
)

attn.copy_weights(pretrained_attn)

output, _ = attn(hidden_states)
```

---

## Performer

```python
from neuralops.attentions.linear import (
    PerformerAttention,
)

attn = PerformerAttention(
    hidden_size=768,
    num_heads=12,
    num_features=256,
    use_orthogonal=True,
)

output, _ = attn(hidden_states)
```

---

## Mamba

```python
from neuralops.attentions.ssm import (
    MambaBlock,
)

block = MambaBlock(
    hidden_size=2048,
    state_size=16,
    expand_factor=2,
    conv_kernel_size=4,
)

output = block(hidden_states)
```

---

## Hybrid Attention

```python
from neuralops.attentions.hybrid import (
    SplitHeadHybridAttention,
)

attn = SplitHeadHybridAttention(
    hidden_size=768,
    num_heads=12,
    num_local_heads=8,
    num_global_heads=4,
    local_window=64,
)

output, _ = attn(hidden_states)
```

---

# Compression API

## Activation-Aware SVD

```python
from neuralops.compression_engine.svd import (
    ASVDCompressor,
)

compressor = ASVDCompressor(
    model=model,
    calibration_loader=calib_loader,
    energy_threshold=0.95,
)

compressed_model = compressor.compress(
    target_rank=64
)

ratio = compressor.get_compression_ratio()

print(f"Compression ratio: {ratio:.2f}×")
```

---

## CUR / Low-Rank Compression

```python
from neuralops.compression_engine.low_rank import (
    CURCompressor,
)

compressor = CURCompressor(model)

compressed = compressor.compress(
    rank=64
)
```

---

## SparseGPT

```python
from neuralops.compression_engine.pruning import (
    SparseGPTPruner,
)

pruner = SparseGPTPruner(
    model,
    sparsity=0.5,
)

pruned = pruner.prune(
    calibration_loader
)
```

---

## Quantization

```python
from neuralops.compression_engine.quantization import (
    FAQQuantizer,
)

quantizer = FAQQuantizer(
    model,
    bits=4,
)

quantized = quantizer.quantize()
```

---

## KV-Cache Compression

```python
from neuralops.compression_engine.kv_cache import (
    ZeroKVNeoCompressor,
)

compressor = ZeroKVNeoCompressor(
    cache_size=4096,
    eviction_policy="attention_sink",
)
```

---

# Supported Transformer Architectures

NeuralOps currently provides injection support for eight major architecture families.

| Architecture | Attention implementation | Important details |
|---|---|---|
| BERT | `BertSelfAttention` | Nested encoder attention |
| RoBERTa | `RobertaSelfAttention` | Nested encoder attention |
| DistilBERT | `MultiHeadSelfAttention` | Separate q/k/v/out projections |
| GPT-2 | `GPT2Attention` | Fused Conv1D QKV |
| LLaMA | `LlamaAttention` | GQA + RoPE |
| Qwen | `Qwen2Attention` | GQA + RoPE |
| T5 | `T5Attention` | Relative attention bias |
| DeBERTa | `DebertaAttention` | Disentangled attention |

---

## BERT

```python
from transformers import BertModel

from neuralops.attentions.svd import (
    SVDLinearAttention,
)

from neuralops.core.replace_attention import (
    replace_attention,
)

model = BertModel.from_pretrained(
    "bert-base-uncased"
)

model = replace_attention(
    model,
    SVDLinearAttention,
    svd_rank=192,
)
```

Typical structure:

```text
model.encoder.layer[i].attention.self
model.encoder.layer[i].attention.output.dense
```

---

## RoBERTa

```python
from transformers import RobertaModel

model = RobertaModel.from_pretrained(
    "roberta-base"
)

model = replace_attention(
    model,
    SVDLinearAttention,
)
```

---

## DistilBERT

```python
from transformers import DistilBertModel

model = DistilBertModel.from_pretrained(
    "distilbert-base-uncased"
)

model = replace_attention(
    model,
    SVDLinearAttention,
)
```

Relevant projection structure:

```text
q_lin
k_lin
v_lin
out_lin
```

---

## GPT-2

```python
from transformers import GPT2Model

model = GPT2Model.from_pretrained(
    "gpt2"
)

model = replace_attention(
    model,
    SVDLinearAttention,
)
```

GPT-2 uses a fused QKV Conv1D projection through `c_attn`.

---

## LLaMA

```python
from transformers import LlamaModel

model = LlamaModel.from_pretrained(
    "meta-llama/Llama-3-8B"
)

model = replace_attention(
    model,
    SVDLinearAttention,
)
```

Relevant components include:

```text
q_proj
k_proj
v_proj
o_proj
rotary_emb
```

---

## Qwen

```python
from transformers import Qwen2Model

model = Qwen2Model.from_pretrained(
    "Qwen/Qwen2.5-0.5B"
)

model = replace_attention(
    model,
    SVDLinearAttention,
)
```

The architecture uses GQA and RoPE.

---

## T5

```python
from transformers import T5Model

model = T5Model.from_pretrained(
    "t5-small"
)

model = replace_attention(
    model,
    SVDLinearAttention,
)
```

T5-specific relative attention bias must be preserved.

---

## DeBERTa

```python
from transformers import DebertaModel

model = DebertaModel.from_pretrained(
    "microsoft/deberta-base"
)

model = replace_attention(
    model,
    SVDLinearAttention,
)
```

DeBERTa uses disentangled attention with architecture-specific projections.

---

# Injection API

### Basic replacement

```python
model = replace_attention(
    model,
    ATTENTION_REGISTRY.get(
        "svd_linear_attention"
    ),
)
```

### Custom configuration

```python
model = replace_attention(
    model,
    ATTENTION_REGISTRY.get(
        "gated_delta_net"
    ),
    svd_rank=192,
    lora_rank=32,
    dropout=0.1,
)
```

### Selective layer replacement

```python
model = replace_attention(
    model,
    ATTENTION_REGISTRY.get(
        "cosformer"
    ),
    layers=[0, 3, 6, 9],
)
```

---

# Benchmarking and Reproduction

NeuralOps includes scripts for systematic evaluation of attention and compression methods.

## Master GPU Benchmark

```bash
./scripts/run_all_benchmarks_gpu.sh
```

The benchmark suite includes:

- WikiText-2 perplexity,
- LM-Eval Harness tasks,
- RULER / Needle-in-a-Haystack evaluation.

---

## Rank Sweeps

```bash
./scripts/runner_benchmark_rank_sweep.sh \
    dobi_svd mnli
```

Ranks are swept from approximately 16 through 384 to characterize quality/parameter trade-offs.

---

## Perplexity Evaluation

```bash
./scripts/runner_benchmark_quality_ppl.sh \
    asvd_attention
```

---

## Multi-Seed Evaluation

```bash
./scripts/runner_benchmark_multiseed.sh \
    svd_linear_attention
```

The benchmark uses multiple seeds to measure recovery stability.

---

## Sequence-Length Scaling

```bash
./scripts/runner_benchmark_seqlen_scaling.sh \
    linformer_attention
```

Sequence lengths are scaled from approximately 128 to 4096.

---

## RULER / Needle-in-a-Haystack

```bash
./scripts/runner_benchmark_ruler.sh \
    simple_ssm
```

---

## Zero-Shot Evaluation

```bash
./scripts/runner_benchmark_zeroshot.sh \
    performer_attention
```

---

# Autonomous Experiment Pipelines

NeuralOps provides higher-level orchestration scripts.

### Master pipeline

```bash
python scripts/run_master_pipeline.py
```

### Long-running GPU pipeline

```bash
python scripts/run_autonomous_48h_pipeline.py
```

### Continuous orchestrator

```bash
python scripts/run_continuous_orchestrator.py
```

---

# Unified Four-Model Evaluation

The project includes unified wrappers for comparing modules across:

- DistilBERT
- RoBERTa-base
- GPT-2
- Qwen2.5-0.5B

Examples:

```bash
python -m neuralops.runners.run_all_4_cosformer \
    --batch_size 32

python -m neuralops.runners.run_all_4_fused_svd_kernel \
    --epochs 3

python -m neuralops.runners.run_all_4_gated_delta_net \
    --lr 5e-5
```

---

# Empirical Findings

The project reports benchmarks covering:

> **245 modules across DistilBERT, RoBERTa, and GPT-2**

Selected findings from the benchmark suite are summarized below.

---

## Near-Lossless Attention

The project reports **41 modules** with a maximum logit difference below **0.5** relative to the baseline in the evaluated setting.

Examples include:

- `flash_attention3`
- `hybrid_attention`
- `lora_attention`
- `svd_linear_attention`

---

## DistilBERT Pareto Frontier

Reported AdaSVD results:

| Rank | Accuracy | Parameters |
|---:|---:|---:|
| Baseline | **90.71%** | 1.00× |
| 128 | **86.47%** | **1.16×** |
| 384 | **90.37%** | — |

---

## RoBERTa Scaling

The project reports AdaSVD recovery from:

```text
81.08% @ rank 16
        ↓
92.32% @ rank 384
```

with `lr=5e-5` in the reported experiment.

---

## Quantization and KV-Cache Results

Reported examples include:

- `faq_quantizer` at approximately 0.98× with accuracy above 93.9%
- `zerokv_neo` reaching up to approximately 200K tokens/s
- `saes_svd_compressor` at rank 256 within approximately 5 PPL of baseline

These numbers should be interpreted in the context of the corresponding benchmark configurations stored under `results/`.

---

# Results Directory

Benchmark artifacts are organized as follows:

```text
results/
├── 01_unified_runs/       # Four-model comparative experiments
├── 02_rank_sweeps/        # Rank scaling experiments
├── 03_zeroshot_evals/     # LM-Eval zero-shot evaluation
├── 04_baselines/          # Unmodified model baselines
├── 05_glue_finetuning/    # Multi-seed fine-tuning experiments
├── 06_adhoc_experiments/  # Sequence scaling, RULER, trajectories
└── 07_misc_tests/         # Aggregate and auxiliary records
```

The directory structure is intended to make experimental results discoverable and reproducible.

---

# Analytics and Publication Artifacts

Aggregate benchmark results:

```bash
python neuralops/analysis/generate_all_modules_analysis.py
```

Outputs include:

```text
paper/_ALL_MODULES_ANALYSIS.csv
paper/_ALL_MODULES_ANALYSIS.md
```

Publication-ready sweep artifacts:

```bash
python neuralops/analysis/generate_sweep_publication_artifacts.py
```

Plotting:

```bash
python -m neuralops.analysis.plotting
```

The analysis subsystem supports:

- aggregate result matrices,
- Markdown summaries,
- LaTeX tables,
- Pareto curves,
- benchmark visualizations.

---

# Testing

Run the complete suite:

```bash
pytest neuralops/tests/ -v
```

Run attention tests:

```bash
pytest neuralops/tests/test_attentions.py -v
```

Run with coverage:

```bash
pytest neuralops/tests/ \
    --cov=neuralops \
    --cov-report=html
```

Run only fast tests:

```bash
pytest neuralops/tests/ -m "not slow"
```

Reported test status:

| Metric | Value |
|---|---:|
| Passing | **1,358** |
| Skipped | 6 |
| Baseline failures | 229 |
| Collection errors | **0** |

The documented baseline failures are associated with CPU/macOS-specific limitations, including GPU mocks and numerical tolerance differences.

---

# Troubleshooting

## Triton is not installed

Error:

```text
ImportError: No module named 'triton'
```

Install Triton:

```bash
pip install triton
```

Alternatively, NeuralOps can use the PyTorch fallback:

```python
output = fused_attention(q, k, v)
```

No application-level code change is required.

---

## CUDA Out of Memory

Possible mitigations:

```text
1. Reduce batch size
2. Enable gradient checkpointing
3. Use FP16 / BF16
4. Reduce SVD rank
```

Example:

```python
model.gradient_checkpointing_enable()
```

or:

```text
--batch_size 4
svd_rank=64
```

---

## Deprecated Registry Name

If you see:

```text
DeprecationWarning:
'aegis_v2' is deprecated.
Use 'hybrid_local_kernel' instead.
```

replace:

```python
ATTENTION_REGISTRY.get("aegis_v2")
```

with:

```python
ATTENTION_REGISTRY.get(
    "hybrid_local_kernel"
)
```

The old name remains available for backward compatibility.

---

## Weight Extraction Failure

If an unsupported model produces a weight-extraction error:

1. Verify that the architecture is supported.
2. Inspect the model's attention module.
3. Use `UniversalAdapter` for custom extraction.

Example:

```python
from neuralops.core.adapters import (
    UniversalAdapter,
)

weight, bias = UniversalAdapter.get_weight_bias(
    src_attn,
    "q",
)
```

---

# FAQ

## How is NeuralOps different from FlashAttention?

FlashAttention focuses on efficient **exact attention computation**.

NeuralOps focuses primarily on:

- approximate attention,
- linear attention,
- sparse attention,
- recurrent attention,
- SSMs,
- hybrid mechanisms,
- structured model compression,
- reproducible comparison.

FlashAttention-3 is nevertheless included as one of the attention implementations available through the framework.

---

## Can NeuralOps be used with custom models?

Yes, provided the model follows one of the supported architecture patterns.

For models with non-standard attention layouts, a custom `UniversalAdapter` can be implemented.

---

## Is NeuralOps compatible with Hugging Face?

Yes.

The framework is designed to operate directly on Hugging Face Transformer models and exposes attention replacement utilities for supported architectures.

---

## Is a GPU required?

No.

CPU and Apple Silicon MPS environments can use PyTorch fallback implementations.

GPU hardware is recommended for:

- large-scale benchmarks,
- Triton kernels,
- long-context evaluation,
- production inference,
- extensive compression experiments.

---

## How do I add a new attention module?

Start from:

```text
neuralops/attentions/experimental/template_attention.py
```

Then register the module:

```python
@ATTENTION_REGISTRY.register(
    "my_custom_attention"
)
class MyCustomAttention(BaseAttention):
    ...
```

Add corresponding tests and documentation before submitting a pull request.

---

## Where are the benchmark results?

Benchmark artifacts are stored under:

```text
results/
```

and organized according to the project's result taxonomy.

---

# Contributing

Contributions are welcome.

A typical contribution workflow is:

```bash
git checkout -b feature/my-feature

git add .

git commit -m "Add my feature"

git push origin feature/my-feature
```

Then open a Pull Request.

## Contribution Guidelines

Please:

- follow PEP 8,
- add type hints,
- add tests,
- update docstrings,
- update the README when introducing public modules,
- preserve backward compatibility,
- document new experimental components,
- include benchmark evidence for performance-related claims.

Avoid introducing public behavior changes without corresponding tests.

---

# Project Status and Roadmap

## v1.0.0 — Initial Release

The initial release includes:

- 114 attention modules,
- 73 compression engines,
- 16 Triton kernels,
- 8 Transformer architectures,
- 641 deprecated aliases,
- 198 test files,
- 1,358 reported passing tests,
- seven-category result taxonomy,
- publication artifact generators.

## Planned Roadmap

### v1.1.0

- INT8 kernels
- Blackwell support

### v1.2.0

- Automatic LoRA rank tuning

### v1.3.0

- Flash-Decoding for long-context inference

---

# Citation

If NeuralOps contributes to your research, please cite:

```bibtex
@software{neuralops2026,
  title  = {NeuralOps: SVD Linear Attention & Model Compression Framework for Foundation Models},
  author = {NeuralOps Research Team},
  year   = {2026},
  url    = {https://github.com/tahamajs/SVD_linear_Attention}
}
```

---

# License

NeuralOps is distributed under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

---

# Project Philosophy

NeuralOps is built around a broader research goal:

> **Make alternative attention mechanisms and model-compression techniques experimentally accessible without sacrificing reproducibility, architectural compatibility, or hardware efficiency.**

The framework intentionally sits at the intersection of three layers:

```text
                 ┌────────────────────────────┐
                 │       Model Architecture   │
                 │                            │
                 │ BERT · GPT · LLaMA · Qwen │
                 └─────────────┬──────────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │     Computational Layer    │
                 │                            │
                 │ Linear · SVD · SSM · Sparse│
                 │ Hybrid · Recurrent · MoE   │
                 └─────────────┬──────────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │      Systems Layer         │
                 │                            │
                 │ Triton · SRAM · HBM · CUDA │
                 │ Compression · Profiling    │
                 └────────────────────────────┘
```

This separation makes it possible to ask research questions such as:

- Which attention mechanisms preserve quality under reduced complexity?
- Which low-rank structures are actually useful under realistic activation distributions?
- When does a theoretically subquadratic method become faster in practice?
- How much performance is lost when attention is replaced without retraining?
- Which compression methods provide the best quality/parameter/latency trade-off?
- How does memory traffic affect the practical performance of linear attention?
- Which architectural assumptions are responsible for observed benchmark improvements?

NeuralOps is intended to make those questions measurable rather than purely theoretical.

---

<p align="center">

### If NeuralOps is useful for your research, consider starring the repository.

<a href="https://github.com/tahamajs/SVD_linear_Attention">
  <img
    src="https://img.shields.io/github/stars/tahamajs/SVD_linear_Attention?style=social"
    alt="GitHub stars"
  />
</a>

<br><br>

<strong>NeuralOps Research Team</strong>

<br>

<sub>
Subquadratic Attention · Model Compression · Efficient Transformers · Hardware-Aware Deep Learning
</sub>

<br><br>

<a href="#neuralops">Back to top ↑</a>

</p>