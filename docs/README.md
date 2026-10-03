---
title: NeuralOps & SVD Linear Attention Research Hub
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
---

# 📚 NeuralOps Documentation & Research Hub

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Attention Modules](https://img.shields.io/badge/Attentions-236%20Architectures-indigo.svg)](./content/attentions/)
[![Compression Engines](https://img.shields.io/badge/Compressors-107%20Engines-pink.svg)](./content/compressions/)
[![Masterclass Blog](https://img.shields.io/badge/Blog%20Series-5%20Tracks-emerald.svg)](./blog/README.md)

Welcome to the **NeuralOps Research Documentation Hub**. This repository provides the technical specifications, mathematical derivations, empirical validation protocols, and interactive visualization guides for **236 Attention Architectures** and **107 Post-Training Compression Engines**.

---

## 🧭 Documentation Map & Navigation

| Section | Location | Description |
| :--- | :--- | :--- |
| **Quick Start** | [`START_HERE.md`](./START_HERE.md) | Entry guide for new researchers and framework adopters. |
| **Attention Registry** | [`content/attentions/`](./content/attentions/) | 236 mathematical specifications and zero-shot HuggingFace drop-in guides. |
| **Compression Registry** | [`content/compressions/`](./content/compressions/) | 107 post-training compression specifications (SVD, Quantization, Pruning, KV Cache). |
| **Masterclass Blog Series** | [`blog/README.md`](./blog/README.md) | 5-part long-form research blog covering theory, proofs, and hardware execution. |
| **Benchmark Execution Guide** | [`content/benchmark_execution_guide.md`](./content/benchmark_execution_guide.md) | Comprehensive runner instructions for GLUE, LongBench, and GPU clusters. |
| **Validation Protocol** | [`content/empirical_validation_protocol.md`](./content/empirical_validation_protocol.md) | Calibration procedures, statistical reporting standards, and numerical invariant tests. |
| **Diagnostic Audit Report** | [`content/comprehensive_model_audit_and_fix_report.md`](./content/comprehensive_model_audit_and_fix_report.md) | Root-cause analysis of historical failure modes and mathematical remediations. |
| **Machine Catalog** | [`catalog.json`](./catalog.json) | Structured metadata index of all registered modules, parameters, and paths. |

---

## 🔬 Architectural Overview

```
                      ┌──────────────────────────────────────┐
                      │    HuggingFace Pretrained Backbone   │
                      │  (BERT, RoBERTa, DistilBERT, T5, LLM)│
                      └──────────────────┬───────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │    ATTENTION INJECTION    │                   │    COMPRESSION PIPELINE   │
   │ (`core/attention_utils`)  │                   │ (`core/compression_engine`)│
   ├───────────────────────────┤                   ├───────────────────────────┤
   │ • SVD & Low-Rank (r=32)   │                   │ • Activation-Aware SVD    │
   │ • State-Space & Recurrent │                   │ • SliceGPT & Orthogonal   │
   │ • Kernel & Random Features│                   │ • SparseGPT (Hessian OBS) │
   │ • Mixture of Experts (MoE)│                   │ • ASTRO INT4/FP4 Quant    │
   │ • Hardware-Tiled (FA-3)   │                   │ • Delta-KV Cache 2-4 bit  │
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                      ┌──────────────────────────────────────┐
                      │      Hardware-Aligned Execution      │
                      │  (NVIDIA Tensor Cores / Hopper TMA /  │
                      │   Triton Fused Kernels / Apple MPS)  │
                      └──────────────────────────────────────┘
```

---

## 🚀 Quick Usage Examples

### 1. Zero-Shot Attention Injection
```python
import torch
from transformers import AutoModelForSequenceClassification
from framework.core.attention_utils import inject_attention

# 1. Load standard transformer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 2. Swap standard quadratic attention with SVD Linear Attention
model = inject_attention(model, attn_type="svd_linear_attention", rank=32)

# 3. Fast subquadratic forward pass
inputs = torch.randint(0, 1000, (2, 512))
outputs = model(inputs)
print("Logits shape:", outputs.logits.shape)
```

### 2. Post-Training Model Compression
```python
from transformers import AutoModelForSequenceClassification
from framework.core.compression_engine import compress_model

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Compress dense weights via Activation-Aware SVD (AASVD)
compressed_model = compress_model(
    model,
    config={
        "compressors": [
            {"method": "aa_svd_compressor", "rank": 32, "energy_threshold": 0.92}
        ]
    }
)
```

---

## 📊 Benchmark Execution & Verification

To reproduce benchmark results across GLUE and causal language modeling datasets:

```bash
# 1. Verify module installation and forward pass
python Codes/framework/check_framework_sync.py

# 2. Run GLUE evaluation on GPU
python Codes/framework/eval_all_models_datasets.py \
    --model bert-base-uncased \
    --dataset sst2 \
    --attn_type svd_linear_attention \
    --device cuda

# 3. Run validation quality scripts (F1, MCC, PPL, Fidelity)
python Codes/framework/run_validation_metrics.py

# 4. Launch comprehensive multi-track sweep on GPU cluster
bash run_on_gpu_server.sh --mode comprehensive --gpus 0,1,2,3

# 5. Monitor sweep with live interactive dashboard
python Codes/framework/web/live_monitor.py
```

For full benchmarking parameters and distributed options, see [`content/benchmark_execution_guide.md`](./content/benchmark_execution_guide.md).

---

## 📖 Research Blog Series

The unified 5-part research blog series provides deep dives into the mathematical mechanics, error bounds, and GPU kernels:

1. **[Track 1: SVD, Spectral & Low-Rank Factorizations](./blog/01_spectral_svd_lowrank_attentions.md)**: Truncated SVD, singular energy splitting, condition number stability ($\kappa(A) = \sqrt{\kappa(W)}$).
2. **[Track 2: Linear, Kernel, Recurrent & SSM Dynamics](./blog/02_linear_kernel_recurrent_ssm_attentions.md)**: FAVOR+, Nyström landmarking, Cosformer, and continuous selective state-space scans.
3. **[Track 3: Sparse, Multi-Scale, Latent & Paged Architectures](./blog/03_sparse_multiscale_latent_paged_attentions.md)**: Mixture of Experts (MoE), block sparsity, sliding windows, and deep latent memory.
4. **[Track 4: SVD, Low-Rank & Structural Compression Engines](./blog/04_svd_lowrank_structural_compressions.md)**: Whitened covariance SVD, SliceGPT PCA coordinate rotations, and error budgeting.
5. **[Track 5: Quantization, Pruning & Distillation Compressors](./blog/05_quantization_pruning_distillation_compressions.md)**: Second-order SparseGPT, ADMM structured pruning, and ASTRO outlier transformations.

---

## 🎨 Interactive Dashboards & Component Studios

- **[Attention Visualization Dashboard](./Attention_Visualization_Dashboard/Attention_Visualization_Dashboard.md)**: 9 interactive zones for inspecting multi-head attention weights, singular value spectra, and gradient dynamics.
- **[Transformer Component Studios](./Mapping_Apply_the_Attention_Style_Interactive_Dashboard_to_ALL_Transformer_Components/Mapping_Apply_the_Attention_Style_Interactive_Dashboard_to_ALL_Transformer_Components.md)**: Specialized visualizers for KV caches, LayerNorm dynamics, LoRA adapters, and loss landscapes.

---

## 📜 Citation & License

If you use the NeuralOps framework or any of its attention/compression modules in your academic research, please cite:

```bibtex
@software{neuralops2026,
  author = {Majs, Taha and NeuralOps Research Team},
  title = {NeuralOps: Unified Framework for Subquadratic Attention and Deep Model Compression},
  year = {2026},
  url = {https://github.com/tahamajs/SVD_linear_Attention}
}
```

Distributed under the [MIT License](https://opensource.org/licenses/MIT).
