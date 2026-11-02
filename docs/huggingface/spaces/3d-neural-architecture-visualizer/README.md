---
title: 3D Neural Architecture Visualizer
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
---

# 🌌 3D Neural Architecture Visualizer & Compression Atlas
### Interactive WebGL Exploration, Latent Manifolds & Empirical Hardware Telemetry
**Hooshaai (هوش‌های مصنوعی پیشرفته) · Advanced AI Research**

[![PyPI Version](https://img.shields.io/pypi/v/neuralops.svg?color=blue&label=pypi%20neuralops)](https://pypi.org/project/neuralops/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-tahamajs%2FNeuralOps-blue?logo=github)](https://github.com/tahamajs/NeuralOps)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-14%20Checkpoints-ffcc00)](https://huggingface.co/Hooshaai)
[![Hugging Face Ledger](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-orange)](https://huggingface.co/datasets/Hooshaai/neuralops-results)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Monograph](https://img.shields.io/badge/Thesis-317%20Pages%20%7C%20205%20MB-red)](https://github.com/tahamajs/NeuralOps/releases/download/academic-monographs/NeuralOps_Master_Thesis_810101504.pdf)

---

## 🌟 Executive Overview

Welcome to the **3D Neural Architecture Visualizer**, an interactive, zero-dependency WebGL spatial computing environment designed to inspect, analyze, and manipulate modern transformer architectures, subquadratic attention operators, and low-rank compression dynamics.

Built as part of the **NeuralOps** research ecosystem, this Space delivers real-time 3D computational graphs, token lifecycle physics, singular value decomposition (SVD) spectral decay surfaces, and empirical GPU roofline trajectories across **DistilBERT**, **RoBERTa**, **GPT-2**, and **Qwen-2.5**.

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                  3D NEURAL ARCHITECTURE ATLAS                          │
   │                                                                        │
   │   [241 Unified Modules]                                                │
   │   ├── 131 Attention Mechanisms (Dense, Linear, SSM, MLA, Fused)        │
   │   ├──  72 Compression Engines (Low-Rank SVD, Dynamic, Pruning, ADMM)   │
   │   └──  38 Hybrid State-Space Topologies (SVD-Mamba, Hymba, Griffin)    │
   │                                                                        │
   │   [Interactive Capabilities]                                           │
   │   ├── Dual-Viewport Comparison (Baseline Backbone vs. Compressed)      │
   │   ├── High-Precision Particle Token Physics across Residual Highways   │
   │   ├── Real-Time SVD Latent Manifold & Spectral Energy Decay Curves     │
   │   └── Empirical Ada Lovelace (SM 8.9) Roofline Trajectories            │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 🏛️ Comprehensive Architecture & Module Taxonomy (241 Modules)

The Visualizer catalogues and renders **241 verified neural architecture and compression modules**, providing instant LaTeX formulas, tensor dimension tracking, FLOP counts, memory footprints, and PyTorch reference classes:

### 1. 131 Attention Mechanisms (14 Architectural Families)
- **Dense Softmax & Standard**: Scaled Dot-Product, Multi-Head Attention (MHA), Multi-Query (MQA), Grouped-Query Attention (GQA).
- **Linear & Kernelized Attention**: Cosformer, Performer, Linear Transformer, Katharopoulos Attention, Random Feature Attention (RFA).
- **Gated & Recurrent Variants**: Gated Attention Unit (GAU), Retention (RetNet), TransNormer, RWKV Attention.
- **Delta & Memory Rules**: DeltaNet, Kronecker Delta Attention (KDA), Associative Memory Attention.
- **State-Space Duality & SSM**: Mamba-2 (SSD), Hymba, Griffin-SSD, Selective State-Space Attention.
- **Sparse & Windowed**: BigBird, Longformer, Sliding Window, Local Causal Window, Block Sparse.
- **Multi-Head Latent Attention (MLA)**: DeepSeek-style low-rank latent KV compression ($d_c = 128$).
- **Entmax & Sparsemax**: Tsallis 1.5-Entmax (AdaSplash), exact zero probability attention.
- **Positional & Relative Encodings**: RoPE (Rotary), ALiBi, T5 Relative Bias, Continuous Relative Position.
- **Fused & Hardware-Tiled**: FlashAttention-2, SOAFusedAttention, Fused Bias-Dropout-GELU.

### 2. 72 Model Compression Engines (8 Algorithmic Paradigms)
- **Low-Rank SVD Factorization**: Fisher-Weighted SVD (FWSVD), ASVD, AATC-xKV, SVD-Linear.
- **Dynamic & Adaptive Rank**: Adaptive Rank Allocation, Evolutionary Rank Search, Singular Thresholding.
- **Structured Channel & Head Pruning**: ADMM-based joint sparsity, Magnitude Pruning, Taylor expansion sensitivity.
- **Subspace Projections**: Random Subspace Sketch (RPC), Principal Component Whitening, Optimal Subspace Truncation.
- **Tensor Decompositions**: Tensor Train (TT), Tucker decomposition, CP factorization for feedforward layers.
- **Quantization & Microscaling**: FP8 (E4M3 / E5M2), Microscaling Formats (MXFP6, MXFP4), INT4 SmoothQuant.
- **KV-Cache Eviction & Streaming**: Heavy Hitter Oracle (H2O), StreamingLLM attention sinks, TOVA.
- **Rotational Invariance**: Fast Walsh-Hadamard Transform (QuaRot), SpinQuant outlier suppression.

### 3. 38 Hybrid & State-Space Architectures
- **SVD-Mamba**: Dual-tier low-rank factorized projections driving selective state-space scan kernels.
- **Hymba TransX**: Fused differential attention head combined with associative SSM scan highway.
- **Griffin-SSD**: Recurrent gated linear unit paired with local sliding window attention.
- **Cosformer-SVD**: Cosine kernel reweighting over low-rank singular factorized Key/Value streams.

---

## 🌌 WebGL 3D Spatial Computing & Latent Manifold Visualizer

### 1. Dual-Viewport Comparative Architecture
Compare uncompressed baseline models directly against surgically altered variants:
- **Side-by-Side Topology**: View the dense $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ projections on the left vs. the factorized low-rank bottleneck $U \in \mathbb{R}^{d \times r}, \Sigma V^\top \in \mathbb{R}^{r \times d}$ on the right.
- **Semantic Zoom (LOD 0 / LOD 1)**: Zoom out to view macro-block residual connectivity; zoom in to inspect individual head projections, layer norm scales, and matrix dimensions.

### 2. Token Physics & Residual Highway Dynamics
- **Continuous Particle Tracing**: Watch input token embeddings propagate through the attention projection highway, cross-head aggregation, residual summation nodes, and MLP feedforward expansors.
- **Real-Time Velocities**: Visual feedback highlights memory bandwidth bottlenecks, cache hits, and token shed locations.

### 3. 3D Latent Manifold & SVD Spectral Visualizer (`latent_manifold.html`)
- **Interactive 3D Riemannian Mesh**: Real-time representation of singular value energy decay across rank thresholds ($r \in [1, 768]$) and whitening temperature ($\alpha$).
- **Covariance Whitening Pullback**: Live visual feedback showing how activation covariance whitening restores conditioning in low-rank subspaces.
- **Dynamic Shading Modes**: Seamless toggle between solid surface, wireframe grid, and point cloud topologies.

---

## ⚡ Empirical Hardware Telemetry on NVIDIA RTX 4070 (SM 8.9)

All performance indicators, FLOP measurements, and memory consumption statistics displayed in the Visualizer are grounded in physical execution on **NVIDIA GeForce RTX 4070 (Ada Lovelace, 12 GB GDDR6X, 160.2 Peak TFLOPS, 504.0 GB/s)**:

| Metric | Dense Baseline (`distilbert`) | SVD Compressed (`rank=64`) | Fused FlashAttention-2 | DeltaNet Recurrent |
| :--- | :---: | :---: | :---: | :---: |
| **Throughput (seq/s)** | 312.4 | **488.1** (+56.2%) | **641.7** (+105.4%) | 420.5 (+34.6%) |
| **Peak VRAM (MB)** | 1,420 MB | **948 MB** (-33.2%) | **712 MB** (-49.8%) | 860 MB (-39.4%) |
| **Operational Intensity** | 42.1 FLOP/Byte | 28.4 FLOP/Byte | 114.6 FLOP/Byte | 38.2 FLOP/Byte |
| **Roofline Regime** | Memory-Bound | Memory-Bound | Memory/Compute Knee | Memory-Bound |
| **Accuracy Retention** | 100.0% (90.48% SST-2) | **99.6%** (90.14% SST-2) | **100.0%** (90.48% SST-2) | **98.2%** (88.85% SST-2) |

---

## 🏥 In-Place Model Surgery & 6-Pillar Knowledge Distillation

NeuralOps enables zero-copy surgery to transplant any of the 131 attention mechanisms or 72 compressors directly into pre-trained PyTorch backbones:

```python
import torch
import neuralops as nops

# 1. Load baseline model
model = torch.hub.load("huggingface/pytorch-transformers", "model", "distilbert-base-uncased")

# 2. In-place surgery: Replace dense attention with Fisher-Weighted SVD
surgeon = nops.AttentionSurgeon(model)
surgery_report = surgeon.replace_attention(
    target_module="attention",
    new_attention="svd_linear_attention",
    rank=64,
    fisher_whitening=True,
)
print(f"Surgery complete: {surgery_report.parameters_saved} params reduced.")

# 3. Post-surgery healing: 6-Pillar Knowledge Distillation with WSD Schedule
healer = nops.DistillationHealer(
    teacher=model,
    student=surgeon.model,
    pillars=["logits_kl", "hidden_mse", "attention_kl", "svd_align"],
    lr_schedule="wsd",
)
healer.heal(train_dataloader, epochs=3)
```

---

## 🚀 Deployment & Offline Launch Modes

The 3D Visualizer requires **zero build tools** and runs client-side in any modern browser:

### 1. In this Hugging Face Space
Interact directly with the live visualizer in the **App** tab!

### 2. Launch Standalone Offline File (No Server Needed!)
Open [`standalone.html`](standalone.html) directly in any web browser:
```bash
open standalone.html
```

### 3. Run Locally with Lightweight HTTP Server
```bash
python3 -m http.server 8000
# Open http://localhost:8000 in your browser
```

---

## 🔗 Official Ecosystem Links

- **GitHub Repository**: [github.com/tahamajs/NeuralOps](https://github.com/tahamajs/NeuralOps)
- **PyPI Package**: [pypi.org/project/neuralops](https://pypi.org/project/neuralops/)
- **Hugging Face Organization & Models**: [huggingface.co/Hooshaai](https://huggingface.co/Hooshaai)
- **Hugging Face Benchmark Ledger**: [huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)
- **Master's Thesis Monograph (317 Pages, 205 MB)**: [NeuralOps Master's Thesis PDF](https://github.com/tahamajs/NeuralOps/releases/download/academic-monographs/NeuralOps_Master_Thesis_810101504.pdf)

---

## 📚 Academic Citation

```bibtex
@article{majlesi2026neuralops,
  author       = {Majlesi, Taha and contributors},
  title        = {NeuralOps: A Unified Hardware-Accelerated Framework for Subquadratic Sequence Modeling and Low-Rank Attention Surgery},
  year         = {2026},
  url          = {https://github.com/tahamajs/NeuralOps}
}
```

---

## 📄 License

This visualizer and the NeuralOps framework are licensed under the [MIT License](https://opensource.org/licenses/MIT).
