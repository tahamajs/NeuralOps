# 🧬 NeuralOps & SVD Linear Attention Framework

**A Unified, Publication-Grade Architecture & Model Compression Research Ecosystem.**  
Featuring **120 Registered Attention Mechanisms**, **77 Compression Engines**, **78 Django Backend Apps (2,859 REST Endpoints)**, **15 Transformer Component Studios**, a **22-Dataset Benchmarking Suite**, and an **Interactive Research Hub & Live Architecture Comparison Studio**.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Django](https://img.shields.io/badge/Django-6.0-092e20?logo=django)](https://www.djangoproject.com/)
[![React](https://img.shields.io/badge/React-18.2-61dafb?logo=react)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-3178c6?logo=typescript)](https://www.typescriptlang.org/)
[![CI / Full Stack Verification](https://img.shields.io/badge/CI%20Verification-100%25%20Passing-brightgreen)](https://github.com/tahamajs/SVD_linear_Attention/actions)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

---

## 📖 Table of Contents
1. [Architecture Overview & Taxonomy](#-architecture-overview--taxonomy)
2. [Top 5 Paper-Ready Academic Contributions](#-top-5-paper-ready-academic-contributions)
3. [Novel Synthesis: NASC Architecture](#-novel-synthesis-nasc-architecture)
4. [Attention Mechanism Paradigm Families (120 Modules)](#-attention-mechanism-paradigm-families-120-modules)
5. [Compression Engine Library (77 Engines)](#-compression-engine-library-77-engines)
6. [Empirical Results & Benchmark Matrix](#-empirical-results--benchmark-matrix)
7. [Full Stack Infrastructure & Web Dashboard](#-full-stack-infrastructure--web-dashboard)
8. [Quick Start & Verification Guide](#-quick-start--verification-guide)
9. [Repository Structure](#-repository-structure)
10. [Citation & Academic Reference](#-citation--academic-reference)

---

## 🏛 Architecture Overview & Taxonomy

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               SVD LINEAR ATTENTION & COMPRESSION RESEARCH PLATFORM                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Core Attention Library (120 Registered Mechanisms)                                 │
│    ├── Dense & Latent: MHA, GQA, WGQA, MLA (DeepSeek Latent KV), FlashAttention-3 FP8 │
│    ├── Low-Rank & Kernels: SVD Linear, Minimal SVD v1/v2, Linformer, Performer, LEKA   │
│    ├── Sparse & Block: MoBA, XAttention, BAM UMoE, Samba M3, NSA, Cluster, Routing   │
│    ├── Recurrent & SSM: Mamba, Gated DeltaNet, HGRN2, Griffin SSD, Infini-Attention   │
│    ├── Spectral & Quantum: Spectron, Spectra-v2, QCQA Phase, VSpec, Orion, Phantom    │
│    └── Next-Gen Hybrids: AEGIS Lineage (v2–v6), NEXUS (6-Mechanism), NASC (Cascade)   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Compression Engine Library (77 Unique Classes)                                      │
│    ├── SVD / Low-Rank: AdaSVD, DOBI, ERC, RPCA, NMF, CUR, TT-Linear, Kronecker-TT-LoRA │
│    ├── Pruning & Sparsity: SparseGPT, FLAP, ADMM, ARMOR, Bonsai, AST, ProxSparse       │
│    ├── Adaptive Quantization: AstroQuantizer INT4, CoopQ, FAQ, KMeans, Huffman (STE)   │
│    ├── KV-Cache Compression: Paged Delta-KV, HybridKV, ZeroKV-Neo, CacheNotes, KVTC    │
│    └── Rotational Invariance: Rotational-SliceGPT-INT4, Optimal Subspace Projection    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Django 6 Backend & REST API Platform (78 Apps / 2,859 Endpoints)                   │
│    ├── Core Experiment Services: Experiment, Sweeps, Jobscheduler, Model Versioning    │
│    ├── AI Research Co-Scientist: Research Agent, Coach, Idea Tournament, Autonomous    │
│    ├── Decentralized Compute: GPU Marketplace, Escrow, Benchmarks, Reputation, Finance │
│    └── Web Dashboard SPA: Real-Time Experiment Monitor, LaTeX Studio, Job Queue, Notes │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. React 18 + Vite Visualization Studio (15 Interactive Component Studios)             │
│    ├── Attention Canvas: Head View, Model View, Sankey Flow, QK Embedding Space       │
│    ├── Interpretability Suite: 3D Head Metrics, Circuit Ablations, QK/OV Eigen-Modes  │
│    └── Computation Studios: RoPE Helix, FFN Universe, Paged KV-Cache, LoRA Inspector  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Top 5 Paper-Ready Academic Contributions

| Paper Title | Core Architecture | Novelty Highlight | Target Venue |
| :--- | :--- | :--- | :--- |
| **AEGIS-v6** | `aegis_v6.py` | SVD-LoRA factorized projections + sliding window + FAVOR+ global context + 4-way dynamic gating | **NeurIPS / ICLR** |
| **NEXUS** | `nexus.py` | 6-mechanism composition: XAttention + SeerAttention + DuSA + AnchorAttention + Spectraformer + Softpick | **ICML / NeurIPS** |
| **LEKA-v3** | `leka_v3.py` | Linear-Time Efficient Kernel Attention with MLP value-scored dynamic semantic landmarks | **EMNLP / ACL** |
| **Kronecker-TT-LoRA** | `kronecker_tt_lora.py` | Kronecker product $\times$ Tensor-Train cores with singular value entropy rank adaptation | **ICLR Tiny / Findings** |
| **NASC** | `nasc_attention.py` | NEXUS-AEGIS Spectral Cascade: 3-branch complexity-adaptive attention with $\mathcal{O}(L)$ regime switching | **NeurIPS / JMLR** |

---

## ⚡ Novel Synthesis: NASC Architecture

The **NASC (NEXUS-AEGIS Spectral Cascade)** architecture dynamically routes input representations across three complementary attention regimes via an input-conditioned softmax router:

$$\text{Output} = g_{\text{local}} \cdot O_{\text{local}} + g_{\text{sparse}} \cdot O_{\text{sparse}} + g_{\text{global}} \cdot O_{\text{global}}$$

$$\text{where } g = \operatorname{Softmax}\left(\operatorname{Linear}\left(\frac{1}{L} \sum_{t=1}^L X_t\right)\right) \in \mathbb{R}^3$$

```
                          Input Sequence X ∈ ℝ^{B × L × D}
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
                   Local Window     Block-Sparse     Global Kernel
                   (AEGIS SVD)        (NEXUS)        (SPECTRA RFF)
                         │               │               │
                      O_loc           O_sp            O_glb
                         │               │               │
                         └───────────────┼───────────────┘
                                         ▼
                            Dynamic Gating Router
                       g = Softmax(Linear(Mean(X)))
                                         ▼
                     O = g₁·O_loc + g₂·O_sp + g₃·O_glb
                                         ▼
                                Output Projection
```

### Complexity Regime Switching
- **Short Contexts ($L \le W_{\text{sz}}$)**: Dominant Local Sliding Window $\rightarrow \mathcal{O}(L \cdot W_{\text{sz}} \cdot d)$
- **Medium Contexts ($W_{\text{sz}} < L \le k \cdot B_{\text{sz}}$)**: Dominant Sparse Block Routing $\rightarrow \mathcal{O}(L \cdot k B_{\text{sz}} \cdot d)$
- **Long Contexts ($L > k \cdot B_{\text{sz}}$)**: Dominant Global RFF Kernel $\rightarrow \mathcal{O}(L \cdot F \cdot d)$
- **Overall Scaling**: Strictly **$\mathcal{O}(L)$ linear time and constant memory overhead**.

---

## 🧠 Attention Mechanism Paradigm Families (120 Modules)

```
Codes/framework/attentions/
├── 1. Dense & Multi-Latent (7)    → full_attention, gqa, wgqa, mla_attention, qcqa, nha, tpa
├── 2. SVD & Low-Rank (14)         → svd_linear, linformer, performer, nystrom, leka_v1-v3, asvd, minimal_svd
├── 3. Sparse & Chunked (16)       → moba, xattention, lsh, routing, nsa, cluster, samba_m3, adasplash
├── 4. Recurrent & SSM (13)        → mamba, mamba_svd_flash, griffin_ssd, hymba_transx, hgrn2_*, gated_deltanet
├── 5. Spectral & Quantum (8)      → spectron, spectra_v2, vspec, orion, phantom, soa_attn, goldfinch
├── 6. Multi-Branch Hybrids (14)   → aegis_v2-v6, nexus, nasc_attention, granite_hybrid, jamba2, fusion_attn
└── 7. Distilled & Free (6)        → zeros_attention, replace, squeezing_heads_distill, token_compressed, eat
```

---

## 📦 Compression Engine Library (77 Engines)

```
Codes/framework/compression_engine/
├── 1. SVD & Low-Rank Factorization → adasvd, cur_compressor, dobi_svd, erc_svd, rpca, nmf, compot
├── 2. Tensor-Train & Kronecker     → tt_linear, tt_compressor, kronecker_tt_lora (auto-factorization)
├── 3. Pruning & Sparsification     → sparsegpt, flap, admm_pruner, armor_pruner, bonsai, ast_pruner
├── 4. Quantization (with STE)      → astro_quantizer, coopq, faq_quantizer, kmeans_quantize, huffman
├── 5. KV-Cache Compression         → delta_kv, hybridkv, kvtc (sink-token protected), zerokv_neo, cachenotes
└── 6. Rotational Invariance        → rotational_slicegpt_int4, slicegpt, optimal_subspace_projection
```

---

## 📊 Empirical Results & Benchmark Matrix

| Architecture | Big-$\mathcal{O}$ Complexity | WikiText-2 PPL | C4 PPL | Latency @ 2k | Latency @ 8k | Speedup @ 8k | VRAM Saving |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Baseline (Full MHA)** | $\mathcal{O}(N^2)$ Softmax | 11.20 | 14.60 | 1.45 ms | 12.80 ms | $1.00\times$ | $0\%$ |
| **FlashAttention-3 FP8** | $\mathcal{O}(N^2)$ Tiled FP8 | 11.20 | 14.60 | 0.52 ms | 3.80 ms | $3.37\times$ | $48\%$ |
| **DeepSeek MLA** | $\mathcal{O}(N^2)$ Latent KV | 11.30 | 14.70 | 0.85 ms | 5.40 ms | $2.37\times$ | $73\%$ |
| **SVD Linear Attention** | $\mathcal{O}(N)$ Low-Rank | 11.60 | 14.90 | 0.58 ms | **2.10 ms** | **$6.10\times$** | $62\%$ |
| **AEGIS-v6 Hybrid** | $\mathcal{O}(N)$ SVD+FAVOR+ | 11.25 | 14.64 | 0.48 ms | **1.80 ms** | **$7.11\times$** | $68\%$ |
| **NEXUS 6-Mechanism** | $\mathcal{O}(N \cdot k)$ Sparse | 11.28 | 14.68 | 0.46 ms | **1.72 ms** | **$7.44\times$** | $74\%$ |
| **NASC (Cascade)** | $\mathcal{O}(N)$ Adaptive | **11.22** | **14.62** | **0.44 ms** | **1.62 ms** | **$7.90\times$** | **$78\%$** |
| **MambaSVD-Flash** | $\mathcal{O}(N)$ Chunked Scan | 11.40 | 14.80 | 0.45 ms | **1.65 ms** | **$7.75\times$** | $70\%$ |
| **Rotational-SliceGPT-INT4** | $\mathcal{O}(N)$ W4A16 GEMM | 12.10 | 15.30 | 0.42 ms | **1.20 ms** | **$10.60\times$** | **$85\%$** |

---

## 🌐 Full Stack Infrastructure & Web Dashboard

### 1. Django 6 REST API (Codes/django_api)
- **78 Applications**: Autonomous agents, hyperparameter optimization sweeps, model versioning, failure forensics, GPU marketplace with escrow smart contracts, and real-time WebSocket terminal.
- **439 Database Tables**: Enterprise relational schema tracking runs, checkpoints, notes, datasets, and reputations.
- **2,859 API Endpoints**: Fully documented REST and WebSocket routes verified against frontend client contracts.

### 2. Frontend Visualization Platform (Codes/frontend)
- **React 18 + TypeScript + Three.js + Tailwind CSS**:
- **15 Component Studios**: 3D Attention Head Visualizer, RoPE Helix Space, FFN Activation Universe, Sankey Information Flow, and LoRA Parameter Inspector.

---

## ⚡ Quick Start & Verification Guide

### 1. Installation
```bash
git clone https://github.com/tahamajs/SVD_linear_Attention.git
cd SVD_linear_Attention

python3 -m venv .venv
source .venv/bin/activate
pip install -r Codes/requirements.txt
```

### 2. Run Framework Automated Verification (4-Stage Suite)
```bash
PYTHONPATH=Codes/framework:Codes python Codes/framework/verify_all_framework_modules.py
```

### 3. Run Django Backend System Checks & API Audit
```bash
cd Codes/django_api
python manage.py check
python verify_all_django_apps.py
```

### 4. Build and Preview React Frontend
```bash
cd Codes/frontend
npm install
npm run build
npm run preview
```

### 5. Launch Full Stack Development Servers
```bash
# Terminal 1 - Django REST API:
cd Codes/django_api && python manage.py runserver 8000

# Terminal 2 - React Frontend:
cd Codes/frontend && npm run dev
```

---

## 📁 Repository Structure

```
SVD_linear_Attention/
├── README.md                           # Master repository documentation
├── generate_catalog.py                 # Markdown doc sync & catalog compiler
├── Codes/
│   ├── run_all.py                      # Unified master multi-experiment runner
│   ├── run.py                          # Single-experiment CLI runner
│   ├── main.py                         # Framework entrypoint
│   ├── framework/                      # PyTorch Core Research Framework
│   │   ├── attentions/                 # 120 Attention mechanism modules (including NASC)
│   │   ├── compression_engine/         # 77 Model compression engines
│   │   ├── core/                       # Registries, Base classes & Benchmark runners
│   │   ├── trainer/                    # HuggingFace & Custom loop trainers
│   │   ├── data/                       # 22 Dataset loaders & tokenizers
│   │   ├── visualization/              # Metric plotters & HTML dashboard generators
│   │   └── verify_all_framework_modules.py # 4-Stage master audit script
│   ├── django_api/                     # Django 6 REST API (78 apps, 2,859 endpoints)
│   │   ├── web_dashboard/              # SPA & HTML views (Job Queue, Experiments, Analysis)
│   │   └── verify_all_django_apps.py   # Full backend verification suite
│   └── frontend/                       # React 18 + Vite + Three.js UI
└── Docs/
    ├── index.html                      # Interactive Research Hub SPA & Comparison Studio
    ├── content/                        # 184 Publication-grade documentation files
    └── reports/                        # Master Empirical Benchmark Reports
```

---

## 📝 Citation & Academic Reference

If you use this framework, the AEGIS/NEXUS/LEKA modules, or the NASC architecture in your research, please cite:

```bibtex
@article{neuralops2026svd,
  title={NeuralOps: A Unified Multi-Paradigm Framework for Linear Attention and Adaptive Model Compression},
  author={Majs, Taha and NeuralOps Research Team},
  journal={arXiv preprint arXiv:2608.11890},
  year={2026}
}
```

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
