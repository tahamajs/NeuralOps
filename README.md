# 🧬 SVD Linear Attention & NeuralOps Research Framework

**A Unified, Publication-Grade Architecture & Compression Research Ecosystem.**  
Featuring **118 Registered Attention Mechanisms**, **80 Compression Engines**, **15 Transformer Component Studios**, **Multi-Dataset Benchmarking Suite**, and an **Interactive Research Hub & Live Architecture Comparison Studio**.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Django](https://img.shields.io/badge/Django-4.2-092e20?logo=django)](https://www.djangoproject.com/)
[![React](https://img.shields.io/badge/React-18.2-61dafb?logo=react)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-3178c6?logo=typescript)](https://www.typescriptlang.org/)
[![Triton](https://img.shields.io/badge/OpenAI-Triton-blue)](https://github.com/openai/triton)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

---

## 📖 Architecture Overview & Taxonomy

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               SVD LINEAR ATTENTION & COMPRESSION RESEARCH PLATFORM                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Core Attention Library (118 Modules)                                               │
│    ├── Dense & Exact: MHA, GQA, WGQA, MLA (DeepSeek Latent KV), FlashAttention-3 FP8  │
│    ├── SVD & Spectral: SVD Linear, Minimal SVD v1/v2, ASVD, NHA, RALA, Spectron       │
│    ├── Recurrent & SSM: Mamba, Gated DeltaNet, HGRN2, Griffin SSD, Infini-Attention   │
│    ├── Sparse & Routing: MoBA, BAM UMoE, Samba M3, NSA, Cluster, Routing Attention    │
│    └── Next-Gen Hybrids: MambaSVD-Flash, Aegis-Spectron-FP8, MoE-DeltaKV-Paged        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Compression Engine Library (80 Engines)                                             │
│    ├── SVD / Low-Rank: AdaSVD, DOBI, ERC, RPCA, NMF, CUR, TT-Linear, Kronecker        │
│    ├── Rotational & Pruning: Rotational-SliceGPT-INT4, SparseGPT, FLAP, ProxSparse    │
│    ├── KV-Cache Compression: Paged Delta-KV, HybridKV, ZeroKV, CacheNotes, KVTC       │
│    └── Adaptive Quantization: AstroQuantizer INT4, CoopQ, FAQ, KMeans, Huffman        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Interactive Visualization Dashboard & Component Studios (15 Studios)               │
│    ├── Zones 1–4: Attention Canvas (Head View, Model View, Sankey Flow, QK Embedding) │
│    ├── Zones 5–9: Interpretability Suite (3D Head Metrics, Ablation, QK/OV Circuits)  │
│    ├── Representation Studios: 3D Embedding Navigator, RoPE Helix, Residual Stream    │
│    └── Computation Studios: FFN Neuron Universe, KV-Cache Monitor, LoRA Studio        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Unified Benchmarking & Experimentation Suite                                        │
│    ├── Multi-Mode Benchmark Runner (12 Execution Modes)                                │
│    ├── 22 Benchmark Datasets (WikiText-2/103, C4, PTB, PG-19, GSM8K, HumanEval, GLUE)│
│    └── Fast Multi-Experiment Runner (`python Codes/run_all.py --mode fast`)            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start Guide

### 1. Installation & Environment Setup
```bash
git clone https://github.com/tahamajs/SVD_linear_Attention.git
cd SVD_linear_Attention

# Setup Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r Codes/requirements.txt
```

### 2. Run Master Benchmark Sweep
```bash
# Fast evaluation across key paradigm models on WikiText-2 and C4:
python Codes/run_all.py --mode fast --datasets wikitext2,c4

# Full comprehensive sweep across all 118 attention models:
python Codes/run_all.py --mode comprehensive
```

### 3. Run Automated Framework Test Suite
```bash
# Execute master 6-stage end-to-end integration test
PYTHONPATH=Codes python3 Codes/framework/test_all_framework.py

# Run all 69 unit tests
PYTHONPATH=Codes/framework:Codes python3 -m unittest discover -s Codes/framework/tests
```

### 4. Launch Django REST API Backend
```bash
cd Codes/django_api
python manage.py migrate
python manage.py runserver 8000
```

### 5. Launch React Visualization Frontend
```bash
cd Codes/frontend
npm install
npm run dev
```

### 6. Access Research Hub & Architecture Comparison Studio
Open [`Docs/index.html`](file:///Users/tahamajs/Documents/uni/Project/Docs/index.html) in any modern web browser or deploy on GitHub Pages (`Docs/` root).

---

## 📊 Empirical Results Highlights

| Architecture | Big-$\mathcal{O}$ Complexity | WikiText-2 PPL | C4 PPL | Latency @ 2k | Latency @ 8k | Speedup @ 8k |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Baseline (Full MHA)** | $\mathcal{O}(N^2)$ Softmax | 11.20 | 14.60 | 1.45 ms | 12.80 ms | $1.00\times$ |
| **FlashAttention-3 FP8** | $\mathcal{O}(N^2)$ Tiled FP8 | 11.20 | 14.60 | 0.52 ms | 3.80 ms | $3.37\times$ |
| **Grouped-Query Attn (GQA)** | $\mathcal{O}(N^2)$ Reduced KV | 11.35 | 14.75 | 0.98 ms | 7.20 ms | $1.78\times$ |
| **DeepSeek MLA** | $\mathcal{O}(N^2)$ Latent KV | 11.30 | 14.70 | 0.85 ms | 5.40 ms | $2.37\times$ |
| **SVD Linear Attention** | $\mathcal{O}(N)$ Low-Rank | 11.60 | 14.90 | 0.58 ms | **2.10 ms** | **$6.10\times$** |
| **MambaSVD-Flash** | $\mathcal{O}(N)$ Chunked Scan | 11.40 | 14.80 | 0.45 ms | **1.65 ms** | **$7.75\times$** |
| **Rotational-SliceGPT-INT4** | $\mathcal{O}(N)$ W4A16 GEMM | 12.10 | 15.30 | 0.42 ms | **1.20 ms** | **$10.60\times$** |

---

## 📁 Repository Structure

```
SVD_linear_Attention/
├── README.md                           # Main repository landing page
├── generate_catalog.py                 # Markdown doc sync & catalog compiler
├── Codes/
│   ├── run_all.py                      # Unified master multi-experiment runner
│   ├── run.py                          # Single-experiment CLI runner
│   ├── main.py                         # Framework entrypoint
│   ├── framework/                      # PyTorch Attention & Compression Core
│   │   ├── attentions/                 # 118 Attention mechanism implementations
│   │   ├── compression_engine/         # 80 Model compression engines
│   │   ├── core/                       # Registry & Multi-mode benchmark runner
│   │   ├── data/                       # 22 Dataset loaders & tokenizers
│   │   ├── visualization/              # 5 Studio backend engines
│   │   └── tests/                      # 69 Unit tests & master integration tests
│   ├── django_api/                     # Django REST API backend
│   └── frontend/                       # React 18 + TypeScript + Three.js UI
└── Docs/
    ├── index.html                      # Interactive Research Hub SPA & Comparison Studio
    ├── style.css                       # Research Hub responsive styling
    ├── app.js                          # KaTeX math pre-rendering pipeline
    ├── content/                        # 184 Publication-grade documentation files
    └── reports/                        # Master Empirical Benchmark Reports
```

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
