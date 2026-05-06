
# ⚡ Efficient Transformer Toolkit

<div align="center">

**A modular, extensible framework for training, compressing, and deploying efficient Transformer models.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[**Quick Start**](#-quick-start) •
[**Attention Zoo**](#-attention-zoo) •
[**Compression Engine**](#-compression-engine) •
[**Data Pipeline**](#-data-pipeline) •
[**CLI**](#-cli-with-config-profiles) •
[**Logging & Analysis**](#-experiment-logging)

</div>

---

## 📖 Overview

The **Efficient Transformer Toolkit** brings together **30+ state‑of‑the‑art attention mechanisms**, **20+ model compression techniques**, **flexible data pipelines**, and a **powerful CLI** – all designed to make Transformer efficiency research and deployment fast and reproducible.

With this toolkit you can:
- **Experiment** with dozens of attention types (from vanilla to extreme hybrids)
- **Compress** pretrained models with zero‑shot or training‑based methods
- **Train** and **evaluate** on GLUE benchmarks or custom datasets
- **Automate** everything through a unified command‑line interface with YAML config profiles

---

## 📁 Project Structure

```
Efficient-Transformer-Toolkit/
├── Codes/cli/                     # Command‑line interface & config profiles
│   ├── cli.py                     # main CLI script
│   ├── config_loader.py           # YAML config loading
│   └── configs/                   # pre‑built YAML profiles
├── framework/
│   ├── attentions/                # 30+ attention modules
│   │   ├── aurora.py, nova.py, ... # compression suites
│   │   ├── lsh_attention.py, ...  # sparse / adaptive
│   │   ├── grouped_query/         # GQA, MLA, QCQA, WGQA
│   │   ├── minimal_*.py           # minimal SVD variants
│   │   └── README.md
│   ├── compression/               # 20+ compressors
│   │   ├── core/                  # BaseCompressor, COMPRESSOR_REGISTRY
│   │   ├── svd_linear.py, ...     # SVD-based
│   │   ├── tt_linear.py, ...      # Tensor Train
│   │   ├── pruning/               # SparseGPT, SliceGPT, etc.
│   │   ├── kv_cache/              # DeltaKV, KVTC, etc.
│   │   └── README.md
│   ├── data/                      # Data loaders
│   │   ├── dataset_loader.py      # unified loader (GLUE, custom, synthetic)
│   │   ├── glue_loader.py
│   │   ├── custom_loader.py
│   │   ├── multi_choice.py
│   │   ├── preprocessing.py
│   │   ├── synthetic_generator.py
│   │   └── README.md
│   ├── models/                    # Model loading, wrapper, Unsloth integration
│   │   ├── model_loader.py
│   │   ├── base_model_wrapper.py
│   │   ├── hybrid_bert_svd.py
│   │   ├── compressed_bert.py
│   │   └── README.md
│   ├── analysis/                  # Metrics tracking, plotting, reports
│   │   ├── callbacks.py
│   │   ├── metrics.py
│   │   ├── plotting.py
│   │   ├── model_comparison.py
│   │   └── README.md
│   ├── routing/                   # Task detection routers
│   │   ├── keyword_router.py
│   │   ├── ml_router.py
│   │   └── ensemble_router.py
│   ├── experiment_logging/        # Professional logging
│   │   ├── logger.py, config.py, tracker.py, callbacks.py
│   │   └── README.md
│   └── (other utilities)
└── README.md                      # ← you are here
```

---

## ⚡ Features

### 🧠 Attention Zoo

- **30+ dropout‑in attention variants** covering:
  - **Compression suites:** AURORA, NOVA, PRISM, ORION, PHANTOM, OHA
  - **Minimal zero‑shot:** MinimalAURORA, MinimalNOVA, MinimalPRISM, MinimalSVD
  - **Linear attention:** Performer, Nyströmformer, ChelaSlay, RALA, ReGLA, ZeroS
  - **Sparse attention:** LSH, Routing, Cluster, NSA, MoBA, XAttention, AdaptiveSpan
  - **Hybrid SSM:** HymbaTransX, Jamba, Samba, GoldFinch, Griffin, GraniteHybrid
  - **Hardware‑optimised:** FlashAttention‑3, FlashAttention‑3 FP8
  - **Grouped Query:** GQA, MLA, QCQA, WGQA
- All modules are **drop‑in** and **registered** under a unique key (e.g., `"aurora"`, `"nova"`).
- Full catalogue → [`framework/attentions/README.md`](framework/attentions/README.md)

### 📦 Compression Engine

- **SVD‑based:** energy‑threshold rank selection, factorised linear layers
- **Tensor Train:** true TT decomposition (torchtt) or SVD fallback
- **Kronecker (SoKA):** low‑rank approximation via Kronecker product
- **Pruning:** SparseGPT, SliceGPT, CoMe, FLAP, Mamba‑Shedder, OATS, ProxSparse, AST, ARMOR, ADMM, Bonsai
- **Quantisation:** UniQL, MLoRQ, KMeans, FAQ, CoopQ, Astro, SLiM
- **KV‑Cache:** DeltaKV, KVTC, HybridKV, ZeroKV‑Neo, CacheNotes
- All compressors are registered in `COMPRESSOR_REGISTRY` and accessible via CLI
- Full catalogue → [`framework/compression/README.md`](framework/compression/README.md)

### 📊 Data Pipeline

- Unified `ExperimentConfig`‑driven dataset loading
- Supports **GLUE benchmarks** (local Parquet files), **custom CSV/JSON/Parquet**, **multi‑choice** (ARC, HellaSwag), and **synthetic generation**
- Automatic column detection, tokenization, and splitting
- Full docs → [`framework/data/README.md`](framework/data/README.md)

### 🧰 CLI with Config Profiles

- YAML‑based config profiles with **precedence**: defaults → YAML → env vars → CLI args
- **Interactive mode** for guided configuration
- Pre‑built profiles for **tasks**, **models**, **attentions**, and **presets** (fast / accurate / extreme compress)
- Commands: `train`, `evaluate`, `compress`, `generate‑synthetic`, `list‑configs`, `init‑config`
- Full docs → [`Codes/cli/README.md`](Codes/cli/README.md)

### 📋 Experiment Logging

- Per‑run timestamped directories (e.g., `logs/experiment_2026‑05‑03_10‑38‑19/`)
- Three output formats: colored console, plain text log, structured JSONL
- **MetricsTracker** to record training/evaluation metrics (JSON/CSV export)
- **Trainer callback** to automatically capture HuggingFace logs and save training plots
- Full docs → [`framework/experiment_logging/README.md`](framework/experiment_logging/README.md)

### 📈 Analysis & Plotting

- Automatic training history plots
- Comprehensive 3×3 grid comparing accuracy vs rank, speedup, parameter reduction, etc.
- Model comparison across multiple runs
- HTML, CSV, and LaTeX report generation
- Full docs → [`framework/analysis/README.md`](framework/analysis/README.md)

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Optional (for full feature support):
```bash
pip install unsloth            # 4‑bit loading
pip install questionary        # interactive CLI
pip install colorlog            # colored console logs
pip install torchtt             # true Tensor Train decomposition
```

### 2. Train a model with minimal compression (zero‑shot)

```bash
cd Codes/cli
python cli.py train --preset fast --task mrpc --epochs 3
```

This uses:
- `distilbert-base-uncased`
- `minimal_nova` attention (SVD‑factorised, 2‑4× parameter reduction)
- MRPC dataset

### 3. Experiment with a different attention type

```bash
python cli.py train --task mrpc --attention aurora --output aurora_mrpc
```

### 4. Compress a model

```bash
python cli.py compress --model distilbert-base-uncased --method tt --target-rank 16 -o model_compressed
```

### 5. Evaluate a trained model

```bash
python cli.py evaluate --model_path ./output --task mrpc
```

---

## 🧩 Module Overview

### Attention (partial list)

| Key | Name | Description |
|-----|------|-------------|
| `aurora` | AURORA | 10‑in‑1 compression suite |
| `nova` | NOVA | 8 techniques (ARMOR, AST, DeltaKV, …) |
| `prism_full` | PRISM Full | 6 integrated techniques |
| `orion` | ORION | SVD + head‑split kernel/full + Softpick |
| `performer` | Performer | FAVOR+ random features |
| `flash3` | FlashAttention3 | Tiled GPU kernel |
| … | *30+ more* | See `framework/attentions/README.md` |

### Compression

| Key | Method |
|-----|--------|
| `svd` | Truncated SVD |
| `tt` | Tensor Train decomposition |
| `soka` | Kronecker (SoKA) approximation |
| `uniql` | SVD + uniform quantisation |
| `sparsegpt` | One‑shot second‑order pruning |
| `delta_kv` | Residual‑based KV compression |
| … | *20+ compressors* |

### Data

- GLUE tasks (CoLA, SST‑2, MRPC, QQP, STSB, MNLI, QNLI, RTE, WNLI)
- Custom CSV/JSON/Parquet
- Multi‑choice (ARC, HellaSwag)
- Synthetic data generation

---

## 🔧 Adding a New Module

1. **Attention:** Copy `framework/attentions/custom/template_attention.py`, implement your class, and register it with `@ATTENTION_REGISTRY.register("your_key")`.
2. **Compressor:** Copy `framework/compression/custom/template_compressor.py`, fill in the `compress()` method, and register it.
3. **CLI config:** Add a YAML file to `Codes/cli/configs/`.
4. **Data source:** Extend `framework/data/dataset_loader.py` if needed.

All modules are automatically picked up by the registry and the CLI.

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file.

## 🙏 Acknowledgements

Built upon numerous open‑source projects and research papers: HuggingFace Transformers, FlashAttention, Performer, Reformer, AURORA, NOVA, PRISM, and many more.

---

*Build faster, smaller, smarter Transformers.*
