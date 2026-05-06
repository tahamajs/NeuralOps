# Efficient Transformer Toolkit

**A modular, extensible framework for training, compressing, and deploying efficient Transformer models.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

The **Efficient Transformer Toolkit** brings together **30+ attention mechanisms**, **20+ model compression techniques**, **flexible data pipelines**, and a **powerful CLI** – all designed to make Transformer efficiency research and deployment fast and reproducible.

With this toolkit you can:
- **Experiment** with dozens of attention types (from vanilla to extreme hybrids)
- **Compress** pretrained models with zero‑shot or training‑based methods
- **Train** and **evaluate** on GLUE benchmarks or custom datasets
- **Automate** everything through a unified CLI with YAML config profiles

---

## 📁 Project Structure

```
Efficient-Transformer-Toolkit/
├── Codes/cli/                     # Command‑line interface & config profiles
├── framework/
│   ├── attentions/                # 30+ attention modules
│   ├── compression/               # 20+ compressors (SVD, TT, pruning, etc.)
│   ├── data/                      # Data loaders (GLUE, custom, multi‑choice, synthetic)
│   ├── models/                    # Model loading, wrapper, Unsloth integration
│   ├── analysis/                  # Metrics tracking, plotting, report generation
│   ├── routing/                   # Task detection (keyword, ML, ensemble)
│   └── experiment_logging/        # Professional logging with timestamped runs
└── README.md
```

---

## ⚡ Features

### 🧠 Attention Zoo
- **Compression suites:** AURORA, NOVA, PRISM, ORION, PHANTOM, OHA
- **Minimal zero‑shot:** MinimalAURORA, MinimalNOVA, MinimalPRISM, MinimalSVD
- **Linear attention:** Performer, Nyströmformer, ChelaSlay, RALA, ReGLA, ZeroS
- **Sparse attention:** LSH, Routing, Cluster, NSA, MoBA, XAttention, AdaptiveSpan
- **Hybrid SSM:** HymbaTransX, Jamba, Samba, GoldFinch, Griffin, GraniteHybrid
- **Hardware‑optimised:** FlashAttention‑3, FlashAttention‑3 FP8
- **Grouped Query:** GQA, MLA, QCQA, WGQA
- All modules are **drop‑in** and registered under a unique key (e.g., `"aurora"`, `"nova"`).

### 📦 Compression Engine
- **SVD‑based:** energy‑threshold rank selection, factorised linear layers
- **Tensor Train:** true TT decomposition (torchtt) or SVD fallback
- **Kronecker (SoKA):** low‑rank approximation via Kronecker product
- **Pruning:** SparseGPT, SliceGPT, CoMe, FLAP, Mamba‑Shedder, OATS, ProxSparse, AST, ARMOR, ADMM, Bonsai
- **Quantisation:** UniQL, MLoRQ, KMeans, FAQ, CoopQ, Astro
- All compressors are registered in `COMPRESSOR_REGISTRY` and accessible via CLI

### 📊 Data Pipeline
- Unified `ExperimentConfig`‑driven dataset loading
- Supports **GLUE benchmarks**, **custom CSV/JSON/Parquet**, **multi‑choice**, and **synthetic generation**
- Automatic column detection, tokenization, and splitting

### 🧰 CLI with Config Profiles
- YAML‑based config profiles with **precedence**: defaults → YAML → env vars → CLI args
- **Interactive mode** for guided configuration
- Pre‑built profiles for **tasks**, **models**, **attentions**, and **presets**
- Commands: `train`, `evaluate`, `compress`, `generate‑synthetic`, `list‑configs`, `init‑config`

### 📋 Experiment Logging
- Per‑run timestamped directories
- Three output formats: colored console, plain text log, structured JSONL
- **MetricsTracker** for step‑wise metrics (JSON/CSV export)
- **Trainer callback** to capture logs and save training plots

### 📈 Analysis & Plotting
- Automatic training history plots
- Comprehensive 3×3 grid comparing accuracy vs rank, speedup, etc.
- Model comparison across multiple experiments
- HTML, CSV, and LaTeX report generation

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Optional:
```bash
pip install unsloth            # for 4‑bit loading
pip install questionary        # for interactive CLI
pip install colorlog            # for colored console output
```

### 2. Train a model with minimal compression (zero‑shot)

```bash
cd Codes/cli
python cli.py train --preset fast --task mrpc --epochs 3
```

### 3. Experiment with a different attention type

```bash
python cli.py train --task mrpc --attention aurora --output aurora_mrpc
```

### 4. Compress a model

```bash
python cli.py compress --model distilbert-base-uncased --method tt --target-rank 16 -o model_compressed
```

---

## 🧩 Module Overview

**Attention** (partial list):

| Key | Name | Description |
|-----|------|-------------|
| `aurora` | AURORA | 10‑in‑1 compression suite |
| `nova` | NOVA | 8 techniques (ARMOR, AST, DeltaKV, …) |
| `prism_full` | PRISM Full | 6 integrated techniques |
| `performer` | Performer | FAVOR+ random features |
| `flash3` | FlashAttention3 | Tiled GPU kernel |
| … | *30+ more* | See `framework/attentions/README.md` |

**Compression**:

| Key | Method |
|-----|--------|
| `svd` | Truncated SVD |
| `tt` | Tensor Train decomposition |
| `uniql` | SVD + uniform quantisation |
| `sparsegpt` | One‑shot second‑order pruning |
| … | *20+ compressors* |

---

## 🔧 Adding a New Module

1. **Attention:** Copy the template, implement your class, register with `@ATTENTION_REGISTRY.register("your_key")`.
2. **Compressor:** Copy the template, fill in `compress()`, and register.
3. **CLI config:** Add a YAML file to `Codes/cli/configs/`.
4. **Data source:** Extend `framework/data/dataset_loader.py`.

All modules are automatically picked up by the registry and the CLI.

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file.

## 🙏 Acknowledgements

Built upon numerous open‑source projects and research papers: HuggingFace Transformers, FlashAttention, Performer, Reformer, AURORA, NOVA, PRISM, and many more.

---

*Build faster, smaller, smarter Transformers.*