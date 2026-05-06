# Efficient Transformer Toolkit

**A modular, extensible framework for training, compressing, and deploying efficient Transformer models.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

The **Efficient Transformer Toolkit** brings together **30+ state‑of‑the‑art attention mechanisms**, **15+ model compression techniques**, **flexible data pipelines**, and a **powerful CLI** – all designed to make Transformer efficiency research and deployment fast and reproducible.

With this toolkit you can:

- **Experiment** with dozens of attention types (from vanilla to extreme hybrids)
- **Compress** pretrained models with zero‑shot or training‑based methods
- **Train** and **evaluate** on GLUE benchmarks or custom datasets
- **Automate** everything through a unified command‑line interface with YAML config profiles

---

## 🗂️ Project Structure

```
Efficient-Transformer-Toolkit/
├── Codes/cli/                     # Command‑line interface & config profiles
├── framework/
│   ├── attentions/                # 30+ attention modules (suites, sparse, hybrid SSM, etc.)
│   ├── compression/               # Model compression (SVD, TT, pruning, quantisation)
│   ├── data/                      # Data loaders (GLUE, custom, multi‑choice, synthetic)
│   ├── models/                    # Model loading, wrapper, injection, Unsloth integration
│   ├── analysis/                  # Metrics tracking, plotting, report generation
│   ├── routing/                   # Task detection (keyword, ML, ensemble)
│   └── logging/                   # Professional logging with timestamped runs
└── README.md
```

---

## ⚙️ Features

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

### 📦 Compression Engine

- **SVD‑based:** Energy‑threshold rank selection, factorised linear layers
- **Tensor Train:** True TT decomposition (torchtt) or SVD fallback
- **Kronecker (SoKA):** Low‑rank approximation via Kronecker product
- **Pruning:** SparseGPT, SliceGPT, CoMe, FLAP, Mamba‑Shedder, OATS, ProxSparse, AST, ARMOR, ADMM, Bonsai
- **Quantisation:** UniQL, MLoRQ, KMeans, FAQ, CoopQ, Astro
- All compressors are registered in `COMPRESSOR_REGISTRY` and accessible via CLI

### 📊 Data Pipeline

- Unified `ExperimentConfig`‑driven dataset loading
- Supports **GLUE benchmarks** (local Parquet files), **custom CSV/JSON/Parquet**, **multi‑choice** (ARC, HellaSwag), and **synthetic generation**
- Automatic column detection, tokenization, and splitting

### 🧰 CLI with Config Profiles

- YAML‑based config profiles with **precedence**: defaults → YAML → env vars → CLI args
- **Interactive mode** for guided configuration
- Pre‑built profiles for **tasks**, **models**, **attentions**, and **presets** (fast / accurate / extreme compress)
- Commands: `train`, `evaluate`, `compress`, `generate‑synthetic`, `list‑configs`, `init‑config`

### 📋 Experiment Logging

- Per‑run timestamped directories (e.g., `logs/experiment_2026‑05‑03_10‑38‑19/`)
- Three output formats: **colored console**, **plain text log**, **structured JSONL**
- **MetricsTracker** to record training/evaluation metrics (JSON/CSV export)
- **Trainer callback** to automatically capture HuggingFace logs and save training plots

### 📈 Analysis & Plotting

- Automatic training history plots
- Comprehensive 3×3 grid comparing accuracy vs rank, speedup, parameter reduction, etc.
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

This uses:
- `distilbert-base-uncased`
- `minimal_nova` attention (SVD‑factorised, 2‑4× parameter reduction)
- MRPC dataset

### 3. Experiment with a different attention type

```bash
python cli.py train --task mrpc --attention aurora --output aurora_mrpc
```

### 4. Use a config file for reproducibility

```bash
python cli.py init-config --preset accurate -o my_exp.yaml
# edit my_exp.yaml if needed
python cli.py train --config my_exp.yaml
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
| `nova` | NOVA | 8 techniques (ARMOR, AST, DeltaKV, KVT, Softpick, Subspace) |
| `prism_full` | PRISM Full | 6 integrated techniques |
| `orion` | ORION | SVD + head‑split kernel/full + Softpick |
| `phantom` | PHANTOM | CUR + LSH + Softpick |
| `oha` | OHA | Sliding window + low‑rank global + gate |
| `performer` | Performer | FAVOR+ orthogonal random features |
| `lsh` | LSH Attention | Reformer‑style LSH |
| `flash3` | FlashAttention3 | Tiled GPU kernel |
| `goldfinch_ahn` | GoldFinch AHN | Sliding window + GatedDeltaNet |
| … | *30+ more* | See `framework/attentions/README.md` |

### Compression

| Key | Method |
|-----|--------|
| `svd` | Truncated SVD per layer |
| `tt` | Tensor Train decomposition |
| `soka` | Kronecker (SoKA) approximation |
| `uniql` | SVD + uniform quantisation |
| `salr` | Sparse low‑rank |
| `mgie` | Entropy‑driven rank selection |
| `sparsegpt` | One‑shot second‑order pruning |
| `flap` | Fluctuation‑based adaptive pruning |
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
3. **CLI config:** Add a YAML file to `Codes/cli/configs/attentions/` (or `tasks/`, `models/`).
4. **Data source:** Extend `framework/data/dataset_loader.py` if needed.

All modules are automatically picked up by the registry and the CLI.

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This toolkit builds upon numerous open‑source projects and research papers:

- HuggingFace Transformers
- FlashAttention, Performer, Reformer, Nyströmformer, Linformer
- AURORA, NOVA, PRISM, ORION, PHANTOM, Hymba, Jamba, Samba, etc.
- SparseGPT, SliceGPT, CoMe, FLAP, Mamba‑Shedder, OATS
- And many more…

