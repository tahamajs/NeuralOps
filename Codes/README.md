# SVD‑Lin Studio – Complete Transformer Research Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.36+-yellow.svg)](https://github.com/huggingface/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**SVD‑Lin Studio** is a unified framework for **exploring, compressing, and visualising transformer models** using **SVD‑based low‑rank approximations** and **advanced compression techniques**. It supports:

- **Multiple attention mechanisms** (SVD, Nyström, Performer, RALA, NHA, FlashAttention‑3)
- **Many compression methods** (standard SVD, adaptive SVD, head pruning, layer skipping, quantized SVD, tensor train, CP decomposition, etc.)
- **Full training pipeline** with baseline, LoRA, DoRA, R‑Drop, R3F, Mixout, LLRD
- **Interactive visualisation** of attention maps, hidden states, gradients, embeddings (Plotly dashboards)
- **Synthetic data generation** (multi‑domain, noise, 10 reasoning tasks, irony, negation, comparative, etc.)
- **Smart router ensemble** (keyword‑based + ML + weighted fusion) for multi‑task adaptation
- **SVD‑Optimized Architecture (SOA)** – low‑rank transformers with a shared global basis
- **Benchmarking & analysis** – speed, memory, accuracy‑rank tradeoffs, layer‑wise inspection

---

## 📖 Table of Contents

- [SVD‑Lin Studio – Complete Transformer Research Platform](#svdlin-studio--complete-transformer-research-platform)
  - [📖 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [🛠️ Installation](#️-installation)
    - [1. Clone the repository](#1-clone-the-repository)
    - [2. Create a virtual environment (optional)](#2-create-a-virtual-environment-optional)
    - [3. Install dependencies](#3-install-dependencies)
    - [4. Install the package in editable mode](#4-install-the-package-in-editable-mode)
  - [🚀 Quick Start](#-quick-start)
    - [CLI – run a compression + fine‑tuning experiment](#cli--run-a-compression--finetuning-experiment)
    - [Python – replace attention with SVD and analyse](#python--replace-attention-with-svd-and-analyse)
    - [Generate synthetic data](#generate-synthetic-data)
  - [📁 Project Structure](#-project-structure)
  - [📖 Detailed Usage](#-detailed-usage)
    - [Command‑Line Interface (CLI)](#commandline-interface-cli)
    - [Python API](#python-api)
      - [Load and compress a model](#load-and-compress-a-model)
      - [Fine‑tune with LoRA + R‑Drop](#finetune-with-lora--rdrop)
      - [Generate reasoning dataset (10 tasks)](#generate-reasoning-dataset-10-tasks)
      - [Visualise attention](#visualise-attention)
  - [🧩 Component Guide](#-component-guide)
    - [Compression](#compression)
    - [Attentions](#attentions)
    - [Training \& Methods](#training--methods)
    - [Routing \& Ensemble](#routing--ensemble)
    - [SOA (SVD‑Optimized Architecture)](#soa-svdoptimized-architecture)
    - [Data \& Synthetic Generation](#data--synthetic-generation)
    - [Visualisation \& Analysis](#visualisation--analysis)
  - [📊 Evaluation \& Benchmarking](#-evaluation--benchmarking)
  - [🔧 Configuration](#-configuration)
  - [📝 Examples](#-examples)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)

---

## ✨ Features

| Category | Included |
|----------|----------|
| **Attention mechanisms** | SVD, Nyström, Performer, Linformer, RALA, Native Hybrid Attention (NHA), FlashAttention‑3, Hamming Attention, etc. |
| **Compression methods** | standard_svd, head_pruning, layer_skipping, quantized_svd, blockwise_lowrank, randomized_svd, tensor_train_decomp, cp_decomposition, dynamic_token_pruning, spectral_clustering_prune, hash_quantization, moe_sparsification, hybrid_advanced |
| **Training techniques** | Baseline, LoRA, DoRA, R‑Drop, R3F, Mixout, LLRD, Lion optimiser |
| **Routing** | Keyword‑based router, TF‑IDF + Logistic Regression, FastText neural router, weighted ensemble, task‑aware attention |
| **SVD‑Optimized Architecture (SOA)** | Shared global basis from BERT QKV / hidden states, full migration, ONNX export |
| **Data generation** | Multi‑domain (movies, tech, health, politics, etc.), noise injection, 10 reasoning tasks (deductive, inductive, abductive, analogical, counterfactual, causal, temporal, spatial, quantitative, social), irony, negation, comparative, long‑context, paraphrases |
| **Visualisation** | Attention heatmaps (interactive Plotly), attention entropy, hidden state PCA, gradient norms, training curves, confusion matrices, rank‑comparison plots |
| **Analysis** | Layer‑wise analysis (cosine similarity, effective rank, intrinsic dimension), speed benchmarking, memory profiling, report generation (HTML, CSV, LaTeX) |
| **Hardware** | MPS (Apple Silicon), CUDA, CPU with gradient checkpointing and memory optimisations |

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/svd-lin-studio.git
cd svd-lin-studio
```

### 2. Create a virtual environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you plan to use **Lion** optimiser, install it separately:

```bash
pip install lion-pytorch
```

### 4. Install the package in editable mode

```bash
pip install -e .
```

Now you can use the `svd-lin` command or import `studio` in Python.

---

## 🚀 Quick Start

### CLI – run a compression + fine‑tuning experiment

```bash
# Compress BERT with rank 32 on the CoLA task
python -m studio.cli compress --model bert-base-uncased --rank 32 --finetune --task cola
```

### Python – replace attention with SVD and analyse

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from studio.attentions import replace_attention_in_model
from studio.visualization import AttentionDashboard

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Replace all attention layers with SVDLinearAttention (rank 64)
model = replace_attention_in_model(model, "svd", rank=64)

# Launch interactive dashboard
dashboard = AttentionDashboard(model, tokenizer)
dashboard.analyze_text("The quick brown fox jumps over the lazy dog.")
```

### Generate synthetic data

```bash
# Generate universal multi‑domain dataset (movies, tech, health, politics)
python -m studio.cli generate-data --type universal --samples 5000 --output_dir ./data

# Generate 10 reasoning tasks (Persian text)
python -m studio.cli generate-data --type reasoning --samples 1000
```

---

## 📁 Project Structure

```
studio/
├── cli.py                 # unified command‑line interface
├── config.py              # central configuration (hardware, data, model, training, SVD)
├── data/
│   ├── loader.py          # GLUE, custom CSV, dataset discovery
│   ├── dataset.py         # SentimentDataset, MultiChoiceDataset
│   ├── synthetic.py       # core synthetic generation (universal, comprehensive, advanced)
│   ├── synthetic_advanced.py   # 10 reasoning tasks (deductive, inductive, …)
│   ├── synthetic_task_specific.py  # irony, long context, domain shift, paraphrase, noise
│   └── analyze_datasets.py       # class balance analysis & plotting
├── compression/
│   ├── svd_linear.py      # SVDLinearLayer
│   ├── adaptive_compressor.py  # AdvancedZeroShotCompressor (14 methods)
│   ├── compress_bert.py        # compress_bert_true_svd
│   └── patterns.py        # architecture detection (BERT, DistilBERT, RoBERTa, etc.)
├── attentions/
│   ├── svd_attention.py   # SVDLinearAttention (drop‑in replacement)
│   ├── compressed_attention.py  # CompressedAttentionLayer (zero‑shot)
│   ├── utils.py           # replace_attention_in_model
│   └── [other attentions] # Nyström, Performer, RALA, NHA, …
├── routing/
│   ├── keyword_router.py
│   ├── ml_router.py
│   ├── ensemble_router.py
│   └── smart_router.py    # SmartRouter with task detection & fallback
├── ensemble/
│   ├── weighted_ensemble.py   # SmartWeightedEnsemble, TaskAwareAttention
│   └── robust_ensemble.py     # RobustEnsembleModel (dynamic shape alignment)
├── training/
│   ├── trainer.py         # train_model, fine_tune_compressed_model
│   ├── methods.py         # LoRA, DoRA, Mixout, LLRD, rdrop_loss, MethodTrainer
│   ├── optimizer.py       # AdamW, Lion
│   └── scheduler.py
├── analysis/
│   ├── evaluate.py        # evaluate_model, benchmark_model
│   ├── metrics.py         # accuracy, f1, pearson, spearman, matthews
│   ├── layerwise_analyzer.py
│   └── reporting.py       # HTML, CSV, LaTeX reports
├── visualization/
│   ├── dashboard.py       # AttentionDashboard, ModelVisualizer
│   └── [other plotting utilities]
├── soa/                   # SVD‑Optimized Architecture
│   ├── config.py, attention.py, model.py
│   ├── svd_migration.py   # compute_global_basis, migrate_bert_to_soa
│   └── export_onnx.py
└── utils/
    ├── device.py, logger.py, checkpoint.py, helpers.py
```

---

## 📖 Detailed Usage

### Command‑Line Interface (CLI)

All commands are run via `python -m studio.cli <command>`.

| Command | Description | Example |
|---------|-------------|---------|
| `compress` | Apply SVD compression (optionally fine‑tune) | `compress --model bert-base-uncased --rank 32 --finetune --task cola` |
| `ensemble` | Run weighted ensemble on a dataset | `ensemble --dataset ./data/combined.csv` |
| `router` | Run keyword‑based router ensemble | `router --dataset ./data` |
| `benchmark` | Run full compression benchmark across ranks & models | `benchmark` |
| `soa` | Build SOA model from BERT | `soa --model bert-base-uncased --rank 64` |
| `generate-data` | Generate synthetic datasets | `generate-data --type universal --samples 5000` |
| `analyze-data` | Check class balance of CSV files | `analyze-data --data_dir ./data --output report.png` |
| `benchmark-speed` | Compare inference speed | `benchmark-speed --original ./orig --compressed ./comp --rank 32` |

### Python API

#### Load and compress a model

```python
from studio.compression import AdvancedZeroShotCompressor
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
compressor = AdvancedZeroShotCompressor(method="standard_svd", base_threshold=0.9)
compressed_model, stats, reduction = compressor.compress_model(model, "standard_svd")
```

#### Fine‑tune with LoRA + R‑Drop

```python
from studio.training.methods import get_trainer_for_method
from studio.data.loader import load_dataset_from_config

train_ds, val_ds = load_dataset_from_config({"name": "cola", "type": "glue"}, tokenizer)
trainer = get_trainer_for_method(
    method="lora+rdrop",
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    output_dir="./output",
    learning_rate=2e-5,
    epochs=3,
    lora_r=8,
)
trainer.train()
```

#### Generate reasoning dataset (10 tasks)

```python
from studio.data.synthetic_advanced import generate_reasoning_datasets
from pathlib import Path

datasets = generate_reasoning_datasets(Path("./reasoning_data"), samples_per_task=1000)
```

#### Visualise attention

```python
from studio.visualization import AttentionDashboard

dashboard = AttentionDashboard(model, tokenizer)
dashboard.analyze_text("This movie was incredible!", output_html="attn.html")
```

---

## 🧩 Component Guide

### Compression

All compressors live in `studio.compression`. The main classes:

- `SVDLinearLayer` – replaces a linear layer with a low‑rank SVD approximation.
- `AdvancedZeroShotCompressor` – applies one of 14 compression strategies (see `config.compression_methods`). Uses singular value energy to decide ranks.
- `compress_bert_true_svd` – compresses all linear layers of a BERT model (except embeddings/classifier).

### Attentions

Attention mechanisms are in `studio.attentions`. You can replace any model’s attention using:

```python
from studio.attentions import replace_attention_in_model

model = replace_attention_in_model(model, "svd", rank=64)
```

Supported attention types: `svd`, `nystrom`, `performer`, `linformer`, `rala`, `nha`, `flash3`, `hamming`, and more.

### Training & Methods

- `studio.training.trainer` – low‑level training loop.
- `studio.training.methods` – includes `apply_lora`, `apply_dora`, `apply_mixout`, `get_llrd_param_groups`, `rdrop_loss`, and `MethodTrainer` (which handles R‑Drop/R3F automatically).
- `get_trainer_for_method` – factory that builds a `Trainer` with the correct method (baseline, lora, dora, rdrop, r3f, mixout, llrd, lora+rdrop).

### Routing & Ensemble

- **KeywordRouter** – rule‑based detection (comparative, negation, irony).
- **MLRouter** – TF‑IDF + Logistic Regression (trainable).
- **SmartRouter** – combines keyword detection with specialised compressed models for each task.
- **Weighted Ensemble** – `SmartWeightedEnsemble` with dynamic per‑sample weights and task‑aware attention.

### SOA (SVD‑Optimized Architecture)

- **SOABert** – low‑rank transformer that operates on a shared latent basis.
- **Migration** – compute global basis from BERT’s QKV weights or from hidden states, then transfer weights.
- **Export** – ONNX export for deployment.

### Data & Synthetic Generation

- **GLUE loader** – loads local Parquet files; fallback to HuggingFace hub.
- **Synthetic** – three main generators:
  1. `run_universal_generation` – multi‑domain (movies, tech, health, politics) with templates.
  2. `run_comprehensive_generation` – 7 domains, 3 complexity levels, 5 noise types.
  3. `run_advanced_generation` – 5 strategies (synthetic, paraphrased, contextual, noise‑robust, complex).
- **Reasoning tasks** (advanced) – 10 distinct cognitive pathways (deduction, induction, abduction, analogy, counterfactual, causal, temporal, spatial, quantitative, social); text in Persian.
- **Task‑specific** – irony, negation, comparative, long‑range dependency, aspect‑based, conditional, noise robustness, domain shift, paraphrase, temporal context.

### Visualisation & Analysis

- `AttentionDashboard` – interactive Plotly dashboard with attention heatmaps, entropy, hidden state PCA, gradient norms, layer importance.
- `LayerwiseAnalyzer` – computes cosine similarity between consecutive layers, effective rank (singular value entropy), intrinsic dimension (MLE).
- `ModelComparison` – compare multiple runs across ranks and tasks, generate pivot tables, ranking.
- Reporting – HTML/CSV/LaTeX summaries.

---

## 📊 Evaluation & Benchmarking

- **Speed benchmark** – `studio.evaluation.speed_benchmark.measure_inference_time` compares original vs compressed model.
- **Comprehensive benchmark** – `studio.evaluation.benchmark.run_full_benchmark` tests all methods, ranks, tasks (GLUE).
- **Class balance analysis** – `studio.data.analyze_datasets.plot_distribution_report` creates a bar chart of label distributions.

Example: run full benchmark (quick mode)

```python
from studio.evaluation.benchmark import run_full_benchmark

df = run_full_benchmark(epochs=1, batch_size=8)
df.to_csv("quick_bench.csv")
```

---

## 🔧 Configuration

The central `studio.config` module provides a dataclass `MainConfig` with nested sections:

- `hardware` – device, gradient checkpointing, mixed precision.
- `data` – GLUE path, max length, batch size, sample limits.
- `model` – model name, attention type, rank, num labels.
- `training` – epochs, learning rate, warmup, clipping.
- `svd` – energy threshold, min/max rank, compression method.
- `experiment` – tasks, seeds, output directories.

You can override values by passing arguments to the CLI or by modifying `config.yaml`.

---

## 📝 Examples

Full examples are available in `examples/` (not part of the merged repository, but can be created):

- `01_compress_finetune.py` – load BERT, apply SVD, fine‑tune on CoLA.
- `02_synthetic_generation.py` – create multi‑domain dataset.
- `03_attention_dashboard.py` – interactive attention analysis.
- `04_router_ensemble.py` – route text to specialised compressed models.
- `05_soa_migration.py` – convert BERT to SOA and test.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request on GitHub. When adding new attention/compression methods, register them with the appropriate registry (`ATTENTION_REGISTRY`, `COMPRESSOR_REGISTRY`). Write tests and update the documentation.

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

**SVD‑Lin Studio** – Your all‑in‑one platform for transformer research, compression, and visualisation. 🚀