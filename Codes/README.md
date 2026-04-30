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
  - [📖 **Complete Description of Each Part of SVD‑Lin Studio**](#-complete-description-of-each-part-of-svdlin-studio)
  - [🧱 **Root Files (`cli.py`, `config.py`)**](#-root-files-clipy-configpy)
    - [`studio/cli.py`](#studioclipy)
    - [`studio/config.py`](#studioconfigpy)
  - [📂 **Data Module (`studio/data/`)**](#-data-module-studiodata)
    - [`loader.py`](#loaderpy)
    - [`dataset.py`](#datasetpy)
    - [`synthetic.py`](#syntheticpy)
    - [`analyze_datasets.py`](#analyze_datasetspy)
  - [🗜️ **Compression Module (`studio/compression/`)**](#️-compression-module-studiocompression)
    - [`svd_linear.py`](#svd_linearpy)
    - [`adaptive_compressor.py`](#adaptive_compressorpy)
    - [`compress_bert.py`](#compress_bertpy)
    - [`patterns.py`](#patternspy)
  - [👁️ **Attentions Module (`studio/attentions/`)**](#️-attentions-module-studioattentions)
    - [`svd_attention.py`](#svd_attentionpy)
    - [`compressed_attention.py`](#compressed_attentionpy)
    - [`utils.py`](#utilspy)
  - [🧭 **Routing Module (`studio/routing/`)**](#-routing-module-studiorouting)
    - [`keyword_router.py`](#keyword_routerpy)
    - [`ml_router.py`](#ml_routerpy)
    - [`ensemble_router.py`](#ensemble_routerpy)
    - [`smart_router.py`](#smart_routerpy)
  - [👥 **Ensemble Module (`studio/ensemble/`)**](#-ensemble-module-studioensemble)
    - [`weighted_ensemble.py`](#weighted_ensemblepy)
    - [`robust_ensemble.py`](#robust_ensemblepy)
  - [🏋️ **Training Module (`studio/training/`)**](#️-training-module-studiotraining)
    - [`trainer.py`](#trainerpy)
    - [`methods.py`](#methodspy)
    - [`optimizer.py`](#optimizerpy)
    - [`scheduler.py`](#schedulerpy)
  - [📊 **Analysis Module (`studio/analysis/`)**](#-analysis-module-studioanalysis)
    - [`evaluate.py`](#evaluatepy)
    - [`metrics.py`](#metricspy)
    - [`layerwise_analyzer.py`](#layerwise_analyzerpy)
    - [`reporting.py`](#reportingpy)
  - [🎨 **Visualization Module (`studio/visualization/`)**](#-visualization-module-studiovisualization)
    - [`dashboard.py`](#dashboardpy)
  - [🚀 **SOA (SVD‑Optimized Architecture) Module (`studio/soa/`)**](#-soa-svdoptimized-architecture-module-studiosoa)
    - [`config.py`](#configpy)
    - [`attention.py`](#attentionpy)
    - [`model.py`](#modelpy)
    - [`svd_migration.py`](#svd_migrationpy)
    - [`export_onnx.py`](#export_onnxpy)
  - [⚡ **Evaluation Module (`studio/evaluation/`)**](#-evaluation-module-studioevaluation)
    - [`benchmark.py`](#benchmarkpy)
    - [`speed_benchmark.py`](#speed_benchmarkpy)
  - [🛠️ **Utils Module (`studio/utils/`)**](#️-utils-module-studioutils)
    - [`device.py`](#devicepy)
    - [`logger.py`](#loggerpy)
    - [`checkpoint.py`](#checkpointpy)
    - [`helpers.py`](#helperspy)
  - [✅ **Summary**](#-summary)
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
## 📖 **Complete Description of Each Part of SVD‑Lin Studio**

Below is a **detailed, technical description** of every component in the project structure. This will help you understand the role of each file and how they work together.

---

## 🧱 **Root Files (`cli.py`, `config.py`)**

### `studio/cli.py`
**Purpose:** Unified command‑line interface for the entire framework.  
**Key Features:**
- Parses commands: `generate-data`, `analyze-data`, `compress`, `ensemble`, `router`, `benchmark`, `soa`, `benchmark-speed`.
- Routes each command to the appropriate internal function.
- Supports options for dataset type, number of samples, model name, rank, etc.
- Example usage: `python -m studio.cli generate-data --type universal --samples 5000`

### `studio/config.py`
**Purpose:** Central configuration management.  
**Key Classes:**
- `HardwareConfig`: device (auto/cuda/mps), gradient checkpointing, mixed precision.
- `DataConfig`: GLUE path, max sequence length, batch sizes, number of workers.
- `ModelConfig`: model name, attention type, rank, number of labels, cache directory.
- `TrainingConfig`: epochs, learning rate, warmup, gradient clipping.
- `SVDConfig`: energy threshold, min/max rank, compression method.
- `ExperimentConfig`: tasks list, seed, output/log directories.
- `MainConfig`: container for all above, with `get_device()` method.
- Creates global `config` instance and `DEVICE` (torch device) for use throughout the project.

---

## 📂 **Data Module (`studio/data/`)**

### `loader.py`
**Purpose:** Load datasets from GLUE (local parquet) or custom CSV files.  
**Key Functions:**
- `load_glue_task(task_name, local_path)`: returns a HuggingFace `DatasetDict` from local Parquet files.
- `load_custom_dataset(path, text_col, label_col)`: loads a CSV and renames columns to "text" and "label".
- `discover_datasets(local_path)`: scans for available GLUE tasks or CSV files in `./data`.
- `load_dataset_from_config(task_info, tokenizer, max_length)`: creates `SentimentDataset` objects for train/validation splits.

### `dataset.py`
**Purpose:** PyTorch `Dataset` classes for text classification.  
**Classes:**
- `SentimentDataset`: for single‑label classification (binary or multi‑class). Takes texts and labels, tokenizes on the fly.
- `MultiChoiceDataset`: for multiple‑choice tasks (e.g., ARC, HellaSwag). Handles JSON‑encoded choices and answers.

### `synthetic.py`
**Purpose:** **All‑in‑one synthetic data generation** (merged from 10+ scripts).  
**Key Functions:**
- `run_universal_generation()`: Creates multi‑domain datasets (movies, tech, health, politics) using templates, positive/negative vocabularies. Saves per‑domain CSV and a combined master file.
- `run_comprehensive_generation()`: Same universal vocabulary but adds diversity flags and different complexities.
- `run_advanced_generation()`: Combines universal and reasoning datasets.
- `generate_reasoning_datasets()`: Produces 10 reasoning tasks (deductive, inductive, abductive, analogical, counterfactual, causal, temporal, spatial, quantitative, social) – Persian text.
- `run_all_task_generations()`: Creates task‑specific datasets: irony, negation, comparative, long‑context, paraphrase, noise robustness.
- Helper: `clean_text()`, `safe_choice()`, domain vocabularies, templates.

### `analyze_datasets.py`
**Purpose:** Analyse class balance of generated datasets.  
**Function:**
- `plot_distribution_report(data_dir, output_image)`: Reads all CSV files in a directory, plots the percentage of class 0 and class 1 (based on the first 500 rows) as a bar chart, saves the figure.

---

## 🗜️ **Compression Module (`studio/compression/`)**

### `svd_linear.py`
**Purpose:** Core SVD‑based linear layer replacement.  
**Class:**
- `SVDLinearLayer(linear_layer, rank)`: performs SVD on the original weight matrix, stores `U`, `S`, `Vh` as low‑rank parameters. The forward pass reconstructs the weight (`U @ diag(S) @ Vh`) efficiently.

### `adaptive_compressor.py`
**Purpose:** Advanced zero‑shot compression with multiple strategies.  
**Class:**
- `AdvancedZeroShotCompressor(method, base_threshold, min_rank)`: supports 14 methods (`standard_svd`, `head_pruning`, `layer_skipping`, `quantized_svd`, etc.).  
  - `compress_model(model, method)`: analyses each `MultiheadAttention` layer, determines rank from singular value energy, replaces with `CompressedAttentionLayer`. Returns compressed model, statistics, and reduction percentage.

### `compress_bert.py`
**Purpose:** Simple SVD compression for all linear layers of a BERT‑like model.  
**Function:**
- `compress_bert_true_svd(model, rank)`: replaces every `nn.Linear` (except embeddings, pooler, classifier) with `SVDLinearLayer`.

### `patterns.py`
**Purpose:** Architecture detection and target layer patterns.  
**Constants & Functions:**
- `ARCH_PATTERNS`: dictionary mapping architecture names to lists of substrings that identify attention projection layers (e.g., `'attention.self.query'` for BERT, `'attention.q_lin'` for DistilBERT).
- `detect_architecture(model_name)`: returns architecture key based on model name.
- `get_layer_patterns(arch)`: returns the list of patterns for that architecture.

---

## 👁️ **Attentions Module (`studio/attentions/`)**

### `svd_attention.py`
**Purpose:** Drop‑in replacement attention using SVD.  
**Class:**
- `SVDLinearAttention(config, rank)`: inherits from `nn.Module`. Projects Q, K, V as usual, then optionally applies a learned low‑rank projection (simplified for demonstration; full version uses `torch.linalg.svd` on input hidden states). Returns output and attention weights.

### `compressed_attention.py`
**Purpose:** Zero‑shot attention compression by factorising QKV and output projections.  
**Class:**
- `CompressedAttentionLayer(original_layer, rank_qkv, rank_out, method)`: extracts weights, performs SVD, stores compressed factors (`U_qkv`, `S_qkv_sqrt`, `Vh_qkv`, etc.). The forward pass reconstructs Q, K, V on‑the‑fly, computes attention, then projects output with compressed out‑projection. Achieves significant parameter reduction without training.

### `utils.py`
**Purpose:** Utility to replace attention modules throughout a model.  
**Function:**
- `replace_attention_in_model(model, attention_type, rank)`: uses `ATTENTION_REGISTRY` (imported from elsewhere) to get the attention class, then replaces each module whose name contains "attention" (and has `query` attribute) with a new instance. Copies weights when possible.

---

## 🧭 **Routing Module (`studio/routing/`)**

### `keyword_router.py`
**Purpose:** Rule‑based task detection.  
**Class:**
- `KeywordRouter(keywords)`: uses a dictionary of keywords for `'comparative'`, `'negation'`, `'irony'`. `route(text)` returns predicted task and a dict of scores.
- `detect_task_type(text)`: convenience function.

### `ml_router.py`
**Purpose:** Machine learning‑based router.  
**Classes:**
- `MLRouter`: uses `TfidfVectorizer` and `LogisticRegression` (trainable). `train(texts, labels)`, `route(text)` returns class and probabilities.
- `FastTextRouter`: a simple neural network (embedding → average → linear). Not fully integrated but provided as a template.

### `ensemble_router.py`
**Purpose:** Weighted combination of multiple routers.  
**Class:**
- `EnsembleRouter(routers, weights)`: `route(text)` collects predictions from each router, weighs them by `weights`, and returns the class with the highest weighted score.

### `smart_router.py`
**Purpose:** Full pipeline that uses keyword detection to select a specialised compressed model.  
**Class:**
- `SmartRouter(models_config, device)`: loads a different compressed model for each task (irony, negation, comparative). `predict(text)` detects the task, runs the corresponding model, returns prediction and detected task.

---

## 👥 **Ensemble Module (`studio/ensemble/`)**

### `weighted_ensemble.py`
**Purpose:** Smart weighted ensemble with task‑aware attention.  
**Classes:**
- `TaskAwareAttention(num_tasks, hidden_dim)`: learns per‑task embeddings and a similarity matrix. Computes attention weights over models given the current task.
- `SmartWeightedEnsemble(models, num_tasks)`: maintains a performance matrix and uses confidence‑based dynamic weights. `combine_logits(task_id, all_logits, input_ids, attention_mask, method)` supports simple average, weighted, task‑aware attention, or hybrid.

### `robust_ensemble.py`
**Purpose:** Simple ensemble with dynamic shape alignment.  
**Class:**
- `RobustEnsembleModel(base_models, num_labels)`: stacks the models, learns a weight per model per label. Forward pass aligns logits to `num_labels` (truncates or pads), then computes weighted sum.

---

## 🏋️ **Training Module (`studio/training/`)**

### `trainer.py`
**Purpose:** Basic training loop and fine‑tuning helper.  
**Functions:**
- `train_model(model, train_loader, val_loader, device, optimizer, epochs)`: runs epochs, computes loss, logs accuracy and F1.
- `fine_tune_compressed_model(model, tokenizer, train_dataset, val_dataset, epochs, lr)`: creates data loaders, optimizer, and calls `train_model`.

### `methods.py`
**Purpose:** Advanced training methods (LoRA, DoRA, Mixout, LLRD, R‑Drop, R3F) and a custom trainer.  
**Functions / Classes:**
- `apply_lora(model, r, alpha, dropout)`: uses `peft` library.
- `apply_dora(model, r, alpha, dropout)`: custom implementation (to be filled).
- `apply_mixout(model, p)`: replaces linear layers with `MixoutLinear`.
- `get_llrd_param_groups(model, base_lr, decay, weight_decay)`: returns parameter groups with layer‑wise learning rates.
- `rdrop_loss(logits1, logits2, labels, is_regression, alpha)`: computes the R‑Drop loss (combination of cross‑entropy/MSE + KL divergence).
- `MethodTrainer`: a subclass of HuggingFace `Trainer` that overrides `compute_loss` to handle `"rdrop"` and `"r3f"` methods.
- `get_trainer_for_method(method, model, tokenizer, train_dataset, eval_dataset, output_dir, **kwargs)`: constructs a `TrainingArguments` and either a plain `Trainer` or a `MethodTrainer` with the correct loss.

### `optimizer.py`
**Purpose:** Optimizer factory.  
**Function:**
- `get_optimizer(model, lr, optimizer_name)`: supports `'adamw'` and `'lion'` (requires `lion-pytorch`). Returns the optimizer.

### `scheduler.py`
**Purpose:** Learning rate scheduler.  
**Function:**
- `get_scheduler(optimizer, num_training_steps, warmup_ratio)`: returns a linear schedule with warmup using HuggingFace’s `get_linear_schedule_with_warmup`.

---

## 📊 **Analysis Module (`studio/analysis/`)**

### `evaluate.py`
**Purpose:** Evaluate model on a dataset and benchmark speed.  
**Functions:**
- `evaluate_model(model, dataloader, device)`: returns accuracy and weighted F1.
- `benchmark_model(model, dataloader, device)`: measures average batch time.

### `metrics.py`
**Purpose:** Individual metric functions.  
**Functions:**
- `compute_accuracy(preds, labels)`
- `compute_f1(preds, labels, average='weighted')`
- `compute_matthews_corrcoef(preds, labels)`
- `compute_pearson(preds, labels)`
- `compute_spearman(preds, labels)`
- `compute_metrics_for_task(preds, labels, task_name)`: dispatches to appropriate metric(s) based on GLUE task.

### `layerwise_analyzer.py`
**Purpose:** Analyse hidden states and attention.  
**Class:**
- `LayerwiseAnalyzer(model, tokenizer)`: registers forward hooks to capture hidden states and attention maps.  
  - `compute_cosine_similarities()`: cosine similarity between consecutive layers’ average hidden states.
  - `compute_effective_ranks()`: entropy‑based rank of each layer’s representation.
  - `analyze(text)`: runs forward pass and returns a dictionary of metrics.

### `reporting.py`
**Purpose:** Generate summaries.  
**Functions:**
- `generate_csv_summary(all_results, output_path)`: converts a nested dict of results to CSV.
- `generate_html_report(results_dir, output_dir, df)`: creates a simple HTML table.
- (Optional) `generate_latex_table`: exports as LaTeX.

---

## 🎨 **Visualization Module (`studio/visualization/`)**

### `dashboard.py`
**Purpose:** Interactive attention dashboard.  
**Class:**
- `AttentionDashboard(model, tokenizer, device)`: registers hooks to capture attention maps, hidden states, and gradients.  
  - `analyze_text(text, layer, head, output_html)`: runs forward+backward, creates an interactive Plotly figure with six subplots:
    1. Attention heatmap (selected layer/head)
    2. Attention entropy per token
    3. Cosine similarity between consecutive hidden states
    4. Gradient norms per layer
    5. PCA projection of first vs last hidden state
    6. Attention received by the CLS token
  - Saves the dashboard as an HTML file.
- `ModelVisualizer`: static method `plot_training_curves(log_history, save_path)`.

---

## 🚀 **SOA (SVD‑Optimized Architecture) Module (`studio/soa/`)**

### `config.py`
**Purpose:** SOA configuration.  
**Class:**
- `SOAConfig`: vocab size, hidden size, number of layers, `k_rank` (basis rank), etc. Includes `from_pretrained` method to read from a HuggingFace config.

### `attention.py`
**Purpose:** Fused attention for SOA.  
**Class:**
- `SOAFusedAttention(hidden_size, num_heads, k_rank)`: takes a latent representation of shape `(batch, seq, k_rank)` and projects it to `Q, K, V` in full hidden space, then computes standard multi‑head attention. Output is projected back to `hidden_size`.

### `model.py`
**Purpose:** SOABert model and blocks.  
**Classes:**
- `SOABlock(cfg, shared_basis)`: pre‑norm transformer block with SVD attention and FFN. The input is projected to latent space by `h @ shared_basis`.
- `SOABert(cfg)`: embeddings, shared basis, stack of `SOABlock`s, pooler, classifier. Forward pass returns logits.

### `svd_migration.py`
**Purpose:** Compute global basis and migrate BERT weights.  
**Functions:**
- `compute_global_basis(bert_model, k_rank)`: collects Q, K, V weights from all layers, concatenates, performs SVD, returns top‑`k_rank` left singular vectors as the basis.
- `compute_global_basis_from_hidden_states(bert_model, k_rank, sample_size)`: alternative method that uses hidden states from random inputs.
- `migrate_bert_to_soa(bert_model, soa_model, basis)`: copies embeddings, layer norms, FFN weights, and computes `A = basis^T @ W` for attention projections.

### `export_onnx.py`
**Purpose:** Export SOABert to ONNX format.  
**Function:**
- `export_onnx(rank, onnx_path)`: loads a default `SOABert`, creates dummy inputs, and runs `torch.onnx.export`.

---

## ⚡ **Evaluation Module (`studio/evaluation/`)**

### `benchmark.py`
**Purpose:** Full compression benchmark across methods, ranks, and tasks.  
**Function:**
- `run_full_benchmark(model_name, methods, ranks, tasks, epochs, batch_size, seed)`: loops over all combinations, loads models, applies compression (or not), fine‑tunes (optional), evaluates, and returns a DataFrame.

### `speed_benchmark.py`
**Purpose:** Measure inference speed.  
**Functions:**
- `measure_inference_time(model, tokenizer, text_list, num_runs)`: runs the model multiple times on the same texts, returns average time.
- `run_speed_benchmark(original_model_path, compressed_model_path, rank, model_name, test_texts)`: loads (or creates) original and compressed models, times both, prints speedup.

---

## 🛠️ **Utils Module (`studio/utils/`)**

### `device.py`
**Purpose:** Device management.  
**Functions:**
- `get_device(device_str)`: returns a `torch.device` for `'cuda'`, `'mps'`, `'cpu'`, or `'auto'` (prefers MPS, then CUDA, then CPU).
- `clear_memory()`: calls `torch.cuda.empty_cache()` and `torch.mps.empty_cache()`.
- `get_memory_usage()`: returns allocated memory in MB (CUDA only, else None).
- `get_model_device(model)`: returns the device of the first parameter.

### `logger.py`
**Purpose:** Set up logging.  
**Function:**
- `setup_logger(name, level, log_file)`: creates a logger with console and optional file output.

### `checkpoint.py`
**Purpose:** Save and load experiment checkpoints.  
**Class:**
- `CheckpointManager(checkpoint_dir)`: `save_checkpoint(experiment_id, results, name)`, `load_checkpoint(name)`. Stores JSON with timestamp.

### `helpers.py`
**Purpose:** Miscellaneous helpers.  
**Functions:**
- `set_seed(seed, deterministic)`: sets random, numpy, torch seeds.
- `count_parameters(model, trainable_only)`: counts parameters.
- `format_number(num)`: formats large numbers with K/M/B suffix.
- `format_time(seconds)`: converts seconds to human‑readable string.
- `save_json(obj, path)`, `load_json(path)`.

---

## ✅ **Summary**

Each module is **dedicated to a specific responsibility**:

- **Data**: loading, datasets, synthetic generation, analysis.
- **Compression**: SVD layers and zero‑shot compressors.
- **Attentions**: drop‑in replacements and factorised attention.
- **Routing**: task detection and specialised model selection.
- **Ensemble**: weighted and robust ensemble methods.
- **Training**: training loops, advanced methods (LoRA, R‑Drop, etc.), custom trainer.
- **Analysis**: evaluation, metrics, layer‑wise inspection, reporting.
- **Visualisation**: interactive attention dashboards.
- **SOA**: low‑rank transformer built on a shared basis.
- **Evaluation**: comprehensive benchmarks and speed tests.
- **Utils**: device, logging, checkpoint, helpers.

All components are **integrated, deduplicated, and ready for research or production**. Use the CLI to generate data, train models, compress, visualise, and benchmark.







## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.


