# 🧬 NeuralOps: SVD Linear Attention & Model Compression Framework

[![NeurIPS/ICLR Reproducibility](https://img.shields.io/badge/Reproducibility-NeurIPS%2FICLR%202026-brightgreen.svg)](./)
[![Python 3.10 / 3.11 / 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.0+](https://img.shields.io/badge/pytorch-2.5.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NeuralOps** is a fully-featured standard Python package for rigorous empirical evaluation, zero-shot architectural introspection, and hardware roofline profiling of **Linear & Subquadratic Attention Mechanisms**, **Subspace SVD Decompositions**, **State-Space Models (SSMs)**, and **Structured Sparsity/Quantization** across Foundation Models.

---

## 📑 Table of Contents
1. [Project Overview & Core Philosophy](#-project-overview--core-philosophy)
2. [Taxonomy: The 5 Research Tracks](#-taxonomy-the-5-research-tracks)
3. [Environment & Installation (GPU/CUDA setup)](#-environment--installation-gpucuda)
4. [Hardware Requirements & Profiles](#-hardware-requirements--profiles)
5. [Commands to Reproduce Benchmarks](#-commands-to-reproduce-benchmarks)
6. [Generating Full Documentation](#-documentation-and-api-reference)
7. [Citation & License](#-citation--license)

---

## 🔬 Project Overview & Core Philosophy

Standard Multi-Head Attention (MHA) incurs quadratic computational $O(N^2)$ and memory complexity with respect to sequence length $N$. NeuralOps systematically solves, audits, and evaluates subquadratic alternatives through:

- **Universal Zero-Shot Module Injection**: Dynamically replace transformer attention blocks across diverse families (`distilbert`, `bert`, `roberta`, `t5`, `gpt2`, `qwen2.5`) with exact projection weight slicing and geometry preservation (RoPE, GQA, and fused $QKV$ handling).
- **In-Place Weight Compression Engines**: Evaluate singular value decomposition (SVD), structural pruning, low-rank quantization, and KV-cache compression without requiring structural attention class replacement.

---

## 🏛 Taxonomy: The 5 Research Tracks

All modules are strictly implemented as a standard Python library in the `neuralops` root package:

```
neuralops/
├── attentions/               # 115 subquadratic, linear, and hybrid attention modules
│   ├── adaptive_span_attention.py, cosformer.py, flash_attention3.py,
│   ├── galerkin_attention.py, performer_attention.py, linformer_attention.py ...
└── compression_engine/       # 76 SVD factorizers, quantizers, and structural pruners
    ├── armor_pruner.py, ast_pruner.py, astro_quantizer.py, dobi_svd.py,
    ├── delta_kv.py, enlighten_compressor.py, faq_quantizer.py, mlorq_compressor.py ...
```

---

## 🚀 Environment & Installation (GPU/CUDA)

NeuralOps is now a standard, pip-installable python package. To install it natively alongside its PyTorch and CUDA dependencies, follow these steps exactly:

```bash
# 1. Clone the repository
git clone https://github.com/tahamajs/SVD_linear_Attention.git
cd SVD_linear_Attention

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install PyTorch with explicit CUDA acceleration
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install the NeuralOps package natively!
pip install -e .
```

---

## 💻 Hardware Requirements & Profiles

NeuralOps benchmarks are designed for execution across standard workstation and GPU compute profiles:

- **GPU Requirement**: NVIDIA RTX 3080/4080 / 4070 (8 GB VRAM) / A10 / A5000 / A100.
- **CUDA Environment**: **CUDA Version 12.1 – 12.7** (12.4 highly recommended for Triton kernels).
- **Host CPU & RAM**: x86_64 or Apple Silicon (M-series), $\ge$ 16 GB Physical RAM.
- **Process Memory Safety Watchdog**: Built-in memory monitors auto-throttle batch sizes and parallel worker counts if free RAM drops below 1.0 GB or VRAM exceeds 7.5 GB.

---

## ⚡ Commands to Reproduce Benchmarks

Because NeuralOps is now a standard package, you can run all orchestrators directly using module paths!

### A. Zero-Shot 5-Task Reasoning Suite (GPT-2)
Evaluates modules without fine-tuning across 5 reasoning benchmarks:
```bash
python scripts/run_zeroshot_parallel.py --parallel 4 --num_samples 500 --batch_size 8 --model models/gpt2
```

### B. Rank Sweeps & SVD Scaling (DistilBERT / RoBERTa)
Profiles accuracy retention across low-rank bottlenecks:
```bash
python neuralops/benchmarks/run_rank_sweeper.py \
  --modules armor_pruner,cluster_attention,dobi_svd,dynamic_rank \
  --ranks 64,128,256,384 \
  --model models/base_checkpoints/distilbert_mnli \
  --model_family bert \
  --dataset mnli \
  --output_base results/02_rank_sweeps/rank_sweep_mnli \
  --batch_size 16
```

### C. Language Modeling & Perplexity (C4 / WikiText-2)
Computes causal autoregressive token cross-entropy loss and perplexity on long-form web text:
```bash
python neuralops/benchmarks/run_quality_sweep.py \
  --modules full,asvd_attention,cosformer,performer_attention \
  --model models/gpt2 \
  --model_family gpt2 \
  --dataset c4 \
  --rank 256 \
  --batch_size 8 \
  --output_dir results/04_baselines/c4_baselines
```

### D. Multi-Seed Pretrained Fine-Tuning Sweep
Runs statistical multi-seed fine-tuning recovery across classification benchmarks:
```bash
python neuralops/benchmarks/run_multiseed.py \
  --modules svd_linear_attention,armor_pruner,cluster_attention \
  --models distilbert \
  --dataset sst2 \
  --seeds 42,123,456,789,999 \
  --output_dir results/05_glue_finetuning/multiseed_rigor_distilbert
```

---

## 📚 Documentation and API Reference

NeuralOps comes with standard automated Python documentation. To generate and view the full interactive API reference:

```bash
pip install pdoc3
pdoc --html --output-dir docs neuralops
```
Once complete, open `docs/neuralops/index.html` in your browser.

---

## ⚖️ Citation & License

Distributed under the MIT License. See [LICENSE](LICENSE) for full details.
