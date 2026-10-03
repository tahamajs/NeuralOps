# 🧬 NeuralOps: SVD Linear Attention & Model Compression Framework

[![NeurIPS/ICLR Reproducibility](https://img.shields.io/badge/Reproducibility-NeurIPS%2FICLR%202026-brightgreen.svg)](./)
[![Python 3.12 / 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.0+](https://img.shields.io/badge/pytorch-2.5.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Attention Mechanisms](https://img.shields.io/badge/Attentions-115%20Implemented-6366f1.svg)](./Codes/framework/attentions/)
[![Compression Engines](https://img.shields.io/badge/Compressors-76%20Implemented-ec4899.svg)](./Codes/framework/compression_engine/)
[![Verified Tracks](https://img.shields.io/badge/Tracks-5%20Verified%20(100%25)-10b981.svg)](./Codes/framework/)
[![CI Status](https://github.com/tahamajs/SVD_linear_Attention/actions/workflows/ci.yml/badge.svg)](https://github.com/tahamajs/SVD_linear_Attention/actions)

**NeuralOps** is an enterprise-grade research framework designed for rigorous empirical evaluation, zero-shot architectural introspection, and hardware roofline profiling of **Linear & Subquadratic Attention Mechanisms**, **Subspace SVD Decompositions**, **State-Space Models (SSMs)**, and **Structured Sparsity/Quantization** across Foundation Models.

---

## 📑 Table of Contents
1. [Project Overview & Core Philosophy](#-project-overview--core-philosophy)
2. [Taxonomy: The 5 Research Tracks](#-taxonomy-the-5-research-tracks)
3. [Understanding Injection vs. Compression](#-understanding-injection-vs-compression)
4. [Hardware Requirements & Profiles](#-hardware-requirements--profiles)
5. [Environment & Installation](#-environment--installation)
6. [Commands to Reproduce Benchmarks](#-commands-to-reproduce-benchmarks)
   - [A. Zero-Shot 5-Task Reasoning Suite](#a-zero-shot-5-task-reasoning-suite-gpt-2)
   - [B. Rank Sweeps & SVD Scaling](#b-rank-sweeps--svd-scaling-distilbert--roberta)
   - [C. Language Modeling & Perplexity (C4 / WikiText-2)](#c-language-modeling--perplexity-c4--wikitext-2)
   - [D. Multi-Seed Pretrained Fine-Tuning Sweep](#d-multi-seed-pretrained-fine-tuning-sweep)
7. [Comprehensive Results Directory Architecture](#-comprehensive-results-directory-architecture)
8. [Automated Artifacts & Dashboard Telemetry](#-automated-artifacts--dashboard-telemetry)
9. [Citation & License](#-citation--license)

---

## 🔬 Project Overview & Core Philosophy

Standard Multi-Head Attention (MHA) incurs quadratic computational $O(N^2)$ and memory complexity with respect to sequence length $N$. NeuralOps systematically solves, audits, and evaluates subquadratic alternatives through:

- **Universal Zero-Shot Module Injection**: Dynamically replace transformer attention blocks across diverse families (`distilbert`, `bert`, `roberta`, `t5`, `gpt2`, `qwen2.5`) with exact projection weight slicing and geometry preservation (RoPE, GQA, and fused $QKV$ handling).
- **In-Place Weight Compression Engines**: Evaluate singular value decomposition (SVD), structural pruning, low-rank quantization, and KV-cache compression without requiring structural attention class replacement.
- **Academic Rigor & Reproducibility**: 5-seed paired Student's t-tests, Bonferroni corrections, per-sample prediction caching, and exact Pareto frontier mapping between parameter count, latency, and reasoning capability.

---

## 🏛 Taxonomy: The 5 Research Tracks

All modules implemented in `Codes/framework/attentions/` and `Codes/framework/compression_engine/` map directly into 5 primary research tracks:

```
Codes/framework/
├── attentions/               # 115 subquadratic, linear, and hybrid attention modules
│   ├── adaptive_span_attention.py, cosformer.py, flash_attention3.py,
│   ├── galerkin_attention.py, performer_attention.py, linformer_attention.py ...
└── compression_engine/       # 76 SVD factorizers, quantizers, and structural pruners
    ├── armor_pruner.py, ast_pruner.py, astro_quantizer.py, dobi_svd.py,
    ├── delta_kv.py, enlighten_compressor.py, faq_quantizer.py, mlorq_compressor.py ...
```

| Track | Focus | Key Module Implementations |
| :--- | :--- | :--- |
| **Track 1: SVD & Low-Rank Factorization** | Matrix decomposition, truncated SVD, dynamic energy truncation, symmetric low-rank projections | `asvd_attention`, `dobi_svd`, `aa_svd_compressor`, `dynamic_rank`, `erc_svd`, `saes_svd_v2`, `swift_svd_act` |
| **Track 2: Spectral & State-Space Models (SSMs)** | Recurrent formulations, selective state spaces, gated linear recurrence, linear decay | `mamba_block`, `griffin_ssd`, `gated_delta_net`, `gated_linear_attention`, `decay_linear_attention`, `aegis_v1`–`v6` |
| **Track 3: Mixture-of-Experts & Hybrids** | Sparse routing, token clustering, block-wise transformer approximations, hybrid layers | `bam_umoe`, `expert_choice_moe`, `block_transformer_hiba`, `cluster_attention`, `cur_moe_compressor` |
| **Track 4: Kernel Approximations & Quantization** | Positive Random Features (FAVOR+), kernel mapping, FP8/INT4 weight quantization | `performer_attention`, `cosformer`, `nystrom_attention`, `flash_attention3_fp8`, `astro_quantizer`, `coopq_quantizer`, `faq_quantizer` |
| **Track 5: Structural Pruning & KV-Cache Reduction** | Head pruning, ADMM channel sparsity, delta KV compression, dimension reduction | `armor_pruner`, `ast_pruner`, `admm_compressor`, `bonsai_pruner`, `flap`, `delta_kv`, `cachenotes_compressor`, `slicegpt` |

---

## 🔍 Understanding Injection vs. Compression

In benchmark reports and summary CSVs, you will observe the metric `num_injected_attention_layers`:

1. **Pure Compressors, SVD Factorizers, & Pruners (`num_injected_attention_layers = 0`)**:
   - **Status**: `PASS` / `PASS_NO_COMPRESSION`
   - **Mechanism**: Modules such as `armor_pruner`, `dobi_svd`, `enlighten_compressor`, and `faq_quantizer` operate directly on `nn.Linear` projection matrices ($W_q, W_k, W_v, W_o$) or MLP blocks.
   - **Reason for 0**: The original transformer class (e.g. `GPT2Attention`) is kept intact; only its weight tensors are compressed. This is **expected and correct**, yielding retention accuracies of **42.0% – 42.6%** on zero-shot reasoning.
2. **Attention Architecture Replacements (`num_injected_attention_layers = 12`)**:
   - **Status**: `PASS`
   - **Mechanism**: Modules such as `adaptive_span_attention`, `asvd_attention`, or `compressed_mha_attention` completely replace the underlying PyTorch attention block with a custom forward computation class via `UniversalAttentionWrapper`.

---

## 💻 Hardware Requirements & Profiles

NeuralOps benchmarks are designed for execution across standard workstation and GPU compute profiles:

- **GPU**: NVIDIA RTX 4070 (8 GB VRAM) / RTX 3080/4080 / A10 / A5000 / A100.
- **CUDA Environment**: CUDA Version 12.1 – 12.7, PyTorch $\ge$ 2.5.0.
- **Host CPU & RAM**: x86_64 or Apple Silicon (M-series), $\ge$ 16 GB Physical RAM.
- **Process Memory Safety Watchdog**: Built-in memory monitors auto-throttle batch sizes and parallel worker counts if free RAM drops below 1.0 GB or VRAM exceeds 7.5 GB.

---

## 🚀 Environment & Installation

```bash
# 1. Clone the repository
git clone https://github.com/tahamajs/SVD_linear_Attention.git
cd SVD_linear_Attention

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install PyTorch with CUDA acceleration
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install framework dependencies
pip install -r requirements.txt
```

---

## ⚡ Commands to Reproduce Benchmarks

### A. Zero-Shot 5-Task Reasoning Suite (GPT-2)
Evaluates modules without fine-tuning across 5 reasoning benchmarks (`PIQA`, `ARC-Easy`, `ARC-Challenge`, `HellaSwag`, `WinoGrande`) with 500 samples per task:

```bash
# Run multi-worker parallel evaluator (e.g. 4-5 workers on 8 GB GPU):
python scripts/run_zeroshot_parallel.py \
  --parallel 4 \
  --num_samples 500 \
  --batch_size 8 \
  --model models/gpt2

# Aggregate all individual task predictions into master CSVs & Parquet:
python scripts/aggregate_zeroshot_details.py
```

### B. Rank Sweeps & SVD Scaling (DistilBERT / RoBERTa)
Profiles accuracy retention across low-rank bottlenecks ($r \in \{64, 128, 256, 384\}$):

```bash
python Codes/framework/benchmarks/run_rank_sweeper.py \
  --modules armor_pruner,cluster_attention,dobi_svd,dynamic_rank \
  --ranks 64,128,256,384 \
  --model models/base_checkpoints/distilbert_mnli \
  --model_family bert \
  --dataset mnli \
  --output_base results/rank_sweep_mnli \
  --batch_size 16
```

### C. Language Modeling & Perplexity (C4 / WikiText-2)
Computes causal autoregressive token cross-entropy loss and perplexity on long-form web text:

```bash
python Codes/framework/benchmarks/run_quality_sweep.py \
  --modules full,asvd_attention,cosformer,performer_attention \
  --model models/gpt2 \
  --model_family gpt2 \
  --dataset c4 \
  --rank 256 \
  --batch_size 8 \
  --output_dir results/c4_baselines
```

### D. Multi-Seed Pretrained Fine-Tuning Sweep
Runs statistical multi-seed fine-tuning recovery across classification benchmarks:

```bash
python Codes/framework/benchmarks/run_multiseed.py \
  --modules svd_linear_attention,armor_pruner,cluster_attention \
  --models distilbert \
  --dataset sst2 \
  --seeds 42,123,456,789,999 \
  --output_dir results/multiseed_rigor_distilbert
```

---

## 📂 Comprehensive Results Directory Architecture

The `results/` hierarchy contains 268 experimental suites structured cleanly as follows:

```
results/
├── README.md                                  # Complete structural taxonomy of all runs
├── benchmark_summary_v2.csv                   # Master multi-model, multi-dataset benchmark summary
├── master_aggregated_results.csv              # Master verified accuracy, parameter, and ratio table
├── SEQLEN_SCALING_MASTER.csv                  # Sequence-length scaling latency & memory matrix
│
├── zeroshot_parallel/                         # 183-Module Zero-Shot Reasoning Suite (GPT-2)
│   ├── <module_name>/
│   │   ├── <task>.json                        # Task summary metrics (e.g. piqa.json, hellaswag.json)
│   │   ├── <task>_detailed.json               # Full sample-level predictions & answers
│   │   ├── <module>_zeroshot.json             # Combined 5-benchmark macro performance
│   │   ├── run_metadata.json                  # Environment, GPU model, Git SHA, and versions
│   │   └── run.log                            # Raw execution trace
│
├── rank_sweep_mnli/ & rank_sweep_agnews/      # Rank sweeps (ranks 64 to 384)
├── c4_baselines/ & gpt2_c4_ppl/               # Autoregressive perplexity evaluations on C4
├── multiseed/ & multiseed_rigor_*/            # 5-seed statistical verification runs
├── distilbert_* & roberta_*                   # Bidirectional encoder GLUE classification runs
└── Unified/                                   # Full multi-model evaluation archive (237 subdirectories)
```

---

## 📊 Automated Artifacts & Dashboard Telemetry

- **Live Evaluation Monitor**: Open [zeroshot_live_monitor.md](Docs/zeroshot_live_monitor.md) for live 20-minute inspection cycles, ranked Pareto tables, and comparative analysis.
- **Web Analytics Studio**:
  ```bash
  python Codes/framework/live_monitor.py --port 8080 --host 0.0.0.0
  ```
- **Automated Publication LaTeX Generation**:
  ```bash
  python Codes/framework/analysis/generate_sweep_publication_artifacts.py \
    --results_dir results/zeroshot_parallel/
  ```

---

## ⚖️ Citation & License

Distributed under the MIT License. See [LICENSE](LICENSE) for full details. If you use NeuralOps in academic research, please cite:

```bibtex
@inproceedings{neuralops2026,
  title={NeuralOps: Universal Benchmarking and Subspace Factorization for Subquadratic Attention Architectures},
  author={Schumacher, Hadi and Majs, Taha and Collaborators},
  booktitle={Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```
