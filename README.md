# 🧬 NeuralOps: SVD Linear Attention & Model Compression Framework

[![NeurIPS/ICLR Reproducibility](https://img.shields.io/badge/Reproducibility-NeurIPS%2FICLR%202026-brightgreen.svg)](./)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.0](https://img.shields.io/badge/pytorch-2.5.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Attention Mechanisms](https://img.shields.io/badge/Attentions-236%20Registered-6366f1.svg)](./Codes/framework/attentions/)
[![Compression Engines](https://img.shields.io/badge/Compressors-107%20Registered-ec4899.svg)](./Codes/framework/compression_engine/)
[![Verified Tracks](https://img.shields.io/badge/Tracks-5%20Verified%20(100%25)-10b981.svg)](./Codes/framework/)
[![CI Status](https://github.com/tahamajs/SVD_linear_Attention/actions/workflows/ci.yml/badge.svg)](https://github.com/tahamajs/SVD_linear_Attention/actions)

**NeuralOps** is an enterprise-grade research framework designed for rigorous empirical evaluation, zero-shot architectural introspection, and hardware roofline profiling of **Linear & Subquadratic Attention Mechanisms**, **Subspace SVD Decompositions**, **State-Space Models (SSMs)**, and **Structured Sparsity/Quantization** across Foundation Models.

---

## 📑 Table of Contents
1. [Project Overview & Purpose](#-project-overview--purpose)
2. [Hardware Requirements](#-hardware-requirements)
3. [Environment & Installation](#-environment--installation)
4. [Docker Container Execution](#-docker-container-execution)
5. [Commands to Reproduce Benchmarks](#-commands-to-reproduce-benchmarks)
   - [Phase 1: Efficiency & Micro-Benchmarking Sweep](#phase-1-efficiency--micro-benchmarking-sweep)
   - [Phase 2: Multi-Seed Quality Sweep (Pretrained Fine-Tuning)](#phase-2-multi-seed-quality-sweep-pretrained-fine-tuning)
6. [Interactive Live Monitor & Telemetry Dashboard](#-interactive-live-monitor--telemetry-dashboard)
7. [Output Files & Directory Hierarchy](#-output-files--directory-hierarchy)
8. [Automated Publication Artifact Generation](#-automated-publication-artifact-generation)

---

## 🔬 Project Overview & Purpose

Modern long-context transformers suffer from $O(N^2)$ quadratic computational and memory bottlenecks in self-attention layers. NeuralOps provides:
- **Universal Zero-Shot Module Injection**: Seamlessly swap standard multi-head self-attention with any of **236 Attention Mechanisms** or **107 Compression Engines** across HuggingFace transformer backbones (`distilbert`, `bert`, `roberta`, `t5`, `gpt2`, `llama`) with automatic weight projection cloning via SVD factorization.
- **5-Track Systematic Taxonomy**:
  1. *Track 1 (SVD & Low-Rank Factorization)*: Exact SVD, Minimal-SVD, Dynamic Rank SVD, SVD-Linear.
  2. *Track 2 (Spectral & State-Space Models)*: Mamba, SVD-Mamba, Samba M3, Griffin SSD, Gated DeltaNet, AEGIS v1–v6.
  3. *Track 3 (Mixture-of-Experts & Hybrids)*: BAM-UMoE, Samba, Jamba, Hymba, NEXUS, LEKA v1–v3.
  4. *Track 4 (Kernel Approximations & Quantization)*: FAVOR+ Performer, Cosformer, Nyström, FlashAttention-3, ASTRO, FAQ.
  5. *Track 5 (Structural Pruning & Sparsity)*: Linformer, SliceGPT, SparseGPT, 2:4 Semi-Structured Sparsity.
- **Statistical Rigor**: 3-seed paired Student's t-tests, Bonferroni multi-hypothesis corrections, 95% bootstrap confidence intervals, and logit cosine similarity fidelity metrics against uncompressed baselines.

---

## 💻 Hardware Requirements

All benchmark runs in the NeuralOps paper were evaluated on consumer and workstation hardware under strict thermal and power limits:

- **GPU**: NVIDIA GeForce RTX 4070 (8 GB GDDR6X VRAM) or equivalent modern Ampere/Ada Lovelace GPU (RTX 3080/4080, A10, A5000).
- **CUDA & Drivers**: CUDA Version 12.1 – 12.7, NVIDIA Display Driver $\ge$ 560.00.
- **Host CPU & RAM**: x86_64 CPU (8+ physical cores recommended), $\ge$ 16 GB System Memory.
- **Storage**: Minimum 20 GB free disk space for cached HuggingFace datasets (`sst2`, `mrpc`, `cola`, `stsb`) and model checkpoints.

---

## 🚀 Environment & Installation

### Option A: Local Python Environment (Recommended)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/tahamajs/SVD_linear_Attention.git
   cd SVD_linear_Attention
   ```

2. **Create Python 3.12 Virtual Environment**:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   # Install PyTorch with CUDA 12.1 / 12.4 support:
   pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   # Install pinned requirements:
   pip install -r requirements.txt
   ```

---

## 🐳 Docker Container Execution

For guaranteed containerized reproducibility across Linux workstations:

```bash
# 1. Build the Docker Image
docker build -t neuralops:2026 .

# 2. Run Smoke Test inside Container
docker run --gpus all --rm neuralops:2026

# 3. Interactive Shell inside Container with Volume Mount
docker run --gpus all -it -v $(pwd)/results:/workspace/results neuralops:2026 bash
```

---

## ⚡ Commands to Reproduce Benchmarks

### Phase 1: Efficiency & Micro-Benchmarking Sweep
Measures forward latency, backward latency, peak VRAM allocation, and arithmetic throughput across sequence lengths:

```bash
python Codes/framework/run_benchmarks_optimized.py \
  --tracks 1,2,3,4,5 \
  --batch_sizes 1,16,64 \
  --seq_lens 128,512,1024 \
  --runs_per_combo 5 \
  --precision fp16 \
  --output_dir results/run_phase1_efficiency
```

### Phase 2: Multi-Seed Quality Sweep (Pretrained Fine-Tuning)
Fine-tunes base transformer backbones (`distilbert-base-uncased` or `bert-base-uncased`) with injected modules across 3 random seeds (`42, 123, 777`) with AdamW and linear learning rate warmup:

```bash
python Codes/framework/run_quality_sweep.py \
  --modules full,svd_linear_attention,minimal_svd,mamba,performer_attention,linformer_attention,aegis_v6,leka_v3,griffin_ssd,samba_m3 \
  --models distilbert \
  --datasets sst2,mrpc \
  --seeds 42,123,777 \
  --epochs 5 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --batch_size 16 \
  --seq_len 128 \
  --pretrained \
  --gradient_checkpointing \
  --output_dir results/run_quality_pretrained
```

*Expected outputs*: Baseline `full` validation accuracy $\ge$ 88.0% on SST-2; factorized architectures achieve 82%–90% retention with up to 1.8× memory reduction.

---

## 📊 Interactive Live Monitor & Telemetry Dashboard

Launch the NeuralOps real-time monitoring studio to track hardware metrics, training loss trajectories, and live comparative tables:

```bash
python Codes/framework/live_monitor.py --port 8080 --host 0.0.0.0
```
Navigate to `http://localhost:8080` in your web browser:
- **Telemetry Stream**: Real-time GPU core %, memory allocation, temperature, and board power draw via `nvidia-smi`.
- **Comparative Analytics Lab**: Real-time multi-seed averages (Accuracy, F1, Logit Cosine Fidelity, Latency, VRAM).
- **Run Comparator (Diff Engine)**: Side-by-side run comparisons across hyperparameters and architectures.

---

## 📂 Output Files & Directory Hierarchy

Benchmark executions generate standardized tabular and structured artifacts:

```
results/run_quality_pretrained/
├── summary.csv                 # 18-column primary evaluation table
├── detailed_results.json       # Full per-step loss, gradient norms, and seed records
├── config.json                 # Exact run hyperparameters and environment flags
├── plots/                      # Publication figures (PDF & 300 DPI PNG)
│   ├── accuracy_vs_latency.pdf
│   ├── pareto_frontier.pdf
│   └── vram_scaling.pdf
└── raw_logs/                   # Per-combination execution logs
```

### `summary.csv` Schema
| Column | Description |
| :--- | :--- |
| `run_id` | Unique ISO-timestamped benchmark identifier |
| `module_name` | Architecture name (e.g. `svd_linear_attention`, `minimal_svd`) |
| `model_family` | Transformer backbone (`DISTILBERT`, `BERT`, `ROBERTA`) |
| `dataset` | GLUE evaluation benchmark task (`SST2`, `MRPC`, `COLA`, `STSB`) |
| `status` | Execution status (`PASS`, `FAIL`, `TIMEOUT`) |
| `mean_accuracy`| Multi-seed average validation accuracy (%) |
| `std_accuracy` | Sample standard deviation across seeds |
| `ci_lower`, `ci_upper` | 95% Confidence Interval bounds |
| `p_value` | Paired two-tailed Student's t-test p-value vs baseline |
| `f1_score` | Macro/Binary F1 score |
| `fidelity_score` | Logit cosine similarity against uncompressed baseline |
| `fwd_ms`, `bwd_ms` | CUDA event-timed latency (ms) |
| `peak_vram_mb` | Peak allocated GPU memory (MB) |
| `throughput_tokens_sec` | Effective training throughput (tokens/second) |

---

## 📄 Automated Publication Artifact Generation

Generate publication-ready NeurIPS/ICLR LaTeX tables and vector figures directly from any benchmark output folder:

```bash
python Codes/framework/generate_publication_report.py \
  --results_dir results/run_quality_pretrained/
```

This compiles:
- `LATEX_QUALITY_TABLE.tex`: Booktabs LaTeX markup for academic submissions.
- `LATEX_EFFICIENCY_TABLE.tex`: Hardware efficiency table (Latency, Speedup, VRAM compression).
- `QUALITY_BENCHMARK_REPORT.md`: Comprehensive markdown audit report with automated architectural advice.

---

## ⚖️ License & Citation

Distributed under the MIT License. See `LICENSE` for more information. If you use NeuralOps in academic research, please cite our manuscript:

```bibtex
@inproceedings{neuralops2026,
  title={NeuralOps: Universal Benchmarking and Subspace Factorization for Subquadratic Attention Architectures},
  author={Schumacher, Hadi and Majs, Taha and Collaborators},
  booktitle={Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```
