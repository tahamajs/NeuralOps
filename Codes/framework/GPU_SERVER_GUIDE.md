# 🚀 GPU Server Execution & Benchmark Deployment Guide

This guide provides step-by-step instructions to deploy, prepare datasets, and run all **123 Attention Mechanisms** and **86 Compression Engines** on high-performance GPU servers (NVIDIA A100, H100, RTX 4090/3090, V100, AWS EC2, Lambda Labs, RunPod).

---

## 📋 1. Prerequisites & Environment Setup

On your GPU server, ensure NVIDIA drivers and CUDA (>= 11.8 or 12.x) are installed.

```bash
# 1. Clone or sync the repository
cd /path/to/SVD_linear_Attention

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA 12.1 acceleration
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install transformer dependencies
pip install transformers datasets accelerate evaluate scikit-learn tabulate matplotlib seaborn
```

---

## 🔍 2. Hardware Diagnostics & Precision Verification

Run the diagnostics probe to inspect all GPUs, Tensor Cores, compute capability, and mixed-precision (BF16 / FP16) support:

```bash
bash Codes/framework/run_on_gpu_server.sh --mode diagnostics
```

**Example Output:**
```
================================================================================
              NEURALOPS GPU SERVER HARDWARE DIAGNOSTICS
================================================================================
 Hostname           : gpu-node-01
 Operating System   : Linux 5.15.0-89-generic
 Python Version     : 3.10.12
 PyTorch Version    : 2.2.0+cu121
 CUDA Available     : YES
 GPU Device Count   : 4
   - GPU [0]        : NVIDIA A100-SXM4-80GB (80.0 GB VRAM, Compute Capability 8.0)
   - GPU [1]        : NVIDIA A100-SXM4-80GB (80.0 GB VRAM, Compute Capability 8.0)
   - GPU [2]        : NVIDIA A100-SXM4-80GB (80.0 GB VRAM, Compute Capability 8.0)
   - GPU [3]        : NVIDIA A100-SXM4-80GB (80.0 GB VRAM, Compute Capability 8.0)
 BF16 Support       : YES (Native Bfloat16)
 FP16 Support       : YES (Half-Precision)
 Tensor Cores       : ENABLED
================================================================================
```

---

## 📦 3. Benchmark Dataset Preparation & Caching

Download, tokenize, and pre-cache standard benchmark datasets (GLUE and Language Modeling) to disk for zero-latency DataLoader streaming:

```bash
# Option A: Prepare all GLUE tasks + WikiText-2 + C4
bash Codes/framework/run_on_gpu_server.sh --mode prepare-data

# Option B: Prepare specific tasks with a custom tokenizer
python Codes/framework/data/prepare_datasets.py --tasks sst2,mrpc,qnli,cola --tokenizer bert-base-uncased --save_dir ./prepared_benchmark_data
```

> **Note**: If your GPU server is air-gapped / offline without direct internet access, the pipeline automatically detects offline state and generates high-fidelity synthetic benchmark tensors for all tasks.

---

## ⚡ 4. Executing Benchmark Sweeps

### A. Fast Sweep (BF16 Precision on GPU)
Runs all 5 tracks across BERT, RoBERTa, and T5 with native mixed precision and Tensor Core acceleration:
```bash
bash Codes/framework/run_on_gpu_server.sh --mode fast
```

### B. Comprehensive Multi-Model Matrix
Evaluates all modules across full dataset splits and model families:
```bash
bash Codes/framework/run_on_gpu_server.sh --mode comprehensive
```

### C. Multi-GPU Distributed Execution (via `torchrun` / DDP)
Run benchmark processes in parallel across all available GPUs:
```bash
torchrun --nproc_per_node=4 Codes/framework/launch_gpu_server.py --mode fast --distributed
```

### D. Single-Track or Single-Module Evaluation
```bash
# Track 1: SVD & Low-Rank Factorization
python Codes/framework/auto_run.py --track 1 --models bert,roberta,t5 --datasets sst2,mrpc

# Track 2: Spectral & SSM Recurrent (AEGIS, Mamba, Spectra)
python Codes/framework/auto_run.py --track 2 --models bert,roberta,t5 --datasets sst2,mrpc

# Track 3: MoE & Hybrid Routing (BAM-UMoE, Jamba, LeKA)
python Codes/framework/auto_run.py --track 3 --models bert,roberta,t5 --datasets sst2,mrpc

# Track 4: Kernel Approximations & Quantization (Performer, Flash3, Astro, KMeans)
python Codes/framework/auto_run.py --track 4 --models bert,roberta,t5 --datasets sst2,mrpc

# Track 5: Structural Pruning & Sparsity (SliceGPT, SparseGPT, FLAP)
python Codes/framework/auto_run.py --track 5 --models bert,roberta,t5 --datasets sst2,mrpc

# Specific single module:
python Codes/framework/auto_run.py --module aegis_v1 --models bert,roberta,t5 --datasets sst2
```

---

## 📊 5. Results & Consolidated Reporting

Every single evaluation writes an independent JSON result record immediately to disk.

- **JSON Results Directory**: `results/gpu_server/json/`
- **Master CSV Summary**: `results/gpu_server/benchmark_summary.csv`
- **Consolidated Markdown Report**: `results/gpu_server/GPU_BENCHMARK_REPORT.md`

To compile or refresh the consolidated markdown report from existing JSON records:
```bash
python Codes/framework/auto_run.py --report --results_dir results/gpu_server
```

---

*Authored for the NeuralOps Core Research Framework.*
