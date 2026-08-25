# 🧠 NeuralOps & SVD Linear Attention Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Attentions](https://img.shields.io/badge/Attentions-236%20Registered-6366f1.svg)](./framework/attentions/)
[![Compressors](https://img.shields.io/badge/Compressors-107%20Registered-ec4899.svg)](./framework/compression_engine/)
[![FastAPI & Django](https://img.shields.io/badge/Backend-Django%20REST%20%26%20Channels-092e20.svg)](./django_api/)
[![React 18 & Vite](https://img.shields.io/badge/Frontend-React%2018%20+%20Vite-61dafb.svg)](./frontend/)

A unified, modular research ecosystem for **Linear Attention Architectures**, **Hardware-Aware Model Compression (SVD, Quantization, Pruning)**, and **Full-Stack Experiment Orchestration**.

---

## 📑 Table of Contents
1. [Overview & Key Features](#-overview--key-features)
2. [Repository Structure](#-repository-structure)
3. [Quickstart & Installation](#-quickstart--installation)
4. [Framework Core Architecture & 5 Tracks](#-framework-core-architecture--5-tracks)
5. [Automated Dataset Engine & GPU Server Execution](#-automated-dataset-engine--gpu-server-execution)
6. [Attention Mechanisms Registry (236 Modules)](#-attention-mechanisms-registry-236-modules)
7. [Model Compression Engines (107 Algorithms)](#-model-compression-engines-107-algorithms)
8. [Django REST & WebSocket API Gateway](#-django-rest--websocket-api-gateway)
9. [Frontend Research Dashboard](#-frontend-research-dashboard)
10. [Empirical Benchmark Results](#-empirical-benchmark-results)

---

## 🌟 Overview & Key Features

The **NeuralOps & SVD Linear Attention Framework** is designed to evaluate, compress, and train large transformer models with high efficiency:

- **236 Custom Attention Modules**: Plug-and-play implementations ranging from Truncated SVD Attention ($O(N)$), FlashAttention-3 (FP8), and Multi-Head Latent Attention (MLA) to State-Space Hybrids (Mamba-SVD-Flash, AEGIS v1–v6).
- **107 Model Compression Engines**: Post-training compression algorithms including Activation-Aware SVD (AASVD), Error-Controlled SVD (`erc_svd`), Rotational SliceGPT INT4, SparseGPT with damped Hessian inversions, DeltaKV differential caching, AWQ, and structured pruning (ARMOR/AST).
- **Zero-Shot Architecture Injection**: Automatically swaps standard Multi-Head Attention in HuggingFace models (BERT, RoBERTa, T5, LLaMA, Mistral, DistilBERT, GPT-2) with custom attention mechanisms via `inject_attention`.
- **Full-Stack Research Workspace**: Includes an asynchronous Django REST/WebSocket backend with real-time GPU telemetry, Bayesian sweep orchestration, and an Overleaf-style collaborative LaTeX studio.
- **Interactive React UI**: React 18 + Vite SPA featuring live attention heatmaps, 3D manifold visualizers, and a masterclass documentation explorer.

---

## 📁 Repository Structure

```
Codes/
├── framework/                # Core PyTorch Research Framework
│   ├── attentions/           # 236 Custom Attention Architectures
│   ├── compression_engine/   # 107 Model Compression & Pruning Engines
│   ├── core/                 # Registries, Base Classes & Model Wrappers
│   ├── data/                 # prepare_datasets.py & Benchmark DataLoader Streamer
│   ├── trainer/              # Distributed Trainers (HF, Custom Loop, FSDP)
│   ├── visualization/        # Heatmaps, Pareto Frontiers & Dashboards
│   ├── analysis/             # Hessian & Outlier Spectral Analyzers
│   ├── results_saver/        # SQLite, Parquet & JSON Exporters
│   ├── launch_gpu_server.py  # GPU Server hardware profiler & DDP launcher
│   ├── run_on_gpu_server.sh  # Single-command shell runner (--mode fast/comprehensive)
│   └── parallel_track_checker.py # 5-process parallel multi-track validator
│
├── django_api/               # Enterprise REST & WebSocket Backend
│   ├── api/                  # Core REST Endpoints & Serializers
│   ├── experiment/           # Experiment Tracking & Metrics Streaming
│   ├── sweeps/               # Optuna Bayesian Sweep Orchestrator
│   ├── jobscheduler/         # Async Worker Queues & Slurm Hooks
│   ├── paper/                # LaTeX Studio & Real-time PDF Compilation
│   └── health_monitor/       # Real-time NVML GPU & Node Telemetry
│
├── frontend/                 # React 18 + Vite Research Dashboard
│   ├── src/features/         # Experiment Monitors, Model Studios & Canvas
│   ├── src/router.tsx        # Dynamic Routing & Lazy-Loaded Pages
│   └── public/Docs/          # Synchronized Documentation & Catalog
│
├── run.py                    # Universal Single Experiment CLI
├── run_all.py                # Multi-Model Benchmark Suite
└── main.py                   # Master Entrypoint & Server Orchestration
```

---

## ⚡ Quickstart & Installation

```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core framework in editable mode
pip install -e Codes/framework/
pip install -r Codes/django_api/requirements.txt
```

### Run Multi-Process Track Verification
```bash
PYTHONPATH=Codes python3 Codes/framework/parallel_track_checker.py \
  --workers 5 \
  --models bert,roberta,t5 \
  --datasets sst2,mrpc \
  --rank 16
```

### Pre-Cache Benchmark Datasets & Run on GPU Server
```bash
# Pre-cache datasets
bash Codes/framework/run_on_gpu_server.sh --mode prepare-data

# Run fast BF16 sweep on GPU
bash Codes/framework/run_on_gpu_server.sh --mode fast
```
