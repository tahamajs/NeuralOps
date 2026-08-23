# 🧬 NeuralOps: SVD Linear Attention & Model Compression Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/Docs-454%20Modules-emerald.svg)](./Docs/catalog.json)
[![Dashboard](https://img.shields.io/badge/UI-React%2018%20+%20Vite-61dafb.svg)](./Codes/frontend/)
[![Backend](https://img.shields.io/badge/API-Django%20REST-092e20.svg)](./Codes/django_api/)

An open-source research and engineering platform for **Linear Attention Algorithms**, **Subspace SVD Compression**, **Quantization & Structured Pruning**, and **Full-Stack Experiment Management** for Foundation Models.

---

## 📑 Table of Contents
1. [Key Capabilities](#-key-capabilities)
2. [Quickstart (30 Seconds)](#-quickstart-30-seconds)
3. [Architecture Overview](#-architecture-overview)
4. [Supported Attention Architectures](#-supported-attention-architectures)
5. [Supported Compression Engines](#-supported-compression-engines)
6. [Empirical Benchmarks](#-empirical-benchmarks)
7. [Running Experiments & Sweeps](#-running-experiments--sweeps)
8. [Full-Stack Research Workspace](#-full-stack-research-workspace)
9. [Documentation & Masterclass Hub](#-documentation--masterclass-hub)
10. [References](#-references)

---

## 🌟 Key Capabilities

- **123 Attention Architectures**: Includes Low-Rank SVD ($O(N)$), FlashAttention-3 (FP8), DeepSeek-style Multi-Head Latent Attention (MLA), GQA, Performer, and Mamba-SSM hybrids.
- **86 Compression Strategies**: Activation-Aware SVD (AASVD), Rotational SliceGPT INT4, DeltaKV differential cache compression, SmoothQuant, and structured pruning (ARMOR/AST).
- **Zero-Shot HuggingFace Integration**: Inject any custom attention or compression pipeline into existing models (`AutoModelForSequenceClassification`, `AutoModelForCausalLM`) with a single function call.
- **Enterprise-Grade Tooling**: Asynchronous Django REST API, WebSocket telemetry, Slurm/Celery distributed job execution, and an Overleaf-style LaTeX paper writing studio.
- **Interactive Web UI**: Modern React 18 + Vite dashboard with live training monitors, attention heatmaps, 3D latent space visualizers, and a masterclass document browser.

---

## ⚡ Quickstart (30 Seconds)

### Installation
```bash
# Clone the repository
git clone https://github.com/tahamajs/SVD_linear_Attention.git
cd SVD_linear_Attention

# Setup Python environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the core framework
pip install -e Codes/framework/
pip install -r Codes/django_api/requirements.txt
```

### Python API Example: Attention Injection
```python
import torch
from transformers import AutoModelForSequenceClassification
from framework.core.attention_utils import inject_attention

# 1. Load any standard HuggingFace transformer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# 2. Swap standard quadratic attention with O(N) SVD Linear Attention
model = inject_attention(model, attn_type="svd_linear", rank=64)

# 3. Standard forward pass with low-rank factorized attention
inputs = torch.randint(0, 1000, (2, 512))
outputs = model(inputs)
print("Forward pass successful:", outputs.logits.shape)
```

### Python API Example: Model Compression
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from framework.core.compression_engine import compress_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# Apply Activation-Aware SVD + 4-bit quantization pipeline
compressed_model = compress_model(
    model,
    config={
        "compressors": [
            {"method": "aasvd", "rank": 64},
            {"method": "rotational_slicegpt_int4", "bits": 4}
        ]
    },
    tokenizer=tokenizer
)
```

---

## 🏗️ Architecture Overview

```
SVD_linear_Attention/
├── Codes/
│   ├── framework/           # Core PyTorch module library (Attentions, Compressors, Trainers)
│   ├── django_api/          # REST & WebSocket API Gateway (78 Apps, GPU Monitor, LaTeX Studio)
│   ├── frontend/            # React 18 + Vite Research Hub & Dashboard
│   ├── run.py               # Single experiment CLI runner
│   ├── run_all.py           # Multi-model batch evaluation suite
│   └── main.py              # Unified launcher
├── Docs/
│   ├── catalog.json         # Master index of 454 guides, reports, studios & masterclasses
│   ├── content/             # 123 Attention + 86 Compression specs, plus deep dives
│   └── results/             # Benchmark summaries, Pareto data & CSV exports
└── datasets/                # Benchmark evaluation datasets (GLUE, WikiText, C4, etc.)
```

---

## 🔬 Supported Attention Architectures

| Paradigm | Example Implementations | Complexity | Primary Benefit |
| :--- | :--- | :---: | :--- |
| **Spectral & SVD** | `svd_linear`, `aegis_v6`, `nasc_attention`, `spectron` | $\mathcal{O}(N \cdot r \cdot d)$ | Optimal rank-truncated parameter savings ($>70\%$). |
| **State-Space (SSM)** | `mamba_svd_flash`, `rwkv`, `gated_delta_net` | $\mathcal{O}(N \cdot d)$ | Constant-memory recurrent generation. |
| **Latent & Low-Rank** | `mla_attention`, `gqa_attention`, `wgqa_attention` | $\mathcal{O}(N \cdot d_v)$ | Drastic KV cache memory reduction. |
| **Random Features** | `performer_attention`, `linformer_attention` | $\mathcal{O}(N \cdot m)$ | Linearized softmax kernel approximation. |
| **Hardware-Tiled** | `flash_attention3_fp8`, `aegis_spectron_fp8` | Exact $\mathcal{O}(N^2)$ | Asynchronous TMA memory transfers with FP8 Tensor Cores. |
| **Sparse & MoE** | `nsa_attention`, `moe_deltakv_paged` | $\mathcal{O}(N \cdot k)$ | Dynamic routing and selective token computation. |

---

## 🗜️ Supported Compression Engines

- **AASVD (`aasvd`)**: Activation-aware SVD preconditioning against empirical covariance matrices.
- **Rotational SliceGPT (`rotational_slicegpt_int4`)**: Unitary basis transformation preventing outlier channel clipping during 4-bit quantization.
- **DeltaKV (`delta_kv`)**: Differential keyframe temporal cache encoding for long-context generation ($16.4\times$ KV savings).
- **AWQ (`awq`)**: Salient channel preservation for INT4 weight quantization.
- **ARMOR & AST (`armor_pruner`, `ast_pruner`)**: Outlier-aware structural attention head and channel pruning without retraining.

---

## 📊 Empirical Benchmarks

Tested on **LLaMA-3-70B** across standard benchmark suites (WikiText-103, MMLU, GSM8K):

| Architecture / Compressor | Type | VRAM Footprint | Compression Ratio | Speedup | WikiText-103 PPL | Accuracy Retention |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense Baseline (FP16)** | Full MHA | $138.4\text{ GB}$ | $1.0\times$ | $1.0\times$ | $3.12$ | $100.0\%$ |
| **SVD Linear Attention** | Attention | **$38.8\text{ GB}$** | **$3.57\times$** | **$4.12\times$** | **$3.16$** | **$99.5\%$** |
| **Mamba SVD Flash** | Attention | **$30.4\text{ GB}$** | **$4.55\times$** | **$5.45\times$** | **$3.18$** | **$99.3\%$** |
| **Rotational SliceGPT INT4** | Compression | **$34.8\text{ GB}$** | **$3.98\times$** | **$3.98\times$** | **$3.16$** | **$99.4\%$** |
| **DeltaKV Cache** | KV Compression | **$8.4\text{ GB}$** | **$16.48\times$** | **$6.25\times$** | **$3.13$** | **$99.7\%$** |
| **AASVD Compressor** | Compression | **$38.2\text{ GB}$** | **$3.62\times$** | **$3.65\times$** | **$3.15$** | **$99.6\%$** |

---

## 🧪 Running Experiments & Sweeps

```bash
# 1. Quick GPU Smoke Test (2-3 minutes)
python Codes/run_all.py --mode fast --device cuda

# 2. Comprehensive 192-Model Evaluation (25-35 minutes on RTX 4070)
python Codes/run_all.py --mode comprehensive --device cuda

# 3. Fine-Tune on Downstream Task (e.g. GLUE SST-2)
python Codes/run.py \
  --model distilbert-base-uncased \
  --task sst2 \
  --attn_type svd_linear \
  --rank 64 \
  --epochs 3 \
  --device cuda
```

---

## 🖥️ Full-Stack Research Workspace

### Launching the Django Backend & React Frontend
```bash
# 1. Start Django Backend API (Port 8000)
python Codes/django_api/manage.py runserver 0.0.0.0:8000

# 2. Start React Research Dashboard (Port 5173)
cd Codes/frontend
npm run dev
```

Visit `http://localhost:5173` to explore the live interactive dashboard, experiment metrics, and paper drafting studio.

---

## 📚 Documentation & Masterclass Hub

All documentation is indexed in `Docs/catalog.json` and rendered interactively inside the web UI (`/docs`):
- **106 Attention Masterclasses**: Located in `Docs/content/attentions_blog/`
- **84 Compression Masterclasses**: Located in `Docs/content/compressions_blog/`
- **6 Deep-Dive Series Guides**: Located in `Docs/blog/`

---

## 📖 References

1. **Vaswani et al.** (2017). *Attention Is All You Need*. NeurIPS.
2. **Dao et al.** (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.
3. **Katharopoulos et al.** (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML.
4. **Xiao et al.** (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*. ICML.
5. **Lin et al.** (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression*. MLSys.
6. **Ashkboos et al.** (2024). *SliceGPT: Compress Large Language Models by Deleting Rows and Columns*. ICLR.
7. **Gu & Dao** (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.
8. **DeepSeek-AI** (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*.

---

*Maintained by the NeuralOps Core Research Team.*
