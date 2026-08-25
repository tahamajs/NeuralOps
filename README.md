# 🧬 NeuralOps: SVD Linear Attention & Model Compression Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Attention Mechanisms](https://img.shields.io/badge/Attentions-236%20Registered-6366f1.svg)](./Codes/framework/attentions/)
[![Compression Engines](https://img.shields.io/badge/Compressors-107%20Registered-ec4899.svg)](./Codes/framework/compression_engine/)
[![Verified Tracks](https://img.shields.io/badge/Tracks-5%20Verified%20(100%25)-10b981.svg)](./Codes/framework/)
[![Dashboard](https://img.shields.io/badge/UI-React%2018%20+%20Vite-61dafb.svg)](./Codes/frontend/)
[![Backend](https://img.shields.io/badge/API-Django%20REST-092e20.svg)](./Codes/django_api/)

An enterprise-grade research and production framework for **Linear & Subquadratic Attention Architectures**, **Subspace SVD Decompositions**, **Quantization & Structured 2:4 Sparsity**, **Automated Dataset Caching**, and **Universal Multi-Model Remote GPU Execution** for Foundation Models.

---

## 📑 Table of Contents
1. [Key Capabilities & 5-Track Architecture](#-key-capabilities--5-track-architecture)
2. [Universal Multi-Model Compatibility](#-universal-multi-model-compatibility)
3. [Quickstart (30 Seconds)](#-quickstart-30-seconds)
4. [GPU Server Execution & Deployment](#-gpu-server-execution--deployment)
5. [Automated Dataset Preparation Pipeline](#-automated-dataset-preparation-pipeline)
6. [Supported Attention & Compression Modules](#-supported-attention--compression-modules)
7. [5-Track Parallel Validation Suite](#-5-track-parallel-validation-suite)
8. [Full-Stack Research Workspace](#-full-stack-research-workspace)
9. [Documentation & Mathematical Proofs](#-documentation--mathematical-proofs)

---

## 🌟 Key Capabilities & 5-Track Architecture

- **236 Attention Mechanisms**: Registered in dynamic `ATTENTION_REGISTRY` covering exact SVD factorized projections, Spectral/SSM scans (AEGIS v1–v6, Spectra, Mamba, Gated DeltaNet), Mixture-of-Experts routing (BAM-UMoE, Samba, Jamba, Hymba, NEXUS), Kernel approximations (FAVOR+ Performer, Nyström, FlashAttention-3, Cosformer), and dynamic sliding-window attention.
- **107 Compression Engines**: Registered in `COMPRESSOR_REGISTRY` including Error-Controlled SVD (`erc_svd`), Entropy Budget Allocation (`entropy_budget_allocation`), Rotational SliceGPT INT4, SparseGPT with damped Hessian inversions, ADMM structured pruning, 2:4 hardware sparsity, and dynamic KV cache pruning (`zerokv_neo`, `delta_kv`, `kvtc`, `hybridkv`).
- **5 Synchronized Evaluation Tracks**:
  - **Track 1**: Low-Rank Factorization & SVD (13 Attentions, 17 Compressors)
  - **Track 2**: Spectral, SSM & Recurrent State-Space Models (28 Attentions, 1 Compressor)
  - **Track 3**: Mixture-of-Experts, Token Pruning & Hybrid Routing (20 Attentions, 1 Compressor)
  - **Track 4**: Kernel Approximations & Dynamic Quantization (11 Attentions, 9 Compressors)
  - **Track 5**: Structural Pruning, 2:4 Sparsity & Layer Shedding (2 Attentions, 18 Compressors)
- **Universal Multi-Model Introspection**: Zero-shot injection into **BERT**, **RoBERTa**, **T5** (`T5EncoderModel`), **DistilBERT**, **GPT-2**, and **LLaMA** architectures with universal `.weight` reconstructed property support.
- **Automated Dataset Engine**: Tokenization, dynamic sequence padding, and streaming serialization for GLUE tasks (`sst2`, `mrpc`, `qnli`, `mnli`, `cola`, `rte`, `stsb`, `qqp`) and Language Modeling (`wikitext2`, `c4`) with zero-latency disk caching and offline synthetic fallbacks.

---

## 🔄 Universal Multi-Model Compatibility

| Model Architecture | HuggingFace Class | Attention Replacement Target | Linear Layer Replacement Target | Introspection Compatibility |
| :--- | :--- | :--- | :--- | :---: |
| **BERT** | `BertForSequenceClassification` | `bert.encoder.layer[*].attention.self` | `nn.Linear` (all query, key, value, dense) | ✅ 100% Passing |
| **RoBERTa** | `RobertaForSequenceClassification` | `roberta.encoder.layer[*].attention.self` | `nn.Linear` (all query, key, value, dense) | ✅ 100% Passing |
| **T5 Encoder** | `T5EncoderModel` | `encoder.block[*].layer[0].SelfAttention` | `DenseReluDense.wi`, `wi_0`, `wi_1`, `wo` | ✅ 100% Passing (`.weight` property) |
| **DistilBERT** | `DistilBertForSequenceClassification` | `distilbert.transformer.layer[*].attention` | `q_lin`, `k_lin`, `v_lin`, `out_lin` | ✅ 100% Passing |
| **GPT-2** | `GPT2LMHeadModel` | `transformer.h[*].attn` | `c_attn`, `c_proj`, `c_fc` | ✅ 100% Passing |
| **LLaMA / Mistral** | `AutoModelForCausalLM` | `model.layers[*].self_attn` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate/up/down` | ✅ 100% Passing |

---

## ⚡ Quickstart (30 Seconds)

### Installation
```bash
# Clone repository
git clone https://github.com/tahamajs/SVD_linear_Attention.git
cd SVD_linear_Attention

# Setup Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core framework in editable mode
pip install -e Codes/framework/
pip install -r Codes/django_api/requirements.txt
```

### 1. Zero-Shot Attention Injection
```python
import torch
from transformers import AutoModelForSequenceClassification
from framework.core.attention_utils import inject_attention

# 1. Load standard pretrained transformer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 2. Swap standard attention with SVD Linear Attention across all layers
model = inject_attention(model, attn_type="svd_linear_attention", rank=32)

# 3. Forward pass with O(N) factorized attention
inputs = torch.randint(0, 1000, (2, 128))
outputs = model(inputs)
print("Forward logits:", outputs.logits.shape)
```

### 2. Post-Training Compression Pipeline
```python
from transformers import AutoModelForSequenceClassification
from framework.core.compression_engine import compress_model

model = AutoModelForSequenceClassification.from_pretrained("roberta-base")

# 2. Apply Error-Controlled SVD (ERC-SVD) + INT8 Quantization
compressed_model = compress_model(
    model,
    config={
        "compressors": [
            {"method": "erc_svd", "allowed_error": 0.05, "rank": 32},
            {"method": "astro_quantizer", "bits": 8}
        ]
    }
)
```

---

## 🖥️ GPU Server Execution & Deployment

The framework includes dedicated tooling for automated multi-GPU probing, native BF16/FP16 AMP mixed precision, and single-command benchmark runs:

```bash
# 1. Inspect hardware capabilities & Tensor Cores
bash Codes/framework/run_on_gpu_server.sh --mode diagnostics

# 2. Pre-cache all GLUE & LM datasets to disk for zero-latency streaming
bash Codes/framework/run_on_gpu_server.sh --mode prepare-data

# 3. Run fast verification sweep across BERT, RoBERTa, and T5
bash Codes/framework/run_on_gpu_server.sh --mode fast

# 4. Run full multi-model, multi-track evaluation matrix
bash Codes/framework/run_on_gpu_server.sh --mode comprehensive
```

For complete GPU server setup, CUDA drivers, and SLURM / DDP multi-GPU scaling instructions, see [`Codes/framework/GPU_SERVER_GUIDE.md`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/GPU_SERVER_GUIDE.md).

---

## 📦 Automated Dataset Preparation Pipeline

Tokenizes and serializes benchmark splits to disk to eliminate runtime network dependencies on remote compute nodes:

```bash
# Download and pre-tokenize all GLUE and Language Modeling benchmarks
PYTHONPATH=Codes python3 Codes/framework/data/prepare_datasets.py \
  --benchmarks glue,lm \
  --models bert-base-uncased,roberta-base,t5-base \
  --output_dir datasets/cache/ \
  --max_seq_len 128 \
  --num_workers 4
```

---

## 🔬 5-Track Parallel Validation Suite

Run the multi-process parallel validation engine across all 5 tracks:

```bash
PYTHONPATH=Codes python3 Codes/framework/parallel_track_checker.py \
  --workers 5 \
  --models bert,roberta,t5 \
  --datasets sst2,mrpc \
  --rank 16
```

---

## 🏛️ Full-Stack Research Workspace

- **Frontend UI (React 18 + Vite + TailwindCSS)**: Located in [`Codes/frontend/`](file:///Users/tahamajs/Documents/uni/Project/Codes/frontend/). Features interactive 3D latent space visualizers, Pareto frontier analysis, and real-time telemetry.
- **Backend API (Django REST Framework + WebSockets)**: Located in [`Codes/django_api/`](file:///Users/tahamajs/Documents/uni/Project/Codes/django_api/). Handles asynchronous experiment queuing, Celery job scheduling, and Overleaf-style paper drafting.
- **Documentation & Proofs**: Located in [`Docs/content/`](file:///Users/tahamajs/Documents/uni/Project/Docs/content/). Includes rigorous mathematical formulations, Lyapunov stability bounds, and energy thresholding proofs.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
