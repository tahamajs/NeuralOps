# ⚙️ SVD Linear Attention Framework Core

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Registries](https://img.shields.io/badge/Registries-236%20Attn%20%7C%20107%20Comp-blueviolet.svg)](./core/registry.py)
[![Tracks](https://img.shields.io/badge/Tracks-5%20Verified%20(100%25)-10b981.svg)](./eval_all_models_datasets.py)

The core PyTorch library for the NeuralOps framework, providing **236 custom attention mechanisms**, **107 post-training compression engines**, model wrapping and attention injection, distributed trainers, dataset loaders, and benchmark visualization tools.

---

## 📑 Table of Contents
1. [Core Subsystems Overview](#-core-subsystems-overview)
2. [5 Evaluation Tracks & Module Mapping](#-5-evaluation-tracks--module-mapping)
3. [Base Classes & Registry Design](#-base-classes--registry-design)
4. [Zero-Shot Attention Injection (`inject_attention`)](#-zero-shot-attention-injection-inject_attention)
5. [Compression Pipelines (`compress_model`)](#-compression-pipelines-compress_model)
6. [Automated Dataset Preparation Pipeline](#-automated-dataset-preparation-pipeline)
7. [GPU Server Execution & Mixed Precision](#-gpu-server-execution--mixed-precision)
8. [Parallel Verification & Testing Suites](#-parallel-verification--testing-suites)

---

## 🏗️ Core Subsystems Overview

```
Codes/framework/
├── attentions/             # 236 Custom Attention modules (SVD, Mamba, MLA, GQA, Flash3, etc.)
├── compression_engine/     # 107 Compression algorithms (AASVD, SliceGPT INT4, DeltaKV, AWQ, etc.)
├── core/                   # Core abstractions, registries, and model wrappers
│   ├── base.py             # BaseAttention, BaseCompressor abstract base classes
│   ├── registry.py         # ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
│   ├── attention_utils.py  # inject_attention() for zero-shot layer replacement
│   ├── architecture.py     # Multi-architecture transformer layer detector
│   ├── compression_engine.py # Chained CompressionPipeline executor
│   └── model_wrapper.py    # UnifiedTransformerModel wrapper
├── data/                   # Dataset loaders & Automated Preprocessing Engine
│   ├── prepare_datasets.py # Tokenizer, dynamic padding, and disk cache serializer
│   ├── benchmark_datasets.py # Unified PyTorch DataLoader streamer
│   └── glue_loader.py      # GLUE dataset local loader
├── launch_gpu_server.py    # GPU Server hardware profiler & DDP launcher
├── run_on_gpu_server.sh    # Single-command shell runner (--mode fast/comprehensive)
├── parallel_track_checker.py # 5-process parallel multi-track validator
├── check_framework_sync.py # Framework integrity & registry mapping auditor
├── auto_run.py             # Automated multi-track CLI benchmark runner
├── trainer/                # HuggingFaceTrainer, CustomLoopTrainer, FSDP/DDP hooks
├── visualization/          # Heatmaps, Pareto frontiers, and HTML report generators
├── analysis/               # Spectral energy decay, Hessian, and outlier analyzers
└── results_saver/          # Structured SQLite, Parquet, and JSON exporters
```

---

## 🎯 5 Evaluation Tracks & Module Mapping

The framework is organized into 5 standardized evaluation tracks mapped in `eval_all_models_datasets.py`:

### Track 1: Low-Rank Factorization & SVD
- **Attentions (13)**: `svd_linear_attention`, `adaptive_svd_attention`, `asvd_attention`, `linformer_attention`, `res_svd`, `factorised_matrix_state`, `svd_projection_attention`, `svd_llm_v2_attention`, `svd_attention`, `projection_factorized_attn`, `svd_linear`, `adaptive_svd_attn`, `asvd`.
- **Compressors (17)**: `direct_svd`, `dobi_svd`, `erc_svd`, `entropy_budget_allocation`, `kronecker_tt_lora`, `swift_svd`, `aasvd`, `rpca`, `cur_moe`, `grecodec`, `frandi`, `dynamic_rank`, `hybrid_svd`, `svd_linear`, `aa_svd`, `erc`, `entropy_budget`.

### Track 2: Spectral, SSM & Recurrent State-Space Models
- **Attentions (28)**: `aegis_v1` through `aegis_v6`, `aegis_hybrid`, `aegis_spectron_fp8`, `spectra_v2`, `spectral_mamba`, `Mamba`, `SVD-Mamba`, `simple_ssm`, `gated_delta_net`, `gated_deltanet_block`, `gated_linear_attention`, `hgrn2_layer`, `hgrn2_gated_deltanet`, `hgrn2_outer`, `griffin_ssd`, `regla_attention`, `halo_regla`, `kda_attention`, `decay_linear_attention`, `spectron`, `galerkin_attention`, `vspec`, `vspec_attention`.
- **Compressors (1)**: `mamba_shedder`.

### Track 3: Mixture-of-Experts, Token Pruning & Hybrid Routing
- **Attentions (20)**: `bam_umoe`, `expert_choice_moe`, `reap_moe`, `samba_moe`, `samba_m3`, `jamba`, `jamba_info`, `hymba_transx`, `moe_deltakv_paged`, `block_transformer_hiba`, `nexus`, `leka_v1`, `leka_v2`, `leka_v3`, `mixattention_moska`, `phantom`, `moba`, `fasa`, `upi_stuffed_mamba`, `granite_hybrid`.
- **Compressors (1)**: `cur_moe`.

### Track 4: Kernel Approximations & Dynamic Quantization
- **Attentions (11)**: `performer_attention`, `nystrom_attention`, `flash_attention3`, `cluster_attention`, `adasplash_attention`, `hamming_attention`, `cosformer`, `decay_linear_attention`, `gated_linear_attention`, `SOAFusedAttention`, `fused_svd_kernel`.
- **Compressors (9)**: `astro_quantizer`, `faq_quantizer`, `coopq_quantizer`, `kmeans_quantize`, `zerokv_neo`, `modegpt`, `delta_kv`, `kvtc`, `hybridkv`.

### Track 5: Structural Pruning, 2:4 Sparsity & Layer Shedding
- **Attentions (2)**: `linformer_attention`, `adaptive_sparse_attention`.
- **Compressors (18)**: `slicegpt`, `slicegpt_pca`, `sparsegpt`, `admm_pruner`, `armor_pruner`, `ast_pruner`, `bonsai_pruner`, `flap`, `flat_llm`, `lillama_compressor`, `cachenotes_compressor`, `come`, `oats`, `proxsparse`, `soka`, `liftr`, `mamba_shedder`, `sdrm`.

---

## 🧩 Base Classes & Registry Design

### 1. `BaseAttention` (`core/base.py`)
All custom attention mechanisms inherit from `BaseAttention`:
```python
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("svd_linear_attention")
class SVDLinearAttention(BaseAttention):
    def __init__(self, config, rank: int = 32):
        super().__init__(config, rank=rank)
        self.U = nn.Parameter(torch.empty(self.hidden_size, rank))
        self.V = nn.Parameter(torch.empty(rank, self.hidden_size))
        self.reset_parameters()

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        latent = F.linear(hidden_states, self.V)
        output = F.linear(latent, self.U)
        return output, None
```

### 2. `BaseCompressor` (`core/base.py`)
All model compressors inherit from `BaseCompressor`:
```python
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("erc_svd")
class ERCCompressor(BaseCompressor):
    def __init__(self, model, allowed_error: float = 0.05, rank: int = 32, **kwargs):
        super().__init__(model, name="erc", **kwargs)
        self.allowed_error = allowed_error
        self.rank = rank

    def compress(self):
        # Error-controlled SVD decomposition
        return self.model, self.get_compression_stats()
```

---

## 💉 Zero-Shot Attention Injection (`inject_attention`)

The `inject_attention` utility in `core/attention_utils.py` automatically detects the underlying model architecture (BERT, RoBERTa, T5, DistilBERT, GPT-2, LLaMA) and replaces its multi-head attention layers:

```python
from transformers import AutoModelForSequenceClassification
from framework.core.attention_utils import inject_attention

# 1. Load standard transformer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 2. Swap standard attention with SVD Linear Attention across all layers
model = inject_attention(model, attn_type="svd_linear_attention", rank=32)
```

---

## 📦 Automated Dataset Preparation Pipeline

Use `prepare_datasets.py` to tokenize, split, and cache all evaluation datasets to disk:

```bash
PYTHONPATH=Codes python3 Codes/framework/data/prepare_datasets.py \
  --benchmarks glue,lm \
  --models bert-base-uncased,roberta-base,t5-base \
  --output_dir datasets/cache/ \
  --max_seq_len 128
```

---

## 🖥️ GPU Server Execution & Mixed Precision

Execute benchmarks on a remote GPU server with native BF16/FP16 AMP mixed precision:

```bash
# 1. Hardware diagnostics
bash Codes/framework/run_on_gpu_server.sh --mode diagnostics

# 2. Pre-cache all datasets
bash Codes/framework/run_on_gpu_server.sh --mode prepare-data

# 3. Fast benchmark sweep
bash Codes/framework/run_on_gpu_server.sh --mode fast

# 4. Comprehensive sweep across all 5 tracks
bash Codes/framework/run_on_gpu_server.sh --mode comprehensive
```

---

## 🔬 Parallel Verification & Testing Suites

```bash
# Run 5 parallel workers validating all 5 tracks simultaneously
PYTHONPATH=Codes python3 Codes/framework/parallel_track_checker.py \
  --workers 5 \
  --models bert,roberta,t5 \
  --datasets sst2,mrpc \
  --rank 16

# Verify registry mapping and integrity
PYTHONPATH=Codes python3 Codes/framework/check_framework_sync.py
```
