---
title: NeuralOps 3D Architecture Atlas & Benchmark Hub
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: static
pinned: true
license: mit
---

<div align="center">

# ⚡ Hooshaai (هوشا)
### Frontier Efficiency Lab for Next-Generation Neural Architectures
*In-Place Model Surgery · Subquadratic Sequence Modeling · Hardware-Silicon Co-Design*

<br/>

[![Interactive 3D Space](https://img.shields.io/badge/%F0%9F%8C%90%203D%20Space-NeuralOps%20Atlas-6366f1?style=for-the-badge&logo=three.js&logoColor=white)](https://huggingface.co/spaces/Hooshaai/README)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Model%20Zoo-14%20Frontier%20Checkpoints-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Hooshaai)
[![Hugging Face Ledger](https://img.shields.io/badge/%F0%9F%A4%97%20Empirical%20Ledger-Hooshaai%2Fneuralops--results-orange?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Hooshaai/neuralops-results)
[![PyPI](https://img.shields.io/badge/PyPI-neuralops%20v2.0.2-3776ab?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/neuralops/)
[![GitHub](https://img.shields.io/github/stars/tahamajs/NeuralOps?style=for-the-badge&logo=github&color=10b981)](https://github.com/tahamajs/NeuralOps)

<br/>

**[✨ Follow Hooshaai on Hugging Face](https://huggingface.co/Hooshaai)** • **[💻 GitHub Repository](https://github.com/tahamajs/NeuralOps)** • **[📑 Academic Monograph (317 pages)](https://github.com/tahamajs/NeuralOps/releases/tag/academic-monographs)**

---

</div>

## 🌌 What is Hooshaai & NeuralOps?

Standard self-attention scales at **$\mathcal{O}(N^2)$**, leading to catastrophic out-of-memory errors on long sequences and locking developers into high compute costs.

**NeuralOps** solves this by bridging the gap between theoretical subquadratic models and physical silicon:

1. **⚡ In-Place Attention Surgery**: Convert standard Softmax attention backbones (BERT, RoBERTa, GPT-2, LLaMA) into linear-complexity subquadratic models in milliseconds.
2. **🎯 6-Pillar Knowledge Healing**: Restore dense baseline accuracy with post-surgery distillation and Warmup-Stable-Decay (WSD) scheduling.
3. **🏎️ 3-Level Fallback Dispatch**: Automatic routing across native CUDA C++, autotuned Triton JIT, and vectorized PyTorch references.

```
Standard Transformer (O(N²)) : Input ──► [ N × N Quadratic Attention ] ──► OOM at 16k
Hooshaai NeuralOps  (O(N))   : Input ──► [ Associative Linear Scan   ] ──► Scalable to 1M+
```

---

## 🚀 Silicon Speedup on NVIDIA Ada Lovelace (RTX 4070)

All results measured on physical silicon (CUDA 12.8, PyTorch 2.4, Triton 3.0):

```
Sequence Length: 8,192 Tokens
Standard FlashAttention-2: ████████████████████ (92.6 ms | 6.85 GB VRAM)
Hooshaai Cosformer Fused : ████ (15.9 ms - 5.8x faster | 780 MB VRAM)
Hooshaai Griffin-SSD     : ███ (12.5 ms - 7.4x faster | 640 MB VRAM)

Sequence Length: 32,768 Tokens
Standard FlashAttention-2: 💥 CUDA OUT OF MEMORY (> 12 GB)
Hooshaai Hymba-TransX    : ██████ (48.1 ms - Scalable to 1M+ tokens | 1.84 GB VRAM) ✅
```

---

## 🏛️ Ecosystem Invariants Verified on Silicon

| Component | Verified Count | Architectural Scope & Implementation |
| :--- | :---: | :--- |
| **Attention Mechanisms** | **131** | 14 architectural families: Dense, Linear, Gated, Delta, SSM, Sparse, SVD, MLA, Hybrid, MoE, TTT, Memory, Positional, Fused |
| **Compression Engines** | **72** | 8 paradigms: Low-rank SVD, Dynamic rank, Structured pruning, Subspace, Tensor decomp, Quantization, KV-cache, Rotational invariance |
| **Total Unified Operators** | **203** | 131 Attentions + 72 Compressors mathematically verified against finite-difference autograd (`atol=1e-3, eps=1e-5`) |
| **Native CUDA C++ Packages** | **130** | 132 native implementations with 128-bit vector loads, warp shuffles, and bank-conflict elimination |
| **Triton JIT Kernel Modules** | **26** | 62 unified operators autotuned across block tiles for FP16, BF16, and FP32 |
| **Master Jupyter Tutorials** | **29** | Step-by-step reproducible notebooks from SVD whitening to end-to-end post-surgery distillation |

---

## 🤗 14 Frontier Hub Checkpoints (Ready to Download)

Pre-trained and surgically healed checkpoints ready for inference via `transformers` and `neuralops`:

| Checkpoint | Base Backbone | Subquadratic Architecture | Compression Strategy | Hugging Face Hub |
| :--- | :--- | :--- | :--- | :---: |
| `distilbert-base-fwsvd-rank64` | DistilBERT | SVD Linear Attention | Fisher-Weighted SVD ($r=64$) | [🤗 Model](https://huggingface.co/Hooshaai/distilbert-base-fwsvd-rank64) |
| `gpt2-medium-fwsvd-rank64` | GPT-2 Medium | SVD Linear Attention | Fisher-Weighted SVD ($r=64$) | [🤗 Model](https://huggingface.co/Hooshaai/gpt2-medium-fwsvd-rank64) |
| `roberta-base-fwsvd-rank32` | RoBERTa Base | SVD Linear Attention | Fisher-Weighted SVD ($r=32$) | [🤗 Model](https://huggingface.co/Hooshaai/roberta-base-fwsvd-rank32) |
| `distilbert-sst2-cosformer` | DistilBERT | Cosformer | Cosine Reweighting Scan | [🤗 Model](https://huggingface.co/Hooshaai/distilbert-sst2-cosformer) |
| `distilbert-sst2-griffin-ssd` | DistilBERT | Griffin-SSD | Sliding Window + State-Space | [🤗 Model](https://huggingface.co/Hooshaai/distilbert-sst2-griffin-ssd) |
| `distilbert-sst2-hymba-transx` | DistilBERT | Hymba TransX | Differential + Associative SSM | [🤗 Model](https://huggingface.co/Hooshaai/distilbert-sst2-hymba-transx) |
| `distilbert-agnews-full-mla` | DistilBERT | Multi-Head Latent (MLA) | Low-Rank Latent KV Bottleneck | [🤗 Model](https://huggingface.co/Hooshaai/distilbert-agnews-full-mla) |
| `distilbert-sst2-adasplash` | DistilBERT | AdaSplash | Tsallis 1.5-Entmax Sparsity | [🤗 Model](https://huggingface.co/Hooshaai/distilbert-sst2-adasplash) |

*👉 Explore all 14 models at **[huggingface.co/Hooshaai](https://huggingface.co/Hooshaai)**.*

---

## ⚡ Quickstart

Install `neuralops` via pip:

```bash
pip install neuralops
```

Perform in-place attention surgery on any pre-trained model:

```python
from transformers import AutoModelForSequenceClassification
from neuralops import replace_attention

# 1. Load any standard pre-trained transformer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# 2. In-place surgery: Swap quadratic attention for subquadratic Cosformer in milliseconds
replace_attention(model, target_attention="cosformer")

print("✓ In-place surgery complete. Model is now linear-complexity O(N).")
```

---

## 📊 Benchmark Dataset (`Hooshaai/neuralops-results`)

Access our empirical SQLite and Parquet ledgers containing over 2,600 verified benchmarks across 675 modules:

```python
from datasets import load_dataset

dataset = load_dataset("Hooshaai/neuralops-results", data_files="attentions_sweep.parquet")
df = dataset["train"].to_pandas()
print(df[["module_name", "throughput_seq_per_sec", "peak_memory_mb"]].head())
```

---

<div align="center">

### ⭐ Follow Hooshaai to Stay Updated
We publish open-source subquadratic models, CUDA/Triton kernels, and empirical papers weekly.

**[🤗 Follow @Hooshaai](https://huggingface.co/Hooshaai)** • **[⭐ Star on GitHub](https://github.com/tahamajs/NeuralOps)**

</div>
