<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Efficient Attention Architectures

### 2.1.1 Sparse Attention

Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., 2020). These 
methods design sparse attention patterns but do not prune tokens. EAT 
combines standard attention with dynamic pruning.

### 2.1.2 Low-Rank Attention

Linformer (Wang et al., 2020) uses low-rank key/value projections. 
Nyströmformer (Xiong et al., 2021) uses Nyström approximation. EAT 
preserves standard attention shape.

### 2.1.3 Kernel Attention

Performer (Choromanski et al., 2021) and Linear Transformer (Katharopoulos 
et al., 2020). EAT uses standard softmax attention.

## 2.2 Token Pruning and Reduction

### 2.2.1 Vision Transformers

DynamicViT (Rao et al., 2021) and EViT (Liang et al., 2022) prune tokens in 
vision transformers based on attention scores. EAT borrows this idea but 
applies it to text.

### 2.2.2 Text Transformers

LTP (Kim et al., 2022) uses a learned scorer to prune tokens. PuMer (Cao et 
al., 2023) uses a similar approach with vision-language models. EAT differs 
by combining pruning with LoRA recovery and multi-architecture support.

### 2.2.3 Token Merging

ToMe (Bolya et al., 2023) merges similar tokens rather than pruning. EAT 
prunes outright; the merge alternative is a future direction.

## 2.3 Parameter-Efficient Fine-Tuning

LoRA (Hu et al., 2021), AdaLoRA (Zhang et al., 2023), DoRA (Liu et al., 2024). 
EAT uses standard LoRA for recovery.

## 2.4 Knowledge Transfer

DistilBERT (Sanh et al., 2019) distills. EAT transfers weights exactly 
(zero-shot). The two are orthogonal.

## 2.5 Position of Our Work

| Axis | Prior work | EAT |
|---|---|---|
| **Architecture** | Single family | Four families |
| **Weight transfer** | Retrain or distill | Zero-shot transfer |
| **Recovery** | Full or LoRA | LoRA only |
| **Pruning** | Train-time | Inference-time |
| **Parameter effect** | Reduces | **Increases** (LoRA) |

**Key distinction.** EAT is not a compressor. It replaces attention to 
support **inference-time pruning** and **multi-architecture deployment**. 
Any parameter reduction reported for EAT would be a misattribution.

## 2.6 Comparison to Companion Modules

| Module | Type | Params | Pruning | Multi-arch |
|---|---|---|---|---|
| NHA | Hybrid attention | +LoRA | No | Yes |
| A3 | Low-rank compression | **−50%** | No | Yes |
| AASVD | Low-rank compression | **−50%** | No | Yes |
| **EAT** | Attention + pruning | **+LoRA** | **Yes** | **Yes** |

---