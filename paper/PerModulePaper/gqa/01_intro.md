<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Autoregressive inference in transformer models is memory-bound, not 
compute-bound. During generation, the model stores keys and values for every 
past token in a **KV cache**, whose size is:

$$
\text{KV size} = 2 \cdot H_{KV} \cdot d_h \cdot L \cdot B \cdot \text{bytes}
\tag{1.1}
$$

where $H_{KV}$ is the number of KV heads, $d_h$ is the per-head dimension, 
$L$ is the sequence length, and $B$ is the batch size. For a 12-layer GPT-2 
model with $H = 12$, $d_h = 64$, $L = 1024$, and $B = 1$, the KV cache at 
FP16 is:

$$
2 \cdot 12 \cdot 64 \cdot 1024 \cdot 12 \cdot 2 = 37.7 \text{ MB}
\tag{1.2}
$$

which is **5% of the model's 774 MB weight footprint** — small in isolation, 
but growing linearly with sequence length and batch size. At $L = 32{,}768$ 
and $B = 8$, the KV cache dominates inference memory, requiring **9.6 GB** 
per layer group, exceeding most consumer GPUs.

Two extremes have been explored to reduce this cost:

1. **Multi-Head Attention (MHA).** Each of the $H$ query heads has its own 
   $K$ and $V$ ($H_{KV} = H$). Maximally expressive, maximally memory-intensive.
2. **Multi-Query Attention (MQA).** All query heads share a single $K, V$ 
   pair ($H_{KV} = 1$). Minimal memory, but empirically worse quality, 
   especially on tasks requiring fine-grained attention.

**Grouped-Query Attention (GQA)** was introduced as a middle ground: partition 
the $H$ query heads into $G$ groups, each with a dedicated $K, V$ pair 
($H_{KV} = G$). The KV cache shrinks by a factor of $H / G$, and quality 
degrades gracefully with $G$.

## 1.2 Problem Statement

Given a pretrained MHA-based model $\mathcal{M}$ with $H$ query heads and 
$H_{KV} = H$, we seek a modified model $\mathcal{M}'$ such that:

$$
\mathcal{M}' \approx \mathcal{M}, \qquad
\text{KV}(\mathcal{M}') \ll \text{KV}(\mathcal{M})
\tag{1.3}
$$

We impose the following constraints:

1. **Quality preservation:** $\Delta\text{Acc} \leq 5\%$ on SST-2.
2. **Weight compatibility:** No retraining of base weights; use LoRA recovery.
3. **Architecture portability:** Must work with flat, nested, and causal transformers.
4. **Inference ready:** KV cache reduction must be realized at generation time.

## 1.3 Our Approach

We present a production-hardened implementation of GQA with three key features:

1. **Exact weight transfer.** The `copy_weights` procedure handles four 
   distinct attention configurations: flat (DistilBERT), nested (RoBERTa), 
   causal (GPT-2), and grouped-query (Qwen3.5). For MHA → GQA conversion, 
   query weights are copied verbatim; KV weights are grouped and repeated.

2. **Flexible group size.** We support $G \in \{1, 2, \ldots, H\}$. The 
   default is $G = H / 2$, giving a 2× KV cache reduction.

3. **LoRA recovery.** After injection, the base weights are frozen and 
   only LoRA adapters are trainable. This preserves pretrained knowledge 
   while allowing task-specific adaptation.

## 1.4 Contributions

- **A production-hardened GQA implementation.** Our PyTorch implementation 
  handles four attention variants, RoPE, LoRA recovery, and safe fallback.
- **A systematic evaluation.** We benchmark GQA on DistilBERT, RoBERTa, and 
  GPT-2 on SST-2 with ablations over $G / H$ ratio.
- **KV cache analysis.** We provide a detailed breakdown of where the memory 
  savings come from and how they scale with $L$ and $B$.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts are 
  released under MIT.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention 
and KV cache. Section 4 presents GQA. Section 5 describes integration with 
four architectures. Section 6 details the experimental setup. Section 7 
reports results. Section 8 analyzes where quality is lost. Section 9 
discusses practical guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $H$ | Number of query heads |
| $H_{KV}$ | Number of KV heads |
| $G$ | Number of KV groups ($= H_{KV}$) |
| $d_h$ | Per-head dimension |
| $L$ | Sequence length |
| $B$ | Batch size |
| $\text{KV}(\cdot)$ | KV cache memory footprint |
| $\mathcal{A}(\cdot)$ | Attention function |
| $\text{repeat\_kv}(\cdot)$ | KV head repetition operator |
