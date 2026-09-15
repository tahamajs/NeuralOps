<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Attention Variants

### 2.1.1 Multi-Head Attention (MHA)

Introduced by Vaswani et al. (2017), MHA splits the hidden dimension into 
$H$ heads, each with its own $Q, K, V$ projections. This is the standard 
configuration for BERT, RoBERTa, and most encoder-only models.

### 2.1.2 Multi-Query Attention (MQA)

Shazeer (2019) proposed MQA, which shares a single $K, V$ pair across all 
query heads. This reduces the KV cache by a factor of $H$ but often 
degrades quality, especially for tasks requiring head diversity.

### 2.1.3 Grouped-Query Attention (GQA)

Ainslie et al. (2023) proposed GQA as a middle ground, evaluated on 
T5-based models. GQA is now standard in LLaMA-2, Mistral, Qwen, and 
Gemma. Our work re-implements GQA in a unified framework with extensive 
architecture support.

## 2.2 KV Cache Optimization

### 2.2.1 Eviction

H2O (Zhang et al., 2023) keeps attention sinks and heavy hitters. 
StreamingLLM (Xiao et al., 2024) preserves initial tokens. These methods 
are orthogonal to GQA.

### 2.2.2 Quantization

KIVI (Liu et al., 2024) quantizes KV to 2-bit. KVQuant (Hooper et al., 2024) 
uses per-channel scaling. GQA can be combined with quantization for 
multiplicative savings.

### 2.2.3 Low-Rank Compression

Low-rank KV cache compression (e.g., MLA in DeepSeek-V2) projects KV into 
a low-rank space. This is an orthogonal approach to reducing the number of 
heads.

## 2.3 Parameter-Efficient Fine-Tuning

LoRA (Hu et al., 2021) freezes base weights and trains low-rank adapters. 
We use LoRA as a recovery mechanism after GQA injection.

## 2.4 Position of Our Work

| Axis | Original GQA (Ainslie et al.) | Our Implementation |
|---|---|---|
| Scope | T5-based enc-dec | Enc, dec, encoder-only, GQA-native |
| Weight transfer | Not detailed | 4 architectures |
| Fallback | Not specified | Returns None on failure |
| RoPE | Partial | Full support |
| Companion methods | None | Integrated with A3, AASVD, etc. |

**Key distinction.** Our work is a **production-hardened re-implementation**, 
not a new algorithmic contribution. We focus on robustness, architecture 
coverage, and integration with the broader compression toolkit.

## 2.5 Comparison with Related Variants

| Variant | $H_{KV}$ | KV reduction | Notes |
|---|---|---|---|
| MHA | $H$ | 1× | Baseline |
| GQA | $G \le H$ | $H / G$ | Tunable |
| MQA | 1 | $H$× | Aggressive |
| MLA (DeepSeek) | Low-rank | varies | Compressed space |
