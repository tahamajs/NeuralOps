<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Background: The KV-Cache Bottleneck

Autoregressive transformer inference stores, for each generated token, the 
Key and Value projections produced by every layer. This **KV cache** grows 
linearly with sequence length and dominates memory usage for long-context 
generation:

$$
\text{KV-cache size} = 2 \cdot N_l \cdot H \cdot d_h \cdot L \cdot B
\tag{1.1}
$$

where $N_l$ is the number of layers, $H$ is the number of heads, $d_h$ is the 
per-head dimension, $L$ is the sequence length, and $B$ is the batch size. 
For a 70B parameter model with $L = 32{,}000$, this can exceed 40 GB per 
batch.

Multiple approaches address this bottleneck:

1. **KV-cache quantization** (KIVI, KVQuant): reduce bit-width per element.
2. **Token eviction** (H2O, StreamingLLM): keep only a subset of tokens.
3. **Low-rank factorization** (MLA, this work): replace K, V with a 
   smaller latent representation.
4. **KV sharing** (GQA, MQA): share KV heads across query heads.

## 1.2 Multi-Head Latent Attention (MLA)

MLA was introduced by DeepSeek-AI (2024) as part of the DeepSeek-V2 
architecture. The key idea is to **replace the separate K and V projections 
with a single shared low-rank latent vector**:

$$
c_t = W^{DKV} h_t \in \mathbb{R}^{c}
\tag{1.2}
$$

where $h_t$ is the hidden state at position $t$, $W^{DKV} \in \mathbb{R}^{c \times d}$ 
is a down-projection to latent dimension $c$, and $c \ll 2 d_h H$.

Then K and V are computed from this latent:

$$
k_t = W^{UK} c_t, \qquad v_t = W^{UV} c_t
\tag{1.3}
$$

The KV cache now stores only $c_t \in \mathbb{R}^c$ instead of both 
$k_t, v_t \in \mathbb{R}^{d_h H}$. For DeepSeek-V2 with $c = 512$ and 
$2 d_h H = 2 \cdot 128 \cdot 128 = 32{,}768$, this is a **64× reduction** in 
KV-cache memory.

## 1.3 Why Reproduce MLA?

MLA is embedded in the DeepSeek-V2 architecture and was not explicitly 
designed for other model families. We ask: **can MLA be ported to 
DistilBERT, RoBERTa, GPT-2, and Qwen3.5 without losing accuracy?**

This is a practical question because:

1. **Model families differ structurally.** DistilBERT uses flat attention 
   with `q_lin`; RoBERTa uses nested `self.query`; GPT-2 uses fused Conv1D 
   `c_attn`; Qwen uses GQA with RoPE.
2. **Head counts differ.** Qwen has $H_{kv} = 2 < H = 8$, requiring GQA 
   expansion.
3. **Position encodings differ.** Qwen uses RoPE; DistilBERT uses absolute 
   embeddings; GPT-2 uses learned positions.

A successful reproduction would enable MLA-style KV compression on any 
pretrained transformer, not just DeepSeek models.

## 1.4 Our Contributions

We make the following contributions:

- **A faithful reproduction of MLA** with explicit low-rank latent 
  projections, SVD-based initialization from concatenated K and V weights, 
  and singular-value normalization.
- **A multi-architecture portability study** covering four families: flat, 
  nested, causal Conv1D, and GQA + RoPE.
- **An empirical evaluation** on SST-2 with three models (DistilBERT, 
  RoBERTa, GPT-2), reporting accuracy, KV-cache reduction, and inference 
  memory.
- **Full reproducibility.** Code, weights, and scripts released under 
  Apache 2.0.

We emphasize: **this paper is a reproduction and portability study, not a 
novel architecture.** We do not claim to improve MLA. We report where it 
works, where it does not, and what engineering effort was required.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on MLA. Section 
4 formalizes the method. Section 5 describes integration with four 
architectures. Section 6 details the setup. Section 7 reports results. 
Section 8 analyzes the portability challenges. Section 9 discusses guidance. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $h_t \in \mathbb{R}^d$ | Hidden state at position $t$ |
| $W^{DKV} \in \mathbb{R}^{c \times d}$ | Down-projection to latent |
| $c \in \mathbb{R}^c$ | Latent vector |
| $W^{UK} \in \mathbb{R}^{H d_h \times c}$ | K up-projection |
| $W^{UV} \in \mathbb{R}^{H d_h \times c}$ | V up-projection |
| $W^Q \in \mathbb{R}^{H d_h \times d}$ | Query projection |
| $W^O \in \mathbb{R}^{d \times H d_h}$ | Output projection |
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads |
| $d_h$ | Per-head dimension |
| $c$ | Latent dimension (MLA rank) |
| $L$ | Sequence length |
| $B$ | Batch size |
| $N_l$ | Number of layers |

---