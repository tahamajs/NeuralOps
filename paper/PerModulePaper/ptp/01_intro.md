<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer inference at long sequence length is dominated by the 
$\mathcal{O}(L^2)$ cost of softmax attention, but the actual **useful** 
content in a sequence is often much smaller than $L$. In classification 
tasks like SST-2, only a handful of tokens — the sentiment-bearing words — 
determine the output. In question answering, only tokens near the answer 
span are critical. The rest are redundant.

Prior work exploits this in three ways:

1. **Fixed heuristics.** Prune tokens by attention magnitude (Kim & Cho, 
   2021), or by position (longformer-style). Simple but not task-aligned.

2. **Learned selectors.** Train a selector network on top of a frozen 
   transformer (Wang et al., 2021). Requires task-specific training.

3. **Full retraining.** Train the transformer with pruning in the loop 
   (Rao et al., 2021). Effective but expensive.

**The gap.** None of these is a **drop-in** solution that (a) learns which 
tokens to keep, (b) preserves pretrained weights exactly, and (c) works 
across flat, nested, and causal architectures. PTP fills this gap.

## 1.2 Problem Statement

Given a pretrained transformer with hidden dimension $d$ and $H$ heads, we 
seek a modified attention module $\mathcal{A}'$ that:

1. **Prunes tokens:** Retains only a fraction $\rho = k/L$ of tokens.
2. **Preserves [CLS]:** The classification token is always retained.
3. **Handles padding:** Invalid tokens are excluded.
4. **Preserves causality:** For decoders, no future information leaks.
5. **Drop-in:** Pretrained projections are copied exactly.
6. **Quality:** $\Delta\text{Acc} \leq 0.02\%$ on SST-2.

## 1.3 Our Approach: PTP

**Progressive Token Pruning** (PTP) introduces a **learnable scoring 
network**:

$$
s_i = \sigma\left(\text{MLP}(x_i)\right) \in (0, 1)
\tag{1.1}
$$

where $x_i$ is the hidden state of token $i$, and the MLP is:

$$
\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
\tag{1.2}
$$

with $W_1 \in \mathbb{R}^{64 \times d}$, $W_2 \in \mathbb{R}^{1 \times 64}$.

**Top-$k$ selection.** Let $k = \lceil \rho L \rceil$. We keep the $k$ 
tokens with the highest scores:

$$
\mathcal{T} = \text{top-}k_i \, s_i
\tag{1.3}
$$

**Key design choices:**

1. **[CLS] preservation.** We set $s_0 = 10^4$ before top-$k$, guaranteeing 
   the first token is always retained.
2. **Padding masking.** Invalid tokens get $s_i = -10^4$ before top-$k$.
3. **Gather-based KV pruning.** Rather than masking, we gather the top-$k$ 
   KV tensors, reducing memory.
4. **Causal-aware attention.** For decoders, we mask queries that would 
   attend to future keys by comparing original positions.

## 1.4 Contributions

- **A learnable token-selection module.** PTP scores tokens with a small 
  MLP and prunes them via top-$k$.
- **[CLS] preservation.** Guaranteed by score clamping, with no additional 
  parameters or logic.
- **Causal-aware pruning.** We preserve causality by comparing original 
  positions after pruning.
- **A drop-in wrapper.** Pretrained projections are copied exactly; only 
  the score network and LoRA adapters are trainable.
- **A systematic evaluation.** Benchmarked on DistilBERT, RoBERTa, and 
  GPT-2 on SST-2 with ablations over keep ratio and score network depth.
- **Token importance analysis.** We show that learned scores correlate with 
  content-word importance and reveal systematic patterns.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention 
and token pruning. Section 4 presents PTP. Section 5 describes integration 
with four architectures. Section 6 details the experimental setup. Section 
7 reports results. Section 8 analyzes the learned token scores. Section 9 
discusses practical guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $H$ | Number of heads |
| $d$ | Hidden dimension |
| $\rho$ | Keep ratio |
| $k$ | Number of retained tokens ($= \lceil \rho L \rceil$) |
| $s_i$ | Score of token $i$ |
| $\mathcal{T}$ | Set of retained token indices |
| $x_i$ | Hidden state of token $i$ |
| $\text{MLP}$ | Score network |

---