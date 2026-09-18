<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer attention at inference has two distinct regimes that pull in 
opposite directions:

- **Local regime.** Attention concentrates on nearby tokens, especially in 
  early layers. Sliding-window attention (Beltagy et al., 2020) exploits this 
  with $\mathcal{O}(L w)$ complexity, where $w$ is the window size.

- **Global regime.** Some attention heads (e.g., attention sinks, coreference 
  heads) attend to distant tokens. Linear attention (Katharopoulos et al., 
  2020; Choromanski et al., 2021) approximates global attention in 
  $\mathcal{O}(L)$ via a kernel factorization.

**The problem with picking one.** Longformer's window truncates global 
context. Performer's random features degrade local precision. Neither 
extremes is universally better.

**The opportunity.** Rather than commit to one regime, **hybrid attention** 
runs both and combines them. If the combination weight is **learned per 
token**, the model can allocate local precision where it matters and global 
approximation elsewhere.

## 1.2 Problem Statement

Given a pretrained transformer with dense attention, we seek a modified 
module $\mathcal{A}'$ such that:

1. **Parallel streams:** Both local and global attention are computed 
   simultaneously.
2. **Learned fusion:** A per-token gate mixes the two streams.
3. **Stability:** The gate is initialized to favor the pretrained behavior.
4. **Quality:** $\Delta\text{Acc} \leq 0.02\%$ on SST-2.
5. **Pretrained compatibility:** No base retraining; only LoRA + gate 
   parameters are trainable.

## 1.3 Our Approach: NHA

**Native Hybrid Attention** (NHA) runs two attention streams in parallel:

1. **Local stream** $\mathcal{A}_\text{local}$: sliding-window attention with 
   window size $w$ and chunked computation.
2. **Global stream** $\mathcal{A}_\text{global}$: RFF-based linear attention 
   with orthogonal projection matrix.

The streams are combined via a **token-level learned gate**:

$$
O_i = g_i \cdot O_i^\text{local} + (1 - g_i) \cdot O_i^\text{global}
\tag{1.1}
$$

where $g_i = \sigma(w^\top x_i + b)$ with $b$ initialized to $3.0$ (biasing 
toward local, $\sigma(3) \approx 0.95$).

**Key design choices:**

1. **Orthogonal RFF projection.** We build the RFF projection via block QR, 
   reducing variance compared to i.i.d. Gaussian projections.
2. **Chunked causal prefix scan.** For decoders, the RFF stream maintains 
   cumulative state across chunks, preserving strict causality.
3. **Token-level gating.** The gate is per token, not per head, capturing 
   position-specific needs without excess parameters.

## 1.4 Contributions

- **A native hybrid attention module.** NHA runs local and global streams in 
  parallel within each layer, fused by a learned token-level gate.
- **Orthogonal RFF.** We use block-QR orthogonal projections for reduced 
  variance.
- **Chunked causal scan.** A memory-bounded RFF implementation for decoders.
- **A production-hardened implementation.** Supports four architectures 
  with RoPE, GQA, LoRA recovery, and safe injection.
- **A systematic evaluation.** Benchmarked on DistilBERT, RoBERTa, and GPT-2 
  on SST-2 with ablations over window size, RFF feature count, and gate bias.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts released 
  under Apache 2.0.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on local and 
linear attention. Section 4 presents NHA. Section 5 describes integration 
with four architectures. Section 6 details the experimental setup. Section 
7 reports results. Section 8 analyzes gate behavior. Section 9 discusses 
practical guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $H$ | Number of attention heads |
| $H_{kv}$ | Number of KV heads |
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $w$ | Window size |
| $m$ | Number of RFF features |
| $g_i$ | Gate value at position $i$ |
| $\Omega$ | RFF projection matrix |
| $S, Z$ | Cumulative RFF state |
| $\phi(\cdot)$ | Feature map |
| $\sigma$ | Sigmoid |

---