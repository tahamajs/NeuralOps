<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer language models concentrate the majority of their parameters in 
the feed-forward network and the multi-head self-attention (MHSA) mechanism. 
In DistilBERT-base, the four attention projections $W_Q, W_K, W_V, W_O$ 
account for roughly **70%** of the parameters in each transformer block. 
Reducing the parameter cost of attention is a high-leverage compression 
strategy.

Standard MHSA stores four dense matrices per layer:

$$
W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}
\tag{1.1}
$$

totaling $4 d^2$ parameters. For DistilBERT ($d = 768$), this is 
$4 \times 589{,}824 = 2{,}359{,}296$ parameters per layer, or 14.2M across 
six layers.

**Low-rank factorization** is the classical primitive for reducing this cost. 
For each $W$, one replaces it with a product:

$$
W \approx U V, \qquad
U \in \mathbb{R}^{d_\text{out} \times r}, \quad
V \in \mathbb{R}^{r \times d_\text{in}}, \quad r \ll \min(d_\text{in}, d_\text{out})
\tag{1.2}
$$

The parameter count drops from $d_\text{in} d_\text{out}$ to 
$r(d_\text{in} + d_\text{out})$. For square matrices, the reduction factor is 
$d / 2r$. Classical SVD (Eckart–Young, 1936) gives the optimal rank-$r$ 
approximation in Frobenius norm.

**The rank-selection problem.** Classical SVD requires choosing $r$ 
a-priori, which is a hard trade-off:

- **Too small** — approximation error is high, downstream accuracy drops.
- **Too large** — parameter reduction is minimal, defeating the purpose.

In practice, different layers have different **intrinsic ranks**. Attention 
projections in early layers of DistilBERT tend to have rapidly-decaying 
spectra (effective rank ~30–48), while middle layers decay more slowly 
(effective rank ~80–100). A single global $r$ is suboptimal for both.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_l$ layers and a target 
energy threshold $\eta \in (0, 1)$, we seek a compressed model 
$\mathcal{M}'$ such that:

$$
\mathcal{M}' \approx \mathcal{M}, \qquad
\frac{|\theta(\mathcal{M}')|}{|\theta(\mathcal{M})|} \ll 1
\tag{1.3}
$$

subject to:

1. **Per-layer adaptive rank.** Different layers may use different 
   $r_\ell$.
2. **Spectral energy preservation.** $E(r_\ell) \geq \eta$ for each layer.
3. **Parameter reduction.** $r_\ell(d_\text{in} + d_\text{out}) < d_\text{in} d_\text{out}$.
4. **Multi-architecture support.** Flat, nested, causal, and GQA transformers.
5. **No base retraining.** Only LoRA adapters may be trained.

## 1.3 Our Approach: MinimalSVD

**MinimalSVD** is a compact, auditable baseline for low-rank attention 
compression. It has two components:

1. **SVD factorization with energy-adaptive rank.** Each $W$ is factorized as 
   $U V$ via truncated SVD. The rank is selected per layer as:

   $$
   r_\ell = \min\!\left(r_\text{max}, \left|\{i : E(i) < \eta\}\right| + 1\right)
   \tag{1.4}
   $$

   where $E(r) = \sum_{i \leq r} \sigma_i^2 / \sum_i \sigma_i^2$. The 
   factorization distributes $\Sigma^{1/2}$ symmetrically.

2. **LoRA recovery.** After factorization, base weights are frozen and LoRA 
   adapters $A, B$ are added with $B = 0$ at initialization.

**Design philosophy.** MinimalSVD intentionally keeps the injection protocol 
**minimal** — the injection hook targets a single transformer layer at a 
time, delegating whole-model traversal to the caller. This makes the code 
easy to audit, extend, and embed in different frameworks. Unlike the more 
complex variants (NOVA, PRISM), MinimalSVD does not perform pre-testing on 
the first layer; correctness is ensured by the strict weight-extraction 
checks in `copy_weights`.

## 1.4 Contributions

- **A minimal reference implementation.** A compact, auditable baseline for 
  energy-adaptive low-rank attention compression.
- **Spectral energy-adaptive rank selection.** A principled criterion that 
  selects a different rank per layer based on spectral decay.
- **Multi-architecture support.** Four families (DistilBERT, RoBERTa, GPT-2, 
  Qwen3.5) with architecture-aware weight extraction.
- **A systematic evaluation.** Benchmark on three models on SST-2 with 
  ablations over $r_\text{max}$, $\eta$, and LoRA rank.
- **Full reproducibility.** Code, checkpoints, and scripts released under 
  Apache 2.0.

## 1.5 Relationship to Companion Variants

MinimalSVD is the **baseline** of a family of low-rank attention modules 
that share the same `FactorisedLinear` core:

| Variant | Additional features |
|---|---|
| **MinimalSVD** | Baseline (this work) |
| MinimalNOVA | Whole-model injection + pre-test |
| MinimalPRISM | Injection-time nested detection |

All three share the same energy-adaptive rank selection, LoRA recovery, and 
weight transfer logic. MinimalSVD exists to isolate the core contribution 
from engineering additions.

## 1.6 Paper Organization

Section 2 reviews related work. Section 3 provides background. Section 4 
presents MinimalSVD. Section 5 describes architecture integration. Section 
6 details the setup. Section 7 reports results. Section 8 analyzes the 
energy-adaptive rank behavior. Section 9 discusses guidance. Section 10 
lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ | Weight matrix |
| $U, V$ | Low-rank factors |
| $r$ | Target rank |
| $r_\ell$ | Adaptive rank for layer $\ell$ |
| $r_\text{max}$ | Rank cap |
| $\sigma_i$ | $i$-th singular value |
| $E(r)$ | Cumulative spectral energy |
| $\eta$ | Energy threshold |
| $d$ | Hidden size |
| $H$ | Query head count |
| $H_{kv}$ | KV head count (GQA) |
| $d_h$ | Per-head dimension |
| $\alpha, r_L$ | LoRA scaling, rank |

---