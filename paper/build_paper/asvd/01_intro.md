<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

The dominant storage cost in transformer models lies in the weight matrices 
of linear layers. For DistilBERT-base, the four attention projections 
($W_Q, W_K, W_V, W_O$) account for approximately 4.7 million parameters 
out of 66.9 million total (7%). For RoBERTa-base, they account for 
9.4 million out of 124.6 million (7.5%). Reducing these is the primary 
lever for model compression.

**Low-rank factorization** replaces each $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
with the product $U \Sigma V^\top$ where $U \in \mathbb{R}^{d_\text{out} \times r}$, 
$\Sigma \in \mathbb{R}^r$, and $V \in \mathbb{R}^{r \times d_\text{in}}$ for 
$r \ll \min(d_\text{in}, d_\text{out})$. Parameter count drops from 
$d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out}) + r$.

**Classical SVD is unweighted.** The optimal rank-$r$ approximation in 
Frobenius norm is the truncated SVD, but the objective 
$\|W - W_r\|_F$ treats every input dimension as equally important. Transformer 
activations, however, exhibit two critical properties that unweighted SVD 
ignores:

1. **Heterogeneous variance across dimensions** — some hidden features 
   carry most of the signal.
2. **Inter-feature correlations** — dimensions are not statistically 
   independent.

**Activation-aware SVD** (ASVD) reweights the reconstruction objective by 
an empirical estimate of the input covariance. The mathematical core is 
simple: replace $\|W - W_r\|_F^2$ with 
$\mathbb{E}_x[\|Wx - W_r x\|_2^2]$, which under a diagonal covariance 
assumption becomes a weighted Frobenius norm.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a compressed model $\mathcal{M}'$ such that:

1. **Parameter reduction:** each layer's projection parameter count is 
   reduced by factor $R > 1$.
2. **Accuracy preservation:** $\Delta\text{Acc} \leq 3\%$ on 
   the target task.
3. **Strict validation:** every compressed layer satisfies the Frobenius 
   reconstruction bound $\|W - U \Sigma V^\top\|_F / \|W\|_F \leq \tau$.
4. **Multi-architecture:** works on flat, nested, causal, and GQA models.
5. **Deterministic fallback:** on validation failure, the original layer is 
   preserved.

## 1.3 Our Approach: ASVD

ASVD differs from prior low-rank methods in three ways:

1. **Explicit singular value matrix.** Unlike A3 and AASVD (which fold 
   $\Sigma^{1/2}$ into the factors), ASVD keeps $\Sigma$ as a separate 
   parameter. This enables:
   - Per-rank inspection of the spectral profile.
   - Rank truncation without re-factorization.
   - Direct manipulation of individual singular values.

2. **Energy-based dynamic rank.** Rather than using a fixed rank, ASVD 
   selects the minimal $r$ such that cumulative energy 
   $\sum_{i=1}^r \sigma_i^2 / \sum_i \sigma_i^2 \geq \varepsilon$ where 
   $\varepsilon = 0.95$. This adapts to each layer's 
   intrinsic rank.

3. **Pre-flight extraction validation.** Before replacing any attention 
   module, ASVD attempts weight extraction from the source. If extraction 
   fails, injection aborts cleanly — no silent fallback to random 
   initialization.

## 1.4 Contributions

- **A production-hardened ASVD implementation.** We provide a complete 
  PyTorch implementation with explicit $\Sigma$, energy-based rank selection, 
  strict Frobenius validation, and pre-flight extraction checks.
- **Multi-architecture weight transfer.** A single `copy_weights` routine 
  handles DistilBERT, RoBERTa, GPT-2, and Qwen/LLaMA without per-model 
  branching.
- **A factorization module with identity mode.** `ASVDLinear.set_identity()` 
  enables clean handling of nested encoders (RoBERTa/BERT), where the output 
  projection must be bypassed.
- **A systematic evaluation.** Benchmarks on three models with ablations 
  over rank, energy threshold, error threshold, and dynamic vs fixed rank.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts 
  released under MIT.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD and 
activation-aware compression. Section 4 presents ASVD. Section 5 details 
multi-architecture integration. Section 6 gives the experimental setup. 
Section 7 reports results. Section 8 analyzes contributions. Section 9 
provides guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W \in \mathbb{R}^{m \times n}$ | Weight matrix |
| $U \in \mathbb{R}^{m \times r}$ | Left singular vectors |
| $\Sigma \in \mathbb{R}^r$ | Singular values |
| $V \in \mathbb{R}^{r \times n}$ | Right singular vectors |
| $r$ | Rank |
| $\varepsilon$ | Energy threshold |
| $\tau$ | Frobenius error threshold |
| $\sigma_i$ | $i$-th singular value |
| $C_X$ | Input activation covariance |
| $r_L$ | LoRA rank |
| $d_h$ | Per-head dimension |

---