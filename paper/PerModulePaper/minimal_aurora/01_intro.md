# 1. Introduction

## 1.1 Motivation

Low-rank factorization is a standard tool for compressing transformer 
attention (ASVD, AASVD, A3, PiSSA, Res-SVD). For each projection 
$W \in \mathbb{R}^{m \times n}$, one replaces $W$ with a product $UV^\top$ 
of rank $r \ll \min(m, n)$, reducing parameters from $mn$ to $r(m + n)$.

**The key hyperparameter is the rank $r$.** Choosing it is a trade-off:

- **Too small:** aggressive compression, large approximation error.
- **Too large:** high fidelity but little compression.

Previous methods use:

- **Fixed rank** (ASVD, Res-SVD): same $r$ for all layers.
- **Adaptive retry** (A3): start at $r$, increase until error threshold.
- **Budget allocation** (SDRM): rank proportional to layer parameter count.
- **Full covariance** (AASVD): rank from global parameter budget.

**MinimalAURORA uses energy thresholding:** for each layer, choose the 
smallest rank $r$ such that the top-$r$ singular values capture at least 
$\varepsilon$ of the total spectral energy:

$$
E(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2} \geq \varepsilon
\tag{1.1}
$$

This adapts automatically to each layer's spectral decay. Layers with 
fast decay (concentrated spectrum) get small rank; layers with slow decay 
get larger rank.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a compressed model $\mathcal{M}'$ such that:

1. Each projection $W^{(\ell)}$ is replaced by $U^{(\ell)} V^{(\ell)\top}$ 
   with rank $r^{(\ell)}$ selected by energy threshold.
2. Parameter count reduces by factor $R > 3$.
3. Accuracy drop is bounded: $\Delta\text{Acc} \leq 0.02\%$.
4. Weight transfer works across four attention families.
5. Trainable LoRA adapters provide recovery capacity.

## 1.3 Our Approach: MinimalAURORA

MinimalAURORA replaces each projection with:

$$
W \approx \underbrace{U V^\top}_{\text{frozen SVD}} + \underbrace{\frac{\alpha}{r_L} B A}_{\text{trainable LoRA}}
\tag{1.2}
$$

The SVD is computed via **standard (unweighted) decomposition**, with rank 
selected by energy thresholding. LoRA is initialized to zero.

**Why "Minimal"?** The implementation avoids the complexity of 
activation-aware weighting (A3/AASVD) or two-level SVD (Res-SVD). It uses 
the minimal machinery needed for compression with PEFT recovery.

## 1.4 Contributions

- **Energy-threshold rank selection.** Each layer gets its own rank, 
  automatically adapted to its spectral decay.
- **A production-hardened `FactorisedLinear` module.** SVD base + LoRA 
  residual, with optional energy threshold.
- **Multi-architecture support.** Same code handles flat, nested, causal, 
  and GQA+RoPE attention.
- **Nested encoder auto-detection.** RoBERTa/BERT-specific handling 
  detected automatically.
- **Honest reporting.** MinimalAURORA **reduces** parameter count. We also 
  note a `pres_gate` parameter that is currently unused.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, 
energy thresholding, and LoRA. Section 4 presents MinimalAURORA. Section 5 
details multi-architecture integration. Section 6 gives setup. Section 7 
reports results. Section 8 analyzes contributions and the `pres_gate` 
issue. Section 9 provides guidance. Section 10 lists limitations. Section 
11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W$ | Projection weight matrix |
| $U, V$ | SVD factors |
| $r$ | Energy-selected rank |
| $\varepsilon$ | Energy threshold |
| $\sigma_i$ | $i$-th singular value |
| $B, A$ | LoRA factors |
| $r_L$ | LoRA rank |
| $\alpha$ | LoRA scaling |
| $d_\text{in}, d_\text{out}$ | Input/output dimensions |

---