<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Self-attention computes pairwise interactions between all tokens in a 
sequence, yielding an $N \times N$ attention matrix. This quadratic cost 
in sequence length is the primary bottleneck for long-context transformers. 
A natural question arises: **can attention be reformulated in a lower-dimensional 
space without losing its expressive power?**

One classical approach is to view attention as a discretization of the 
continuous non-local integral operator:

$$
(\mathcal{K} f)(x) = \int_\Omega k(x, y) f(y) \, dy
\tag{1.1}
$$

where $k(x, y)$ is a kernel and $\Omega$ is the domain. Petrov-Galerkin 
methods approximate such operators by projecting onto finite-dimensional 
subspaces:

$$
\mathcal{K} \approx \Phi \, K_{\Phi \Phi} \, \Phi^\top
\tag{1.2}
$$

where $\Phi \in \mathbb{R}^{N \times r}$ is a basis and $K_{\Phi \Phi}$ is 
the projected kernel matrix.

0.85

<!-- 
EXAMPLE continuation:
"This observation motivated us to ask: if we project Q, K, V onto an 
orthonormal basis $\Phi$ with $r \ll N$, we reduce the attention matrix 
from $N \times N$ to $r \times r$, achieving $O(N r)$ memory and 
$O(N r^2)$ time. The natural choice for $\Phi$ on finite sequences is the 
discrete cosine transform (DCT)."
-->

## 1.2 Problem Statement

Given a pretrained transformer with hidden size $d$ and $H$ heads, we seek 
a modified attention function $\mathcal{A}'$ such that:

$$
\mathcal{A}'(Q, K, V) \approx \mathcal{A}(Q, K, V), \quad
\text{Cost}(\mathcal{A}') = \mathcal{O}(Nr^2), \quad r \ll N
\tag{1.3}
$$

We do not require retraining of base weights; LoRA adapters provide 
recovery capacity.

## 1.3 Our Approach

We propose **Galerkin Attention**, which:

1. **Projects** $Q, K, V$ onto an orthonormal DCT basis $\Phi \in \mathbb{R}^{N \times r}$:
   $$
   Q_\Phi = \Phi^\top Q, \quad K_\Phi = \Phi^\top K, \quad V_\Phi = \Phi^\top V
   $$

2. **Computes attention** in the reduced basis space:
   $$
   O_\Phi = \mathrm{softmax}\!\left(\frac{Q_\Phi K_\Phi^\top}{\sqrt{d}}\right) V_\Phi
   $$

3. **Reconstructs** to the original sequence length:
   $$
   O = \Phi \, O_\Phi
   $$

4. **Falls back** to standard causal attention for decoder models.

## 1.4 Contributions

- **A Petrov-Galerkin formulation of attention** using fixed DCT basis 
  projection.
- **An empirical evaluation** on DistilBERT, RoBERTa, and GPT-2 on SST-2.
- **A negative result:** encoder models lose 25–30% accuracy; the method 
  fails as a compression technique.
- **An analysis** attributing failure to the fixed, data-independent basis.
- **Full reproducibility**, including code and checkpoints.

## 1.5 Paper Organization

Section 2 reviews related work on Galerkin methods, spectral attention, 
and low-rank attention. Section 3 provides background on attention, DCT, 
and Petrov-Galerkin projection. Section 4 presents Galerkin Attention. 
Section 5 covers architecture integration. Section 6 details the 
experimental setup. Section 7 reports results. Section 8 analyzes why the 
method fails. Section 9 discusses lessons learned. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $N$ | Sequence length | §1.1 |
| $d$ | Head dimension | §1.1 |
| $H$ | Number of heads | §1.1 |
| $r$ | Basis dimension (rank) | §1.1 |
| $\Phi \in \mathbb{R}^{N \times r}$ | Orthonormal basis | §1.1 |
| $\mathcal{K}$ | Integral operator | §1.1 |
| $k(x, y)$ | Kernel function | §1.1 |
| $Q_\Phi, K_\Phi, V_\Phi$ | Basis-projected Q/K/V | §1.3 |
</parameter>

---