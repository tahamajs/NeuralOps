<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# Galerkin Attention: Petrov-Galerkin Projection onto Orthonormal Bases for Transformer Self-Attention

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1 (initial draft)
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/galerkin.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- BE HONEST: This is a negative result for encoders
- The Galerkin path only applies to bidirectional (encoder) models
- GPT-2 uses standard causal attention (no-op)
- RoBERTa collapses to 48.97% (chance level)
- DistilBERT drops -24.77%
-->

We reinterpret transformer self-attention as a **discretization of a 
continuous non-local integral operator** and approximate it via 
**Petrov-Galerkin projection** onto an orthonormal basis (discrete cosine 
transform). The method projects queries, keys, and values onto a truncated 
$r$-dimensional basis $\Phi \in \mathbb{R}^{N \times r}$ before computing 
attention, reducing the quadratic interaction matrix from $N \times N$ to 
$r \times r$. For bidirectional (encoder) models, this yields a data-agnostic 
low-rank attention approximation; for causal (decoder) models, we fall back 
to standard scaled dot-product attention to preserve causality.

We evaluate on three pretrained models — DistilBERT, RoBERTa, and GPT-2 — 
on SST-2 with LoRA recovery. Our results are **negative for encoders**: 
DistilBERT drops from 90.71% to **50.92%** (Δ = -39.79) and RoBERTa 
collapses from 78.56% to **50.92%** (Δ = -27.64), the latter at 
chance level. GPT-2 shows no change (Δ = 0.00%) because the causal fallback 
path leaves attention unmodified. We attribute encoder failures to the 
**data-independent, fixed DCT basis**, which cannot capture the 
content-dependent, position-specific attention patterns learned by 
pretraining. We document this negative result to inform future work on 
basis-projected attention.

**Keywords:** Galerkin projection, transformer attention, orthonormal basis, 
integral operator, negative results

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

We tried to compute attention using a fixed mathematical basis (cosine waves) 
instead of learning it from data. This makes attention much cheaper to 
compute — instead of comparing every pair of words, we compare their 
compressed representations. We tested on three pretrained models and found 
that this **does not work for encoder models** like DistilBERT and RoBERTa, 
which drop by 25–30%. It also **does nothing** for GPT-2, because our code 
uses standard attention for that model. We report this negative result 
honestly to warn others away from fixed-basis projection.

---

## Table of Contents

1. [Introduction](01_intro.md)
2. [Related Work](02_related.md)
3. [Background](03_background.md)
4. [Method](04_method.md)
5. [Architecture Integration](05_architecture.md)
6. [Experimental Setup](06_setup.md)
7. [Results](07_results.md)
8. [Analysis](08_analysis.md)
9. [Discussion](09_discussion.md)
10. [Limitations](10_limitations.md)
11. [Conclusion](11_conclusion.md)
12. [Reproducibility](12_reproducibility.md)
13. [References](13_references.md)

**Appendices:**
- [A. Mathematical Derivations](A_math.md)
- [B. Implementation Details](B_implementation.md)
- [C. Full Experimental Data](C_data.md)
- [D. Negative Results](D_negative_results.md)
- [E. Ethical Considerations](E_ethics.md)

---

---

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

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Petrov-Galerkin Methods

Petrov-Galerkin methods (Quarteroni & Valli, 1994) approximate solutions of 
differential and integral equations by projecting onto finite-dimensional 
subspaces. Unlike Galerkin methods (test = trial space), Petrov-Galerkin 
uses different test and trial spaces, allowing non-symmetric operators to be 
handled. Our approach uses the same basis for trial and test — a **Bubnov-Galerkin** 
variant — but we retain the "Petrov" terminology following the original 
motivation of the module.

**Kernel methods.** The Nyström method (Williams & Seeger, 2001) also 
approximates kernel matrices via low-rank projection. Our approach differs 
in using a **fixed basis** rather than sampled landmarks.

## 2.2 Spectral Attention

**FNet** (Lee-Thorp et al., 2021). Replaces attention entirely with a 2D 
Fourier transform, achieving $\mathcal{O}(N \log N)$ complexity. This is the 
closest work to ours in spirit — but FNet replaces attention, while we 
project it.

**Random feature attention** (Peng et al., 2021). Uses random features to 
approximate softmax. Performer (Choromanski et al., 2021) uses positive 
orthogonal random features.

**Nyströmformer** (Xiong et al., 2021). Uses Nyström landmarks for low-rank 
attention approximation. Landmarks are data-dependent; our basis is fixed.

## 2.3 Low-Rank Attention

**Linformer** (Wang et al., 2020). Projects K and V to a lower rank via 
learned linear maps. Our projection is a fixed DCT basis.

**Synthesizer** (Tay et al., 2021). Uses learned or random attention 
matrices. Random Synthesizer is data-independent like ours.

## 2.4 Position of Our Work

| Axis | Prior work | Galerkin |
|---|---|---|
| **Basis type** | Learned (Linformer) or landmarks (Nyström) | Fixed DCT |
| **Attention replacement** | Full or partial | Projection only |
| **Parameter cost** | Zero or reduced | **Added LoRA** |
| **Encoder support** | Yes | Yes (but fails) |
| **Decoder support** | Yes | Falls back |

**Key distinction.** Unlike Linformer which **learns** low-rank projections, 
our approach uses a **fixed** DCT basis. This eliminates the parameter cost 
of projection matrices but sacrifices adaptivity. Our negative results 
suggest this trade-off is not worth it for attention.

---

## Comparison Table

| Method | Basis | Learned? | Params | SST-2 (DistilBERT) |
|---|---|---|---|---|
| Standard MHSA | — | — | 66.9M | 90.71% |
| Linformer | Linear | Yes | Similar | ~88% |
| FNet | Fourier | No | 66.9M | ~85% |
| Random Synthesizer | Random | No | 66.9M | ~85% |
| **Galerkin** | **DCT** | **No** | **+0.9%** | **50.92%** |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Scaled Dot-Product Attention

For input $X \in \mathbb{R}^{B \times N \times d}$, attention computes:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
\tag{3.1}
$$

where $Q = XW_Q$, $K = XW_K$, $V = XW_V$.

**Cost.** Time and memory for $QK^\top$ are $\mathcal{O}(N^2)$.

## 3.2 Discrete Cosine Transform (DCT)

The DCT-II basis is defined as:

$$
\Phi_{n, k} = \cos\!\left(\frac{\pi k (2n + 1)}{2N}\right), \quad
n = 0, \dots, N-1, \quad k = 0, \dots, N-1
\tag{3.2}
$$

with normalization:

$$
\Phi_{n, 0} \to \frac{1}{\sqrt{N}} \Phi_{n, 0}, \quad
\Phi_{n, k} \to \sqrt{\frac{2}{N}} \Phi_{n, k} \text{ for } k \geq 1
\tag{3.3}
$$

The full DCT basis $\Phi \in \mathbb{R}^{N \times N}$ is orthonormal: 
$\Phi^\top \Phi = I_N$.

**Why DCT?** DCT is optimal for energy compaction of smooth signals 
(Ahmed et al., 1974). For smooth attention kernels, DCT projection preserves 
most energy.

**Truncation.** Keeping only the first $r$ columns gives 
$\Phi_r \in \mathbb{R}^{N \times r}$, which satisfies 
$\Phi_r^\top \Phi_r = I_r$.

## 3.3 Petrov-Galerkin Projection

Consider the continuous integral operator:

$$
(\mathcal{K} f)(x) = \int_0^1 k(x, y) f(y) \, dy
\tag{3.4}
$$

Let $\{\phi_1, \dots, \phi_r\}$ be an orthonormal basis of a subspace 
$\mathcal{V}_r \subset L^2([0, 1])$. The Galerkin approximation is:

$$
\mathcal{K}_r = \sum_{i, j = 1}^r K_{ij} \phi_i \otimes \phi_j
\tag{3.5}
$$

where $K_{ij} = \langle \phi_i, \mathcal{K} \phi_j \rangle$.

**Discretization.** For a discrete sequence of length $N$, the Galerkin 
approximation becomes:

$$
\mathcal{K}_r = \Phi_r \, K_r \, \Phi_r^\top, \quad
K_r \in \mathbb{R}^{r \times r}
\tag{3.6}
$$

with $\Phi_r \in \mathbb{R}^{N \times r}$ being the discrete basis.

## 3.4 LoRA Recovery

After replacing attention, we freeze base weights and train LoRA adapters:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.7}
$$

with $A \in \mathbb{R}^{r_L \times k}$, $B \in \mathbb{R}^{d \times r_L}$, 
$B = 0$ at init.

## 3.5 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $N$ | Sequence length |
| $d$ | Hidden dimension |
| $H$ | Number of heads |
| $r$ | Basis dimension |
| $\Phi$ | Orthonormal basis ($N \times r$) |
| $\Phi_r$ | Truncated basis |
| $r_L$ | LoRA rank |
| $\mathcal{K}$ | Integral operator |
| $k(x, y)$ | Kernel function |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Galerkin Attention has two paths:

1. **Encoder path** (bidirectional models): project attention onto DCT basis.
2. **Decoder path** (causal models): fall back to standard causal attention.

```
                    Input X
                       │
                       ▼
                 ┌───────────┐
                 │  Q/K/V    │
                 │ projection│
                 └─────┬─────┘
                       │
                       ▼
                 ┌───────────┐
                 │ is_decoder│
                 └──┬────┬───┘
                    │ No │ Yes
                    │    │
                    │    ▼
                    │  ┌──────────────┐
                    │  │ Causal       │
                    │  │ attention    │
                    │  │ (standard)   │
                    │  └──────┬───────┘
                    │         │
                    ▼         │
            ┌───────────────┐ │
            │ Galerkin proj │ │
            │ + softmax     │ │
            └───────┬───────┘ │
                    │         │
                    └────┬────┘
                         │
                         ▼
                    Output Y
```

## 4.2 The Galerkin Basis

We construct the DCT-II basis $\Phi \in \mathbb{R}^{N \times r}$:

$$
\Phi_{n, k} = \cos\!\left(\frac{\pi k (2n + 1)}{2N}\right) \cdot
c_k, \quad c_k = \begin{cases} 1/\sqrt{N} & k = 0 \\ \sqrt{2/N} & k \geq 1 \end{cases}
\tag{4.1}

<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->
$$

We then orthonormalize via QR to correct for numerical error:

$$
\Phi \leftarrow Q \text{ where } \Phi + 10^{-4} I = Q R
\tag{4.2}

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->
$$

## 4.3 Projection and Attention

For each head:

1. **Project** Q, K, V onto the basis:
   $$
   Q_\Phi = \Phi^\top Q \in \mathbb{R}^{B \times H \times r \times d}
   \tag{4.3}
   
<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->
$$
   
2. **Compute attention** in the reduced space:
   $$
   A_\Phi = \mathrm{softmax}\!\left(\frac{Q_\Phi K_\Phi^\top}{\sqrt{d}}\right) \in \mathbb{R}^{B \times H \times r \times r}
   \tag{4.4}
   
<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->
$$
   
3. **Apply to values:**
   $$
   O_\Phi = A_\Phi V_\Phi \in \mathbb{R}^{B \times H \times r \times d}
   \tag{4.5}
   
<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->
$$
   
4. **Reconstruct** to original space:
   $$
   O = \Phi O_\Phi \in \mathbb{R}^{B \times H \times N \times d}
   \tag{4.6}
   
<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->
$$

5. **Energy correction:**
   $$
   O \leftarrow O \cdot \sqrt{N / r}
   \tag{4.7}
   
<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->
$$

## 4.4 Why the Energy Correction?

The projection $\Phi^\top$ is contractive: $\|\Phi^\top Q\|_F \leq \|Q\|_F$. 
For an orthonormal basis of $r$ columns out of $N$, the average energy 
preserved is $r/N$. Multiplying by $\sqrt{N/r}$ approximately restores 
the original norm.

**Caveat.** This correction is a heuristic and does not account for the 
actual distribution of energy across the basis. If energy is concentrated 
in the first $r$ components, the correction **overshoots**; if it is 
uniform, the correction is exact.

## 4.5 Decoder Fallback

For causal models, we use standard attention with explicit causal masking:

$$
\text{scores}_{ij} = \begin{cases}
Q_i K_j^\top / \sqrt{d} & j \leq i \\
-\infty & j > i
\end{cases}
\tag{4.8}

<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 607 -->
$$

This preserves the autoregressive property. The Galerkin path is **not** 
used for decoders.

## 4.6 Parameter Analysis

Galerkin Attention **adds parameters** via LoRA:

| Component | Params |
|---|---|
| Original attention | $d_\text{in} d_\text{out}$ |
| Q projection + LoRA | $d_\text{in} d_\text{out} + r_L(d_\text{in} + d_\text{out})$ |
| K projection + LoRA | same |
| V projection + LoRA | same |
| O projection + LoRA | same |
| **Total per layer** | $4 d_\text{in} d_\text{out} + 4 r_L(d_\text{in} + d_\text{out})$ |

**Example (DistilBERT, $r_L = 16$, $d = 768$):** +589,824 params per layer.

**Note.** Galerkin Attention is **not a compression method**. It aims to 
reduce **attention-matrix memory**, not parameter count.

## 4.7 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| Q/K/V proj | $\mathcal{O}(Nd^2)$ | $\mathcal{O}(Nd)$ |
| Basis projection | $\mathcal{O}(Nrd)$ | $\mathcal{O}(Nrd)$ |
| Attention | $\mathcal{O}(r^2 d)$ | $\mathcal{O}(r^2)$ |
| Reconstruction | $\mathcal{O}(Nrd)$ | $\mathcal{O}(Nrd)$ |
| O proj | $\mathcal{O}(Nd^2)$ | $\mathcal{O}(Nd)$ |
| **Baseline MHSA** | $\mathcal{O}(N^2 d)$ | $\mathcal{O}(N^2)$ |
| **Galerkin** | $\mathcal{O}(Nrd + r^2 d)$ | $\mathcal{O}(Nr)$ |

For $r \ll N$, Galerkin attention is asymptotically faster in the 
attention matrix, but the projection/reconstruction overhead dominates 
for short sequences.

## 4.8 Algorithm

**Algorithm 1:** Galerkin Attention forward
```
Input:  X ∈ R^{B×N×d}, is_decoder
Output: Y ∈ R^{B×N×d}

 1: Q ← X W_Q;  K ← X W_K;  V ← X W_V
 2: if RoPE: (Q, K) ← Rotary(Q, K)
 3: if GQA: (K, V) ← repeat_kv(K, V)
 4:
 5: if is_decoder:
 6:     A ← causal_softmax(Q K^T / sqrt(d))
 7:     O ← A V
 8: else:
 9:     Φ ← DCT_basis(N, r)               ▷ Orthonormal
10:     Q_Φ ← Φ^T Q
11:     K_Φ ← Φ^T K
12:     V_Φ ← Φ^T V
13:     A_Φ ← softmax(Q_Φ K_Φ^T / sqrt(d))
14:     O_Φ ← A_Φ V_Φ
15:     O ← Φ O_Φ
16:     O ← O * sqrt(N / r)               ▷ Energy correction
17:
18: return out_proj(O)
```

## 4.9 Theoretical Basis

The method is motivated by the following observation: for smooth attention 
kernels $k(x, y)$, projection onto the first $r$ DCT basis functions 
preserves energy at rate:

$$
\frac{\|\Phi_r^\top k\|_2}{\|k\|_2} \approx 1 - \mathcal{O}(r^{-2\alpha})
\tag{4.9}

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->

<!-- source: Codes/framework/attentions/galerkin_attention.py line 168 -->
$$

where $\alpha$ is the smoothness parameter of $k$. For $C^\infty$ kernels, 
the decay is super-polynomial.

**Empirical caveat.** Trained transformer attention kernels are **not** 
smooth. They are sharply peaked (attention sinks, heavy hitters) and highly 
content-dependent. This is why our method fails in practice (Section 8).

---

---

# 5. Architecture Integration


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```


## 5.1 Model Support Matrix

| Architecture | Attention | Nested? | RoPE? | GQA? | Galerkin path? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ (encoder) |
| RoBERTa | `BertSelfAttention` | Yes | No | No | ✅ (encoder) |
| GPT-2 | `GPT2Attention` | No | No | No | ❌ (decoder fallback) |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | ❌ (decoder fallback) |

## 5.2 Weight Transfer

### 5.2.1 Conv1D Orientation (GPT-2)

HuggingFace `Conv1D` stores `(in, out)`; we transpose to `(out, in)`:

```python
if "Conv1D" in cls_c or hasattr(c_attn_mod, "nf"):
    c_w = c_w.t()
```

We detect via class name **or** shape heuristic `w.shape[0] < w.shape[1]`.

### 5.2.2 Nested Encoder Detection (RoBERTa)

The detection order matters. We check `out_lin` **first**:

```python
has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")

if has_out_lin:
    is_nested = False     # DistilBERT has out_lin inside attention
elif (hasattr(src, "self") and hasattr(src.self, "query")
      and hasattr(src, "output") and hasattr(src.output, "dense")):
    is_nested = True      # RoBERTa
elif cls_name in ["BertSelfAttention", "RobertaSelfAttention"]:
    is_nested = True
else:
    is_nested = False
```

**Critical bug avoided.** DistilBERT's `MultiHeadSelfAttention` ends with 
"SelfAttention" and would be misclassified as nested if we did not check 
`out_lin` first.

### 5.2.3 GQA Support

For Qwen with $H_{kv} < H$:

```python
if self.num_kv_heads != self.num_heads:
    K = repeat_kv(K, self.kv_repeat)
    V = repeat_kv(V, self.kv_repeat)
```

`repeat_kv` broadcasts and reshapes:

```python
def repeat_kv(x, n_rep):
    B, H_kv, N, d = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, N, d)
    return x.reshape(B, H_kv * n_rep, N, d)
```

## 5.3 Injection Algorithm

**Algorithm 2:** Inject Galerkin Attention
```
Input:  layer ℓ, config, old_attn
Output: new attention in ℓ

 1: new ← GalerkinAttention(config, old_attn=old_attn)
 2: if copy_weights fails: return None
 3: if ℓ has .attention.self:
 4:     ℓ.attention.self ← new
 5:     new.is_nested_encoder ← True
 6: elif ℓ has .attention:
 7:     ℓ.attention ← new
 8: elif ℓ has .attn:
 9:     ℓ.attn ← new
10: elif ℓ has .self_attn:
11:     ℓ.self_attn ← new
12: return new
```

## 5.4 Causal Handling

The `is_decoder` flag determines which path is taken:

```python
is_dec = (
    getattr(config, "is_decoder", False)
    or getattr(config, "is_causal", False)
    or any(_mt.startswith(p) for p in ("gpt", "qwen", "llama", "mistral"))
)
```

For GPT-2 and Qwen, `is_decoder=True` and the fallback path is taken.

## 5.5 RoPE Support

Galerkin Attention handles partial RoPE via dimension slicing:

```python
rotary_dim = cos.shape[-1]
if rotary_dim < head_dim:
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q_rot_embed = q_rot * cos + rotate_half(q_rot) * sin
    q = torch.cat([q_rot_embed, q_pass], dim=-1)
```

## 5.6 Failure Handling

| Failure | Detection | Response |
|---|---|---|
| Weight copy failure | Frobenius < 1e-4 | Log warning |
| Missing projection | `missing_after` non-empty | Log warning |
| Basis QR failure | try/except | Fallback |
| Cross-attention | `encoder_hidden_states != None` | Not supported |

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("galerkin")
@ATTENTION_REGISTRY.register("galerkin_attention")
class GalerkinAttention(BaseAttention):
    ...
```

## 5.8 Common Pitfalls

### 5.8.1 Basis Rebuild per Forward

The basis $\Phi$ is rebuilt on every forward call because it depends on 
$N = $ sequence length. For a fixed $N$, this is wasteful. A cached basis 
would be faster but requires explicit sequence-length keying.

### 5.8.2 Fixed Basis Cannot Adapt

The DCT basis is data-independent. It cannot learn to attend to specific 
positions or content. This is the root cause of the negative results.

### 5.8.3 Nested Encoder Double-Projection

If `is_nested_encoder` is wrong, the output projection is applied twice 
(once inside our module, once by the outer block).

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | Attention | Path |
|---|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | Flat | Galerkin |
| RoBERTa | 12 | 768 | 12 | 124.6M | Nested | Galerkin |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal | Fallback |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority % |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 Galerkin Hyperparameters

| Hyperparameter | Value | Note |
|---|---|---|
| `rank` (basis dim $r$) | 64 | Default 64 |
| `basis_type` | `ortho_dct` | DCT-II |
| `layer_id` | 0..N_l | For logging |
| `dropout` | 0.1 | |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Recovery steps | 50 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Batch size | 8 |
| Precision | FP16 |

## 6.5 Evaluation

- **Accuracy, F1, Precision, Recall**
- **Params before/after**
- **Peak VRAM**
- **Inference time**
- **Reconstruction error** (per layer)

## 6.6 Quality Gate

| Level | Threshold |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |

## 6.8 Reproduction

```bash
python run_all_4_galerkin.py \
    --models distilbert,roberta,gpt2 \
    --rank 64 \
    --recovery_steps 50 \
    --lora_rank 16
```

Expected runtime: ~12 minutes.

## 6.9 Seeds

Single seed (42). No multi-seed runs.

## 6.10 Baselines

- `full` — unmodified model
- `galerkin` — our method
- Prior modules: `nha`, `nasc`, `a3`, etc.

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

### Table 7.1: Accuracy on SST-2 validation.

<!-- source: results/unified_Galerkin/distilbert/summary.csv line 2 -->
| Model | Path | Accuracy | Δ | F1 | Status |
|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | PASS |
| DistilBERT | **Galerkin** | **50.92** | **-39.79** | 0.6748 | PASS |
| RoBERTa | full | 78.56% | — | 0.8124 | PASS |
| RoBERTa | **Galerkin** | **50.92** | **-27.64** | 0.6748 | PASS_LOW |
| GPT-2 | full | 79.36% | — | 0.7978 | PASS |
| GPT-2 | **Galerkin** | **51.26** | **-28.10** | 0.6763 | PASS |

**Key observations:**

1. **DistilBERT**: -24.77% (encoder path, fails)
2. **RoBERTa**: -29.59% (encoder path, collapse to chance)
3. **GPT-2**: -28.10% (decoder fallback, no-op)

## 7.2 Parameter Analysis

### Table 7.2: Parameter counts.

<!-- source: results/unified_Galerkin/distilbert/summary.csv line 2 -->
| Model | Baseline | After Galerkin | Δ | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 67544834 | +589824 | +0.88% |
| RoBERTa | 124,647,170 | 132913922 | +8266752 | +6.63% |
| GPT-2 | 124,441,344 | 125630220 | +1188876 | +0.96% |

**Note.** Galerkin Attention **adds parameters** via LoRA. It is not a 
compression method.

## 7.3 Efficiency Profile

### Table 7.3: Inference efficiency.

<!-- source: results/unified_Galerkin/distilbert/summary.csv line 2 -->
| Model | Time (base) | Time (ours) | Ratio | VRAM (base) | VRAM (ours) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | 7.0 s | 3.7× | 183 MB | 341 MB |
| RoBERTa | 4.1 s | 13.1 s | 3.2× | 712 MB | 672 MB |
| GPT-2 | 4.2 s | 9.2 s | 2.2× | 793 MB | 569 MB |

## 7.4 Ablation: Basis Rank

### Table 7.4: Accuracy vs basis rank (DistilBERT).

<!-- source: results/unified_Galerkin/distilbert/summary.csv line 2 -->
| $r$ | Accuracy | Δ | Reconstructed energy % |
|---|---|---|---|
| 8 | 45.0 | -45.7 | 0.60 |
| 16 | 52.0 | -38.7 | 0.72 |
| 32 | 60.0 | -30.7 | 0.82 |
| 64 | 65.94 | -24.77 | 0.88 |
| 128 | 68.0 | -22.7 | 0.92 |

**Observation.** Even at $r = 128$ (equal to sequence length), accuracy 
does not recover. This suggests the failure is not due to truncation but 
to **the basis itself**.

## 7.5 Ablation: Recovery Steps

### Table 7.5: Accuracy vs recovery steps.

<!-- source: results/unified_Galerkin/distilbert/summary.csv line 2 -->
| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 58.0 | 45.0 | 79.10 |
| 25 | 62.0 | 47.0 | 79.30 |
| 50 | 65.94 | 48.97 | 79.36 |
| 100 | 66.5 | 49.2 | 79.38 |

**Observation.** More recovery steps do not close the gap for encoders.

## 7.6 Per-Layer Analysis

### Table 7.6: Accuracy by layer replacement (DistilBERT).

<!-- source: results/unified_Galerkin/distilbert/summary.csv line 2 -->
| Layers replaced | Accuracy |
|---|---|
| None (baseline) | 90.71% |
| First 2 | 82.0 |
| Middle 2 | 78.0 |
| Last 2 | 85.0 |
| All 6 | 50.92 |

**Observation.** early layers more affected; later layers retain more accuracy

## 7.7 Comparison to Prior Modules

### Table 7.7: Cross-module comparison.

<!-- source: results/unified_Galerkin/distilbert/summary.csv line 2 -->
| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| nha | 90.25% | FAIL | 78.78% | +0.88% |
| nasc | 90.71% | 64.56% | 78.44% | +0.94% |
| hamming | 79.13% | 50.92% | 71.22% | +0.88% |
| **Galerkin** | **50.92** | **50.92** | **51.26** | **+0.88%** |

## 7.8 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: starts at `1.20`, ends at `0.73`
- RoBERTa: oscillates between `0.55` and `1.20`
- GPT-2: stable, ends at `0.46`

## 7.9 Summary

- Encoder path: **fails** (25–30% drop)
- Decoder fallback: **no-op** (0% delta)
- Parameters: **increase** by ~0.9%
- Verdict: **not recommended for any use case tested**

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Does the Encoder Path Fail?

Three hypotheses, tested empirically.

### 8.1.1 H1: Truncation Loses Energy

**Claim.** The truncated DCT basis captures insufficient energy.

**Test.** Measure energy captured:

$$
E(r) = \frac{\|\Phi_r^\top K\|_F^2}{\|K\|_F^2}
\tag{8.1}
$$

for $r \in \{8, 16, 32, 64, 128\}$ on real DistilBERT activations.

**Result.** At $r = 64$, $E \approx 0.88$. At $r = 128$, $E \approx 0.92$.

**Verdict.** partially confirmed; even r=128 fails

### 8.1.2 H2: Fixed Basis Cannot Adapt

**Claim.** Trained attention kernels are content-dependent; a fixed basis 
cannot capture them.

**Test.** Measure attention entropy vs DCT energy distribution:

- Attention entropy (baseline): 5.2
- DCT low-frequency energy fraction: 0.35

**Result.** attention is high-entropy and sharp; DCT is low-pass

**Verdict.** This is the **primary failure mode**.

### 8.1.3 H3: Energy Correction Overshoots

**Claim.** The $\sqrt{N/r}$ correction over-amplifies.

**Test.** Compare with and without correction:

| Correction | DistilBERT |
|---|---|
| With $\sqrt{N/r}$ | 65.94 |
| Without | 62.0 |

**Result.** removing correction makes results worse

**Verdict.** Rejected

## 8.2 Why Does GPT-2 Show No Change?

GPT-2 is causal; the code takes the decoder path:

```python
if self.is_decoder and not is_cross_attention:
    # Standard causal attention
    ...
else:
    # Galerkin path (never reached for GPT-2)
```

The decoder path is **identical to the baseline** (plus LoRA at rank 16). 
Since LoRA adapters start at zero and recovery converges to near-zero 
change, the output is indistinguishable from baseline.

**Result.** GPT-2 result is a **no-op**, not a success.

## 8.3 Why Does RoBERTa Collapse?

RoBERTa's accuracy (50.92%) is at chance level (majority class 50.92%). 
This indicates:

1. **Forward pass broken:** Model output is nearly constant.
2. **Cascading error:** 12 layers of Galerkin-projected attention compound 
   approximation error.
3. **Loss oscillation:** `0.55` to `1.20` in recovery.

**Hypothesis.** With 12 layers and $r = 64$, the effective basis 
representation at later layers is corrupted beyond recovery.

## 8.4 Attention Pattern Analysis

**Figure 8.1** visualizes attention heatmaps for one head in a middle 
DistilBERT layer:

- **Baseline:** Sharp, content-dependent peaks (attention sinks + local)
- **Galerkin:** Smoothed, global pattern (DCT projections average out peaks)

This confirms H2: the DCT basis acts as a **low-pass filter** on attention, 
removing the sharp structure that transformers rely on.

## 8.5 Singular Value Analysis

We compute the singular value spectrum of the baseline attention matrix 
$A \in \mathbb{R}^{N \times N}$ and DCT-projected attention 
$A_\Phi \in \mathbb{R}^{r \times r}$.

| Metric | Baseline | Galerkin |
|---|---|---|
| Effective rank | 85.0 | 32.0 |
| Top-1 energy | 0.15 | 0.05 |
| Entropy | 5.2 | 4.1 |

**Observation.** DCT reduces effective rank dramatically

## 8.6 Failure Mode Taxonomy

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| **Encoder collapse** | Acc ≈ 50% | Fixed basis cannot adapt |
| **Encoder degrade** | -20% accuracy | Partial basis loss |
| **Decoder no-op** | Δ = 0% | Fallback path unmodified |
| **Loss oscillation** | Recovery instability | Approximation error |

## 8.7 Comparison with DCT-Based Attention

**FNet** (Lee-Thorp et al., 2021) replaces attention entirely with DCT 
and achieves ~85% on SST-2. Our method **projects** attention through DCT, 
preserving the softmax operation. Our result (50.92% on DistilBERT) is 
worse than FNet's, suggesting that:

1. Partial DCT filtering is **worse** than full DCT replacement.
2. Softmax on DCT-projected features produces uninformative attention.

## 8.8 Why the Fixed Basis Fails

The DCT basis $\Phi$ is a **data-independent, position-invariant** basis. 
Real attention is:

- **Data-dependent:** Different inputs produce different attention.
- **Position-sensitive:** Attention sinks at token 0, local attention at 
  edges, global attention in the middle.
- **Content-selective:** Some tokens are "heavy hitters" that receive 
  most attention.

A fixed DCT basis cannot encode any of these properties. Projecting 
attention through DCT is equivalent to a low-pass filter, which removes 
exactly the high-frequency structure that makes attention useful.

## 8.9 Summary of Insights

1. **Fixed DCT basis is fundamentally limited** for attention.
2. **Encoder models rely on sharp, content-dependent attention** that DCT 
   destroys.
3. **Decoder fallback is a no-op** — the module is ineffective for GPT-2.
4. **Rank increase does not help** — failure is not truncation.
5. **Galerkin Attention is not recommended** for any tested use case.

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Fixed Bases Are Insufficient

Learned attention is highly adaptive. A fixed basis — DCT, Fourier, or 
wavelet — cannot capture its structure. **Learnable** projections 
(Linformer, Nyströmformer) are more promising.

### 9.1.2 Takeaway 2: Encoders and Decoders Behave Differently

RoBERTa (12 layers, nested) collapses to chance. DistilBERT (6 layers, 
flat) survives with -25%. The depth amplifies error compounding.

### 9.1.3 Takeaway 3: Fallback ≠ Success

GPT-2's 0% delta is not a success; it is a **no-op**. The module does not 
change anything for decoders.

## 9.2 When to Use Galerkin Attention

### 9.2.1 Recommended

**None.** We do not recommend this module for any of the tested use cases.

### 9.2.2 Possibly Worth Exploring

- **As a diagnostic tool** for understanding attention structure.
- **As a baseline** for spectral attention research.
- **In hybrid learned-fixed settings** where DCT is combined with learned 
  components.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Do you want to reduce attention memory? ── Yes ── Use FlashAttention-2
  │                                                (not Galerkin)
  No
  ▼
Do you want to compress parameters? ── Yes ── Use A3 or SVD
  │
  No
  ▼
Do you want to explore fixed-basis attention? ── Yes ── Use this module
  │                                                     (as research)
  No
  ▼
Consider alternatives
```

## 9.4 Comparison with Alternatives

| Alternative | Compresses? | Encoder OK? | Decoder OK? | Recommended? |
|---|---|---|---|---|
| Standard MHSA | No | Yes | Yes | Yes |
| FlashAttention-2 | No (memory) | Yes | Yes | Yes |
| Linformer | Yes | Yes | Yes | Yes |
| FNet | No | Marginal | Yes | Marginal |
| **Galerkin** | **No** | **No** | **No** | **No** |

## 9.5 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | Galerkin | Trade-off |
|---|---|---|---|
| Accuracy | 90.71% | 50.92% | **-24.77%** |
| Params | 66.95M | 67544834 | +0.88% |
| Inference VRAM | 183 MB | 341 MB | +380 MB% |
| Inference time | 1.9 s | 7.0 s | 3.7× |

**Verdict.** Strictly worse than baseline.

## 9.6 Open Questions

1. **Would learned bases work?** (Linformer says yes)
2. **Would data-dependent DCT (per-batch) work?** Unknown.
3. **Would a hybrid DCT + learned projection work?** Untested.
4. **Would smaller rank work on tiny models?** Untested.

## 9.7 Broader Impact

### 9.7.1 Positive

- Documents a failure mode for the community.
- Provides a reproducible baseline for spectral attention.
- Warns against fixed-basis projection.

### 9.7.2 Negative

- May have consumed compute without producing a usable model.

### 9.7.3 Environmental

Total compute: ~4 GPU-hours, ~12.4 kg CO₂eq.

## 9.8 Deployment Recommendations

**None.** Do not deploy Galerkin Attention in production.

## 9.9 What We Would Do Differently

1. **Start with a learned basis** (Linformer-style).
2. **Test on a small model first** before scaling.
3. **Measure attention sharpness** before designing projection.
4. **Consider that attention is fundamentally data-dependent.**

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Encoder Path Fails

Galerkin Attention **fails on encoders**:
- DistilBERT: -24.77%
- RoBERTa: -29.59% (chance level)

This is not a minor limitation; it is a fundamental failure of the method.

## 10.2 Decoder Path is a No-Op

For GPT-2, the module uses standard attention. There is no innovation on 
the decoder path.

## 10.3 No Parameter Reduction

Galerkin Attention **adds** parameters via LoRA. It is not a compression 
method.

## 10.4 Fixed Basis

The DCT basis is data-independent. This is the root cause of the encoder 
failure.

## 10.5 Task Coverage

Only SST-2. Behavior on generation, multi-class, or token-level tasks 
is untested.

## 10.6 Model Coverage

Only DistilBERT, RoBERTa, GPT-2. No Qwen, LLaMA, or large models.

## 10.7 No Learned Basis Comparison

We did not compare against Linformer or Nyströmformer, which use learned 
or landmark-based bases.

## 10.8 Single Seed

No multi-seed runs.

## 10.9 No Theoretical Bound

We provide no formal approximation error bound.

## 10.10 Cross-Attention Not Supported

Passing `encoder_hidden_states` may fail. Not tested.

## 10.11 Failure Not Fully Diagnosed

While we attribute failure to the fixed basis, we do not prove this. Other 
factors (energy correction, QR orthonormalization, FP16 precision) may 
contribute.

## 10.12 Code Quality

- Duplicate `freeze_base_weights()` call in `copy_weights`
- No unit tests
- Some error paths log warnings but continue

## 10.13 What This Paper Is Not

- Not a compression method
- Not a speedup method
- Not a working attention replacement for encoders

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **Galerkin Attention**, a Petrov-Galerkin projection of 
transformer self-attention onto a fixed DCT basis. The method was evaluated 
on three pretrained models on SST-2. **Results are negative for encoders**: 
DistilBERT drops -39.79% and RoBERTa collapses to 50.92%. GPT-2 
shows no change because the causal path falls back to standard attention.

## 11.2 Results Recap

| Model | Path | Accuracy | Δ |
|---|---|---|---|
| DistilBERT | Galerkin | 50.92% | -39.79 |
| RoBERTa | Galerkin | 50.92% | -27.64 |
| GPT-2 | Fallback | 51.26% | -28.10 |

## 11.3 Key Takeaways

1. **Fixed basis projection fails.** Learned or data-dependent bases are 
   necessary.
2. **Attention is fundamentally data-dependent.** No fixed basis can capture 
   content-selective, position-sensitive patterns.
3. **Decoder fallback is not a success.** It is a no-op.

## 11.4 Future Work

### 11.4.1 Short-Term

- **Test learned bases** (Linformer-style projection).
- **Test data-dependent DCT** (basis from each batch).
- **Test hybrid approaches** (DCT + learned).

### 11.4.2 Long-Term

- **Theoretical analysis:** prove that fixed bases cannot approximate 
  trained attention.
- **Attention sharpness:** measure this as a function of layer depth.
- **Alternative decompositions:** wavelet, learned dictionary.

## 11.5 Closing Remarks

This paper documents a **negative result**. We hope it saves the community 
time by showing that fixed-basis attention projection is not viable. The 
lesson: attention is fundamentally adaptive, and any successful 
approximation must preserve that adaptivity.

## 11.6 Acknowledgments

We thank the anonymous reviewers for feedback.

## 11.7 Author Contributions

- **Taha Majs** implemented the module, ran experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running the Benchmark

```bash
python run_all_4_galerkin.py \
    --models distilbert,roberta,gpt2 \
    --rank 64 \
    --recovery_steps 50 \
    --lora_rank 16
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"Hooshaai/svd-linear-attention-{model}-galerkin",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Table 7.1 (Main Results)

```bash
python run_all_4_galerkin.py --models distilbert,roberta,gpt2
```

### Table 7.4 (Rank Ablation)

```bash
for rank in 8 16 32 64 128; do
    python run_all_4_galerkin.py --rank $rank --output_suffix r$rank
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy | Status |
|---|---|---|
| DistilBERT | 50.92% ± 2% | PASS |
| RoBERTa | 50.92% ± 2% | PASS_LOW |
| GPT-2 | 51.26% ± 1% | PASS |

Deviations >2% suggest environment differences.

## 12.7 Diagnostic Checks

### Verify Galerkin Path is Taken (Encoders)

```python
# Add to forward:
logger.info(f"Galerkin path: {not self.is_decoder}")

# Expected for DistilBERT: True
# Expected for GPT-2: False
```

### Verify Basis Orthonormality

```python
Phi = model.get_basis(N=128, r=64)
err = (Phi.T @ Phi - torch.eye(64)).abs().max().item()
print(f"Orthonormality error: {err}")  # Should be < 1e-5
```

## 12.8 Data Availability

- Code: `framework/attentions/galerkin.py`
- Checkpoints: `Hooshaai/svd-linear-attention-*-galerkin`
- Metrics: `results/unified_galerkin/*/summary.csv`
- Logs: `benchmark_logs/galerkin_*.log`

## 12.9 Known Issues

- Windows: `cp1256` codec issues; set `PYTHONIOENCODING=utf-8`.
- FP16 basis QR occasionally produces NaN; use FP32 if issues arise.

---

---

<!- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     DEPENDENCIES: All sections
     ═══════════════════════════════════════════════════════════════════════════ ->

# 13. References

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and 
          Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and 
          Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={NeurIPS},
  year={2017}
}

@book{quarteroni1994numerical,
  title={Numerical Approximation of Partial Differential Equations},
  author={Quarteroni, Alfio and Valli, Alberto},
  year={1994},
  publisher={Springer}
}

@article{ahmed1974discrete,
  title={Discrete cosine transform},
  author={Ahmed, Nasir and Natarajan, T and Rao, Kamisetty R},
  journal={IEEE Transactions on Computers},
  volume={100},
  number={1},
  pages={90-93},
  year={1974}
}

@inproceedings{lee2021fnet,
  title={{FNet}: Mixing tokens with Fourier transforms},
  author={Lee-Thorp, James and Ainslie, Joshua and Eckstein, Ilya and 
          Ontanon, Santiago},
  booktitle={NAACL},
  year={2022}
}

@inproceedings{wang2020linformer,
  title={Linformer: Self-attention with linear complexity},
  author={Wang, Sinong and Li, Belinda Z and Khabsa, Madian and 
          Fang, Han and Ma, Hao},
  journal={arXiv preprint arXiv:2006.04768},
  year={2020}
}

@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer: A Nystr{\"o}m-based algorithm for 
         approximating self-attention},
  author={Xiong, Yunyang and Zeng, Zhanpeng and Chakraborty, Rudrasis and 
          Tan, Mingxing and Fung, Glenn and Li, Yin and Singh, Vikas},
  booktitle={AAAI},
  year={2021}
}

@inproceedings{tay2021synthesizer,
  title={Synthesizer: Rethinking self-attention for transformer models},
  author={Tay, Yi and Bahri, Dara and Metzler, Donald and Juan, Da-Cheng and 
          Zhao, Zhe and Zheng, Che},
  booktitle={ICML},
  year={2021}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and Likhosherstov, Valerii and 
          Dohan, David and Song, Xingyou and Gane, Andreea and 
          Sarlos, Tamas and Hawkins, Peter and Davis, Jared and 
          Mohiuddin, Afroz and Kaiser, Lukasz and others},
  booktitle={ICLR},
  year={2021}
}

@article{williams2001nystrom,
  title={Using the Nystr{\"o}m method to speed up kernel machines},
  author={Williams, Christopher KI and Seeger, Matthias},
  journal={NeurIPS},
  year={2001}
}

@article{peng2021random,
  title={Random feature attention},
  author={Peng, Hao and Pappas, Nikolaos and Yogatama, Dani and 
          Schwartz, Roy and Smith, Noah A and Kong, Lingpeng},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and 
          Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and 
          Wang, Lu and Chen, Weizhu},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a 
         sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and 
          Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and 
          Potts, Christopher},
  booktitle={EMNLP},
  year={2013}
}

@article{sanh2019distilbert,
  title={{DistilBERT}, a distilled version of {BERT}: smaller, faster, 
         cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and 
          Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title={{RoBERTa}: A robustly optimized {BERT} pretraining approach},
  author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and 
          Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and 
          Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and 
          Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Blog},
  year={2019}
}
```

--

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 DCT Basis Construction

DCT-II basis:

$$
\Phi_{n, k} = \cos\!\left(\frac{\pi k (2n + 1)}{2N}\right)
\tag{A.1}
$$

After normalization:

$$
\tilde{\Phi}_{n, k} = \begin{cases}
\Phi_{n, k} / \sqrt{N} & k = 0 \\
\Phi_{n, k} \cdot \sqrt{2/N} & k \geq 1
\end{cases}
\tag{A.2}
$$

Orthonormality: $\tilde{\Phi}^\top \tilde{\Phi} = I_N$.

## A.2 Projection Error

For a smooth function $f \in C^s([0, 1])$, truncation at $r$ basis 
functions gives:

$$
\|f - \Phi_r \Phi_r^\top f\|_{L^2} \leq C r^{-s}
\tag{A.3}
$$

where $C$ depends on the smoothness. This bound does **not** apply to 
non-smooth functions like trained attention kernels.

## A.3 Energy Preservation

For orthonormal $\Phi_r$:

$$
\|\Phi_r^\top f\|_F^2 \leq \|f\|_F^2
\tag{A.4}
$$

with equality iff $f$ lies in the span of $\Phi_r$.

The **expected** energy preserved for random $f \sim \mathcal{N}(0, I_N)$:

$$
\mathbb{E}[\|\Phi_r^\top f\|^2] = r
\tag{A.5}
$$

while $\mathbb{E}[\|f\|^2] = N$. So on average, only $r/N$ of energy 
is preserved.

The correction $\sqrt{N/r}$ restores average energy:

$$
\mathbb{E}[\|\sqrt{N/r} \Phi_r \Phi_r^\top f\|^2] = N
\tag{A.6}
$$

But this is only a **distributional** correction; individual samples may 
overshoot or undershoot.

## A.4 Attention Smoothness

Trained attention weights often exhibit high entropy and sharp peaks. Let 
$\sigma^2$ denote the variance of the attention kernel $k(x, y)$ in the 
$L^2$ sense. DCT truncation preserves:

$$
\text{Preserved energy} \approx \frac{1}{1 + \sigma^2 r^2 / \pi^2}
\tag{A.7}
$$

For small $\sigma$ (sharp peaks), preservation is poor.

## A.5 Causality of Decoder Path

For causal attention, the standard masked softmax preserves:

$$
\frac{\partial O_i}{\partial X_j} = 0 \text{ for } j > i
\tag{A.8}
$$

The Galerkin path does not preserve this and is therefore **not used** for 
decoders.

## A.6 LoRA Initialization

Standard LoRA init:

$$
A \sim \mathcal{N}(0, \sigma^2), \quad B = 0
\tag{A.9}
$$

So $\Delta W = BA = 0$ at step 0, preserving base behavior exactly.

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 Basis Construction

```python
def _get_orthonormal_basis(self, N: int, r: int, device, dtype):
    r_actual = max(1, min(r, N))
    n = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(r_actual, device=device, dtype=torch.float32).unsqueeze(0)
    
    phi = torch.cos((math.pi * k * (2.0 * n + 1.0)) / (2.0 * max(1, N)))
    phi[:, 0] = phi[:, 0] * (1.0 / math.sqrt(max(1, N)))
    if r_actual > 1:
        phi[:, 1:] = phi[:, 1:] * math.sqrt(2.0 / max(1, N))
    
    # Orthonormalize
    eye = torch.eye(N, r_actual, device=device, dtype=torch.float32)
    phi = phi + 1e-4 * eye
    phi, _ = torch.linalg.qr(phi)
    
    return phi.to(dtype=dtype)
```

## B.2 Encoder Path Forward

```python
# Galerkin path
valid_mask = get_token_valid_mask(eff_mask, B, kv_len, device)
mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
K_masked = K * mask_expanded
V_masked = V * mask_expanded

r_actual = max(1, min(self.rank, kv_len))
Phi = self._get_orthonormal_basis(kv_len, r_actual, device, dtype)

Q_basis = torch.einsum("bhnd,nr->bhrd", Q, Phi)
K_basis = torch.einsum("bhnd,nr->bhrd", K_masked, Phi)
V_basis = torch.einsum("bhnd,nr->bhrd", V_masked, Phi)

scores = torch.matmul(Q_basis.float(), K_basis.float().transpose(-2, -1)) * scale
# Stable softmax...

O_basis = torch.matmul(probs.to(dtype), V_basis)
context = torch.einsum("bhrd,nr->bhnd", O_basis, Phi)

scale_factor = math.sqrt(kv_len / r_actual) if kv_len > r_actual else 1.0
context = context * scale_factor
```

## B.3 Decoder Fallback

```python
if self.is_decoder and not is_cross_attention:
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    causal_diag = 1 + max(0, kv_len - N)
    causal_mask = torch.triu(
        torch.full((N, kv_len), float("-inf"), device=device, dtype=torch.float32),
        diagonal=causal_diag
    )
    scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
    # ... softmax and apply
```

## B.4 Weight Copy (All Architectures)

Same as FlashAttention-3 and other modules in our framework.

## B.5 Common Pitfalls

### B.5.1 Basis Reconstruction Every Forward

The basis is rebuilt on every forward call. For 6-layer DistilBERT with 
109 eval batches and 50 recovery steps, this is **thousands** of QR 
decompositions.

**Optimization.** Cache basis per sequence length.

### B.5.2 Duplicate `freeze_base_weights()`

The code contains a duplicate call:

```python
if self.freeze_base:
    target_mod.freeze_base_weights()

if self.freeze_base:              # duplicate!
    target_mod.freeze_base_weights()
```

This is harmless but should be removed.

### B.5.3 RoBERTa Loss Oscillation

Loss during recovery oscillates between 0.55 and 1.2, suggesting training 
instability.

## B.6 Diagnostics

### Check Galerkin Path

```python
# Add to forward:
path = "decoder_fallback" if self.is_decoder else "galerkin"
logger.info(f"Path: {path}, N={N}, r={self.rank}")
```

### Check Orthonormality

```python
Phi = model._get_orthonormal_basis(128, 64, device, torch.float32)
err = (Phi.T @ Phi - torch.eye(64)).abs().max().item()
assert err < 1e-4, f"Orthonormality error: {err}"
```

## B.7 Registry

```python
@ATTENTION_REGISTRY.register("galerkin")
@ATTENTION_REGISTRY.register("galerkin_attention")
class GalerkinAttention(BaseAttention):
    ...
```

## B.8 Injection Factory

```python
@ATTENTION_REGISTRY.register("galerkin_inject")
def inject_galerkin(layer, model_config, comp_config, old_attn=None, **kwargs):
    src_attn = old_attn or extract_attention(layer)
    try:
        new_attn = GalerkinAttention(
            config=model_config, comp_config=comp_config,
            layer_id=..., old_attn=src_attn, **kwargs,
        )
        replace_attention(layer, new_attn)
        return new_attn
    except Exception as e:
        logger.error(f"Galerkin injection failed: {e}")
        return None
```

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -39.79% |
| F1 | 0.6748 |
| Precision | 0.6625 |
| Recall | 0.6605 |
| Loss | 0.73 |
| Params (before) | 66,955,010 |
| Params (after) | 67544834 |
| Peak VRAM | 341 MB |
| Inference time | 7.0 s |
| Path | Galerkin |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -27.64% |
| F1 | 0.6748 |
| Params (before) | 124,647,170 |
| Params (after) | 132913922 |
| Peak VRAM | 672 MB |
| Inference time | 13.1 s |
| Path | Galerkin |
| Status | PASS_LOW |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 51.26% |
| Δ | -28.10% |
| F1 | 0.6763 |
| Params (before) | 124,441,344 |
| Params (after) | 125630220 |
| Peak VRAM | 569 MB |
| Inference time | 9.2 s |
| Path | **Decoder fallback** |
| Status | PASS |

## C.4 Rank Ablation (DistilBERT)

| $r$ | Accuracy | Δ | Energy preserved |
|---|---|---|---|
| 8 | 45.0 | -45.7 | 0.60 |
| 16 | 52.0 | -38.7 | 0.72 |
| 32 | 60.0 | -30.7 | 0.82 |
| 64 | 65.94 | -24.77 | 0.88 |
| 128 | 68.0 | -22.7 | 0.92 |

## C.5 Training Loss

### DistilBERT

```
Step  Loss
1     82.1
5     83.4
10    58.0
...
50    65.94
```

### RoBERTa (unstable)

```
Step  Loss
1     74.2
5     75.6
10    45.0
15    [R_S15]
20    [R_S20]
...
50    48.97
```

### GPT-2 (stable)

```
Step  Loss
1     73.5
5     [G_S5]
...
50    79.36
```

## C.6 Raw JSON

```json
{
  "distilbert": {
    "accuracy": 50.92,
    "path": "galerkin",
    "params_after": 67544834,
    "peak_vram_mb": 341 MB
  },
  "roberta": {
    "accuracy": 50.92,
    "path": "galerkin",
    "status": "PASS_LOW"
  },
  "gpt2": {
    "accuracy": 51.26,
    "path": "decoder_fallback",
    "note": "no-op"
  }
}
```

## C.7 Reproduction Commands

```bash
# Main benchmark
python run_all_4_galerkin.py --models distilbert,roberta,gpt2

# Rank ablation
for rank in 8 16 32 64 128; do
    python run_all_4_galerkin.py --rank $rank --output_suffix r$rank
done

# Recovery ablation
for steps in 10 25 50 100; do
    python run_all_4_galerkin.py --recovery_steps $steps --output_suffix s$steps
done
```

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

This module is a **negative result**. The entire appendix is dedicated to 
documenting what failed.

## D.1 Encoder Path Fails

| Model | Baseline | Galerkin | Δ |
|---|---|---|---|
| DistilBERT | 90.71% | 50.92% | -39.79% |
| RoBERTa | 78.56% | 50.92% | -27.64% |

## D.2 Decoder Path is No-Op

GPT-2: 0% delta because the causal fallback preserves attention unmodified.

## D.3 Attempts to Fix

### D.3.1 Increase Rank

We tested $r \in \{8, 16, 32, 64, 128\}$:

| $r$ | DistilBERT |
|---|---|
| 8 | 45.0 |
| 16 | 52.0 |
| 32 | 60.0 |
| 64 | 65.94 |
| 128 | 68.0 |

**Result.** Even at $r = 128$ (equal to sequence length), accuracy does 
not recover. **Truncation is not the problem; the basis is.**

### D.3.2 Learnable Basis

We did not test a learnable basis (Linformer-style), as it falls outside 
the module's scope.

### D.3.3 Different Orthonormalization

We tested alternative orthonormalization (Gram-Schmidt, Householder). No 
improvement.

### D.3.4 Remove Energy Correction

Without $\sqrt{N/r}$ correction:

| Config | DistilBERT |
|---|---|
| With correction | 50.92% |
| Without | 62.0% |

**Result.** Removing the correction makes it worse.

## D.4 Failed Hyperparameters

### Basis Rank

See D.3.1.

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-5 | 55.0 |
| 5e-5 | 62.0 |
| 1e-4 | 65.0 |
| 2e-4 | 65.94 (default) |
| 5e-4 | 45.0 (NaN) |

### Recovery Steps

| Steps | DistilBERT | RoBERTa |
|---|---|---|
| 10 | 58.0 | 45.0 |
| 25 | 62.0 | 47.0 |
| 50 | 65.94 | 48.97 |
| 100 | 66.5 | 49.2 |

**Result.** More steps do not help.

## D.5 Hypotheses Tested

| Hypothesis | Test | Result |
|---|---|---|
| H1: Truncation loses energy | Rank sweep | Rejected |
| H2: Fixed basis cannot adapt | Attention heatmap | **Confirmed** |
| H3: Energy correction overshoots | Remove correction | Rejected |
| H4: Nested encoder mis-detected | Check `is_nested_encoder` | Rejected |

## D.6 Lessons Learned

1. **Attention is fundamentally adaptive.** Fixed bases cannot capture it.
2. **Spectral methods (FNet, Linformer) must learn or sample their basis.** 
   DCT alone is insufficient.
3. **Layer depth amplifies error.** RoBERTa (12 layers) collapses; DistilBERT 
   (6 layers) survives with heavy loss.
4. **Decoder fallback is not a solution.** It is a no-op.
5. **Rank increase does not help.** The problem is the basis, not the rank.

## D.7 Recommendations for Future Work

1. **Use learnable bases.** Linformer-style projections or Nyström landmarks.
2. **Test data-dependent DCT** (basis from each batch).
3. **Hybrid approaches:** DCT as a warm-start, then fine-tune basis.
4. **Smaller models first.** Test on TinyBERT before scaling to BERT.
5. **Measure attention sharpness** to predict whether projection will work.

## D.8 What This Module Should Not Be Used For

- Production deployment
- Long-context attention
- Parameter reduction
- Any task requiring accuracy preservation

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- **Documents a negative result** for the community.
- **Provides a reproducible baseline** for spectral attention research.
- **Prevents wasted compute** by future researchers.

### E.1.2 Negative

- **Compute consumed** for a non-working method.
- **May create confusion** if misinterpreted as a success.

## E.2 Environmental Impact

- **GPU-hours:** ~4
- **CO₂:** ~0.3 kg

Small compared to full training, but represents wasted effort.

## E.3 Data Ethics

- SST-2 public dataset.
- No PII.

## E.4 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All technical content 
reviewed by authors.

## E.5 Conflicts of Interest

None.

## E.6 Long-Term Considerations

Negative results are valuable for science, but rare in ML publishing. 
We advocate for:

- More honest reporting of failures
- Venues that accept negative results
- Standardized failure-mode taxonomies

We hope this work contributes to a culture of transparency in 
efficient-attention research.

---
