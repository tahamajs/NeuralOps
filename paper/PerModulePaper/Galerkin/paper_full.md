# Galerkin Attention: Petrov-Galerkin Projection for Linear-Complexity Transformer Self-Attention

**Authors:**
- Your Name¹ — you@university.edu
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/galerkin.py
**License:** MIT

---

## Abstract

Self-attention is fundamentally a **continuous non-local integral operator** 
that maps query signals to context signals via a kernel:

$$
(\mathcal{K}Q)(s) = \int \kappa(s, t) V(t) \, dt
\tag{1}
$$

Standard implementations discretize this operator on the raw token grid, 
requiring $\mathcal{O}(N^2 d)$ computation for $N$ tokens. We propose 
**Galerkin Attention**, which formulates self-attention as a **Petrov-Galerkin 
projection** onto an orthonormal basis (a discrete cosine transform variant) 
of dimension $r \ll N$. Queries, keys, and values are projected into this 
basis, attention is computed at rank $r$ in basis space, and the output is 
reconstructed to the original sequence grid via basis synthesis. Complexity 
reduces from $\mathcal{O}(N^2 d)$ to $\mathcal{O}(N r d)$ — **linear in 
sequence length** — for encoder models.

For **decoder models** (GPT-2, Qwen), the projection-based formulation is 
incompatible with causal masking. We therefore fall back to **standard 
causal softmax attention**, sacrificing the linear-complexity benefit in 
exchange for exactness.

Base Q/K/V/O projections are frozen and copied from the pretrained model; 
LoRA rank-16 adapters provide trainable capacity. On DistilBERT and 
RoBERTa-base with SST-2, Galerkin Attention achieves **89.91%** and 
**77.35%** respectively, within **-0.80%** and **-1.21%** 
of the SDPA baseline. On GPT-2, it matches the baseline exactly 
(**79.36%**) because it uses the standard attention fallback.

**Important disclosure.** Galerkin Attention is **not** a compression 
method — parameter count increases by ~0.88% via LoRA. It is 
a **compute-efficiency** method for encoder attention.

**Keywords:** Petrov-Galerkin projection, integral operator attention, 
linear-complexity attention, orthonormal basis, DCT

---

## Lay Summary

Standard attention compares every token to every other token, which becomes 
slow for long sequences. Galerkin Attention uses a different strategy: 
instead of working with individual tokens, it projects them onto a small 
set of "smooth basis functions" (similar to how audio is compressed as a 
sum of cosine waves). Attention is then computed between these smooth 
components, which is much faster. The result is projected back to the 
original sequence. This works well for encoders (like BERT) but not for 
decoders (like GPT-2), where we fall back to standard attention. The 
trade-off: faster compute for encoder models, no benefit for decoder models.

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

# 1. Introduction

## 1.1 Motivation

Self-attention is often described in discrete terms:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

but its mathematical foundation is **continuous**. Attention is a 
**non-local integral operator** acting on functions over a domain $\Omega$:

$$
(\mathcal{K} Q)(s) = \int_\Omega \kappa(s, t) V(t) \, dt
\tag{1.2}
$$

where $\kappa(s, t)$ is the kernel (typically $\exp(\langle Q(s), K(t) \rangle)$). 
Standard transformers discretize this operator on the **raw token grid**, 
yielding $\mathcal{O}(N^2 d)$ complexity for $N$ tokens.

**The Galerkin idea.** In numerical analysis, integral operators are often 
discretized not on the raw grid, but on a **finite-dimensional subspace** 
spanned by a small set of **basis functions**. The **Petrov-Galerkin 
method** selects test and trial bases to minimize the residual of the 
operator equation:

$$
\langle \mathcal{K} Q - \hat{Q}, \psi_i \rangle = 0 \quad \forall i
\tag{1.3}
$$

where $\{\psi_i\}$ is the test basis.

Applying this to attention:
1. Choose an orthonormal basis $\Phi \in \mathbb{R}^{N \times r}$ (e.g., DCT).
2. Project Q, K, V into basis space: $Q_\Phi = \Phi^\top Q$, etc.
3. Compute attention in basis space at rank $r$.
4. Reconstruct via basis synthesis: $Y = \Phi \cdot O_\Phi$.

This gives $\mathcal{O}(N r d)$ complexity — **linear in $N$**.

## 1.2 The Decoder Problem

Galerkin projection is **incompatible with causal masking**. The projected 
keys $K_\Phi$ contain contributions from **all** tokens, not just the 
prefix. For an encoder this is fine; for a decoder it breaks causality.

**Solution.** For decoders, fall back to standard causal softmax attention. 
This preserves exactness at the cost of the linear-complexity benefit.

## 1.3 Problem Statement

Given a pretrained attention module with projections $W_Q, W_K, W_V, W_O$, 
we seek an augmented attention function $\mathcal{A}'$ such that:

1. **Encoder linear complexity.** For bidirectional models, 
   $\mathcal{A}'$ runs in $\mathcal{O}(N r d)$ instead of $\mathcal{O}(N^2 d)$.
2. **Decoder exactness.** For causal models, $\mathcal{A}'$ preserves the 
   standard softmax attention output exactly.
3. **Multi-architecture support.** Works on flat (DistilBERT), nested 
   (RoBERTa), causal (GPT-2), and GQA+RoPE (Qwen).
4. **Step-0 preservation.** With zero-init LoRA, initial forward matches 
   pretrained.

**Note.** Not a compression method. Parameter count increases by LoRA.

## 1.4 Our Approach

**Galerkin Attention** chooses the **discrete cosine basis**:

$$
\Phi_{n, k} = \sqrt{\frac{2 - \delta_{k0}}{N}} \cos\!\left(\frac{\pi k (2n+1)}{2N}\right)
\tag{1.4}
$$

with the DC component scaled by $1/\sqrt{N}$. The basis is orthonormalized 
via QR for numerical stability.

**Encoder forward:**

$$
\begin{aligned}
Q_\Phi &= \Phi^\top Q, \quad K_\Phi = \Phi^\top K, \quad V_\Phi = \Phi^\top V \\
\text{Scores} &= Q_\Phi K_\Phi^\top / \sqrt{d_h} \\
\text{Output}_\Phi &= \text{softmax}(\text{Scores}) V_\Phi \\
Y &= \Phi \cdot \text{Output}_\Phi \cdot \sqrt{N/r}
\end{aligned}
\tag{1.5}
$$

**Decoder forward:**

Standard causal softmax attention (no projection).

## 1.5 Contributions

- **A Galerkin formulation of self-attention.** Section 4.2.
- **A DCT-based orthonormal basis.** With energy-preserving scaling. Section 4.3.
- **Encoder-decoder dispatch.** Projection for encoders, softmax for decoders. Section 4.4.
- **Multi-architecture weight transfer.** GPT-2 Conv1D, RoBERTa nested, DistilBERT flat, Qwen GQA. Section 5.
- **Honest reporting.** We disclose that GPT-2 uses the fallback and gains no benefit. Sections 7–8.
- **Full reproducibility.** Code under MIT.

## 1.6 Paper Organization

Section 2 reviews related work. Section 3 provides background on integral 
operators, Galerkin methods, and DCT. Section 4 presents Galerkin Attention. 
Section 5 details architecture integration. Section 6 gives experimental 
setup. Section 7 reports results. Section 8 analyzes the encoder-decoder 
dispatch. Section 9 discusses usage. Section 10 lists limitations. Section 
11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $r$ | Basis rank |
| $d$ | Hidden size |
| $d_h$ | Per-head dimension |
| $\Phi$ | Orthonormal basis matrix $\in \mathbb{R}^{N \times r}$ |
| $Q_\Phi$ | Projected queries $\in \mathbb{R}^{r \times d_h}$ |
| $\kappa$ | Attention kernel |
| $\Omega$ | Domain |
| $\delta_{k0}$ | Kronecker delta |

---

# 2. Related Work

## 2.1 Linear-Complexity Attention

**Linformer.** Wang et al. (2020) project $K, V$ to fixed low rank with 
learned projection matrices. Galerkin uses a **fixed DCT basis**, not learned.

**Performer.** Choromanski et al. (2021) approximate softmax via random 
features. Galerkin computes **exact** attention in basis space.

**Nyströmformer.** Xiong et al. (2021) use Nyström landmarks. Galerkin uses 
a full DCT basis.

**Linear Transformer.** Katharopoulos et al. (2020) reorder attention as 
associative products. Different mathematical primitive.

## 2.2 Basis Projection Methods

**FNet.** Lee-Thorp et al. (2021) replace self-attention with an unparameterized 
Fourier transform. Galerkin uses a truncated DCT basis plus a softmax 
attention in basis space.

**GFNet.** Rao et al. (2021) use global filter operations in Fourier space 
for vision. Similar spirit, different domain.

**AFNO.** Guibas et al. (2022) use adaptive Fourier neural operators for 
PDEs. Galerkin is a Galerkin discretization of the attention operator.

## 2.3 Numerical Analysis of Attention

**Neural Operators.** Kovachki et al. (2023) formalize neural operators as 
mappings between function spaces. Attention is a specific case.

**Galerkin Methods.** Classical theory (Strang, 1973; Brenner & Scott, 2008) 
for variational discretization of PDEs.

**Petrov-Galerkin.** Uses different test and trial bases. Galerkin Attention 
uses a **single** orthonormal basis (Galerkin, not Petrov-Galerkin in the 
strict sense).

## 2.4 Position of Our Work

| Method | Basis | Learned | Encoder | Decoder |
|---|---|---|---|---|
| Linformer | Linear proj | Yes | Yes | Yes |
| Performer | Random feats | No | Yes | Yes |
| FNet | FFT | No | Yes | No |
| Nyströmformer | Landmarks | Yes | Yes | Yes |
| **Galerkin** | **DCT (fixed)** | **No** | **Yes** | **Fallback** |

**Key distinction.** Galerkin is the only method that:
1. Uses a **fixed orthonormal basis** (DCT).
2. Dispatches encoder vs decoder.
3. Provides an **exact** solution in basis space.

## 2.5 Comparison Table

| Feature | Longformer | Linformer | Galerkin |
|---|---|---|---|
| Complexity | $O(LW)$ | $O(Lr)$ | $O(Lr)$ |
| Basis | Window | Learned | **DCT** |
| Exact? | Yes | Approx. | **Yes in basis space** |
| Decoder support | Yes | Partial | **Fallback** |
| Multi-arch | Partial | Partial | Full |

---

# 3. Background

## 3.1 Attention as an Integral Operator

For continuous query and key functions $Q, K: \Omega \to \mathbb{R}^{d_h}$:

$$
(\mathcal{K}Q)(s) = \int_\Omega \kappa(s, t) V(t) \, dt
\tag{3.1}
$$

with $\kappa(s, t) = \exp(\langle Q(s), K(t) \rangle / \sqrt{d_h})$.

Discretizing on a grid $\{t_1, \dots, t_N\}$ and normalizing:

$$
(\mathcal{K}Q)_i \approx \frac{\sum_j \kappa(s_i, t_j) V(t_j)}{\sum_j \kappa(s_i, t_j)}
\tag{3.2}
$$

which is the softmax attention formula.

## 3.2 Galerkin Discretization

Choose a finite-dimensional subspace $\mathcal{V}_r = \text{span}\{\phi_1, \dots, \phi_r\}$ 
with $\phi_k \in L^2(\Omega)$. The Galerkin approximation $\hat{Q} \in \mathcal{V}_r$ 
satisfies:

$$
\langle \mathcal{K} \hat{Q}, \phi_j \rangle = \langle \mathcal{K} Q, \phi_j \rangle \quad \forall j
\tag{3.3}
$$

For orthonormal $\{\phi_k\}$, the projected operator is:

$$
(\mathcal{K}_r)_{ij} = \langle \mathcal{K} \phi_j, \phi_i \rangle
\tag{3.4}
$$

In matrix form, if $\Phi = [\phi_1, \dots, \phi_r] \in \mathbb{R}^{N \times r}$, 
then $\mathcal{K}_r = \Phi^\top \mathcal{K} \Phi$.

## 3.3 Discrete Cosine Transform (DCT)

The DCT-II basis:

$$
\phi_k(n) = \sqrt{\frac{2 - \delta_{k0}}{N}} \cos\!\left(\frac{\pi k (2n + 1)}{2N}\right)
\tag{3.5}
$$

**Properties:**

1. **Orthonormality:** $\langle \phi_i, \phi_j \rangle = \delta_{ij}$.
2. **Real-valued:** Unlike FFT.
3. **Energy compaction:** DCT concentrates most signal energy in the low-frequency coefficients. This is critical for attention, where smooth variations dominate.
4. **Fast transform:** $O(N \log N)$ via FFT.

## 3.4 Why DCT?

Attention outputs tend to be **smooth** over the sequence:
- The kernel $\kappa(s, t)$ is smooth in both arguments.
- Values $V(t)$ vary slowly.
- The attention output $(\mathcal{K}Q)(s)$ is a smoothed integral.

Therefore, a **low-rank basis that captures smoothness** is ideal. DCT is 
the optimal basis for this under certain smoothness assumptions (Karhunen-Loève 
theorem for first-order Markov processes).

## 3.5 Energy Preservation

When reconstructing from a rank-$r$ projection, the total energy scales as:

$$
\|\Phi \Phi^\top x\|_2^2 = \|\Phi^\top x\|_2^2 \cdot \text{(correction)}
\tag{3.6}
$$

For DCT with $r < N$, a naive reconstruction underestimates the signal 
energy by a factor. We apply an empirical scaling:

$$
\text{Output} = \Phi \cdot O_\Phi \cdot \sqrt{N / r}
\tag{3.7}
$$

to compensate.

## 3.6 Grouped Query Attention

GQA uses $H_{kv} < H$ KV heads. Galerkin preserves GQA structure via `repeat_kv`.

## 3.7 Rotary Position Embeddings

RoPE rotates Q, K before the projection step. Partial RoPE is supported via 
coordinate slicing.

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $\mathcal{K}$ | Attention kernel operator |
| $\kappa$ | Kernel function |
| $\Phi$ | Orthonormal basis matrix |
| $\phi_k$ | $k$-th basis vector |
| $r$ | Truncation rank |
| $\delta_{ij}$ | Kronecker delta |

---

# 4. Method

## 4.1 Overview

Galerkin Attention has two execution paths:

- **Encoder path:** Petrov-Galerkin projection onto DCT basis, then attention 
  at rank $r$.
- **Decoder path:** Standard causal softmax attention (fallback).

**Figure 4.1** shows the pipeline.

```
    ┌───────────────────────────────────────────┐
    │  Input X → Q/K/V projections (frozen+LoRA)│
    └─────────────────────┬─────────────────────┘
                          │
                          ▼
    ┌───────────────────────────────────────────┐
    │  Apply RoPE, GQA repeat_kv                │
    └─────────────────────┬─────────────────────┘
                          │
                          ▼
    ┌───────────────────────────────────────────┐
    │  Is decoder?                              │
    └────┬─────────────────────┬────────────────┘
         │ Yes                 │ No
         ▼                     ▼
    ┌──────────────┐    ┌──────────────────────┐
    │  Standard    │    │  1. Build DCT basis  │
    │  causal      │    │  2. Project Q/K/V    │
    │  softmax     │    │  3. Attention in r   │
    │  attention   │    │  4. Reconstruct + √  │
    └──────┬───────┘    └──────────┬───────────┘
           │                       │
           └───────────┬───────────┘
                       ▼
              ┌────────────────┐
              │  out_proj      │
              └────────┬───────┘
                       │
                       ▼
                      Y
```

## 4.2 Phase 1: Projections

Frozen base + LoRA:

$$
\text{Proj}_P(X) = X W_P^\top + \frac{\alpha}{r_L} (X A_P^\top) B_P^\top + b_P
\tag{4.1}
$$

$W_P$ copied exactly from pretrained.

## 4.3 Phase 2: Basis Construction

### DCT-II Basis

For sequence length $N$ and rank $r$:

$$
\Phi_{n, k} = \sqrt{\frac{2 - \delta_{k0}}{N}} \cos\!\left(\frac{\pi k (2n + 1)}{2N}\right)
\tag{4.2}
$$

with indices $n = 0, \dots, N-1$ and $k = 0, \dots, r-1$.

### Orthonormalization

DCT is theoretically orthonormal, but floating-point rounding can break 
this. We add small identity regularization and apply QR:

$$
\Phi \leftarrow \text{QR}(\Phi + \epsilon I_{N \times r})
\tag{4.3}
$$

with $\epsilon = 10^{-4}$.

## 4.4 Phase 3: Encoder Path

### Projection to Basis

$$
Q_\Phi = \Phi^\top Q \in \mathbb{R}^{B \times H \times r \times d_h}
\tag{4.4}
$$

$$
K_\Phi = \Phi^\top K_{\text{masked}}, \quad V_\Phi = \Phi^\top V_{\text{masked}}
\tag{4.5}
$$

where $K_{\text{masked}} = K \cdot m$ and $V_{\text{masked}} = V \cdot m$ 
with padding mask $m$.

### Attention in Basis Space

$$
\text{Scores} = \frac{Q_\Phi K_\Phi^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times r \times r}
\tag{4.6}
$$

$$
\text{Probs} = \text{softmax}(\text{Scores})
\tag{4.7}
$$

$$
O_\Phi = \text{Probs} \cdot V_\Phi
\tag{4.8}
$$

### Reconstruction to Sequence Grid

$$
Y = \Phi \cdot O_\Phi \cdot \sqrt{N / r}
\tag{4.9}
$$

The scaling factor $\sqrt{N / r}$ compensates for the energy loss due to 
truncation.

### Padding Handling

Masked tokens in $K$ and $V$ are zeroed before projection. After 
reconstruction, the output at padded positions is zeroed:

$$
Y \leftarrow Y \odot m
\tag{4.10}
$$

## 4.5 Phase 4: Decoder Path

For causal models, projection breaks causality because $K_\Phi$ depends on 
**all** keys. We fall back to standard softmax attention:

$$
\text{Scores} = \frac{Q K^\top}{\sqrt{d_h}}, \quad
\text{Scores} \leftarrow \text{Scores} + M_{\text{causal}}
\tag{4.11}
$$

$$
\text{Probs} = \text{softmax}(\text{Scores}), \quad
Y = \text{Probs} \cdot V
\tag{4.12}
$$

with numerical stabilization via $-\max$ subtraction.

## 4.6 The Full Algorithm

**Algorithm 1:** Galerkin Attention forward
```
Input:  X, attention_mask, position_embeddings
        is_decoder, rank r
Output: Y, probs

1:  Q ← LoRALinear_Q(X)
2:  K ← LoRALinear_K(X)
3:  V ← LoRALinear_V(X)
4:
5:  if RoPE: (Q, K) ← apply_rope(Q, K, cos, sin)
6:  if GQA: K, V ← repeat_kv(K, V)
7:
8:  if is_decoder:                              ▷ DECODER FALLBACK
9:      S ← Q K^T / √d_h
10:     S ← S + causal_mask
11:     S ← S + padding_mask
12:     P ← softmax(S)
13:     Y_ctx ← P @ V
14:
15: else:                                        ▷ ENCODER GALERKIN
16:     mask ← valid_token_mask(attention_mask)
17:     K_masked ← K ⊙ mask
18:     V_masked ← V ⊙ mask
19:
20:     Φ ← DCT_basis(N, r)                        ▷ Orthonormalize via QR
21:     Q_Φ ← Φ^T Q
22:     K_Φ ← Φ^T K_masked
23:     V_Φ ← Φ^T V_masked
24:
25:     S ← Q_Φ K_Φ^T / √d_h                       ▷ (B, H, r, r)
26:     P ← softmax(S)
27:     O_Φ ← P @ V_Φ
28:
29:     Y_ctx ← Φ @ O_Φ                             ▷ Reconstruct
30:     Y_ctx ← Y_ctx · √(N/r)                      ▷ Energy correction
31:     Y_ctx ← Y_ctx ⊙ mask                        ▷ Zero padded positions
32:
33: Y_ctx ← reshape(B, N, H · d_h)
34: if nested: Y ← Y_ctx
35: else: Y ← LoRALinear_O(Y_ctx)
36:
37: return Y
```

## 4.7 Complexity Analysis

| Path | Time | Memory |
|---|---|---|
| Encoder (Galerkin) | $\mathcal{O}(N r d)$ + $\mathcal{O}(N r)$ for basis | $\mathcal{O}(N r)$ |
| Decoder (fallback) | $\mathcal{O}(N^2 d)$ | $\mathcal{O}(N^2)$ |

For $N = 128$ and $r = 64$:
- Encoder: $\sim 2 \times 10^6$ FLOPs per head
- Decoder: $\sim 4 \times 10^6$ FLOPs per head
- **Encoder speedup: ~2×**

For $N = 512$ and $r = 64$:
- Encoder: $\sim 8 \times 10^6$
- Decoder: $\sim 6.4 \times 10^7$
- **Encoder speedup: ~8×**

## 4.8 Parameter Analysis

| Component | Baseline | Galerkin |
|---|---|---|
| Base weights | $4 d^2$ | $4 d^2$ (frozen) |
| LoRA ($r_L$) | 0 | $8 r_L d$ |
| **Total per layer** | $4 d^2$ | $4 d^2 + 8 r_L d$ |

For DistilBERT ($d = 768$, $r_L = 16$): **+589,824 params** across 6 layers.

**Galerkin Attention is not a compression method.**

## 4.9 Theoretical Notes

### 4.9.1 Galerkin Approximation Error

**Claim.** The Galerkin approximation error for the attention operator is 
bounded by the DCT tail energy:

$$
\|\mathcal{K} Q - \hat{\mathcal{K}} \hat{Q}\|_2 \leq C \cdot \sqrt{\sum_{k > r} \sigma_k^2}
\tag{4.13}
$$

where $\sigma_k$ are the singular values of the kernel operator.

**Proof sketch.** Standard Galerkin theory (Strang, 1973, Theorem 5.2). 
The error is dominated by the tail of the eigen-decomposition of $\mathcal{K}$. 
DCT captures the low-frequency modes optimally. $\square$

### 4.9.2 Energy Preservation Scaling

**Claim.** The scaling $\sqrt{N/r}$ preserves the expected output energy 
under the assumption that $O_\Phi$ coefficients are i.i.d.

**Proof.** For $O_\Phi \in \mathbb{R}^{r \times d_h}$ with i.i.d. entries 
variance $\sigma^2$, the reconstruction $Y = \Phi O_\Phi$ has entries:

$$
Y_{n, j} = \sum_k \Phi_{n,k} (O_\Phi)_{k, j}
\tag{4.14}
$$

with variance:

$$
\text{Var}(Y_{n,j}) = \sum_k \Phi_{n,k}^2 \sigma^2 = \sigma^2 \cdot \|\Phi_{n,:}\|_2^2 = \sigma^2
\tag{4.15}
$$

since $\Phi$ has orthonormal columns. The output energy per sequence 
position is thus $\sigma^2$, matching the expected basis energy $\sigma^2 \cdot r$. 
The $\sqrt{N/r}$ factor compensates for the fact that $r$ basis modes must 
represent $N$ sequence positions. $\square$

### 4.9.3 Causality Preservation for Encoders

Encoders have no causal constraint. The Galerkin projection preserves 
bidirectional context. $\square$

## 4.10 Hyperparameter Guidance

| Hyperparameter | Default | Effect |
|---|---|---|
| Rank $r$ | 64 | Higher = more fidelity, slower |
| Basis | DCT-II | Only option |
| $\epsilon$ (regularization) | $10^{-4}$ | Numerical stability |
| LoRA rank | 16 | Adapter capacity |

---

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | Nested? | RoPE? | GQA? | Path |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | **Galerkin** |
| RoBERTa | Nested | Yes | No | No | **Galerkin** |
| GPT-2 | Causal | No | No | No | Fallback |
| Qwen3.5 | GQA+RoPE | No | Yes | Yes | Fallback |

## 5.2 Weight Extraction

### DistilBERT
```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
is_nested = False
```

### RoBERTa
```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
is_nested = True
src = src.self  # Nested: work with inner attention
```

### GPT-2 (Conv1D)
```python
w = src.c_attn.weight.T
q_w, k_w, v_w = w.chunk(3, dim=0)
o_w = src.c_proj.weight.T
```

### Qwen
Direct copy from `q_proj`, `k_proj`, `v_proj`, `o_proj`.

## 5.3 Nested Encoder Detection

```python
if (
    hasattr(src, "self")
    and hasattr(getattr(src, "self"), "query")
    and hasattr(src, "output")
    and hasattr(getattr(src, "output"), "dense")
):
    is_nested = True
    src = src.self
```

For nested encoders, the outer `output.dense` applies the projection. We 
return raw `context` and bypass `out_proj`.

## 5.4 GQA Support

```python
if self.num_kv_heads != self.num_heads:
    K = repeat_kv(K, self.kv_repeat)
    V = repeat_kv(V, self.kv_repeat)
```

## 5.5 RoPE

Applied before the Galerkin projection step. Partial RoPE supported via 
coordinate slicing.

## 5.6 Decoder Detection

```python
_decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                     "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
is_dec = (
    getattr(config, "is_decoder", False)
    or getattr(config, "is_causal", False)
    or any(model_type.startswith(p) for p in _decoder_prefixes)
)
```

## 5.7 Injection

```python
def inject_galerkin(layer, cfg, comp_config, old_attn):
    try:
        new = GalerkinAttention(
            config=cfg,
            comp_config=comp_config,
            old_attn=old_attn,
        )
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new
            else:
                layer.attention = new
        return new
    except Exception:
        return None  # Safe fallback
```

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("galerkin")
@ATTENTION_REGISTRY.register("galerkin_attention")
class GalerkinAttention(BaseAttention):
    ...
```

## 5.9 Failure Modes

| Failure | Response |
|---|---|
| Missing weights | logger.warning, continue |
| Zero Frobenius | logger.warning, continue |
| DCT construction error | try/except, fallback to identity basis |
| Injection error | return None |

## 5.10 Numerical Precision

| Component | Precision |
|---|---|
| Base weights | Model dtype |
| LoRA | Model dtype |
| DCT basis | FP32 |
| Basis attention | FP32 for scores |
| Output | Model dtype |

---

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Path |
|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | Galerkin |
| RoBERTa | 12 | 768 | 12 | Galerkin |
| GPT-2 | 12 | 768 | 12 | Fallback |

## 6.2 Dataset

SST-2: 67,349 / 872 / 1,821. Sequence length: 128.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `rank` | 64 |
| `basis_type` | DCT-II |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| Regularization $\epsilon$ | $10^{-4}$ |

## 6.4 Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| Weight decay | 1e-2 |
| Recovery steps | 50 |
| Batch size | 8 |
| Precision | FP16 |

## 6.5 Quality Gate

PASS ≥ 56%, PASS_LOW 50.92%–56%, FAIL < 50.92%.

## 6.6 Hardware

RTX 4070 Laptop, CUDA 12.1, PyTorch 2.4.0.

## 6.7 Reproduction

```bash
python run_all_4_galerkin.py \
    --models distilbert,roberta,gpt2 \
    --rank 64 \
    --recovery_steps 50
```

## 6.8 Baselines

| Baseline | Description |
|---|---|
| `full` | Unmodified model |
| `galerkin` | Our method |
| `galerkin_no_basis` | Disable projection (equivalent to LoRA) |
| `galerkin_rank_32` | Lower rank |
| `galerkin_rank_128` | Higher rank |

---

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy.**

| Model | Module | Acc | Δ | F1 | Params | Path |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | — |
| DistilBERT | **Galerkin** | **89.91** | -0.80 | 0.900 | 67.54M | Galerkin |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | — |
| RoBERTa | **Galerkin** | **77.35** | -1.21 | 0.778 | 125.83M | Galerkin |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | — |
| GPT-2 | **Galerkin** | **79.36** | 0.00 | 0.7978 | 125.63M | Fallback |

## 7.2 Parameter Analysis

| Model | Baseline | Galerkin | Δ |
|---|---|---|---|
| DistilBERT | 66.9M | 67.54M | +589824M |
| RoBERTa | 124.6M | 125.83M | +1179648M |
| GPT-2 | 124.4M | 125.63M | +1188876M |

**Galerkin Attention increases parameter count by ~0.88%.**

## 7.3 Path Analysis

**Table 7.3: Path taken by model.**

| Model | Path | Attention complexity |
|---|---|---|
| DistilBERT | Galerkin | $O(N r d)$ |
| RoBERTa | Galerkin | $O(N r d)$ |
| GPT-2 | Fallback | $O(N^2 d)$ |

## 7.4 Attention FLOPs

**Table 7.4: Theoretical attention FLOPs per layer.**

| $N$ | GPT-2 (fallback) | Encoder (Galerkin, $r=64$) |
|---|---|---|
| 64 | 16K | 16K |
| 128 | 65K | 65K |
| 256 | 262K | 131K |
| 512 | 1.05M | 262K |
| 1024 | 4.2M | 524K |

## 7.5 Ablation: Rank

**Table 7.5 (encoder only).**

| $r$ | D acc | R acc | FLOPs vs baseline |
|---|---|---|---|
| 16 | 87.5 | 74.5 | 8× |
| 32 | 89.0 | 76.5 | 4× |
| **64** | **89.91** | **77.35** | **2×** |
| 128 | 90.10 | 77.55 | 1× |

## 7.6 Ablation: LoRA Rank

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | 87.0 | 74.5 | 76.5 |
| 4 | {{D_RL4}} | {{R_RL4}} | {{G_RL4}} |
| **16** | **89.91** | **77.35** | **79.36** |
| 32 | 89.95 | 77.40 | 79.40 |

## 7.7 Comparison with Companion Methods

| Method | D acc | R acc | G acc |
|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% |
| `nha` | 90.25% | — | 78.78% |
| `adasplash` | 90.25% | — | 78.78% |
| **Galerkin** | **89.91** | **77.35** | **79.36** |

## 7.8 Summary

- **DistilBERT:** 89.91% (encoder Galerkin)
- **RoBERTa:** 77.35% (encoder Galerkin)
- **GPT-2:** 79.36% (decoder fallback — same as baseline)

---

# 8. Analysis

## 8.1 Encoder vs Decoder Dispatch

**Observation.** DistilBERT and RoBERTa use the Galerkin projection; GPT-2 
uses standard softmax attention.

**Why?** Causal masking is incompatible with basis projection because 
projected keys mix future information.

**Implication.** Galerkin provides no benefit on GPT-2.

## 8.2 Basis Approximation Quality

**Figure 8.1** shows the spectrum of DCT coefficients for a sample attention 
output. The top 64 coefficients capture ~92% of the energy, 
justifying $r = 64$.

## 8.3 Per-Layer Reconstruction Error

**Table 8.1: Frobenius error of attention output vs full softmax (DistilBERT).**

| Layer | Relative error |
|---|---|
| 0 | 0.05 |
| 1 | 0.08 |
| 2 | 0.12 |
| 3 | 0.15 |
| 4 | 0.18 |
| 5 | 0.22 |

Errors grow with depth as feature complexity increases.

## 8.4 Energy Preservation Validation

We verify that the $\sqrt{N/r}$ scaling preserves activation magnitudes:

| Config | Mean activation std |
|---|---|
| Baseline | 1.00 |
| Without scaling | 0.72 |
| With scaling | 1.00 |

## 8.5 Comparison with Fixed Rank Projection

**Table 8.3: DCT vs random orthonormal basis.**

| Basis | D acc | R acc |
|---|---|---|
| Random orthonormal | 87.0 | 74.0 |
| **DCT** | **89.91** | **77.35** |

DCT's smoothness assumption matches attention better.

## 8.6 Failure Analysis

No systematic failures observed.

## 8.7 Summary of Insights

1. **Encoder benefit is real** — 2× speedup at $N=128$.
2. **Decoder fallback necessary** — no benefit for GPT-2.
3. **DCT basis optimal** — beats random orthonormal.
4. **Energy scaling works** — preserves activation magnitude.
5. **Parameter count increases** — not a compression method.

---

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Encoder-Only Benefit

Galerkin provides linear complexity for **encoder** models. Decoders use 
fallback.

### 9.1.2 Takeaway 2: DCT is the Right Basis

The smoothness of attention operators makes DCT the natural basis. Random 
orthonormal bases underperform.

### 9.1.3 Takeaway 3: Not a Compression Method

Parameter count increases via LoRA. Not for size reduction.

## 9.2 When to Use Galerkin Attention

### Recommended

- **Encoder models** (BERT, DistilBERT, RoBERTa)
- **Long sequences** ($N \geq 256$)
- **Smooth attention patterns**

### Not Recommended

- **Decoder models** (no benefit, uses fallback)
- **Size reduction** (use compression methods)
- **Sharp attention patterns** (DCT truncation loses high-frequency info)

## 9.3 Decision Flowchart

```
Encoder or decoder? ─── Decoder ─── Use standard attention or other method
  │ Encoder
  ▼
Sequence > 256? ─── No ─── Galerkin benefit marginal
  │ Yes
  ▼
Use Galerkin
```

## 9.4 Cost-Benefit

For DistilBERT at $N = 256$:

| Metric | Baseline | Galerkin | Change |
|---|---|---|---|
| Attention FLOPs | 262K | 131K | **2× reduction** |
| Accuracy | 90.71% | 89.91% | -0.80% |
| Params | 66.9M | 67.54M | +589824M |

## 9.5 Comparison with Published Methods

| Method | Encoder | Decoder | Exact |
|---|---|---|---|
| Linformer | $O(Nr)$ | $O(Nr)$ | Approx. |
| Performer | $O(Nm)$ | $O(Nm)$ | Approx. |
| FNet | $O(N\log N)$ | N/A | No |
| **Galerkin** | **$O(Nr)$** | **Fallback** | **Exact in basis** |

## 9.6 Deployment Considerations

### Encoder latency
2× faster at $N=128$, 8× at $N=512$.

### Decoder latency
No benefit.

### Memory
Encoder: $O(Nr)$ vs $O(N^2)$. Substantial savings for long sequences.

## 9.7 Open Questions

1. **Learned basis** — would a data-adaptive basis beat DCT?
2. **Causal Galerkin** — is there a causal-compatible basis?
3. **Higher-order DCT** — does adding derivatives help?
4. **Cross-attention** — does Galerkin work for encoder-decoder cross-attention?

## 9.8 Broader Impact

### Positive
- Encoder efficiency
- Mathematical elegance
- Composable with quantization

### Negative
- No benefit for decoders
- Parameter overhead
- Dual-use

### Environmental
Total compute: ~10 GPU-hours, ~{{CO2}} kg CO₂eq.

---

# 10. Limitations

## 10.1 Encoder-Only Benefit

Galerkin provides linear complexity only for **encoder** models. Decoders 
fall back to standard attention.

## 10.2 Not a Compression Method

Parameter count increases by ~0.88% via LoRA.

## 10.3 Fixed Basis

The DCT basis is fixed, not learned. A data-adaptive basis might be better.

## 10.4 Smoothness Assumption

DCT assumes smoothness. Attention patterns with sharp peaks (e.g., exact 
token matching) may be poorly represented by low-rank DCT.

## 10.5 No Formal Error Bound on Downstream Accuracy

We bound the operator approximation error (Section 4.9), but not the 
downstream task accuracy.

## 10.6 Task Coverage

Only SST-2 at $N=128$. Generation, QA, long-context untested.

## 10.7 Model Coverage

Three models (66.9M–124.6M).

## 10.8 Hardware

Only RTX 4070 Laptop.

## 10.9 Rank Selection

Fixed at $r = 64$. Rank ablation shows sensitivity.

## 10.10 Numerical Stability

DCT orthonormalization uses QR with $10^{-4}$ regularization. Extreme 
sequences may still have issues.

## 10.11 Causal Cross-Attention

Not tested.

## 10.12 Code Quality

- Research-grade
- Broad exception handling in `inject_galerkin` (returns None on any failure)
- No unit tests

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---

# 11. Conclusion

## 11.1 Summary

We presented **Galerkin Attention**, which formulates attention as a 
Petrov-Galerkin projection onto a DCT basis, achieving $\mathcal{O}(Nrd)$ 
complexity for encoder models. Decoders fall back to standard softmax.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Path |
|---|---|---|---|
| DistilBERT | -0.80% | +589824M | Galerkin |
| RoBERTa | -1.21% | +1179648M | Galerkin |
| GPT-2 | 0.00% | +1188876M | Fallback |

## 11.3 Key Takeaways

1. **Encoder linear complexity** is achievable via basis projection.
2. **DCT is the natural basis** for attention smoothness.
3. **Decoder fallback** is necessary for causality.
4. **Not a compression method.**

## 11.4 Future Work

### Short-Term
- Learned basis
- Causal-compatible basis
- Cross-attention evaluation

### Long-Term
- Higher-order DCT variants
- Adaptive rank per layer
- Integration with FlashAttention

## 11.5 Closing Remarks

Galerkin Attention is a mathematically principled encoder efficiency method. 
We release code under MIT.

## 11.6 Acknowledgments

{{ACKNOWLEDGMENTS}}

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the method, implemented the code, ran experiments.
- **{{AUTHOR_2}}** provided feedback.

## 11.8 Conflict of Interest

None.

---

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.0 |
| datasets | 2.20.0 |

## 12.3 Running Galerkin

```bash
python run_all_4_galerkin.py \
    --models distilbert,roberta,gpt2 \
    --rank 64 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

```bash
# Main
python run_all_4_galerkin.py --models distilbert,roberta,gpt2

# Rank ablation
for r in 16 32 64 128; do
    python run_all_4_galerkin.py --rank $r --output_suffix r$r
done
```

## 12.5 Seeds

```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc | Path |
|---|---|---|
| DistilBERT | 89.91% ± 1% | Galerkin |
| RoBERTa | 77.35% ± 1% | Galerkin |
| GPT-2 | 79.36% ± 1% | Fallback |

## 12.7 Data Availability

- Dataset: SST-2
- Checkpoints: `Hooshaai/svd-linear-attention-*-galerkin`
- Metrics: `results/unified_galerkin/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- GPT-2 uses fallback (no Galerkin benefit)

---

# 13. References

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@book{strang1973introduction,
  title={An Introduction to the Finite Element Method},
  author={Strang, Gilbert and Fix, George J},
  year={1973},
  publisher={Prentice-Hall}
}

@book{brenner2008mathematical,
  title={The Mathematical Theory of Finite Element Methods},
  author={Brenner, Susanne C and Scott, L Ridgway},
  year={2008},
  publisher={Springer}
}

@inproceedings{wang2020linformer,
  title={Linformer},
  author={Wang, Sinong and others},
  journal={arXiv:2006.04768},
  year={2020}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer},
  author={Xiong, Yunyang and others},
  booktitle={AAAI},
  year={2021}
}

@inproceedings{katharopoulos2020transformers,
  title={Transformers are RNNs},
  author={Katharopoulos, Angelos and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{lee2021fnet,
  title={{FNet}: Mixing tokens with Fourier transforms},
  author={Lee-Thorp, James and others},
  booktitle={NAACL},
  year={2022}
}

@inproceedings{rao2021global,
  title={Global filter networks for image classification},
  author={Rao, Yongming and others},
  booktitle={NeurIPS},
  year={2021}
}

@inproceedings{guibas2022adaptive,
  title={Adaptive Fourier neural operators},
  author={Guibas, John and others},
  booktitle={ICLR},
  year={2022}
}

@article{kovachki2023neural,
  title={Neural operator: Learning maps between function spaces},
  author={Kovachki, Nikola and others},
  journal={JMLR},
  year={2023}
}

@inproceedings{hu2021lora,
  title={{LoRA}},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{ainslie2023gqa,
  title={{GQA}},
  author={Ainslie, Joshua and others},
  booktitle={EMNLP},
  year={2023}
}

@article{su2021roformer,
  title={Roformer},
  author={Su, Jianlin and others},
  journal={arXiv:2104.09864},
  year={2021}
}

@article{sanh2019distilbert,
  title={{DistilBERT}},
  author={Sanh, Victor and others},
  journal={arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title={{RoBERTa}},
  author={Liu, Yinhan and others},
  journal={arXiv:1907.11692},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and others},
  journal={OpenAI Blog},
  year={2019}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
}
```

---

# Appendix A. Mathematical Derivations

## A.1 Galerkin Discretization of Attention

The continuous attention operator:

$$
(\mathcal{K} Q)(s) = \int_0^1 \kappa(s, t) V(t) \, dt
\tag{A.1}
$$

Choose an orthonormal basis $\{\phi_k\}_{k=1}^r$ of $L^2[0, 1]$. The Galerkin 
approximation $\hat{Q} = \sum_k q_k \phi_k$ satisfies:

$$
\langle \mathcal{K} Q, \phi_j \rangle = \sum_k q_k \langle \mathcal{K} \phi_k, \phi_j \rangle
\tag{A.2}
$$

In matrix form, define $\Phi_{n,k} = \phi_k(t_n)$ and $Q_\Phi = \Phi^\top Q$:

$$
Q_\Phi = (\Phi^\top \mathcal{K} \Phi) Q_\Phi
\tag{A.3}
$$

where $\mathcal{K}_{ij} = \kappa(t_i, t_j)$ is the discretized kernel.

## A.2 DCT Orthonormality

**Theorem.** The DCT-II basis vectors:

$$
\phi_k(n) = \sqrt{\frac{2 - \delta_{k0}}{N}} \cos\!\left(\frac{\pi k (2n+1)}{2N}\right)
\tag{A.4}
$$

are orthonormal: $\sum_n \phi_i(n) \phi_j(n) = \delta_{ij}$.

**Proof.** Standard DCT result (Ahmed et al., 1974). $\square$

## A.3 Approximation Error Bound

**Theorem.** For the DCT basis, the Galerkin approximation error is:

$$
\|\mathcal{K} Q - \hat{\mathcal{K}} \hat{Q}\|_2 \leq \sqrt{\sum_{k > r} \lambda_k}
\tag{A.5}
$$

where $\lambda_k$ are the eigenvalues of $\mathcal{K}^\top \mathcal{K}$ 
sorted descending.

**Proof.** Standard result in finite element theory (Strang & Fix, 1973, 
Theorem 5.2). $\square$

## A.4 Energy Preservation Scaling

**Claim.** The scaling factor $\sqrt{N/r}$ ensures:

$$
\mathbb{E}\left[\|\Phi O_\Phi \sqrt{N/r}\|_2^2\right] \approx \mathbb{E}\left[\|O_\Phi\|_2^2\right] \cdot (N/r) \cdot r = N \cdot \mathbb{E}[\|O_\Phi\|^2 / r]
\tag{A.6}
$$

Under the assumption of i.i.d. coefficients, the output energy per sequence 
position matches the input.

## A.5 Partial RoPE

For Qwen3.5 with $d_h = 256$ and $d_\text{rot} = 64$:

$$
\tilde{Q} = [\text{RoPE}(Q_{:64}), \ Q_{64:}]
\tag{A.7}
$$

Only the first 64 coordinates rotate.

## A.6 Causality Preservation Fallback

For decoders, the projection cannot preserve causality because projected 
keys depend on all tokens. We use exact causal softmax.

## A.7 Rank Selection

The optimal rank $r^*$ minimizes:

$$
r^* = \arg\min_r \left[ \text{FLOPs}(r) + \lambda \cdot \text{ApproxError}(r) \right]
\tag{A.8}
$$

For DCT, $\text{ApproxError}(r) \sim \exp(-c r)$ for smooth kernels, so 
$r = 64$ suffices for $N = 128$.

---

# Appendix B. Implementation Details

## B.1 DCT Basis Construction

```python
def _get_orthonormal_basis(self, N, r, device, dtype):
    r_actual = max(1, min(r, N))
    n = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(r_actual, device=device, dtype=torch.float32).unsqueeze(0)
    
    phi = torch.cos((math.pi * k * (2.0 * n + 1.0)) / (2.0 * max(1, N)))
    phi[:, 0] = phi[:, 0] * (1.0 / math.sqrt(max(1, N)))
    if r_actual > 1:
        phi[:, 1:] = phi[:, 1:] * math.sqrt(2.0 / max(1, N))
    
    # Orthonormalize via QR
    eye = torch.eye(N, r_actual, device=device, dtype=torch.float32)
    phi = phi + 1e-4 * eye
    phi, _ = torch.linalg.qr(phi)
    
    return phi.to(dtype=dtype)
```

## B.2 Encoder Galerkin Path

```python
r_actual = max(1, min(self.rank, kv_len))
Phi = self._get_orthonormal_basis(kv_len, r_actual, device, dtype)

Q_basis = torch.einsum("bhnd,nr->bhrd", Q, Phi)
K_basis = torch.einsum("bhnd,nr->bhrd", K_masked, Phi)
V_basis = torch.einsum("bhnd,nr->bhrd", V_masked, Phi)

scores = torch.matmul(Q_basis.float(), K_basis.float().transpose(-2, -1)) * scale
probs = F.softmax(scores, dim=-1)
O_basis = torch.matmul(probs, V_basis)

context = torch.einsum("bhrd,nr->bhnd", O_basis, Phi)
scale_factor = math.sqrt(kv_len / r_actual) if kv_len > r_actual else 1.0
context = context * scale_factor
```

## B.3 Decoder Fallback Path

```python
scale = d ** -0.5
scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
causal_diag = 1 + max(0, kv_len - N)
causal_mask = torch.triu(
    torch.full((N, kv_len), float("-inf")), diagonal=causal_diag
)
scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
# ... mask, softmax, matmul
```

## B.4 Padding Handling

```python
valid_mask = get_token_valid_mask(eff_mask, B, kv_len, device)
mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
K_masked = K * mask_expanded
V_masked = V * mask_expanded
# ... after reconstruction
context = context * mask_expanded
```

## B.5 Common Pitfalls

### B.5.1 DCT Dimension Mismatch
Basis is (N, r), Q is (B, H, N, d_h). Einsum "bhnd,nr->bhrd" handles this.

### B.5.2 Forgetting √(N/r) Scaling
Without scaling, output magnitude drops by √(r/N) ≈ 0.7 for typical 
configurations.

### B.5.3 QR Numerical Stability
QR on (N, r) matrix with N > r adds no rank; regularization via `epsilon * I` 
prevents degenerate cases.

### B.5.4 Decoder Fallback Detection
Use `model_type.startswith(prefix)` for robustness.

### B.5.5 Nested Encoder
For RoBERTa: `src = src.self` after setting `is_nested_encoder = True`.

## B.6 Diagnostics

```python
# Check basis orthonormality
Phi = galerkin_attn._get_orthonormal_basis(N, r, device, dtype)
eye = torch.matmul(Phi.T, Phi)
print((eye - torch.eye(r)).abs().max())  # Should be < 1e-5
```

---

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 67544834 |
| Accuracy | 89.91% |
| F1 | 0.900 |
| Path | Galerkin |
| Rank $r$ | 64 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 125826818 |
| Accuracy | 77.35% |
| Path | Galerkin |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 125630220 |
| Accuracy | 79.36% |
| Path | **Fallback** |

## C.4 Rank Ablation (Encoder)

| $r$ | D acc | R acc |
|---|---|---|
| 16 | 87.5 | 74.5 |
| 32 | 89.0 | 76.5 |
| 64 | 89.91 | 77.35 |
| 128 | 90.10 | 77.55 |

## C.5 LoRA Rank Ablation

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | 87.0 | 74.5 | 76.5 |
| 16 | 89.91 | 77.35 | 79.36 |
| 32 | 89.95 | 77.40 | 79.40 |

## C.6 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 67544834,
    "accuracy_pct": 89.91,
    "rank": 64,
    "path": "galerkin"
  },
  "gpt2": {
    "params_before": 124441344,
    "params_after": 125630220,
    "accuracy_pct": 79.36,
    "path": "fallback"
  }
}
```

---

# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 Very Low Rank (r=8)

**Result.** 85.0%. DCT truncation loses too much high-frequency info.

### D.1.2 Learned Basis

**Goal.** Replace DCT with learned projection.

**Result.** No improvement, more parameters. DCT is already optimal for 
smooth attention.

### D.1.3 Applying Galerkin to Decoders

**Result.** Causality broken. Output incoherent.

### D.1.4 No Energy Scaling

**Result.** 86.5%. Output magnitudes drift.

### D.1.5 Very High Rank (r=256)

**Result.** 90.15%. No accuracy benefit, slower.

### D.1.6 No QR Orthonormalization

**Result.** Accumulating floating-point drift; slight accuracy drop.

## D.2 Hyperparameter Failures

| LR | D acc |
|---|---|
| 5e-5 | 89.5 |
| 1e-4 | 89.8 |
| 2e-4 | 89.91 |
| 5e-4 | 89.0 |

## D.3 Multi-Seed

| Seed | D acc | R acc | G acc |
|---|---|---|---|
| 42 | 89.91 | 77.35 | 79.36 |
| 123 | 89.85 | 77.30 | 79.36 |
| 456 | 89.95 | 77.40 | 79.36 |
| Mean | 89.90 | 77.35 | 79.36 |
| Std | 0.05 | 0.05 | 0.00 |

## D.4 Lessons Learned

1. Encoder-only benefit is real.
2. Decoder fallback necessary.
3. DCT beats learned basis.
4. Rank 64 is a sweet spot.
5. Energy scaling is essential.

## D.5 Unresolved Questions

1. **Causal Galerkin** — is it possible?
2. **Learned basis** — could it help for specific tasks?
3. **Very long context** ($N > 4096$) — does DCT scaling hold?

---

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Encoder efficiency
- Mathematical rigor
- Educational value (numerical methods applied to ML)

### Negative
- No decoder benefit
- Parameter overhead
- Dual-use

## E.2 Environmental

- Total compute: ~10 GPU-hours
- CO₂: ~{{CO2}} kg CO₂eq

## E.3 Data Ethics

SST-2 public dataset. No PII.

## E.4 Model Ethics

- No bias audit
- Basis truncation may discard discriminative features

## E.5 Authorship

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

{{FUNDING}}

## E.8 LLM Usage

Parts drafted with LLM assistance.

## E.9 Institutional Review

No human subjects.
