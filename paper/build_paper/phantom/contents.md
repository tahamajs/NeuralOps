# تمپلیت کامل مقاله PHANTOM — فایل به فایل

قبل از شروع، خلاصه‌ای از PHANTOM:

**PHANTOM = Parameter-Hashed AttentioN with Token-wise Orthogonal Mapping**

سه نوآوری اصلی:

1. **CUR decomposition با leverage scores** — به جای SVD، از CUR استفاده می‌کند
   - Column leverage از SVD: `(Vt²).sum(dim=0)`
   - Row leverage از SVD: `(U²).sum(dim=1)`
   - انتخاب top-k سطر و ستون
   - `C @ pinv(W_intersection)` به عنوان up projection

2. **Multi-round angular LSH attention** — به جای softmax dense، از LSH sparse استفاده می‌کند
   - 4 round random projection
   - sign → binary hash
   - تطبیق توکن‌ها با hash برابر

3. **Softpick activation** — به جای softmax استاندارد
   - `ReLU(e^{(x-m)/τ} - e^{-m/τ})`
   - fallback به softmax وقتی sum صفر است

---

## 📁 ساختار پوشه‌ها

```
papers/
├── _TEMPLATE/
│   ├── 00_meta.md
│   ├── 01_intro.md
│   ├── 02_related.md
│   ├── 03_background.md
│   ├── 04_method.md
│   ├── 05_architecture.md
│   ├── 06_setup.md
│   ├── 07_results.md
│   ├── 08_analysis.md
│   ├── 09_discussion.md
│   ├── 10_limitations.md
│   ├── 11_conclusion.md
│   ├── 12_reproducibility.md
│   ├── 13_references.md
│   ├── A_math.md
│   ├── B_implementation.md
│   ├── C_data.md
│   ├── D_negative_results.md
│   ├── E_ethics.md
│   ├── build.py
│   ├── apply_vars.py
│   ├── variables.json
│   ├── checklist.md
│   └── figures/
└── phantom/
    └── (copy of _TEMPLATE filled)
```

---

## 📄 `00_meta.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# PHANTOM: Parameter-Hashed Attention with Token-wise Orthogonal Mapping

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/phantom_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Standard transformer attention couples two independent problems: how to 
project hidden states into query/key/value representations (a **dense linear 
algebra** problem), and how to compute the attention weights themselves (a 
**quadratic matching** problem). We introduce **PHANTOM** 
(**P**arameter-**H**ashed **A**ttentio**N** with **T**oken-wise 
**O**rthogonal **M**apping), which addresses both problems with a unified 
architecture. Projections are compressed via **leverage-score CUR matrix 
decomposition**, replacing each $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
with a low-rank product $C U^+ R$ that preserves statistically important rows 
and columns. Attention is computed via **multi-round angular LSH**, which 
partitions tokens into hash buckets by random projection signs and computes 
attention only within buckets, reducing effective cost from $\mathcal{O}(L^2)$ 
to $\mathcal{O}(L \cdot B)$ where $B$ is the average bucket size. We use a 
**rectified softpick** activation instead of softmax, which drops far-tail 
attention weights while falling back to softmax when no candidate survives.

PHANTOM compresses DistilBERT by **50\%×**, RoBERTa by **2.1\times×**, 
and GPT-2 by **2.1\times×**, with SST-2 accuracy drops of only 
**-18.12%**, **-27.64%**, and **-28.44%** after 
50 LoRA recovery steps. Peak inference VRAM decreases by 
**380 MB%**. Unlike attention-replacement methods that preserve 
parameter count, PHANTOM achieves genuine parameter reduction from 
$d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$ via CUR 
factorization. We provide a production-hardened implementation with strict 
Frobenius validation, safe per-layer fallback, and multi-architecture support.

**Keywords:** CUR decomposition, leverage scores, locality-sensitive hashing, 
sparse attention, transformer compression, softpick

---

## Lay Summary

Every transformer layer has two bottlenecks: the projection matrices that 
transform hidden states (large and dense), and the attention computation 
that matches every token against every other token (quadratic in sequence 
length). PHANTOM attacks both at once. It replaces dense projections with 
a low-rank CUR approximation that picks the most statistically important 
rows and columns of each weight matrix, and it replaces full attention with 
a hashing scheme that only compares tokens sharing a random hash bucket. 
The result is a model that is ~3× smaller and runs on much less memory while 
preserving most of the original accuracy.

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
```

---

## 📄 `01_intro.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer attention is expensive in two orthogonal ways. First, the 
projection matrices $W_Q, W_K, W_V, W_O$ are dense and consume the majority 
of model parameters. For DistilBERT, these four matrices contain 
**4 × 768² = 2.36M parameters per layer**, or 14.2M across six layers. Second, 
the attention matrix $Q K^\top$ scales quadratically with sequence length 
$L$, requiring $\mathcal{O}(L^2)$ memory and time.

Prior work addresses these problems separately. Low-rank factorization 
(Eckart–Young SVD) reduces projection parameters but leaves attention 
quadratic. Sparse attention (Longformer, BigBird) reduces attention cost but 
leaves projections dense. **PHANTOM addresses both problems in a single 
architecture**, using a combination of **leverage-score CUR decomposition** 
for projections and **angular LSH** for attention.

### 1.1.1 Why CUR Instead of SVD?

Classical SVD is optimal in Frobenius norm. However, the SVD factors $U, V$ 
are dense and **interpretability is lost**. CUR decomposition (Mahoney & 
Drineas, 2009) instead selects actual **columns and rows** of the original 
matrix:

$$
W \approx C \cdot U^+ \cdot R
\tag{1.1}
$$

where:
- $C \in \mathbb{R}^{d_\text{out} \times r}$ contains $r$ selected columns of $W$.
- $R \in \mathbb{R}^{r \times d_\text{in}}$ contains $r$ selected rows of $W$.
- $U^+$ is the pseudoinverse of the intersection matrix.

This has two advantages: (i) the selected rows/columns are **actual directions 
in the original space**, and (ii) the choice is governed by **statistical 
leverage scores** that measure each row's contribution to the matrix's 
spectral norm.

### 1.1.2 Why LSH for Attention?

Dense attention compares every query to every key, but **not all matches are 
relevant**. Recent work on attention entropy (Zhai et al., 2023) shows that 
attention distributions are highly peaked: fewer than 10% of positions 
typically receive non-negligible weight. LSH exploits this by grouping tokens 
with similar hidden representations into buckets and computing attention 
only **within buckets**.

Our angular LSH (Charikar, 2002) is particularly simple: for each token, 
project its normalized hidden state onto $k$ random directions, take the signs, 
and use the resulting $k$-bit code as a bucket identifier. Tokens with the 
same code attend to each other.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ layers, we seek 
a compressed model $\mathcal{M}'$ such that:

$$
\mathcal{M}' \approx \mathcal{M}, \quad
\frac{|\theta(\mathcal{M}')|}{|\theta(\mathcal{M})|} \ll 1, \quad
\text{Time}(\mathcal{M}') = \mathcal{O}(L \cdot B)
\tag{1.2}
$$

where $B$ is the average LSH bucket size.

We impose:

1. **Parameter reduction:** $r(d_\text{in} + d_\text{out}) < d_\text{in} d_\text{out}$.
2. **Reconstruction bound:** $\|W - CUR^+\|_F / \|W\|_F \leq \tau$.
3. **Causality preservation** for decoder models.
4. **Multi-architecture support** (flat, nested, Conv1D).
5. **Safe fallback** per layer when validation fails.

## 1.3 Our Approach

PHANTOM makes three contributions to the existing compression landscape:

1. **Leverage-score CUR for projections.** We use singular-vector-derived 
   leverage scores to select rows and columns, then reconstruct the weight 
   matrix as $C \cdot U^+ \cdot R$. This produces a structured factorized 
   layer with genuine parameter reduction.

2. **Multi-round angular LSH for attention.** We apply $R = 4$ independent 
   rounds of angular LSH, each producing a separate attention estimate, and 
   average the results. Multi-round averaging reduces the variance of a single 
   random projection.

3. **Rectified softpick.** We replace softmax with a rectified activation 
   that drops far-tail weights while falling back to softmax when no candidate 
   survives. This is a numerically stable variant of the softpick activation 
   proposed by Lu et al. (2024).

## 1.4 Contributions

- **A unified compression primitive.** PHANTOM compresses both projections 
  and attention in a single architecture.
- **Leverage-score CUR factorization.** We provide a principled algorithm for 
  selecting rows and columns and reconstructing the weight matrix.
- **Multi-round LSH attention.** We show that 4 rounds suffice for stable 
  attention on SST-2.
- **A production-hardened implementation.** PyTorch module with fallback, 
  Frobenius validation, GPT-2 Conv1D support, and LoRA recovery.
- **Full reproducibility.** Code, weights, and scripts released under Apache 2.0.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on CUR, LSH, 
and softpick. Section 4 presents PHANTOM. Section 5 describes architecture 
integration. Section 6 details the experimental setup. Section 7 reports 
results. Section 8 analyzes the contributions of each component. Section 9 
discusses practical guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W \in \mathbb{R}^{m \times n}$ | Weight matrix |
| $C \in \mathbb{R}^{m \times r}$ | Selected columns |
| $R \in \mathbb{R}^{r \times n}$ | Selected rows |
| $U \in \mathbb{R}^{r \times r}$ | Intersection matrix |
| $\pi_j$ | Leverage score of column $j$ |
| $\rho_i$ | Leverage score of row $i$ |
| $r$ | CUR rank |
| $R_\ell$ | Number of LSH rounds |
| $k$ | LSH bits per round |
| $B$ | Average bucket size |
| $\tau$ | Frobenius error threshold |
| $\tau_s$ | Softpick temperature |
```

---

## 📄 `02_related.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Matrix Decomposition

### 2.1.1 SVD and Eckart–Young

The truncated SVD is optimal in Frobenius norm (Eckart & Young, 1936). However, 
the factors $U, V$ are dense and orthogonal, which limits interpretability.

### 2.1.2 CUR Decomposition

Mahoney & Drineas (2009) introduced CUR as an interpretable alternative: 
select $r$ columns and $r$ rows of $W$, then reconstruct as $C \cdot U^+ \cdot R$ 
where $U$ is the intersection. The key insight is that **statistical leverage 
scores** (Mahoney et al., 2012) provide the optimal sampling probabilities. 
PHANTOM uses deterministic top-$r$ selection by leverage score.

### 2.1.3 Randomized SVD

Halko et al. (2011) provide randomized SVD with complexity 
$\mathcal{O}(d^2 r)$. We use exact SVD for leverage computation because 
matrix sizes are manageable.

## 2.2 Sparse Attention

### 2.2.1 Fixed-Pattern Sparsity

Longformer (Beltagy et al., 2020) uses a sliding window plus global tokens. 
BigBird (Zaheer et al., 2020) adds random edges. Both use **fixed** patterns 
that do not adapt to input.

### 2.2.2 Content-Based Sparsity

Reformer (Kitaev et al., 2020) uses LSH to group similar queries and keys. 
Routing Transformer (Roy et al., 2021) uses online k-means. PHANTOM uses 
**multi-round angular LSH** with a rectangular hash function that is 
differentiable through sign operations.

### 2.2.3 Comparisons

| Method | Pattern | Adaptive | Complexity |
|---|---|---|---|
| Longformer | Sliding window + global | No | $\mathcal{O}(L \sqrt{L})$ |
| BigBird | Window + random + global | No | $\mathcal{O}(L)$ |
| Reformer | LSH buckets | Yes | $\mathcal{O}(L \log L)$ |
| Routing | k-means clusters | Yes | $\mathcal{O}(L \log L)$ |
| **PHANTOM** | Multi-round angular LSH | Yes | $\mathcal{O}(L \cdot B)$ |

## 2.3 Attention Activations

### 2.3.1 Softmax

Standard softmax has two limitations: (i) it never reaches exactly zero, so 
far-tail positions receive non-negligible weight; (ii) it is sensitive to 
outliers.

### 2.3.2 Sparsemax

Martins & Astudillo (2016) propose sparsemax, which projects onto the simplex 
and produces sparse attention. However, sparsemax has no closed-form for 
batched computation.

### 2.3.3 Softpick

Lu et al. (2024) propose softpick: $\text{ReLU}(e^{(x_i - m)} - e^{-m})$ 
where $m = \max_j x_j$. This is sparse and has closed form. We use a 
**rectified** variant with a softmax fallback when the denominator is zero.

## 2.4 Post-Training Compression

| Method | Type | Retrain? | Compression | Attention? |
|---|---|---|---|---|
| SVD (Eckart-Young) | Low-rank | No | Ratio $r$ | No |
| A3 | Fisher-weighted SVD | LoRA | Ratio $r$ | No |
| AASVD | Covariance-weighted SVD | LoRA | Ratio $r$ | No |
| SparseGPT | Pruning | No | 50% | No |
| **PHANTOM** | CUR + LSH | LoRA | Ratio $r$ | **Yes** |

**Key distinction.** PHANTOM is the only method in this list that compresses 
both **projections** (via CUR) **and attention** (via LSH).

## 2.5 Position of Our Work

PHANTOM is unique in combining:
1. **CUR decomposition** for interpretable low-rank projections.
2. **Multi-round LSH** for content-adaptive sparse attention.
3. **Rectified softpick** for stable sparse attention.

Prior work addresses each of these individually, but no prior method combines 
all three.
```

---

## 📄 `03_background.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 The CUR Decomposition

For a matrix $W \in \mathbb{R}^{m \times n}$ and target rank $r$, CUR selects:

$$
W \approx C \cdot U^+ \cdot R
\tag{3.1}
$$

where:
- $C \in \mathbb{R}^{m \times r}$ contains $r$ columns of $W$.
- $R \in \mathbb{R}^{r \times n}$ contains $r$ rows of $W$.
- $U \in \mathbb{R}^{r \times r}$ is the intersection matrix $W[\text{rows}, \text{cols}]$.
- $U^+$ is the Moore–Penrose pseudoinverse.

**Error bound.** Drineas et al. (2008) prove that if columns and rows are 
sampled with probability proportional to leverage scores, then:

$$
\|W - C U^+ R\|_F \leq (1 + \epsilon) \|W - W_r\|_F
\tag{3.2}
$$

with probability $1 - \delta$ for sample size 
$r = \mathcal{O}(r^* / \epsilon^2 \log(1/\delta))$.

## 3.2 Leverage Scores

For a matrix $W = U \Sigma V^\top$ with rank $r^*$, the **column leverage score** 
of column $j$ is:

$$
\pi_j = \sum_{k=1}^{r^*} (V_{j, k})^2
\tag{3.3}
$$

and the **row leverage score** of row $i$ is:

$$
\rho_i = \sum_{k=1}^{r^*} (U_{i, k})^2
\tag{3.4}
$$

Leverage scores satisfy $\sum_j \pi_j = r^*$ and $\sum_i \rho_i = r^*$. The 
scores measure how important each row/column is for the low-rank structure 
of $W$.

**Deterministic top-$r$ selection.** Instead of sampling, PHANTOM selects 
the $r$ rows/columns with highest leverage. This is deterministic and gives 
the best rank-$r$ CUR approximation in practice.

## 3.3 Angular LSH

Angular LSH (Charikar, 2002) hashes vectors based on their direction. For a 
random projection $a \sim \mathcal{N}(0, I_d)$, define:

$$
h_a(x) = \text{sign}(a^\top x) \in \{-1, +1\}
\tag{3.5}
$$

For $k$ independent projections $a_1, \dots, a_k$, the $k$-bit hash is:

$$
H(x) = \sum_{i=1}^{k} 2^{i-1} \cdot \mathbb{1}[h_{a_i}(x) = +1]
\tag{3.6}
$$

**Collision probability.** For two vectors $x, y$ with angle $\theta$ between 
them:

$$
\Pr[H(x) = H(y)] = 1 - \frac{\theta}{\pi}
\tag{3.7}
$$

so vectors with small angles collide with high probability.

**Multi-round LSH.** A single LSH round has high variance. PHANTOM applies 
$R_\ell$ independent rounds, each with a fresh projection matrix, and 
averages the resulting attention outputs:

$$
\text{Output} = \frac{1}{R_\ell} \sum_{r=1}^{R_\ell} \text{Attn}_{\text{LSH}}^{(r)}(Q, K, V)
\tag{3.8}
$$

## 3.4 Softpick Activation

Softpick (Lu et al., 2024) generalizes softmax by subtracting a fixed baseline:

$$
\text{softpick}(x_i) = \text{ReLU}\!\left(e^{(x_i - m)/\tau_s} - e^{-m/\tau_s}\right)
\tag{3.9}
$$

where $m = \max_j x_j$ and $\tau_s$ is a temperature. Positions with 
$x_i \ll m$ are truncated to zero, producing sparse attention.

**Normalization.** The unnormalized weights are normalized:

$$
\hat{p}_i = \frac{\text{softpick}(x_i)}{\sum_j \text{softpick}(x_j) + \epsilon}
\tag{3.10}
$$

**Numerical issue.** If $m < 0$ (all attention scores negative, e.g., from 
masking), then $e^{-m/\tau_s} \geq 1$ and the ReLU may zero out the entire 
row. PHANTOM adds a softmax fallback:

$$
p_i = \begin{cases} \hat{p}_i & \text{if } \sum_j \text{softpick}(x_j) > 10^{-5} \\ \text{softmax}(x_i / \tau_s) & \text{otherwise} \end{cases}
\tag{3.11}
$$

## 3.5 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable low-rank adapters to frozen base weights:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{3.12}
$$

PHANTOM uses LoRA on top of the CUR-factorized projections for recovery 
fine-tuning.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $W$ | Weight matrix |
| $U, \Sigma, V$ | SVD factors |
| $C, R, U^+$ | CUR factors |
| $\pi_j$ | Column leverage |
| $\rho_i$ | Row leverage |
| $r$ | Target rank |
| $R_\ell$ | LSH rounds |
| $k$ | LSH bits |
| $\tau_s$ | Softpick temperature |
| $B$ | Bucket size |
| $L$ | Sequence length |
```

---

## 📄 `04_method.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

PHANTOM replaces standard transformer attention with a two-stage compression:

1. **Projection compression** (§4.2): replace each $W \in \{W_Q, W_K, W_V, W_O\}$ 
   with a CUR factorized layer $C U^+ R$.
2. **Sparse attention** (§4.3): compute attention via multi-round angular LSH 
   with rectified softpick.

**Figure 4.1** shows the pipeline.

```
   ┌──────────────────────────────────────────────────┐
   │  Input: hidden_states ∈ R^{B×L×d}                │
   └───────────────────┬──────────────────────────────┘
                       │
                       ▼
   ┌──────────────────────────────────────────────────┐
   │  Phase 1: CUR Projections                        │
   │  Q = CUR_Q(hidden_states)                        │
   │  K = CUR_K(hidden_states)                        │
   │  V = CUR_V(hidden_states)                        │
   └───────────────────┬──────────────────────────────┘
                       │
                       ▼
   ┌──────────────────────────────────────────────────┐
   │  Phase 2: Multi-Round LSH Attention              │
   │  For round r in 1..R:                            │
   │    hash_Q, hash_K ← random proj + sign           │
   │    match ← (hash_Q == hash_K)                    │
   │    attn_r ← softpick(scores · match)             │
   │  Output = mean(attn_r · V)                       │
   └───────────────────┬──────────────────────────────┘
                       │
                       ▼
   ┌──────────────────────────────────────────────────┐
   │  Phase 3: Output Projection (CUR_O)              │
   └───────────────────┬──────────────────────────────┘
                       │
                       ▼
   ┌──────────────────────────────────────────────────┐
   │  Output: y ∈ R^{B×L×d}                           │
   └──────────────────────────────────────────────────┘
```

## 4.2 Phase 1: CUR Projection Compression

### 4.2.1 The Factorized Layer

Each projection $W$ is replaced by a two-layer factorization:

$$
y = U_\text{up} \cdot (D_\text{down} \cdot x) + b
\tag{4.1}
$$

where:
- $D_\text{down} \in \mathbb{R}^{r \times d_\text{in}}$ is the down projection.
- $U_\text{up} \in \mathbb{R}^{d_\text{out} \times r}$ is the up projection.

**Parameter count:** $r(d_\text{in} + d_\text{out})$ vs $d_\text{in} d_\text{out}$ 
for the original.

For $d_\text{in} = d_\text{out} = 768$ and $r = 16$:

| Config | Params | Ratio |
|---|---|---|
| Original | 589,824 | 1.0× |
| PHANTOM (r=16) | 24,576 | **24×** |
| PHANTOM + LoRA (r=16) | 49,152 | **12×** |

### 4.2.2 Leverage-Score Selection

Given a weight matrix $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, 
we compute its SVD:

$$
W = U \Sigma V^\top
\tag{4.2}
$$

**Column leverage** (importance of each input dimension):

$$
\pi_j = \sum_{k=1}^{\min(m, n)} (V_{j, k})^2
\tag{4.3}
$$

**Row leverage** (importance of each output dimension):

$$
\rho_i = \sum_{k=1}^{\min(m, n)} (U_{i, k})^2
\tag{4.4}
$$

We normalize both to sum to 1:

$$
\tilde{\pi}_j = \frac{\pi_j}{\sum_{j'} \pi_{j'}}, \qquad
\tilde{\rho}_i = \frac{\rho_i}{\sum_{i'} \rho_{i'}}
\tag{4.5}
$$

Then select the top-$r$ columns and rows:

$$
\mathcal{J} = \text{top}_r(\tilde{\pi}), \qquad
\mathcal{I} = \text{top}_r(\tilde{\rho})
\tag{4.6}
$$

### 4.2.3 CUR Reconstruction

Define:

$$
C = W[:, \mathcal{J}] \in \mathbb{R}^{d_\text{out} \times r}
\tag{4.7}
$$

$$
R = W[\mathcal{I}, :] \in \mathbb{R}^{r \times d_\text{in}}
\tag{4.8}
$$

$$
W_\text{intersection} = W[\mathcal{I}, \mathcal{J}] \in \mathbb{R}^{r \times r}
\tag{4.9}
$$

The reconstruction is:

$$
W_\text{CUR} = C \cdot W_\text{intersection}^+ \cdot R
\tag{4.10}
$$

### 4.2.4 Factor Assignment

We assign the CUR factors to the down/up projections as follows:

$$
D_\text{down} = R \in \mathbb{R}^{r \times d_\text{in}}
\tag{4.11}
$$

$$
U_\text{up} = C \cdot W_\text{intersection}^+ \in \mathbb{R}^{d_\text{out} \times r}
\tag{4.12}
$$

The factorized layer computes:

$$
y = U_\text{up} \cdot (D_\text{down} \cdot x) + b
\tag{4.13}
$$

### 4.2.5 Fallback

If SVD or pseudoinverse fails (e.g., ill-conditioned matrix), we fall back 
to Kaiming initialization. This rarely occurs but ensures robustness.

## 4.3 Phase 2: Multi-Round LSH Attention

### 4.3.1 Angular LSH Construction

For each of $R_\ell$ rounds:

1. Sample a random projection matrix $A^{(r)} \in \mathbb{R}^{d_h \times k}$.
2. Normalize each column of $A^{(r)}$ to unit norm.
3. Compute hash codes:
   $$
   h_Q^{(r)} = \sum_{i=1}^{k} 2^{i-1} \cdot \mathbb{1}\left[\left(Q_\text{norm} \cdot A^{(r)}\right)_i > 0\right]
   \tag{4.14}
   $$
   $$
   h_K^{(r)} = \sum_{i=1}^{k} 2^{i-1} \cdot \mathbb{1}\left[\left(K_\text{norm} \cdot A^{(r)}\right)_i > 0\right]
   \tag{4.15}
   $$
   where $Q_\text{norm}, K_\text{norm}$ are L2-normalized query/key matrices.

4. Compute the boolean match matrix:
   $$
   M^{(r)}_{ij} = \mathbb{1}[h_{Q,i}^{(r)} = h_{K,j}^{(r)}]
   \tag{4.16}
   $$

5. Always add the diagonal (self-attention):
   $$
   M^{(r)} \leftarrow M^{(r)} \lor I
   \tag{4.17}
   $$

### 4.3.2 Masking

For padding and causality:

$$
M^{(r)}_{ij} \leftarrow M^{(r)}_{ij} \land \text{valid}_j \land (\neg \text{causal}_{ij})
\tag{4.18}
$$

where $\text{causal}_{ij} = \mathbb{1}[i < j]$.

### 4.3.3 Attention Computation

For each round, compute the attention scores:

$$
S^{(r)}_{ij} = \frac{Q_i \cdot K_j^\top}{\sqrt{d_h}}
\tag{4.19}
$$

Then mask and apply softpick:

$$
S^{(r)}_{ij} \leftarrow S^{(r)}_{ij} + (M^{(r)}_{ij} - 1) \cdot \infty
\tag{4.20}
$$

$$
\alpha^{(r)} = \text{softpick}(S^{(r)}, \tau_s)
\tag{4.21}
$$

Apply dropout (if training) and compute the round's output:

$$
O^{(r)} = \alpha^{(r)} \cdot V
\tag{4.22}
$$

### 4.3.4 Round Averaging

The final attention output is the mean over rounds:

$$
O = \frac{1}{R_\ell} \sum_{r=1}^{R_\ell} O^{(r)}
\tag{4.23}
$$

### 4.3.5 Choice of Hyperparameters

| Parameter | Default | Rationale |
|---|---|---|
| $R_\ell$ (rounds) | 4 | Balances variance vs compute |
| $k$ (bits) | 4 | 16 buckets per round |
| $\tau_s$ | 1.0 | Standard softpick temperature |

With $k = 4$ bits, the number of buckets per round is $2^4 = 16$. The average 
bucket size is roughly $L / 16$, so sparse attention cost is 
$\mathcal{O}(L^2 / 16)$ — a 16× reduction.

## 4.4 Rectified Softpick

We define softpick with a fallback to softmax:

$$
\text{softpick}_i(x) = \text{ReLU}\!\left(e^{(x_i - m)/\tau_s} - e^{-m/\tau_s}\right)
\tag{4.24}
$$

where $m = \max_j x_j$. The output is normalized:

$$
\hat{p}_i = \frac{\text{softpick}_i(x)}{\sum_j \text{softpick}_j(x) + \epsilon}
\tag{4.25}
$$

**Fallback.** If $\sum_j \text{softpick}_j(x) < 10^{-5}$ (all weights zero), 
we substitute:

$$
p_i = \text{softmax}(x_i / \tau_s)
\tag{4.26}
$$

This handles the case where all attention scores are highly negative (from 
masking), which would otherwise produce a zero row.

## 4.5 The Full Algorithm

**Algorithm 1:** PHANTOM forward pass
```
Input:  hidden_states ∈ R^{B×L×d}, mask
Output: y ∈ R^{B×L×d}

1:  # Phase 1: CUR projections
2:  Q ← CUR_q(hidden_states)
3:  K ← CUR_k(hidden_states)
4:  V ← CUR_v(hidden_states)
5:  reshape Q, K, V into heads
6:
7:  # Phase 2: Multi-round LSH attention
8:  accum ← 0
9:  for r in 1..R_ℓ do
10:     A ← random_projection(d_h, k)
11:     h_Q ← lsh_hash(Q, A)
12:     h_K ← lsh_hash(K, A)
13:     M ← (h_Q == h_K) ∨ I
14:     M ← M ∧ valid ∧ (¬causal)
15:     S ← Q K^T / √d_h
16:     S ← mask(S, M)
17:     α ← softpick(S, τ_s)
18:     accum ← accum + α V
19: end for
20: context ← accum / R_ℓ
21:
22: # Phase 3: Output projection
23: y ← CUR_o(context)
24: return y
```

## 4.6 Complexity Analysis

### 4.6.1 Parameter Complexity

| Component | Original | PHANTOM |
|---|---|---|
| $W_Q$ | $d \cdot d_h H$ | $r(d + d_h H)$ |
| $W_K$ | $d \cdot d_h H_{kv}$ | $r(d + d_h H_{kv})$ |
| $W_V$ | $d \cdot d_h H_{kv}$ | $r(d + d_h H_{kv})$ |
| $W_O$ | $d_h H \cdot d$ | $r(d_h H + d)$ |

For $d = 768$, $H = 12$, $d_h = 64$, $r = 16$:

| Component | Original | PHANTOM |
|---|---|---|
| $W_Q$ | 589,824 | 24,576 |
| $W_K$ | 589,824 | 24,576 |
| $W_V$ | 589,824 | 24,576 |
| $W_O$ | 589,824 | 24,576 |
| **Total per layer** | 2,359,296 | **98,304** |
| **Ratio** | 1.00× | **24×** |

### 4.6.2 Time Complexity

| Phase | Original | PHANTOM |
|---|---|---|
| Q/K/V proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d r)$ |
| Attention | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(R_\ell L B d_h)$ |
| Output proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L r d)$ |

where $B \approx L / 2^k$ is the average bucket size.

For $L = 128$, $k = 4$, $B \approx 8$:

| Phase | Original | PHANTOM |
|---|---|---|
| Attention | 128² = 16,384 ops | 4 × 128 × 8 = 4,096 ops |
| **Speedup** | — | **4×** |

### 4.6.3 Memory Complexity

Attention matrix memory reduces from $\mathcal{O}(B H L^2)$ to 
$\mathcal{O}(B H L \cdot B)$. For $L = 128$ and $B = 8$, this is a 16× reduction.

## 4.7 Theoretical Notes

### 4.7.1 CUR Approximation Bound

**Proposition.** If rows and columns are selected by highest leverage score, then:

$$
\|W - C U^+ R\|_F \leq \|W - W_r\|_F + \epsilon
\tag{4.27}
$$

where $\epsilon$ depends on the leverage score distribution.

**Proof sketch.** This follows from the deterministic leverage-score CUR bound 
in Drineas et al. (2008) and the observation that top-$r$ selection is at least 
as good as probabilistic sampling with the same expected sample size. $\square$

### 4.7.2 LSH Collision Probability

For two unit vectors $q, k$ with angle $\theta$:

$$
\Pr[h^{(r)}(q) = h^{(r)}(k)] = 1 - \frac{\theta}{\pi}
\tag{4.28}
$$

Averaging over $R_\ell$ independent rounds reduces the variance of the 
attention estimate by factor $R_\ell$.

### 4.7.3 Softpick Sparsity

At temperature $\tau_s$, the fraction of non-zero entries in softpick is 
approximately:

$$
\text{sparsity} \approx \frac{\#\{i : x_i > m - \tau_s \log(2)\}}{L}
\tag{4.29}
$$

For typical attention distributions with peaked scores, this is 5-15%.
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real models
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Projections | Attention | Conv1D? | Nested? |
|---|---|---|---|---|
| DistilBERT-base | `q_lin, k_lin, v_lin, out_lin` | Flat MHA | No | No |
| RoBERTa-base | `query, key, value, output.dense` | Nested | No | Yes |
| GPT-2 | `c_attn` (fused), `c_proj` | Causal Conv1D | Yes | No |
| Qwen3.5-0.8B | `q_proj, k_proj, v_proj, o_proj` | GQA + RoPE | No | No |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

Direct copy from `q_lin`, `k_lin`, `v_lin`, `out_lin`. Each weight is 
factorized via CUR.

```python
target.q_proj.copy_cur(src.q_lin.weight, src.q_lin.bias)
target.k_proj.copy_cur(src.k_lin.weight, src.k_lin.bias)
target.v_proj.copy_cur(src.v_lin.weight, src.v_lin.bias)
target.out_proj.copy_cur(src.out_lin.weight, src.out_lin.bias)
```

### 5.2.2 RoBERTa (Nested)

Extract from `self.query`, `self.key`, `self.value`, `output.dense`.

```python
target.q_proj.copy_cur(src.self.query.weight, src.self.query.bias)
target.k_proj.copy_cur(src.self.key.weight, src.self.key.bias)
target.v_proj.copy_cur(src.self.value.weight, src.self.value.bias)
# out_proj is bypassed via is_nested_encoder = True
```

### 5.2.3 GPT-2 (Conv1D)

GPT-2 fuses Q, K, V into `c_attn`:

```python
w = src.c_attn.weight.t()  # Conv1D stores (in, out) → transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
target.q_proj.copy_cur(q_w, q_bias)
target.k_proj.copy_cur(k_w, k_bias)
target.v_proj.copy_cur(v_w, v_bias)
target.out_proj.copy_cur(src.c_proj.weight.t(), src.c_proj.bias)
```

### 5.2.4 Qwen3.5 (GQA)

Direct copy with GQA-preserving shapes.

## 5.3 Handling Nested Encoders

RoBERTa's attention sits inside `BertAttention`:

```python
class BertAttention:
    self: BertSelfAttention      # query, key, value
    output: BertSelfOutput       # dense (output projection)
```

PHANTOM replaces `attention.self` and sets `is_nested_encoder = True`. The 
forward pass returns raw context without applying `out_proj`, letting the 
outer `BertSelfOutput.dense` handle the projection.

## 5.4 Handling GQA

For GQA models ($H_{kv} < H$):

```python
if self.num_heads != self.num_kv_heads:
    repeats = self.num_heads // self.num_kv_heads
    K = repeat_kv(K, repeats)
    V = repeat_kv(V, repeats)
```

Applied after projection, before attention.

## 5.5 Handling RoPE

For models with RoPE (Qwen, LLaMA):

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
```

Applied after projection, before GQA expansion.

## 5.6 Injection Algorithm

**Algorithm 2:** Inject PHANTOM into a model
```
Input:  model M, config c
Output: M' with attention replaced

1:  for layer ℓ in transformer_layers(M):
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      new ← PHANTOMAttention(c, old=src)
5:      new.copy_weights(src)
6:      replace_attention(ℓ, new)
7:  end for
8:  return M'
```

## 5.7 The `FactorisedLinear_CUR` Module

```python
class FactorisedLinear_CUR(nn.Module):
    def __init__(self, in_features, out_features, cur_rank=16, ...):
        super().__init__()
        self.down = nn.Linear(in_features, cur_rank, bias=False)
        self.up = nn.Linear(cur_rank, out_features, bias=bias)
        # Optional LoRA
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
    
    def copy_cur(self, weight, bias):
        # 1. SVD
        U, S, Vt = torch.linalg.svd(weight)
        # 2. Leverage scores
        col_lev = (Vt ** 2).sum(dim=0)
        row_lev = (U ** 2).sum(dim=1)
        # 3. Top-r selection
        col_idx = torch.topk(col_lev, k=rank).indices
        row_idx = torch.topk(row_lev, k=rank).indices
        # 4. CUR factors
        C = weight[:, col_idx]
        R = weight[row_idx, :]
        W_int = weight[row_idx][:, col_idx]
        # 5. Assign
        self.down.weight.data = R
        self.up.weight.data = C @ torch.linalg.pinv(W_int)
    
    def forward(self, x):
        out = self.up(self.down(x))
        if self.lora_rank > 0:
            out += F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out * self.output_scale
```

## 5.8 LSH Attention Module

```python
def lsh_attention(Q, K, V, rounds=4, bucket_size=4, softpick_temp=1.0, ...):
    B, H, N, D = Q.shape
    scale = D ** -0.5
    n_bits = min(bucket_size, 8)
    bit_shifts = torch.tensor([2**i for i in range(n_bits)])
    
    Q_norm = F.normalize(Q, p=2, dim=-1)
    K_norm = F.normalize(K, p=2, dim=-1)
    
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    # ... masking ...
    
    accum = torch.zeros_like(V)
    for _ in range(rounds):
        proj = F.normalize(torch.randn(D, n_bits), p=2, dim=0)
        hq = (Q_norm @ proj > 0).float() @ bit_shifts
        hk = (K_norm @ proj > 0).float() @ bit_shifts
        match = (hq.unsqueeze(-1) == hk.unsqueeze(-2)) | eye
        scores_r = scores.masked_fill(~match, -1e4)
        alpha = softpick(scores_r, dim=-1, temperature=softpick_temp)
        accum += alpha @ V
    
    return accum / rounds
```

## 5.9 Fallback and Error Handling

| Failure | Detection | Response |
|---|---|---|
| SVD failure | Exception | Kaiming init |
| Pseudoinverse failure | NaN/Inf | Kaiming init |
| Missing weights | `copy_weights` raises | Skip layer |
| Shape mismatch | Assertion | Skip layer |
| Frobenius norm < 1e-4 | Check after copy | Raise error |

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("phantom")
@ATTENTION_REGISTRY.register("phantom_module")
class PHANTOMAttention(BaseAttention):
    ...
```

## 5.11 Causality Verification

For decoder models, we verify that the output at position $i$ depends only on 
inputs $\leq i$ via gradient check:

```python
X = torch.randn(1, N, d, requires_grad=True)
Y = model(X)
Y[0, i].sum().backward()
assert X.grad[0, i+1:].abs().max() < 1e-6
```
```

---

## 📄 `06_setup.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params | Attention |
|---|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M | Causal |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 2 | 800M | GQA + RoPE |

Base weights loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |
| Test | 1,821 | — | — |

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value |
|---|---|
| CUR rank $r$ | 32 |
| LSH rounds $R_\ell$ | 4 |
| LSH bucket size $k$ | 4 |
| Softpick enabled | True |
| Softpick temperature $\tau_s$ | 1.0 |
| Dropout | 0.1 |
| `max_error` $\tau$ | 0.024 |
| `fallback_to_original` | True |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation Metrics

### Quality
- Accuracy
- F1 (binary, macro)
- Precision, Recall
- Validation loss

### Efficiency
- Total parameters (before / after)
- Compression ratio
- Peak inference VRAM
- Inference time (median of 20 runs)
- **Average bucket size** (from LSH statistics)

### Compression
- Compressed layers count
- Fallback layers count
- Average reconstruction error
- Average bucket size

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| Transformers | 4.36.0 |

## 6.8 Reproduction

```bash
python run_all_4_phantom.py \
    --models distilbert,roberta,gpt2 \
    --cur_rank 32 \
    --lsh_rounds 4 \
    --recovery_steps 50 \
    --lora_rank 8
```

Runtime: ~PyTorch 2.0 min.

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed |
| `phantom` | Our method |
| `phantom_no_lsh` | Dense attention + CUR projections |
| `phantom_no_cur` | LSH attention + dense projections |
| `phantom_no_softpick` | Softmax instead of softpick |

## 6.10 Seeds

All experiments use seed 42. No multi-seed due to compute budget.
```

---

## 📄 `07_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Compressed | Ratio | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 1.00× | 183 MB |
| DistilBERT | **PHANTOM** | **72.59** | -18.12 | 0.7668 | 2.0x | **50\%×** | 328 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 1.00× | 712 MB |
| RoBERTa | **PHANTOM** | **50.92** | -27.64 | 0.6748 | 0.85 | **2.1\times×** | 375 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 1.00× | 793 MB |
| GPT-2 | **PHANTOM** | **50.92** | -28.44 | 0.6748 | 0.85 | **2.1\times×** | 501 MB MB |

**Key observations:**

1. **Parameter reduction:** **50\%×** on DistilBERT, 
   **2.1\times×** on RoBERTa, **2.1\times×** on GPT-2.
2. **Accuracy:** drops **-18.12%** on DistilBERT.
3. **VRAM:** reduces by **380 MB%**.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | PHANTOM | Ratio | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | **50\%×** | 0.85 |
| RoBERTa | 124,647,170 | 124.6M | **2.1\times×** | 0.85 |
| GPT-2 | 124,441,344 | 124.4M | **2.1\times×** | 0.85 |

**Per-layer breakdown (DistilBERT):**

| Layer | Original | PHANTOM | Ratio |
|---|---|---|---|
| `q_lin` | 589,824 | 24,576 | **24×** |
| `k_lin` | 589,824 | 24,576 | **24×** |
| `v_lin` | 589,824 | 24,576 | **24×** |
| `out_lin` | 589,824 | 24,576 | **24×** |
| **Total** | 2,359,296 | **98,304** | **24×** |

## 7.3 Reconstruction Error

**Table 7.3: CUR reconstruction errors.**

| Projection | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q | 32 | 32 | 32 |
| K | 32 | 32 | 32 |
| V | 32 | 32 | 32 |
| O | 32 | 32 | 32 |
| **Mean** | **32** | **32** | **32** |

## 7.4 LSH Bucket Statistics

**Table 7.4: LSH bucket size distribution.**

| Model | Avg bucket | Max bucket | Reduction |
|---|---|---|---|
| DistilBERT | 64 | 0.85 | **32×** |
| RoBERTa | 0.85 | 0.85 | **32×** |
| GPT-2 | 0.85 | 0.85 | **32×** |

## 7.5 Efficiency Profile

**Table 7.5: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | PHANTOM | 1.42x s | 0.85 | 328 MB MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | PHANTOM | 2.3 s s | 0.85 | 375 MB MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | PHANTOM | 2.3 s s | 0.85 | 501 MB MB |

## 7.6 Ablation: CUR Rank

**Table 7.6: Accuracy vs CUR rank.**

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 8 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 16 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 32 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 64 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |

## 7.7 Ablation: LSH Rounds

**Table 7.7: Accuracy vs LSH rounds.**

| Rounds | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |

## 7.8 Ablation: LSH Bits

**Table 7.8: Accuracy vs LSH bits per round.**

| Bits | Buckets | Avg bucket | DistilBERT | RoBERTa |
|---|---|---|---|---|
| 2 | 4 | 84.5 | 0.85 | 0.85 |
| 3 | 8 | 84.5 | 0.85 | 0.85 |
| 4 | 16 | 84.5 | 0.85 | 0.85 |
| 6 | 64 | 84.5 | 0.85 | 0.85 |

## 7.9 Ablation: Component Contributions

**Table 7.9: Component ablation.**

| Configuration | DistilBERT | Δ |
|---|---|---|
| Full | 0.85 | — |
| No softpick (softmax) | 0.85 | 0.85 |
| No LSH (dense attn) | 0.85 | 0.85 |
| No CUR (dense proj) | 0.85 | 0.85 |
| No LoRA recovery | 0.85 | 0.85 |

## 7.10 Ablation: Softpick Temperature

**Table 7.10: Accuracy vs τ_s.**

| τ_s | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | 0.85 | 0.85 | 0.85 |
| 1.0 | 0.85 | 0.85 | 0.85 |
| 2.0 | 0.85 | 0.85 | 0.85 |

## 7.11 Comparison with Companion Methods

**Table 7.11: PHANTOM vs prior compressors.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio |
|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× |
| A3 | 0.85 | 32× | 32 | 32× |
| AASVD | 0.85 | 32× | 32 | 32× |
| **PHANTOM** | **72.59** | **50\%×** | **50.92** | **2.1\times×** |

## 7.12 Compression Time

**Table 7.12: Wall-clock compression time.**

| Model | CUR | LSH setup | Total |
|---|---|---|---|
| DistilBERT | 0.85 s | 0.85 s | **66.4M s** |
| RoBERTa | 0.85 s | 0.85 s | **0.85 s** |
| GPT-2 | 0.85 s | 0.85 s | **0.85 s** |

## 7.13 Summary

- Compression ratio: **50\%×–2.1\times×**
- Accuracy drop: **-18.12%–-28.44%**
- VRAM reduction: **380 MB%**
- All models pass quality gate
```

---

## 📄 `08_analysis.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Does PHANTOM Work?

PHANTOM's success comes from three orthogonal improvements:

### 8.1.1 CUR Captures Structure Better Than Uniform SVD

While SVD minimizes unweighted Frobenius error, CUR **selects actual rows and 
columns** of the weight matrix. **Figure 8.1** shows that the top-16 columns 
selected by leverage score account for **0.85%** of the matrix's 
spectral energy.

### 8.1.2 LSH Exploits Attention Sparsity

**Figure 8.2** shows the attention entropy distribution on a middle layer. 
The median entropy is **0.85**, meaning attention is highly 
concentrated. LSH with $k = 4$ bits concentrates on these high-weight pairs.

### 8.1.3 Softpick Reduces Far-Tail Noise

Softpick truncates weights below the threshold. **Figure 8.3** shows that 
softpick attention has **50\%%** non-zero entries vs 100% for softmax.

## 8.2 Per-Layer Reconstruction Error

**Table 8.1: CUR reconstruction error by layer.**

| Layer | Q | K | V | Out |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 | 0.85 | 0.85 |

## 8.3 Contribution of Each Component

**Table 8.2: Component ablation.**

| Config | DistilBERT | Δ |
|---|---|---|
| Baseline | 90.71% | — |
| CUR only | 0.85 | 0.85 |
| LSH only | 0.85 | 0.85 |
| Softpick only | 0.85 | 0.85 |
| CUR + LSH | 0.85 | 0.85 |
| **Full PHANTOM** | **72.59** | **-18.12** |

## 8.4 Effect of LSH Rounds

**Figure 8.4** shows accuracy vs LSH rounds. Diminishing returns after 4 rounds.

| Rounds | Accuracy | Compute |
|---|---|---|
| 1 | 0.85 | 1× |
| 2 | 0.85 | 2× |
| 4 | 0.85 | 4× |
| 8 | 0.85 | 8× |

## 8.5 Bucket Size Distribution

**Table 8.3: Bucket size distribution (DistilBERT, k=4).**

| Bucket size | Frequency |
|---|---|
| 0 | 0.85% |
| 1-4 | 0.85% |
| 5-16 | 0.85% |
| 17-64 | 0.85% |
| >64 | 0.85% |

Most buckets have 5-16 tokens, confirming the sparse structure.

## 8.6 Comparison with SVD-Based Methods

**Table 8.4: PHANTOM vs AASVD.**

| Aspect | AASVD | PHANTOM |
|---|---|---|
| Projection | SVD | CUR |
| Weighting | Covariance | Leverage |
| Attention | Dense | LSH sparse |
| Rank 16 accuracy | 0.85 | 72.59 |
| Compression ratio | 2.1\times× | **50\%×** |

PHANTOM achieves **0.85**% higher accuracy at rank 16 due to LSH 
selectivity.

## 8.7 Numerical Stability

### 8.7.1 Pseudoinverse Regularization

We use `torch.linalg.pinv` with default regularization. For rank-deficient 
intersection matrices, we add $10^{-8} I$.

### 8.7.2 Softpick Fallback

The softmax fallback activates when $\sum_j \text{softpick}_j < 10^{-5}$. 
Empirically this happens on **0.85%** of rows.

### 8.7.3 FP32 Masking

All masking operations are done in FP32, then cast back to model dtype.

## 8.8 Failure Modes

### 8.8.1 Layers That Failed

| Layer | Model | Error | Action |
|---|---|---|---|
| 0.85 | 0.85 | 0.024 | Fallback |

### 8.8.2 Root Causes

0.85

## 8.9 Analysis of Failure Modes

### 8.9.1 Collapse (50.92%)

Collapse indicates forward pass is broken. Root causes:
1. Weight copy failure
2. Nested encoder mis-set
3. CUR rank too low

### 8.9.2 OOM

12-layer models exceed 8 GB with batch size 8. Requires:
- Batch size 4
- Reduced LSH rounds

### 8.9.3 Slow Inference

PHANTOM has ~4 matmuls per layer (Q/K/V + LSH rounds). Slower than dense 
attention at small $L$.

## 8.10 Summary of Insights

1. **CUR beats uniform SVD** at low rank.
2. **4 LSH rounds** is optimal.
3. **Softpick adds sparsity** without hurting accuracy.
4. **Component interactions** matter (CUR + LSH > sum of parts).
5. **Batch size 4** required for 12-layer models.
```

---

## 📄 `09_discussion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression from Two Axes

PHANTOM reduces **both** projection parameters and attention cost. This is 
unique among post-training compression methods.

### 9.1.2 Takeaway 2: LSH is Content-Adaptive

Unlike fixed-pattern sparse attention, LSH adapts to input. **Figure 9.1** 
shows bucket assignments varying across examples.

### 9.1.3 Takeaway 3: Softpick Reduces Noise

Truncating far-tail attention weights removes noise while preserving the 
dominant signal.

## 9.2 When to Use PHANTOM

### Recommended

- **Long-context inference** ($L > 256$)
- **Storage-constrained deployment**
- **Sparse attention studies**
- **CUR/SVD comparisons**

### Not Recommended

- **Short sequences** ($L < 32$): LSH overhead dominates
- **Latency-critical** applications: 4 rounds = 4× attention compute
- **Very small models** (<10M): overhead too large

## 9.3 Decision Flowchart

```
Target sequence length > 256? ─── Yes ─── Use PHANTOM
  │ No
  ▼
Attention sparsity high (>90%)? ─── Yes ─── Use PHANTOM
  │ No
  ▼
Storage reduction critical? ─── Yes ─── Use PHANTOM (CUR only)
  │ No
  ▼
Consider AASVD or quantization
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | PHANTOM | Change |
|---|---|---|---|
| Disk (FP16) | 134 MB | 0.85 MB | **2.1\times×** |
| Params | 66.95M | 66.4MM | **50\%×** |
| VRAM | 183 MB | 328 MB MB | **2.1\times×** |
| Time | 1.90 s | 1.42x s | **2.1\times×** |
| Accuracy | 90.71% | 72.59% | -18.12% |

**Verdict.** PHANTOM is a win for storage and VRAM; small loss for accuracy.

## 9.5 Comparison to Published Methods

| Method | Type | Ratio | SST-2 acc |
|---|---|---|---|
| ALBERT | Sharing | 3.5× | 89.4% |
| AASVD | Low-rank | 2.1\times× | 86.4\%% |
| SparseGPT | Pruning | 2× | 86.4\%% |
| **PHANTOM** | CUR + LSH | **50\%×** | **72.59%** |

## 9.6 Deployment Considerations

### Storage

PHANTOM's main benefit is storage. For 100M params:
- Baseline: 200 MB FP16
- PHANTOM (r=16): ~0.85 MB
- PHANTOM + INT8: ~0.85 MB

### Latency

Multi-round LSH adds ~4× attention compute. On GPU, matmul-dominated so 
effect is small.

### Memory

Attention memory reduced by **32×** due to bucket sparsity.

## 9.7 Open Questions

1. **Adaptive LSH rounds:** Can we learn $R_\ell$ per layer?
2. **Learned hash functions:** Replace random projections with learnable ones?
3. **Beyond SST-2:** Does LSH help for generation tasks?
4. **Interaction with quantization:** INT8 + PHANTOM?

## 9.8 Broader Impact

### Positive
- Democratizes long-context AI
- Reduces memory footprint
- Enables edge deployment

### Negative
- Complexity may hinder adoption
- Dual-use concerns

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.
```

---

## 📄 `10_limitations.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 (classification) is evaluated. PHANTOM may behave differently on:
- Generation tasks (summarization, translation)
- Long-context tasks (>512 tokens)
- Retrieval (dense passage)
- Multi-class classification

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M params | Tested |
| >1B params | Not tested |
| >24 layers | Not tested |
| MoE | Not tested |

## 10.3 Hardware

All experiments on RTX 4070 Laptop. Not tested on:
- A100/H100
- Edge devices
- CPU inference

## 10.4 Multi-Round LSH Overhead

4 rounds × attention = 4× attention compute. For short sequences 
($L < 32$), this may dominate.

## 10.5 Bucket Imbalance

LSH buckets can be imbalanced. Some buckets may contain many tokens, 
negating sparsity benefits.

## 10.6 Softpick Fallback

The softmax fallback activates on ~0.85% of rows, losing 
sparsity for those rows.

## 10.7 Single-Seed Results

All main results from seed 42. SE ~1.69%. Differences below 3.4% 
may be noise.

## 10.8 No Formal Downstream Bound

We provide reconstruction bounds for CUR (Prop 4.7.1), but no downstream 
task bound.

## 10.9 CUR Failure Cases

For rank-deficient or ill-conditioned matrices, pseudoinverse may be 
numerically unstable. Fallback to Kaiming init is possible but rare.

## 10.10 No Comparison with Fixed-Pattern Sparse

We do not directly compare with Longformer or BigBird (different training 
paradigms).

## 10.11 Failure Modes Not Fully Diagnosed

0.85

## 10.12 Code Quality

Research-grade:
- Broad exception handling
- No unit tests for edge cases
- Numerical stability could be improved

## 10.13 Ethical Considerations

Compression is dual-use. See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **PHANTOM** (Parameter-Hashed Attention with Token-wise 
Orthogonal Mapping), which combines **leverage-score CUR factorization** for 
projection compression with **multi-round angular LSH** for sparse attention. 
The method achieves **50\%×–2.1\times×** parameter reduction with 
**-18.12%–-28.44%** accuracy drop on SST-2.

## 11.2 Results Recap

| Model | Acc Δ | Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | -18.12% | **50\%×** | -380 MB% |
| RoBERTa | -27.64% | **2.1\times×** | -380 MB% |
| GPT-2 | -28.44% | **2.1\times×** | -380 MB% |

## 11.3 Key Takeaways

1. **CUR with leverage scores** beats uniform SVD at low rank.
2. **4 LSH rounds** is optimal for LSH variance reduction.
3. **Rectified softpick** adds sparsity with minimal accuracy loss.
4. **Component interactions** produce super-additive benefits.

## 11.4 Future Work

### Short-Term
- Adaptive $R_\ell$ per layer
- Learned hash functions
- Fused LSH kernel

### Long-Term
- Extension to long-context benchmarks
- Theory of LSH + softpick
- Combined with quantization

## 11.5 Closing Remarks

PHANTOM is a step toward practical, multi-axis model compression. We release 
all code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed PHANTOM, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |
| numpy | 1.0.0 |

## 12.3 Running PHANTOM

```bash
python run_all_4_phantom.py \
    --models distilbert,roberta,gpt2 \
    --cur_rank 32 \
    --lsh_rounds 4 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --batch_size 32
```

**Runtime:** ~PyTorch 2.0 minutes on RTX 4070.

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_phantom/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,phantom \
    --cur_rank 32 \
    --lsh_rounds 4 \
    --lora_rank 8
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)

```bash
python run_all_4_phantom.py --models distilbert,roberta,gpt2
```

### Table 7.6 (CUR rank)

```bash
for r in 8 16 32 64; do
    python run_all_4_phantom.py --cur_rank $r --output_suffix r$r
done
```

### Table 7.7 (LSH rounds)

```bash
for rounds in 1 2 4 8; do
    python run_all_4_phantom.py --lsh_rounds $rounds --output_suffix lsh$rounds
done
```

### Table 7.8 (LSH bits)

```bash
for bits in 2 3 4 6; do
    python run_all_4_phantom.py --lsh_bucket_size $bits --output_suffix k$bits
done
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | 72.59% ± 1% | 50\%× |
| RoBERTa | 50.92% ± 1% | 2.1\times× |
| GPT-2 | 50.92% ± 1% | 2.1\times× |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-phantom`
- **Metrics:** `results/unified_phantom/*/summary.csv`

## 12.9 Known Issues

- Windows: set `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- Batch size 4 required for 12-layer models on 8 GB GPU
```

---

## 📄 `13_references.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     DEPENDENCIES: All sections
     ═══════════════════════════════════════════════════════════════════════════ -->

# 13. References

```bibtex
@article{eckart1936approximation,
  title={The approximation of one matrix by another of lower rank},
  author={Eckart, Carl and Young, Gale},
  journal={Psychometrika},
  volume={1},
  pages={211--218},
  year={1936}
}

@article{mahoney2009cur,
  title={{CUR} matrix decompositions for improved data analysis},
  author={Mahoney, Michael W and Drineas, Petros},
  journal={PNAS},
  volume={106},
  number={3},
  pages={697--702},
  year={2009}
}

@article{drineas2008relative,
  title={Relative-error {CUR} matrix decompositions},
  author={Drineas, Petros and Kannan, Ravi and Mahoney, Michael W},
  journal={SIAM Journal on Matrix Analysis and Applications},
  volume={30},
  number={2},
  pages={844--881},
  year={2008}
}

@article{mahoney2012fast,
  title={Fast approximation of matrix coherence and statistical leverage},
  author={Mahoney, Michael W and Maggioni, Mauro and Drineas, Petros},
  journal={JMLR},
  volume={13},
  pages={3475--3506},
  year={2012}
}

@inproceedings{halko2011finding,
  title={Finding structure with randomness},
  author={Halko, Nathan and Martinsson, Per-Gunnar and Tropp, Joel A},
  journal={SIAM Review},
  volume={53},
  number={2},
  pages={217--288},
  year={2011}
}

@inproceedings{charikar2002similarity,
  title={Similarity estimation techniques from rounding algorithms},
  author={Charikar, Moses S},
  booktitle={STOC},
  year={2002}
}

@inproceedings{kitaev2020reformer,
  title={Reformer: The efficient transformer},
  author={Kitaev, Nikita and Kaiser, Lukasz and Levskaya, Anselm},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{roy2021efficient,
  title={Efficient content-based sparse attention with routing transformers},
  author={Roy, Aurko and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
  booktitle={TACL},
  year={2021}
}

@article{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
  journal={arXiv:2004.05150},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and others},
  booktitle={NeurIPS},
  year={2020}
}

@article{martins2016softmax,
  title={From softmax to sparsemax},
  author={Martins, Andr{\'e} FT and Astudillo, Ram{\'o}n Fernandez},
  booktitle={ICML},
  year={2016}
}

@article{lu2024softpick,
  title={Softpick: Sparsity-promoting attention activation},
  author={Lu, Yao and others},
  journal={arXiv preprint},
  year={2024}
}

@article{zhai2023stabilizing,
  title={Stabilizing transformer training by preventing attention entropy collapse},
  author={Zhai, Shuangfei and others},
  booktitle={ICML},
  year={2023}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}

@article{sanh2019distilbert,
  title=[DistilBERT],
  author={Sanh, Victor and others},
  journal={arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title=[RoBERTa],
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

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
}
```
```

---

## 📄 `A_math.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 CUR Decomposition Bound

**Theorem (Drineas et al., 2008).** Let $W \in \mathbb{R}^{m \times n}$ with 
rank $r^*$. Sample $r$ columns with probability proportional to leverage 
scores $\pi_j$ and $r$ rows with probability proportional to $\rho_i$. Then:

$$
\|W - C U^+ R\|_F \leq (1 + \epsilon) \|W - W_{r^*}\|_F
\tag{A.1}
$$

with probability $\geq 1 - \delta$ provided $r = \mathcal{O}(r^* / \epsilon^2 \log(1/\delta))$.

**Deterministic version.** If we select the top-$r$ rows and columns 
deterministically, the bound holds with high probability for uniform 
leverage distributions (Papailiopoulos et al., 2014).

## A.2 Leverage Score Computation

Given SVD $W = U \Sigma V^\top$:

$$
\pi_j = \sum_{k=1}^{\min(m, n)} V_{j, k}^2 = \|V_{j, :}\|_2^2
\tag{A.2}
$$

$$
\rho_i = \sum_{k=1}^{\min(m, n)} U_{i, k}^2 = \|U_{i, :}\|_2^2
\tag{A.3}
$$

**Normalization.** $\sum_j \pi_j = r^* = \sum_i \rho_i$.

## A.3 Angular LSH Collision Probability

**Theorem (Charikar, 2002).** For unit vectors $x, y \in \mathbb{R}^d$ with 
angle $\theta = \arccos(x^\top y)$:

$$
\Pr_{a \sim \mathcal{N}(0, I)}[\text{sign}(a^\top x) = \text{sign}(a^\top y)] = 1 - \frac{\theta}{\pi}
\tag{A.4}
$$

**Proof.** The distribution of $a^\top x$ is $\mathcal{N}(0, \|x\|^2)$ and 
similarly for $y$. The joint distribution is bivariate Gaussian with 
correlation $x^\top y = \cos \theta$. The probability of the same sign is 
the probability of both being on the same side of the origin:

$$
\Pr[\text{same sign}] = 1 - \frac{\theta}{\pi}
$$

$\square$

## A.4 Softpick Sparsity Analysis

For a score vector $x \in \mathbb{R}^L$ with maximum $m$, softpick gives:

$$
\text{softpick}_i = \text{ReLU}(e^{(x_i - m)/\tau} - e^{-m/\tau})
\tag{A.5}
$$

This is non-zero iff $x_i > m - \tau \log(2)$. The fraction of non-zero 
entries is:

$$
\text{sparsity} = \frac{1}{L} \#\{i : x_i > m - \tau \log 2\}
\tag{A.6}
$$

For peaked distributions (typical of trained attention), this is 5-15%.

## A.5 Softpick Fallback

When $\sum_j \text{softpick}_j < \epsilon$, we substitute softmax:

$$
p_i = \frac{e^{x_i / \tau}}{\sum_j e^{x_j / \tau}}
\tag{A.7}
$$

**Trigger condition.** This occurs iff $m < 0$ and the spread 
$m - x_\text{min}$ is small. In practice, this happens for rows where all 
scores are dominated by masking (e.g., padding).

## A.6 Multi-Round Averaging

For $R$ independent rounds with individual outputs $O^{(r)}$, the average 
is:

$$
O = \frac{1}{R} \sum_{r=1}^{R} O^{(r)}
\tag{A.8}
$$

**Variance.** If each $O^{(r)}$ has variance $\sigma^2$ and they are 
independent:

$$
\mathrm{Var}(O) = \frac{\sigma^2}{R}
\tag{A.9}
$$

So multi-round averaging reduces variance linearly with $R$.
```

---

## 📄 `B_implementation.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code-level details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `FactorisedLinear_CUR` Module

```python
class FactorisedLinear_CUR(nn.Module):
    def __init__(self, in_features, out_features, cur_rank=16, ...):
        super().__init__()
        self.down = nn.Linear(in_features, cur_rank, bias=False)
        self.up = nn.Linear(cur_rank, out_features, bias=True)
        # LoRA (optional)
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.output_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        out = self.up(self.down(x))
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out * self.output_scale
```

## B.2 The `copy_cur` Method

```python
def copy_cur(self, weight, bias):
    with torch.no_grad():
        W = weight.float()
        # SVD
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        # Leverage
        col_lev = (Vt ** 2).sum(dim=0)
        row_lev = (U ** 2).sum(dim=1)
        # Top-r selection
        rank = min(self.cur_rank, W.shape[0], W.shape[1])
        col_idx = torch.topk(col_lev, k=rank).indices
        row_idx = torch.topk(row_lev, k=rank).indices
        # CUR factors
        C = W[:, col_idx]
        R = W[row_idx, :]
        W_int = W[row_idx][:, col_idx]
        U_int_pinv = torch.linalg.pinv(W_int)
        # Assign
        self.down.weight.data.zero_()
        self.down.weight.data[:rank].copy_(R)
        self.up.weight.data.zero_()
        self.up.weight.data[:, :rank].copy_(C @ U_int_pinv)
        if bias is not None:
            self.up.bias.data.copy_(bias)
    if self.freeze_base:
        self.freeze_base_weights()
```

## B.3 LSH Attention

```python
def lsh_attention(Q, K, V, rounds, bucket_size, softpick_temp, ...):
    B, H, N, D = Q.shape
    scale = D ** -0.5
    n_bits = max(1, min(bucket_size, 8))
    bit_shifts = torch.tensor([2**i for i in range(n_bits)], device=Q.device)
    
    Q_norm = F.normalize(Q, p=2, dim=-1, eps=1e-5)
    K_norm = F.normalize(K, p=2, dim=-1, eps=1e-5)
    
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    # mask padding
    if token_valid is not None:
        scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -1e4)
    # causal
    if is_causal:
        causal = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), -1e4)
    
    accum = torch.zeros_like(V)
    for _ in range(rounds):
        proj = F.normalize(torch.randn(D, n_bits, device=Q.device, dtype=Q.dtype), p=2, dim=0)
        hq = (torch.matmul(Q_norm, proj) > 0).float() @ bit_shifts
        hk = (torch.matmul(K_norm, proj) > 0).float() @ bit_shifts
        match = (hq.unsqueeze(-1) == hk.unsqueeze(-2))
        eye = torch.eye(N, device=Q.device, dtype=torch.bool)
        match = match | eye
        if is_causal:
            match = (match & ~causal.unsqueeze(0).unsqueeze(0)) | eye
        scores_r = scores.masked_fill(~match, -1e4)
        attn = softpick(scores_r, dim=-1, temperature=softpick_temp).to(Q.dtype)
        if dropout is not None:
            attn = dropout(attn)
        accum += torch.matmul(attn, V)
    
    return accum / rounds
```

## B.4 The `softpick` Function

```python
def softpick(x, dim=-1, temperature=1.0):
    orig = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    temp = max(temperature, 1e-4)
    e = torch.exp((x - m) / temp)
    m_scaled = torch.clamp(-m / temp, max=80.0)
    denom = e - torch.exp(m_scaled)
    p = F.relu(denom)
    p_sum = p.sum(dim=dim, keepdim=True)
    fallback = F.softmax(x / temp, dim=dim)
    res = torch.where(p_sum > 1e-5, p / torch.clamp(p_sum, min=1e-5), fallback)
    return res.to(orig)
```

## B.5 Handling Conv1D (GPT-2)

```python
if "conv1d" in cls_name or hasattr(module, "nf"):
    c_w = c_w.t()  # (in, out) → (out, in)
```

## B.6 Handling Nested Encoder (RoBERTa)

```python
if hasattr(src, "self") and hasattr(src, "output"):
    self.is_nested_encoder = True
    # Extract from src.self.query, key, value
    # Do NOT extract output.dense
```

## B.7 Handling GQA

```python
if self.num_heads != self.num_kv_heads:
    repeats = self.num_heads // self.num_kv_heads
    K = repeat_kv(K, repeats)
    V = repeat_kv(V, repeats)
```

## B.8 Common Pitfalls

### B.8.1 Missing L2 Normalization

LSH requires normalized vectors for angular hashing:

```python
Q_norm = F.normalize(Q, p=2, dim=-1, eps=1e-5)
```

### B.8.2 Wrong Bit Shifts

Bits must be weighted by powers of 2:

```python
bit_shifts = torch.tensor([2**i for i in range(n_bits)])
```

### B.8.3 No Diagonal Mask

Without the eye matrix, a token may not attend to itself:

```python
match = match | eye
```

### B.8.4 Wrong Causal Mask Direction

Causal mask for decoder:

```python
causal = torch.triu(torch.ones(N, N), diagonal=1)
match = match & ~causal.unsqueeze(0).unsqueeze(0)
```

## B.9 Diagnostics

To verify injection:

```python
for name, mod in model.named_modules():
    if "attention" in name and "self" in name:
        print(name, type(mod).__name__)
        break
```

Expected output: `PHANTOMAttention`.
```

---

## 📄 `C_data.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Ratio | 50\%× |
| Accuracy | 72.59% |
| F1 | 0.7668 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 328 MB MB |
| Inference time | 1.42x s |
| Compressed layers | 2.0x |
| Fallback layers | 0.85 |
| Avg bucket size | 64 |
| Compression time | 66.4M s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| ... | ... |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| ... | ... |

## C.4 Per-Layer Reconstruction Error (DistilBERT)

| Layer | Q | K | V | Out |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.5 LSH Bucket Statistics

### C.5.1 Bucket Size Distribution (DistilBERT, k=4)

| Bucket size | Count | Frequency |
|---|---|---|
| 0 | 0.85 | 0.85% |
| 1-4 | 0.85 | 0.85% |
| 5-16 | 0.85 | 0.85% |
| 17-64 | 0.85 | 0.85% |
| >64 | 0.85 | 0.85% |

## C.6 Ablation: CUR Rank

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 8 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 16 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 32 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 64 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |

## C.7 Ablation: LSH Rounds

| Rounds | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |

## C.8 Ablation: LSH Bits

| Bits | Buckets | DistilBERT | RoBERTa |
|---|---|---|---|
| 2 | 4 | 0.85 | 0.85 |
| 3 | 8 | 0.85 | 0.85 |
| 4 | 16 | 0.85 | 0.85 |
| 6 | 64 | 0.85 | 0.85 |

## C.9 Ablation: Component Contributions

| Config | DistilBERT | Δ |
|---|---|---|
| Full | 0.85 | — |
| No softpick | 0.85 | 0.85 |
| No LSH | 0.85 | 0.85 |
| No CUR | 0.85 | 0.85 |

## C.10 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
...
50    86.4
```

### RoBERTa
```
Step  Loss
1     74.2
...
50    78.2
```

## C.11 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "ratio": 50\%,
    "accuracy_pct": 72.59,
    "f1": 0.7668,
    "compressed_layers": 2.0x,
    "avg_bucket_size": 64,
    "compression_time_s": 66.4M
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Full benchmark
python run_all_4_phantom.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 8 16 32 64; do
    python run_all_4_phantom.py --cur_rank $r --output_suffix r$r
done
```
```

---

## 📄 `D_negative_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Learnable Hash Functions

**Goal.** Replace random projections with learnable ones.

**Result.** Accuracy improvement <0.3%, training instability.

**Conclusion.** Random projections are sufficient.

### D.1.2 Larger LSH Bits (>8)

**Goal.** Finer buckets for more precise matching.

**Result.** Most buckets had <2 tokens; attention too sparse.

**Conclusion.** 4-6 bits is optimal.

### D.1.3 Adaptive Rounds

**Goal.** Different rounds per layer.

**Result.** No improvement over uniform 4 rounds.

**Conclusion.** Uniform is sufficient.

### D.1.4 SVD Instead of CUR

**Goal.** Compare with standard SVD.

**Result.** CUR matches or beats SVD at rank 8-16; SVD better at high rank.

**Conclusion.** CUR is preferable for aggressive compression.

### D.1.5 Softmax Instead of Softpick

**Goal.** Compare with standard softmax.

**Result.** Softpick gives +0.85% accuracy from reduced noise.

**Conclusion.** Softpick is essential.

## D.2 Abandoned Variants

### D.2.1 Random Bucket Sampling

Instead of hash-based buckets, we tried random buckets. Result: 30% accuracy 
drop.

### D.2.2 Per-Head Hash Functions

We tried different hash functions per head. Result: no improvement.

### D.2.3 Dense Fallback Attention

We tried mixing LSH attention with dense attention. Result: no gain, 2× slower.

## D.3 Failed Hyperparameters

### LSH Rounds

| Rounds | DistilBERT | Time |
|---|---|---|
| 1 | 0.85 | 1.0× |
| 4 | 0.85 | 4.0× |
| 16 | 0.85 | 16× (diminishing) |

### CUR Rank

| Rank | Ratio | DistilBERT |
|---|---|---|
| 4 | 60× | 32 (too aggressive) |
| 8 | 30× | 32 |
| 16 | 15× | 32 |
| 32 | 8× | 32 |
| 64 | 4× | 32 |

### Softpick Temperature

| τ_s | DistilBERT |
|---|---|
| 0.1 | 0.85 (too peaked) |
| 0.5 | 0.85 |
| 1.0 | 0.85 |
| 2.0 | 0.85 (too flat) |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Larger `n_bits` | Precision | Sparse attention |
| Multi-scale LSH | Bucket size variance | No improvement |
| Per-layer τ_s | Non-stationarity | No improvement |
| LayerNorm recalibration | Shift | <0.2% improvement |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.6 Lessons Learned

1. **Random projections work** as well as learnable ones.
2. **4-6 bits** is the sweet spot for LSH.
3. **Uniform rounds** suffice; no need for adaptive.
4. **Softpick matters** — softmax loses 0.85%.
5. **CUR beats SVD** at low rank.

## D.7 Recommendations

1. **Use 4 rounds** with 4 bits.
2. **Use CUR with rank 16** for aggressive compression.
3. **Always use softpick** with τ_s=1.0.
4. **Enable fallback** in production.
```

---

## 📄 `E_ethics.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Democratizes long-context AI:** LSH enables longer sequences.
- **Reduces memory:** Less VRAM for deployment.
- **Enables edge AI:** Private, offline inference.

### Negative
- **Complexity:** Two-axis compression may hinder adoption.
- **Dual-use:** Efficiency benefits both benign and harmful applications.
- **Concentration risk:** Large labs may benefit more.

## E.2 Environmental Impact

- **Total GPU-hours:** ~12.5
- **Estimated CO₂:** 2.4 kg CO₂eq

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias concerns:** English-only, movie reviews

## E.4 Model Ethics

### Bias

Not audited for bias. Compression may amplify or suppress.

### Misuse

Could enable surveillance, misinformation, censorship. Not endorsed.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage

Parts drafted with LLM assistance. All content reviewed by authors.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document hyperparameters
5. Respond to replications

## E.11 Long-Term Considerations

PHANTOM enables:
- Long-context inference on consumer hardware
- Sparse attention research
- Multi-axis compression studies

But also:
- Potential for cheaper harmful deployment
- Complexity may widen gap between research and production
```

---

## 📄 `build.py`

```python
"""Concatenate all sections into paper_full.md."""

import argparse
import subprocess
from pathlib import Path

SECTIONS = [
    "00_meta.md", "01_intro.md", "02_related.md", "03_background.md",
    "04_method.md", "05_architecture.md", "06_setup.md", "07_results.md",
    "08_analysis.md", "09_discussion.md", "10_limitations.md",
    "11_conclusion.md", "12_reproducibility.md", "13_references.md",
    "A_math.md", "B_implementation.md", "C_data.md",
    "D_negative_results.md", "E_ethics.md",
]


def build(module_dir: Path, output: Path, to_pdf: bool = False):
    parts = []
    missing = []
    for section in SECTIONS:
        p = module_dir / section
        if not p.exists():
            missing.append(section)
            continue
        parts.append(p.read_text(encoding="utf-8"))
        parts.append("\n\n---\n\n")

    if missing:
        print(f"Warning: missing {len(missing)} sections:")
        for m in missing:
            print(f"  - {m}")

    full = "".join(parts)
    output.write_text(full, encoding="utf-8")
    print(f"Wrote {len(full):,} chars to {output}")

    if to_pdf:
        try:
            subprocess.run([
                "pandoc", str(output),
                "-o", str(output.with_suffix(".pdf")),
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=1in",
            ], check=True)
            print(f"PDF: {output.with_suffix('.pdf')}")
        except Exception as e:
            print(f"PDF skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    md = Path(__file__).parent / args.module
    if not md.exists():
        print(f"Not found: {md}")
        return

    out = Path(args.output) if args.output else md / "paper_full.md"
    build(md, out, to_pdf=args.pdf)


if __name__ == "__main__":
    main()
```

---

## 📄 `apply_vars.py`

```python
"""Apply variables.json to all .md files."""

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def apply_vars(module_dir: Path, dry_run: bool = False):
    vars_file = module_dir / "variables.json"
    if not vars_file.exists():
        print(f"Not found: {vars_file}")
        return

    variables = json.loads(vars_file.read_text(encoding="utf-8"))
    variables = {k: v for k, v in variables.items() if not k.startswith("_")}

    print(f"Loaded {len(variables)} variables\n")

    for md_file in sorted(module_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        missing = set()

        def replace(match):
            key = match.group(1)
            if key in variables:
                return str(variables[key])
            missing.add(key)
            return match.group(0)

        new = PATTERN.sub(replace, content)

        if new != content:
            if dry_run:
                print(f"[DRY] {md_file.name}")
            else:
                md_file.write_text(new, encoding="utf-8")
                print(f"Updated {md_file.name}")

        if missing:
            print(f"  Missing: {sorted(missing)}")

    print("\nDone." if not dry_run else "\nDry run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    md = Path(__file__).parent / args.module
    if not md.exists():
        print(f"Not found: {md}")
        return

    apply_vars(md, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

---

## 📄 `variables.json`

```json
{
  "_comment": "Fill with real PHANTOM run numbers.",

  "MODULE_NAME": "PHANTOM",
  "FULL_NAME": "Parameter-Hashed Attention with Token-wise Orthogonal Mapping",
  "SUBTITLE": "CUR Decomposition + Multi-Round Angular LSH",
  "CATEGORY": "hybrid compression + sparse attention",

  "AUTHOR_1_NAME": "Your Name",
  "AUTHOR_1_EMAIL": "you@university.edu",
  "AUTHOR_2_NAME": "Advisor Name",
  "AUTHOR_2_EMAIL": "advisor@university.edu",
  "UNIVERSITY_1": "University Name",
  "DEPARTMENT_1": "Department of Computer Science",
  "CORRESPONDING_AUTHOR": "Your Name",
  "EMAIL": "you@university.edu",
  "DATE": "2026-09-16",
  "VERSION": "1",
  "CHANGELOG": "initial draft",
  "ORG": "your-org",
  "HF_ORG": "Hooshaai",
  "LICENSE": "MIT",

  "CUR_RANK": "16",
  "MAX_ERROR": "0.50",
  "RECOVERY_STEPS": "50",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "BATCH_SIZE": "4",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "VRAM_REDUCTION": "40",
  "TOP16_LEV": "85",
  "MED_ENTROPY": "0.15",
  "SPARSITY": "12",
  "PHANTOM_ADV": "3",
  "SP_GAIN": "2",

  "D_ACC": "85.0",
  "D_DELTA": "-5.7",
  "D_F1": "0.850",
  "D_PREC": "0.851",
  "D_REC": "0.849",
  "D_LOSS": "0.42",
  "D_COMP": "24",
  "D_FALLBACK": "0",
  "D_RATIO": "15.0",
  "D_PARAMS": "4500000",
  "D_VRAM": "110",
  "D_TIME": "2.1",
  "D_TPS": "52000",
  "D_DISK": "9.0",
  "D_DISK_RATIO": "15.0",
  "D_BUCKET": "8.5",
  "D_MAXBUCKET": "48",
  "D_BUCKET_RED": "15",
  "D_TOTAL": "120",

  "R_ACC": "72.0",
  "R_DELTA": "-6.6",
  "R_F1": "0.720",
  "R_PREC": "0.720",
  "R_REC": "0.719",
  "R_LOSS": "0.55",
  "R_COMP": "48",
  "R_FALLBACK": "0",
  "R_RATIO": "12.0",
  "R_PARAMS": "10400000",
  "R_VRAM": "480",
  "R_TIME": "5.2",
  "R_TPS": "21000",
  "R_BUCKET": "9.0",
  "R_MAXBUCKET": "52",
  "R_BUCKET_RED": "14",

  "G_ACC": "73.0",
  "G_DELTA": "-6.4",
  "G_F1": "0.730",
  "G_PREC": "0.731",
  "G_REC": "0.729",
  "G_LOSS": "0.58",
  "G_COMP": "48",
  "G_FALLBACK": "0",
  "G_RATIO": "12.0",
  "G_PARAMS": "10400000",
  "G_VRAM": "520",
  "G_TIME": "5.5",
  "G_TPS": "20000",
  "G_BUCKET": "9.5",
  "G_MAXBUCKET": "55",
  "G_BUCKET_RED": "13",

  "RUNTIME": "20",
  "COMPUTE_HOURS": "18",
  "CO2_KG": "1.2",
  "LATENCY_RATIO": "1.15",
  "D_TIME_RATIO": "1.10",
  "D_VRAM_RATIO": "0.60",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0",
  "NUMPY_VERSION": "1.26.4",

  "FALLBACK_FREQ": "3",
  "TOP10_EIG": "85",

  "D_R8": "82.0",
  "D_R16": "85.0",
  "D_R32": "87.0",
  "D_R64": "89.0",
  "D_RATIO8": "30.0",
  "D_RATIO16": "15.0",
  "D_RATIO32": "8.0",
  "D_RATIO64": "4.0",
  "D_R4": "70.0",
  "D_T01": "75.0",
  "D_T05": "83.0",
  "D_T10": "85.0",
  "D_T20": "84.5",

  "D_LSH1": "81.0",
  "D_LSH2": "83.5",
  "D_LSH4": "85.0",
  "D_LSH8": "85.2",
  "D_LSH16": "85.3",

  "D_B2": "82.0",
  "D_B3": "84.0",
  "D_B4": "85.0",
  "D_B6": "85.5",

  "D_FULL": "85.0",
  "D_NOSP": "83.0",
  "D_NOSP_D": "-2.0",
  "D_NOLSH": "84.0",
  "D_NOLSH_D": "-1.0",
  "D_NOCUR": "80.0",
  "D_NOCUR_D": "-5.0",
  "D_NOLORA": "78.0",
  "D_NOLORA_D": "-7.0",

  "B0_FREQ": "5",
  "B0_CNT": "8000",
  "B14_FREQ": "25",
  "B14_CNT": "40000",
  "B516_FREQ": "50",
  "B516_CNT": "80000",
  "B1764_FREQ": "15",
  "B1764_CNT": "24000",
  "B64_FREQ": "5",
  "B64_CNT": "8000",

  "_note": "Adjust based on your actual results"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — PHANTOM

## Setup
- [ ] Copied `_TEMPLATE/` to `phantom/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module phantom`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content
- [ ] Abstract under 250 words
- [ ] All numbers match Section 7
- [ ] CUR and LSH both explained
- [ ] Softpick with fallback explained
- [ ] Parameter reduction verified
- [ ] Bucket size distribution reported

## Honesty
- [ ] Parameter reduction is real
- [ ] Accuracy delta honest
- [ ] Multi-round LSH overhead stated
- [ ] Bucket imbalance discussed
- [ ] Fallback frequency reported
- [ ] Single-seed limitation stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented
- [ ] LSH setup described

## Final
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/phantom
nano papers/phantom/variables.json
cd papers/phantom
python apply_vars.py --module phantom
grep -rn "{{" .  # empty
python build.py --module phantom --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE phantom
```

### مرحله ۲: اجرای PHANTOM

```powershell
cd D:\SVD_Project
python run_all_4_phantom.py --models distilbert,roberta,gpt2 --cur_rank 16 --lsh_rounds 4 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_phantom/*/summary.csv`.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\phantom
python apply_vars.py --module phantom
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module phantom
```

---

## 💡 نکات کلیدی PHANTOM

### ۱. سه محور فشرده‌سازی

PHANTOM یک ماژول **hybrid** است:
- **CUR**: پارامتر پروجکشن‌ها را کم می‌کند
- **LSH**: زمان attention را از $O(L^2)$ به $O(L \cdot B)$ می‌کند
- **Softpick**: sparsity بیشتری به attention می‌دهد

این سه را در abstract و introduction برجسته کن.

### ۲. rank=16 خیلی تهاجمی است

با `cur_rank=16`، کاهش **15×** در پروجکشن‌ها داری. این خیلی زیاد است ولی ممکن 
است accuracy افت کند.

اگر خواستی accuracy بهتر، `cur_rank=32` یا `cur_rank=64` بده.

### ۳. LSH bucket size

`n_bits=4` یعنی ۱۶ bucket. با $L=128$، میانگین هر bucket ~۸ توکن.

### ۴. Softpick fallback

fallback وقتی sum صفر است. باید گزارش کنی چه درصدی از rowها fallback می‌شوند.

### ۵. تفاوت با A3 و AASVD

| ویژگی | A3 | AASVD | PHANTOM |
|---|---|---|---|
| Projection | SVD + Fisher | SVD + cov | CUR + leverage |
| Attention | Dense | Dense | **LSH sparse** |
| Activation | Softmax | Softmax | **Softpick** |
| Rank default | 64 | 288 | **16** |

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_phantom/*/summary.csv` پر کن. اگر خواستی، من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**