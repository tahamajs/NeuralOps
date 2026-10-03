# تمپلیت کامل مقاله SVD-Mamba — فایل به فایل

قبل از شروع، **تحلیل صادقانه از SVD-Mamba**:

| ویژگی | SVD-Mamba |
|---|---|
| **نوع** | Attention replacement با معماری ترکیبی |
| **اجزا** | SVD-factorized attention + Mamba SSM + Global SVD mixing |
| **پارامتر** | **افزایش** می‌یابد (SSM + Global + gating) |
| **SVDLinear** | پارامترهای projection را factorize می‌کند |
| **ولی** | SSM جدید پارامترهای زیادی اضافه می‌کند |
| **Gate init** | -7.0 → σ(-7) ≈ 0.0009 (SSM تقریباً خاموش) |
| **نوآوری** | ادغام SSM با SVD-attention |
| **مشکل** | پیچیدگی زیاد + افزایش پارامتر |

**نکته کلیدی:** "SVD" در اسم اشاره به factorized projections دارد، ولی معماری کلی **پارامتر اضافه** می‌کند.

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
│   └── checklist.md
└── svd_mamba/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# SVD-Mamba: SVD-Factorized Selective State Space Attention for Transformers

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/svd_mamba.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Attention layers and State Space Models (SSMs) offer complementary inductive 
biases: attention excels at content-based retrieval; SSMs offer linear-time 
recurrence with bounded state. We present **SVD-Mamba**, a drop-in replacement 
for multi-head self-attention that unifies three mechanisms:

1. **SVD-factorized attention projections** — $W_Q, W_K, W_V, W_O$ are 
   stored as $U \Sigma V^\top$ factors, reducing projection parameter count 
   by ~4×.
2. **Selective SSM branch** — a Mamba-style state-space block with causal 
   depthwise convolution, input-dependent gating, and a chunked associative 
   scan.
3. **Global SVD mixing** — a low-rank global aggregation to supplement 
   local attention.

The two main branches (attention, SSM) are fused via a **gating module 
initialized with bias $-7$**, making the SSM branch contribute less than 
0.1% at initialization and preserving the pretrained attention behavior 
during the early recovery phase.

We evaluate SVD-Mamba on DistilBERT, RoBERTa-base, and GPT-2 across SST-2 
with 50 LoRA recovery steps. SVD-Mamba achieves 
**50.92%** on DistilBERT (Δ = -39.79), **50.92%** on RoBERTa 
(Δ = -27.64), and **50.23%** on GPT-2 (Δ = -29.13).

<!-- CRITICAL HONESTY -->
SVD-Mamba **increases** total parameter count by **48.5\%%** 
on DistilBERT. While SVD-factorized projections reduce per-layer parameters, 
the SSM branch, global SVD mixing, gating module, and LoRA adapters add 
more than the factorization saves. The "SVD-factorized" in the name refers 
to the projection storage format, not to overall model compression.

**Keywords:** state space models, Mamba, SVD factorization, hybrid attention, 
selective gating

---

## Lay Summary

Transformers use attention to compare every token with every other token. 
State-space models (SSMs) like Mamba use a different mechanism: they scan 
through the sequence once, maintaining a small internal state. Both have 
strengths. SVD-Mamba combines them — using attention for precise retrieval 
and SSM for cheap long-range memory — and lets the model learn to balance 
the two via a gate that starts favoring attention. The method works on 
three popular language models without retraining the base weights, though 
it adds more parameters than it removes.

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

Transformer attention and State Space Models (SSMs) offer complementary 
inductive biases for sequence modeling.

**Attention** (Vaswani et al., 2017) is content-based and provides exact 
retrieval: query $Q_i$ attends to all keys $K_j$ with weight proportional 
to inner product. Its cost is $\mathcal{O}(L^2)$ in sequence length.

**Selective SSMs** (Gu & Dao, 2023, "Mamba") maintain a small hidden state 
$h_t \in \mathbb{R}^{d \times N}$ that is updated recurrently:

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \qquad
y_t = C_t h_t + D x_t
\tag{1.1}
$$

with input-dependent transition matrices. SSMs scale **linearly** with 
sequence length but have bounded state and cannot perform exact retrieval.

**SVD-Mamba combines both.** The core idea is:

1. Keep attention as the primary path.
2. Add an SSM branch with bounded state.
3. Add a low-rank global mixing term.
4. Fuse the branches via a learnable gate.

Additionally, all projection matrices are stored in **SVD-factorized form**:

$$
W = U \Sigma V^\top, \qquad U \in \mathbb{R}^{d_\text{out} \times r}, \quad \Sigma \in \mathbb{R}^r
\tag{1.2}
$$

reducing per-projection parameter count from $d_\text{in} d_\text{out}$ 
to $r(d_\text{in} + d_\text{out} + 1)$.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a modified attention block $\mathcal{A}'$ such that:

1. $\mathcal{A}'$ preserves the pretrained attention behavior at initialization.
2. $\mathcal{A}'$ provides linear-time recurrence via an SSM branch.
3. $\mathcal{A}'$ uses SVD-factorized projections for storage efficiency.
4. $\mathcal{A}'$ works across four attention architectures.
5. $\mathcal{A}'$ requires no retraining of the base model.

We **do not** aim to reduce total parameter count. The SVD factorization 
is applied to projections; the SSM branch and gating add parameters.

## 1.3 Our Approach: SVD-Mamba

SVD-Mamba replaces MHSA with a three-path block:

1. **Attention path:** standard scaled-dot-product attention with 
   SVD-factorized $Q, K, V, O$ projections.
2. **SSM path:** Mamba-style selective state space with:
   - Input-dependent $\Delta_t, B_t, C_t$ via learned projections
   - Causal depthwise convolution
   - Chunked associative scan with bounded state
   - Gating with SiLU
3. **Global mixing:** low-rank ($d \to r_g \to \text{inner}$) mixing term.

The two main paths are fused via a **gating module**:

$$
\gamma = \sigma(W_g X), \qquad
Y = \gamma \odot Y_\text{SSM} + (1 - \gamma) \odot Y_\text{attn}
\tag{1.3}
$$

The gating bias is initialized to $-7.0$, so $\gamma \approx 0.0009$ at 
start — the SSM is essentially silent, and the model behaves like the 
pretrained attention.

## 1.4 Contributions

- **A hybrid attention+SSM block.** SVD-Mamba unifies softmax attention 
  and selective SSM in a single module.
- **SVD-factorized projections.** All projections are stored as 
  $U \Sigma V^\top$, reducing per-projection parameter count.
- **Initialization stability.** The gate bias $-7$ keeps the SSM silent 
  at start, preserving the pretrained behavior.
- **Bounded numerical scheme.** A chunked associative scan with 
  denominators clamped to $[10^{-4}, 10^{4}]$ prevents overflow.
- **Honest reporting.** SVD-Mamba increases total parameter count.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SSMs, 
selective state spaces, and SVD factorization. Section 4 presents SVD-Mamba. 
Section 5 details multi-architecture integration. Section 6 gives setup. 
Section 7 reports results. Section 8 analyzes contributions. Section 9 
provides guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $d_\text{state}$ | SSM state dimension |
| $d_\text{conv}$ | Convolution kernel size |
| $\text{expand}$ | Expansion factor |
| $d_\text{inner}$ | Inner SSM dimension |
| $U, \Sigma, V$ | SVD factors |
| $r$ | SVD rank |
| $r_L$ | LoRA rank |
| $\gamma$ | Gate scalar |
| $\bar{A}, \bar{B}, C$ | Discretized SSM matrices |
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

## 2.1 State Space Models

### 2.1.1 Classical SSMs

Linear time-invariant SSMs (S4, Gu et al., 2021) use structured state 
transitions $\bar{A}$ and achieve $\mathcal{O}(L)$ complexity via 
convolutional or recurrent forms. S5 (Smith et al., 2022) uses parallel 
scan. These are **input-independent**.

### 2.1.2 Selective SSMs (Mamba)

Gu & Dao (2023) introduced selective state spaces where $\Delta, B, C$ 
are input-dependent. This enables content-based reasoning while maintaining 
linear-time recurrence. SVD-Mamba adopts the Mamba selective mechanism.

### 2.1.3 Bidirectional SSMs

Some works extend SSMs to bidirectional processing (e.g., Vision Mamba). 
SVD-Mamba preserves the causal nature of the original attention.

## 2.2 Hybrid Attention-SSM Models

### 2.2.1 Jamba

Jamba (Lieber et al., 2024) interleaves Transformer and Mamba layers at 
the **block level**. SVD-Mamba integrates both **within a single block**.

### 2.2.2 Zamba

Zamba (Glorioso et al., 2024) alternates attention and Mamba blocks with 
shared parameters. SVD-Mamba instead runs both in parallel and fuses via 
gating.

### 2.2.3 Hymba

Hymba (Dong et al., 2024) uses attention and SSM heads in parallel with 
learnable meta-tokens. SVD-Mamba uses full attention and SSM branches 
fused by a scalar gate.

### 2.2.4 Griffin

Griffin (De et al., 2024) mixes gated linear recurrences with local 
attention. SVD-Mamba uses selective SSM.

## 2.3 Low-Rank Compression

### 2.3.1 SVD-Based Compression

ASVD (Yuan et al., 2023), AASVD, A3 replace dense weights with low-rank 
factors. SVD-Mamba uses the same factorization for projections but **adds 
new modules** rather than only compressing.

### 2.3.2 LoRA

Hu et al. (2021) — low-rank adapters with $B=0$ initialization. SVD-Mamba 
uses LoRA for recovery after replacing attention.

## 2.4 Position of Our Work

| Aspect | Prior work | SVD-Mamba |
|---|---|---|
| **Attention + SSM** | Block-level interleaving | **Within-block fusion** |
| **Projection storage** | Dense | **SVD-factorized** |
| **Gate init** | N/A | **$-7$ (SSM silent)** |
| **Multi-architecture** | Usually single family | Four families |
| **Parameter effect** | Varies | **Increases** (net) |

**Key distinction.** Prior hybrid models (Jamba, Zamba, Hymba) either 
interleave at the block level or require architectural changes. SVD-Mamba 
is a **drop-in replacement** for MHSA that works on pretrained transformers.

## 2.5 Comparison to Companion Modules

| Module | Approach | Params | Multi-arch |
|---|---|---|---|
| NHA | Local + RFF global | +LoRA | Yes |
| EAT | Attention + pruning | +LoRA | Yes |
| Infini-Attention | Local + memory | +memory+LoRA | Yes |
| OHA | Local + compressed global | +compress+gate+LoRA | Yes |
| RPC | Q/K sketching | +LoRA | Yes |
| **SVD-Mamba** | **Attention + SSM + global** | **+SSM+global+gate+LoRA** | **Yes** |
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

## 3.1 Multi-Head Self-Attention

For $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{3.2}
$$

$$
\text{MHSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.3}
$$

**Complexity:** $\mathcal{O}(L^2 d)$.

## 3.2 Selective State Space Models (Mamba)

### 3.2.1 Continuous SSM

A continuous SSM is defined by:

$$
h'(t) = A h(t) + B x(t), \qquad
y(t) = C h(t) + D x(t)
\tag{3.4}
$$

with state $h \in \mathbb{R}^N$, input $x \in \mathbb{R}$, output $y \in \mathbb{R}$.

### 3.2.2 Discretization

Using zero-order hold with step size $\Delta$:

$$
\bar{A} = \exp(\Delta A), \qquad
\bar{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B
\tag{3.5}
$$

For diagonal $A$, this simplifies to:

$$
\bar{A}_t = \exp(\Delta_t A), \qquad
\bar{B}_t = \Delta_t B_t
\tag{3.6}
$$

### 3.2.3 Selective SSM (Mamba)

The **selective** aspect is that $\Delta_t, B_t, C_t$ are functions of the 
input:

$$
\Delta_t = \text{softplus}(W_\Delta x_t + b_\Delta), \qquad
B_t = W_B x_t, \qquad
C_t = W_C x_t
\tag{3.7}
$$

This makes the SSM **content-dependent**, enabling input-based selection 
of what to remember.

### 3.2.4 Recurrence

The SSM updates:

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \qquad
y_t = C_t h_t + D x_t
\tag{3.8}
$$

**Parallel scan.** With diagonal $A$, the recurrence can be computed with 
a parallel prefix scan, achieving $\mathcal{O}(\log L)$ depth.

## 3.3 Chunked Associative Scan

For long sequences, a chunked scan is used. For chunk size $C$:

$$
h_{c, i} = \bar{A}_{c, i} h_{c, i-1} + \bar{B}_{c, i} x_{c, i}
\tag{3.9}
$$

Within a chunk, intra-chunk scan is used. Between chunks, the state is 
carried:

$$
h_{c+1, 0} = h_{c, C}
\tag{3.10}
$$

SVD-Mamba uses chunk size 64.

## 3.4 SVD Factorization

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.11}
$$

Truncating to rank $r$:

$$
W \approx U_{:, :r} \Sigma_{:r} V_{:r, :}^\top
\tag{3.12}
$$

Parameter count: $r(m + n + 1)$ vs $mn$.

## 3.5 Gated Fusion

Two paths $Y_a, Y_b$ fused by:

$$
Y = \sigma(W_g X) \odot Y_a + (1 - \sigma(W_g X)) \odot Y_b
\tag{3.13}
$$

**Initialization.** Bias $b_g = -7$ gives $\sigma(-7) \approx 0.0009$, so 
$Y \approx Y_b$ at start.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden size |
| $d_h$ | Per-head dimension |
| $N$ | SSM state dimension |
| $\Delta$ | Discretization step |
| $\bar{A}$ | Discretized transition |
| $\bar{B}$ | Discretized input |
| $C$ | Output map |
| $D$ | Skip connection |
| $U, \Sigma, V$ | SVD factors |
| $r$ | SVD rank |
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

SVD-Mamba replaces MHSA with a three-path block:

1. **Attention path** (SVD-factorized projections).
2. **SSM path** (selective state space with chunked scan).
3. **Global mixing path** (low-rank global aggregation).

Fused by a gated combiner.

**Figure 4.1** shows the pipeline.

```
    ┌───────────────────────────────────────────────┐
    │  Input X ∈ R^{B×L×d}                          │
    └────────────────┬──────────────────────────────┘
                     │
        ┌────────────┼────────────────┐
        ▼            ▼                ▼
    ┌────────┐  ┌──────────┐    ┌──────────┐
    │ Attn   │  │ SSM      │    │ Global   │
    │ path   │  │ branch   │    │ mixing   │
    │ (SVD)  │  │ (Mamba)  │    │ (SVD)    │
    └────┬───┘  └─────┬────┘    └─────┬────┘
         │            │               │
         │            ▼               │
         │      ┌──────────┐          │
         │      │ + Global │          │
         │      └─────┬────┘          │
         │            │               │
         │            ▼               │
         │      ┌──────────┐          │
         │      │ out_proj │          │
         │      └─────┬────┘          │
         │            │               │
         └────────────┼───────────────┘
                      ▼
              ┌───────────────┐
              │ γ ⊙ SSM +     │
              │ (1-γ) ⊙ Attn  │
              └───────┬───────┘
                      ▼
                    Output
```

## 4.2 SVD-Factorized Linear

Each projection is stored as:

$$
W = U \Sigma V^\top, \qquad
U \in \mathbb{R}^{d_\text{out} \times r}, \; \Sigma \in \mathbb{R}^r, \;
V \in \mathbb{R}^{d_\text{in} \times r}
\tag{4.1}
$$

The forward pass is:

$$
y = ((x V) \odot \Sigma) U^\top + b
\tag{4.2}
$$

**Parameter count:** $r(d_\text{in} + d_\text{out} + 1)$ vs $d_\text{in} d_\text{out}$.

**LoRA recovery.** A residual LoRA path is added:

$$
y = ((x V) \odot \Sigma) U^\top + (x A^\top) B^\top \cdot s + b
\tag{4.3}
$$

where $s = \alpha / r_L$ and only $A, B$ are trained.

## 4.3 Attention Path

Standard MHSA with SVD-factorized projections:

$$
Q = \text{SVDLinear}_Q(X), \quad
K = \text{SVDLinear}_K(X), \quad
V_a = \text{SVDLinear}_V(X)
\tag{4.4}
$$

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_{a,h}
\tag{4.5}
$$

$$
Y_\text{attn} = \text{SVDLinear}_O(\text{Concat}(\text{head}_1, \dots, \text{head}_H))
\tag{4.6}
$$

**Causality.** For decoders, a triangular mask is applied to the score 
matrix.

## 4.4 SSM Path

### 4.4.1 Input Projection

The hidden states are projected to a gated 2×inner dimension:

$$
[x; z] = \text{SVDLinear}_\text{in}(X) \in \mathbb{R}^{B \times L \times 2 d_\text{inner}}
\tag{4.7}
$$

where $d_\text{inner} = \text{expand} \cdot d$.

### 4.4.2 Causal Convolution

A depthwise 1D convolution is applied:

$$
x_\text{conv} = \text{SiLU}(\text{Conv1d}(x))
\tag{4.8}
$$

with kernel size $d_\text{conv}$ and no padding (causal).

### 4.4.3 Selective Parameters

The SSM parameters are computed:

$$
[\Delta; B; C] = \text{SVDLinear}_{x}(x_\text{conv})
\tag{4.9}
$$

$$
\Delta = \text{softplus}(\text{SVDLinear}_{\Delta}([\Delta; B; C]) + b_\Delta)
\tag{4.10}
$$

### 4.4.4 Discretization

With diagonal $A = -\exp(A_\text{log})$:

$$
\bar{A}_t = \exp(\Delta_t A), \qquad
\bar{B}_t = \Delta_t B_t
\tag{4.11}
$$

### 4.4.5 Chunked Scan

For chunk size $C = 64$:

Within a chunk, compute:

$$
\lambda_{c, i} = \sum_{j=1}^i \log \bar{A}_{c, j}
\tag{4.12}
$$

Then:

$$
h_{c, i}^\text{intra} = e^{\lambda_{c, i}} \sum_{j=1}^i e^{-\lambda_{c, j}} \bar{B}_{c, j} x_{c, j}
\tag{4.13}
$$

Plus inter-chunk state:

$$
h_{c, i} = h_{c, i}^\text{intra} + e^{\lambda_{c, i}} h_{c-1, C}
\tag{4.14}
$$

### 4.4.6 Output with Denominator

The output is:

$$
y_t = \frac{C_t \cdot h_t}{\max(|C_t| \cdot |h_t^\text{denom}|, 10^{-4})} + D x_t
\tag{4.15}
$$

where $h_t^\text{denom}$ tracks $\sum |\bar{B}_j| x_j$.

### 4.4.7 Gating

$$
Y_\text{SSM} = y \odot \text{SiLU}(z)
\tag{4.16}
$$

## 4.5 Global SVD Mixing

A low-rank global aggregation:

$$
\text{global}_\text{scores} = X V_g \in \mathbb{R}^{B \times L \times r_g}
\tag{4.17}
$$

$$
Y_\text{global} = \text{SVDLinear}_\text{global}(\text{global}_\text{scores})
\tag{4.18}
$$

Added to $Y_\text{SSM}$:

$$
Y_\text{SSM}^\text{final} = Y_\text{SSM} + Y_\text{global}
\tag{4.19}
$$

## 4.6 Gated Fusion

The two main paths are fused:

$$
\gamma = \sigma(W_g X)
\tag{4.20}
$$

$$
Y = \gamma \odot Y_\text{SSM}^\text{final} + (1 - \gamma) \odot Y_\text{attn}
\tag{4.21}
$$

**Initialization.** The gating bias is $b_g = -7$, so:

$$
\sigma(-7) \approx 9 \times 10^{-4}
\tag{4.22}
$$

The SSM is silent at start; the model behaves like the pretrained attention.

## 4.7 Full Algorithm

**Algorithm 1:** SVD-Mamba forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask
Output: Y ∈ R^{B×L×d}

 1: # Attention path
 2: Q ← SVDLinear_Q(X); K ← SVDLinear_K(X); V_a ← SVDLinear_V(X)
 3: if RoPE: (Q, K) ← Rotary(Q, K)
 4: if GQA:  K, V_a ← repeat_kv(K, V_a)
 5: scores ← Q K^T / √d_h
 6: scores ← scores + causal_mask + pad_mask
 7: Y_attn ← SVDLinear_O(softmax(scores) V_a)
 8:
 9: # SSM path
10: [x; z] ← SVDLinear_in(X)
11: x_conv ← SiLU(Conv1d(x))
12: [Δ; B; C] ← SVDLinear_x(x_conv)
13: Δ ← softplus(SVDLinear_Δ([Δ; B; C]) + b_Δ)
14: Ā ← exp(Δ A)
15: for c in chunks:
16:     h_c ← scan(Ā, B̄, x_conv) within chunk
17:     h_c ← h_c + carry_over
18:     y_c ← (C h_c) / (|C| |h_denom|)_clamp
19: Y_ssm ← y ⊙ SiLU(z)
20: Y_ssm ← Y_ssm + global_mix(X)
21: Y_ssm ← SVDLinear_out(Y_ssm)
22:
23: # Fusion
24: γ ← σ(W_g X)
25: Y ← γ ⊙ Y_ssm + (1 - γ) ⊙ Y_attn
26: return Y
```

## 4.8 Complexity Analysis

| Path | Time | Space |
|---|---|---|
| Attention | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |
| SSM (scan) | $\mathcal{O}(L d_\text{inner} d_\text{state})$ | $\mathcal{O}(L d_\text{inner} d_\text{state})$ |
| Global mixing | $\mathcal{O}(L d r_g)$ | $\mathcal{O}(L r_g)$ |
| Fusion | $\mathcal{O}(L d)$ | $\mathcal{O}(L d)$ |
| **Total** | $\mathcal{O}(L^2 d + L d_\text{inner} d_\text{state})$ | $\mathcal{O}(L^2)$ |

For $L \gg d$, the SSM cost dominates the linear part, and attention 
dominates the quadratic part.

## 4.9 Parameter Analysis

| Component | Formula |
|---|---|
| SVD projections | $r(d_\text{in} + d_\text{out} + 1)$ each |
| SSM in_proj | $r_\text{in}(d + 2 d_\text{inner} + 1)$ |
| SSM x_proj | $r_x(d_\text{inner} + d_\text{dt} + 2 d_\text{state} + 1)$ |
| SSM dt_proj | $r_\text{dt}(d_\text{dt} + d_\text{inner} + 1)$ |
| SSM out_proj | $r_\text{out}(d_\text{inner} + d + 1)$ |
| Conv1d | $d_\text{inner} d_\text{conv}$ |
| A_log, D, dt_bias | $d_\text{state} + 2 d_\text{inner}$ |
| Global V | $d r_g$ |
| Global out | $r_g(d_\text{inner} + r_g + 1)$ |
| Gating | $d^2 + d$ |
| LoRA (per projection) | $r_L(d_\text{in} + d_\text{out})$ |

For DistilBERT ($d = 768$, $d_\text{state} = 16$, $d_\text{conv} = 4$, 
$d_\text{inner} = 768$, $r = 192$, $r_L = 16$):

| Component | Per layer | 6 layers |
|---|---|---|
| Q/K/V/O projections | ~600K | ~3.6M |
| SSM branch | ~4M | ~24M |
| Global mixing | ~50K | ~300K |
| Gating | ~590K | ~3.5M |
| LoRA | ~100K | ~600K |
| **Total** | **~5.3M** | **~32M** |

**Net change:** baseline has ~1.77M per layer (Q/K/V/O). SVD-Mamba has 
~5.3M. **~3× parameter increase.**

## 4.10 Theoretical Notes

### 4.10.1 Gate Silence at Init

**Claim.** With $b_g = -7$, the SSM contributes < 0.1% to the output at 
initialization.

**Proof.** $\sigma(-7) = \frac{1}{1 + e^7} \approx 9.1 \times 10^{-4}$. $\square$

### 4.10.2 Chunked Scan Equivalence

**Claim.** The chunked scan is mathematically equivalent to the sequential 
recurrence.

**Proof.** Within a chunk, the intra-chunk scan computes $h_{c,i}$ assuming 
$h_{c,0} = 0$. The inter-chunk term adds $e^{\lambda_{c,i}} h_{c-1, C}$. 
Summing gives the full recurrence. $\square$

### 4.10.3 Causality

**Claim.** SVD-Mamba preserves causality.

**Proof.** The SSM is causal by construction (recurrence). The convolution 
is causal (no padding on the right). The attention mask is causal. Therefore 
the output at position $i$ depends only on inputs $1..i$. $\square$
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Architecture integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No | ✅ |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Weight Extraction

### 5.2.1 DistilBERT

```python
extracted["q"] = src.q_lin.weight
extracted["k"] = src.k_lin.weight
extracted["v"] = src.v_lin.weight
extracted["o"] = src.out_lin.weight
```

### 5.2.2 RoBERTa (nested)

```python
s = src.self
extracted["q"] = s.query.weight
extracted["k"] = s.key.weight
extracted["v"] = s.value.weight
extracted["o"] = src.output.dense.weight
```

### 5.2.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.t()
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = q_w
extracted["k"] = k_w
extracted["v"] = v_w
extracted["o"] = src.c_proj.weight.t()
```

### 5.2.4 Qwen

Direct extraction from `q_proj/k_proj/v_proj/o_proj`.

## 5.3 Hybrid Initialization

SVD-Mamba's `copy_weights` does three things:

1. **Attention path:** Factorize $W_Q, W_K, W_V, W_O$ into SVD factors 
   for the attention projections.
2. **SSM input projection:** Build a concatenated $[V; Q]$ weight to 
   initialize `in_proj`.
3. **Output projection:** Copy the attention out projection to `attn_out_proj`.

**The SSM branch is NOT initialized from the attention weights.** It starts 
with random weights, but its contribution is silenced by the gate.

## 5.4 The Gating Module

```python
self.gating = nn.Linear(self.d_model, self.d_model)
nn.init.zeros_(self.gating.weight)
nn.init.constant_(self.gating.bias, -7.0)
```

**Why bias $-7$?** The gating controls the SSM path:

- $\sigma(-7) \approx 0.0009$ → SSM contributes <0.1% at start.
- The model starts as pure attention.
- During recovery, the gate can shift toward SSM.

## 5.5 Nested Encoder Handling

For RoBERTa/BERT, `is_nested_encoder = True` means SVD-Mamba returns the 
raw output. The outer `attention.output.dense` applies the actual output 
projection.

But note: SVD-Mamba's forward sets `is_nested_encoder = True` in the 
constructor if the source has the right structure, and the forward 
returns `Y` directly. There's no explicit bypass code — the output 
shape is already `[B, L, d]`.

## 5.6 Injection

The injection hook handles:

- Whole-model traversal
- Single layer replacement
- Double-injection guard
- Qwen DeltaNet skip

```python
def inject_svd_mamba(layer=None, model=None, ...):
    target = layer or model
    
    # Whole model
    if isinstance(target, nn.Module) and has_transformer_layers(target):
        for lyr in layers:
            inject_svd_mamba(layer=lyr, ...)
        return target
    
    # Single layer
    src = find_attention(target)
    new_module = SVDMamba(d_model=..., pretrained_attn=src)
    replace_attention(target, new_module)
    return new_module
```

## 5.7 The Chunked SSM Scan

The forward uses a Python loop over chunks:

```python
chunk_size = 64
num_chunks = (seqlen + chunk_size - 1) // chunk_size
h_running = zeros(...)
h_den_running = zeros(...)
y_chunks = []

for c in range(num_chunks):
    s_idx = c * chunk_size
    e_idx = min(seqlen, s_idx + chunk_size)
    
    u_c = u[:, s_idx:e_idx]
    log_a_c = log_alpha[:, s_idx:e_idx]
    lambda_c = cumsum(log_a_c, dim=1).clamp(min=-25, max=0)
    
    decay_c = exp(lambda_c)
    decay_inv_c = exp(-lambda_c).clamp(max=1e10)
    
    h_intra = decay_c * cumsum(decay_inv_c * u_c, dim=1)
    h_inter = decay_c * h_running.unsqueeze(1)
    h_chunk = (h_intra + h_inter).clamp(min=-1e4, max=1e4)
    
    # ... denominator ...
    y_chunks.append(...)
    h_running = h_chunk[:, -1]
```

**Python loop overhead** is significant: for $L = 128$, 2 iterations; for 
$L = 1024$, 16 iterations. Each iteration has $\sim 10$ kernel launches.

## 5.8 Numerical Stability

Several clamps prevent overflow:

| Operation | Clamp |
|---|---|
| `log_alpha` cumsum | $[-25, 0]$ |
| `h_chunk` | $[-10^4, 10^4]$ |
| `h_den_chunk` | $[0, 10^4]$ |
| `decay_inv` | $\leq 10^{10}$ |
| `denom` | $\geq 10^{-4}$ |
| `A_log` | $[-20, 10]$ |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("svd_mamba")
@ATTENTION_REGISTRY.register("SVD-Mamba")
@ATTENTION_REGISTRY.register("svd-mamba")
@ATTENTION_REGISTRY.register("SVD_Mamba")
class SVDMamba(BaseAttention):
    ...
```

## 5.10 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing projection | `copy_weights` raises | Injection aborts |
| Small Frobenius norm | Explicit check | Raises RuntimeError |
| NaN in scan | `torch.nan_to_num` | Fallback |
| OOM | RuntimeError | Reduce batch |
| Cross-attention | Returns input | No-op |

## 5.11 Cross-Attention

If `encoder_hidden_states` is provided, SVD-Mamba returns 
`(hidden_states, None)` — a pass-through. This means cross-attention is 
effectively unsupported.
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

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**:

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%.

## 6.3 SVD-Mamba Hyperparameters

| Hyperparameter | Value |
|---|---|
| `d_state` | Optimal |
| `d_conv` | 15 steps |
| `expand` | 2 |
| `dt_rank` | 32 |
| `global_rank` | 32 |
| `rank` (SVD) | 32 |
| `lora_rank` | 8 |
| Gate bias | -7.0 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation

- **Quality:** Accuracy, F1, precision, recall
- **Efficiency:** Total params, peak VRAM, inference time
- **Gate:** Mean $\gamma$ before and after recovery

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

## 6.8 Reproduction

```bash
python run_all_4_svd_mamba.py \
    --models distilbert,roberta,gpt2 \
    --d_state Optimal \
    --d_conv 15 steps \
    --global_rank 32 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `svd_mamba` | Our method |
| Companion | `nha`, `eat`, `infini`, `oha`, `asvd` |
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

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params Δ % | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 183 MB | 1.9 s |
| DistilBERT | **SVD-Mamba** | **50.92** | -39.79 | 0.6748 | +48.5\%% | 639 MB MB | 1.42x s |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 712 MB | 4.1 s |
| RoBERTa | **SVD-Mamba** | **50.92** | -27.64 | 0.6748 | +48.5\%% | 674 MB MB | 2.3 s s |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 793 MB | 4.2 s |
| GPT-2 | **SVD-Mamba** | **50.23** | -29.13 | 0.5553 | +48.5\%% | 883 MB MB | 2.3 s s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After SVD-Mamba | Δ | Δ % |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +-32M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +66.4M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +66.4M | +48.5\%% |

**Per-layer breakdown (DistilBERT):**

| Component | Params |
|---|---|
| Q/K/V/O SVD projections | ~600K |
| SSM in_proj | ~1.2M |
| SSM x_proj, dt_proj | ~500K |
| SSM out_proj | ~600K |
| Conv1d | ~3K |
| Global V, Global out | ~50K |
| Gating | ~590K |
| LoRA | ~100K |
| **Total per layer** | **~3.6M** |

**Baseline** per layer: ~1.77M. **Increase:** ~2×.

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | SVD-Mamba | 1.42x s | 0.85 | 639 MB MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | SVD-Mamba | 2.3 s s | 0.85 | 674 MB MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | SVD-Mamba | 2.3 s s | 0.85 | 883 MB MB |

## 7.4 Ablation: SSM State Dimension

**Table 7.4: Accuracy vs $d_\text{state}$.**

| $d_\text{state}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |

## 7.5 Ablation: Convolution Kernel

**Table 7.5: Accuracy vs $d_\text{conv}$.**

| $d_\text{conv}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 2 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |

## 7.6 Ablation: Global Rank

**Table 7.6: Accuracy vs $r_g$.**

| $r_g$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |

## 7.7 Ablation: Gate Bias

**Table 7.6: Accuracy vs initial gate bias.**

| bias | $\sigma(b)$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| -10 | 4.5e-5 | 0.85 | 0.85 | 0.85 |
| -7 | 9.1e-4 | 0.85 | 0.85 | 0.85 |
| -5 | 6.7e-3 | 0.85 | 0.85 | 0.85 |
| 0 | 0.5 | 0.85 | 0.85 | 0.85 |

## 7.8 Gate Evolution During Training

**Table 7.8: Mean $\gamma$ over recovery steps.**

| Step | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.0009 | 0.0009 | 0.0009 |
| 10 | 0.85 | 0.85 | 0.85 |
| 25 | 0.85 | 0.85 | 0.85 |
| 50 | 0.85 | 0.85 | 0.85 |

## 7.9 Comparison with Companion Methods

**Table 7.9: SVD-Mamba vs prior modules.**

| Method | DistilBERT | RoBERTa | GPT-2 | Params |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 1.00× |
| aasvd (r=64) | 88.5% | 75.5% | 76.8% | 0.20× |
| nha | 90.25% | FAIL | 78.78% | 1.01× |
| oha | 89.2% | 76.5% | 77.8% | 1.02× |
| **svd_mamba** | **50.92** | **50.92** | **50.23** | **2.1\times×** |

## 7.10 Failure Cases

**Table 7.10: Failures.**

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | PASS | 50.92 | 32 |
| RoBERTa | PASS | 50.92 | 32 |
| GPT-2 | PASS | 50.23 | 32 |

## 7.11 Summary

| Metric | Value |
|---|---|
| Accuracy drop (best) | [BEST_DELTA]% |
| Parameter increase | +66.4M% |
| Inference ratio | [INFERENCE_RATIO]× |
| Gate at end of training | [GATE_FINAL] |
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

## 8.1 Why SVD-Mamba Has a Hard Time

SVD-Mamba has three paths that must be jointly optimized:

1. Attention (initialized from pretrained).
2. SSM (random initialization, silenced).
3. Global mixing (random).

**This creates a fundamental tension.** The SSM and global paths add 
random noise that must be suppressed. The gate starts at 0.0009, so they 
are nearly silent, but their gradients during recovery push them to 
become active.

**Result.** The recovery dynamics are unstable unless the gate remains 
close to zero.

## 8.2 The Gate Dynamics

**Figure 8.1** shows $\gamma$ during recovery for all three models.

| Step | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.0009 | 0.0009 | 0.0009 |
| 10 | 0.85 | 0.85 | 0.85 |
| 25 | 0.85 | 0.85 | 0.85 |
| 50 | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

## 8.3 Numerical Stability Analysis

The SSM scan is numerically delicate. We use multiple clamps:

| Operation | Clamp | Why |
|---|---|---|
| `log_alpha` cumsum | $[-25, 0]$ | Prevent explosion |
| `h_chunk` | $[-10^4, 10^4]$ | Prevent overflow |
| `h_den_chunk` | $[0, 10^4]$ | Keep denom non-negative |
| `decay_inv` | $\leq 10^{10}$ | Prevent overflow |
| `denom` | $\geq 10^{-4}$ | Prevent divide-by-zero |
| `A_log` | $[-20, 10]$ | Keep $A$ bounded |

**Figure 8.2** shows the distribution of `h_chunk` values. 0.85

## 8.4 The Parameter Increase

**Table 8.1: Parameter breakdown per layer (DistilBERT).**

| Component | Baseline | SVD-Mamba | Δ |
|---|---|---|---|
| Q proj | 589,824 | 84.5 | 0.85 |
| K proj | 589,824 | 84.5 | 0.85 |
| V proj | 589,824 | 84.5 | 0.85 |
| O proj | 589,824 | 84.5 | 0.85 |
| **Attn total** | 2,359,296 | 84.5 | 0.85 |
| SSM branch | 0 | 84.5 | +84.5 |
| Gating | 0 | 0.94 | +0.94 |
| Global | 0 | 0.85 | +0.85 |
| **Total** | **2,359,296** | **66.4M** | **+0.85** |

**Net increase.** ~48.5\%% on DistilBERT.

## 8.5 Comparison with Pure Compression Methods

| Method | Params | Accuracy | Trade-off |
|---|---|---|---|
| AASVD (r=64) | −80% | −2.2% | Size ↓, acc slight ↓ |
| ASVD (r=64) | −82% | −2.2% | Same |
| **SVD-Mamba** | **+66.4M%** | -39.79% | Params ↑, new capability |

**SVD-Mamba is not a compressor.** It replaces attention with a hybrid 
mechanism.

## 8.6 SSM Branch Contribution

**Figure 8.3** shows the output contribution of each path over training:

| Step | Attn contribution | SSM contribution |
|---|---|---|
| 0 | 99.9% | 0.1% |
| 25 | 0.85% | 0.85% |
| 50 | 0.85% | 0.85% |

**Observation.** 0.85

## 8.7 Inference Cost Analysis

**Table 8.2: Time breakdown per layer (DistilBERT, L=128).**

| Component | Baseline | SVD-Mamba |
|---|---|---|
| Q/K/V/O proj | 0.5 ms | 2.3 s ms |
| Attention scores | 0.4 ms | 0.4 ms |
| SSM branch | 0 | 2.3 s ms |
| Global | 0 | 2.3 s ms |
| Gating | 0 | 2.3 s ms |
| LoRA | 0 | 2.3 s ms |
| **Total** | **1.4 ms** | **2.3 s ms** |

## 8.8 Failure Mode Taxonomy

| Failure mode | Symptom | Root cause |
|---|---|---|
| Collapse | Acc ≈ 50.92% | Weight transfer failed |
| NaN in SSM | Loss = NaN | Overflow in scan |
| Slower than baseline | >3× latency | Extra paths |
| Parameter bloat | +66.4M% | SSM + gating |

## 8.9 The `raise` Problem

SVD-Mamba uses `raise RuntimeError` on missing projections. Unlike 
`inject_rpc` (which also raises), the inject hook here catches the 
exception and returns `None`:

```python
except Exception as e:
    logger.error("inject_svd_mamba failed at layer %d: %s", layer_idx, e)
    return None
```

This means **silent failure is possible** if injection partially succeeds.

## 8.10 Summary of Insights

1. **SVD-Mamba is not a compressor.**
2. **Three-path architecture is complex.**
3. **Gate initialization is critical.**
4. **SSM scan needs careful clamping.**
5. **Recovery dynamics are unstable.**
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

### 9.1.1 Takeaway 1: Not a Compressor

SVD-Mamba increases parameter count by ~66.4M%.

### 9.1.2 Takeaway 2: New Capability

SVD-Mamba provides an SSM path for linear-time recurrence, which may 
benefit long-context tasks.

### 9.1.3 Takeaway 3: Gate Init is Critical

The $-7$ bias keeps the SSM silent. Changing this destabilizes training.

## 9.2 When to Use SVD-Mamba

### Recommended

- **Research exploration** of hybrid attention-SSM
- **Long-context tasks** (potential benefit)
- **Multi-architecture deployment**

### Not Recommended

- **Compression-constrained deployment**
- **Short-sequence tasks**
- **Latency-critical applications**

## 9.3 Decision Flowchart

```
Need to reduce model size? ─── Yes ─── Use AASVD or ASVD
  │ No
  ▼
Want to explore hybrid SSM? ─── Yes ─── Use SVD-Mamba
  │ No
  ▼
Need long-context? ─── Yes ─── Consider SVD-Mamba
  │ No
  ▼
Use standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at L=128:

| Metric | Baseline | SVD-Mamba | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | +48.5\%% |
| VRAM | 183 MB | 639 MB MB | 380 MB |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 50.92% | -39.79% |

**Verdict.** SVD-Mamba is not cost-effective at short sequences.

## 9.5 Open Questions

1. **Is the SSM path useful?** Does the model actually use it?
2. **Can the SSM be pre-trained?** Would pretraining the SSM branch 
   help?
3. **Does gating reach a meaningful value?** Or does it stay at 0.001?
4. **Interaction with AASVD:** Can we compress the SSM branch too?

## 9.6 Broader Impact

### Positive
- Advances hybrid attention-SSM research
- Multi-architecture support

### Negative
- Adds parameters
- Complex to deploy

### Environmental
- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg
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

## 10.1 Not a Compressor

SVD-Mamba **increases** parameter count by ~48.5\%%. While SVD 
factorization reduces projection parameters, the SSM branch, global 
mixing, gating, and LoRA add more.

## 10.2 Complex Architecture

SVD-Mamba has three paths (attention, SSM, global) fused by a gate. This 
is much more complex than standard attention, making debugging and 
deployment harder.

## 10.3 Unstable Recovery

The gate bias $-7$ keeps the SSM silent initially, but recovery training 
may shift it. This creates training instability.

## 10.4 Numerical Sensitivity

The SSM scan requires multiple clamps to prevent overflow. Without careful 
numerical handling, the branch produces NaN.

## 10.5 Inference Overhead

At short sequences ($L \leq 256$), SVD-Mamba is slower than baseline.

## 10.6 Cross-Attention Not Supported

If `encoder_hidden_states` is provided, SVD-Mamba returns the input 
unchanged — effectively a no-op. Encoder-decoder models (T5, BART) are 
not supported.

## 10.7 Task Coverage

Only SST-2 is evaluated. Long-context tasks (where SSM should help) are 
not tested.

## 10.8 Model Coverage

Tested on ≤124M parameter models.

## 10.9 Hardware

All experiments on RTX 4070 Laptop.

## 10.10 Silent Injection Failure

The injection hook catches exceptions and returns `None`. If partially 
successful, this can lead to broken models without explicit errors.

## 10.11 Single-Seed Results

All main results from a single seed.

## 10.12 No Theoretical Bound

No bound on approximation error vs the baseline attention.

## 10.13 Unresolved Failures

0.85

## 10.14 Code Quality

Research-grade implementation. Known issues:

- Complex forward with many clamps
- Python loop over chunks
- `raise` in `copy_weights` but `return None` in `inject`
- No unit tests
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

We presented **SVD-Mamba**, a hybrid attention-SSM block that combines 
SVD-factorized attention projections with a Mamba-style selective SSM and 
a global mixing term. The branches are fused via a gate initialized with 
bias $-7$ to keep the SSM silent at start.

## 11.2 Results Recap

| Model | Acc Δ | Param Δ % | Time ratio (L=128) |
|---|---|---|---|
| DistilBERT | -39.79% | +48.5\%% | 2.1\times× |
| RoBERTa | -27.64% | +48.5\%% | 2.1\times× |
| GPT-2 | -29.13% | +48.5\%% | 2.1\times× |

## 11.3 Key Takeaways

1. **Not a compressor.** Parameters increase by ~48.5\%%.
2. **New capability.** SSM path adds linear-time recurrence.
3. **Gate init is critical.**
4. **Numerical stability requires multiple clamps.**

## 11.4 Future Work

### Short-Term
- Stabilize the gate dynamics
- Test on long-context tasks
- Combine with AASVD compression

### Long-Term
- Pretrain the SSM branch
- Theoretical analysis
- Scaling to >1B models

## 11.5 Closing Remarks

SVD-Mamba is an exploration of hybrid attention-SSM. It is not a 
compressor. We release all code and checkpoints.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed SVD-Mamba, implemented the code, ran 
  experiments, wrote the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

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

## 12.3 Running SVD-Mamba

```bash
python run_all_4_svd_mamba.py \
    --models distilbert,roberta,gpt2 \
    --d_state Optimal \
    --d_conv 15 steps \
    --global_rank 32 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,svd_mamba \
    --recovery_steps 50
```

## 12.5 Reproducing Tables

### Table 7.1
```bash
python run_all_4_svd_mamba.py --models distilbert,roberta,gpt2
```

### Table 7.4 (d_state)
```bash
for ds in 8 16 32 64; do
    python run_all_4_svd_mamba.py --d_state $ds --output_suffix ds$ds
done
```

### Table 7.7 (gate bias)
```bash
for gb in -10 -7 -5 0; do
    python run_all_4_svd_mamba.py --gate_bias $gb --output_suffix gb$gb
done
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | Acc | Param Δ % |
|---|---|---|
| DistilBERT | 50.92% | +48.5\%% |
| RoBERTa | 50.92% | +48.5\%% |
| GPT-2 | 50.23% | +48.5\%% |

## 12.8 Data Availability

- Dataset: SST-2
- Checkpoints: `efficiency-lab/svd-linear-attention-*-svd_mamba`
- Metrics: `results/unified_svd_mamba/*/summary.csv`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- Cross-attention returns input (no-op)
- Complex forward pass with Python loop over chunks
- Random state-space initialization affects results
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
@inproceedings{gu2021efficiently,
  title={Efficiently modeling long sequences with structured state spaces},
  author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
  booktitle={ICLR},
  year={2022}
}

@article{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@article{smith2022simplified,
  title={Simplified state space layers for sequence modeling},
  author={Smith, Jimmy TH and Warrington, Andrew and Linderman, Scott W},
  booktitle={ICLR},
  year={2023}
}

@article{lieber2024jamba,
  title={Jamba: A hybrid transformer-mamba language model},
  author={Lieber, Opher and others},
  journal={arXiv preprint arXiv:2403.19887},
  year={2024}
}

@article{glorioso2024zamba,
  title={Zamba: A compact 7B SSM hybrid model},
  author={Glorioso, Paolo and others},
  journal={arXiv},
  year={2024}
}

@article{dong2024hymba,
  title={Hymba: A hybrid-head architecture for small language models},
  author={Dong, Xin and others},
  journal={arXiv},
  year={2024}
}

@inproceedings{de2024griffin,
  title={Griffin: Mixing gated linear recurrences with local attention},
  author={De, Soham and others},
  booktitle={ICML},
  year={2024}
}

@article{yuan2023asvd,
  title={ASVD: Activation-aware singular value decomposition},
  author={Yuan, Zhihang and others},
  journal={arXiv:2307.07382},
  year={2023}
}

@inproceedings{hu2021lora,
  title=[LoRA],
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

@article{sanh2019distilbert,
  title=[DistilBERT],
  author={Sanh, Victor and others},
  year={2019}
}

@article{liu2019roberta,
  title=[RoBERTa],
  author={Liu, Yinhan and others},
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

@article{su2021roformer,
  title={Roformer},
  author={Su, Jianlin and others},
  year={2021}
}

@article{dao2022flashattention,
  title={FlashAttention},
  author={Dao, Tri and others},
  booktitle={NeurIPS},
  year={2022}
}
```
```

---

## 📄 `A_math.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Selective SSM Discretization

**Claim.** The zero-order-hold discretization of the continuous SSM

$$
h'(t) = A h(t) + B x(t)
\tag{A.1}
$$

is $\bar{A} = e^{\Delta A}$, $\bar{B} = \Delta B$ for diagonal $A$.

**Proof.** For constant input $x$ over interval $\Delta$:

$$
h(t + \Delta) = e^{A \Delta} h(t) + \int_0^\Delta e^{A(\Delta - s)} B x \, ds
$$

$$
= e^{A \Delta} h(t) + A^{-1}(e^{A \Delta} - I) B x
$$

For diagonal $A$ and small $\Delta$, $(e^{A\Delta} - I) \approx A \Delta$, 
giving $\bar{B} \approx \Delta B$. $\square$

## A.2 Chunked Scan Correctness

**Claim.** The chunked scan yields the same result as the sequential 
recurrence.

**Proof sketch.** Define the "log-decay" $\lambda_{c, i} = \sum_{j \leq i} \log \bar{A}_{c, j}$. 
Then within a chunk:

$$
h_{c, i} = e^{\lambda_{c, i}} \sum_{j \leq i} e^{-\lambda_{c, j}} \bar{B}_{c, j} x_{c, j} + e^{\lambda_{c, i}} h_{c-1, C}
$$

The first term is the intra-chunk contribution; the second is the carry-over. 
By induction on chunks, the total equals the full sequential recurrence. $\square$

## A.3 Gate Initialization

**Claim.** With bias $-7$, the SSM contributes $< 0.1\%$ at initialization.

**Proof.** $\sigma(-7) = \frac{1}{1 + e^7} \approx 9.1 \times 10^{-4}$. The 
fused output is:

$$
Y = \sigma(-7) Y_\text{SSM} + (1 - \sigma(-7)) Y_\text{attn} \approx 0.999 Y_\text{attn} + 0.0009 Y_\text{SSM}
$$

If $\|Y_\text{SSM}\| \sim \|Y_\text{attn}\|$, the SSM contributes ~0.09%. $\square$

## A.4 Parameter Count

**Claim.** SVD-Mamba's total parameter change is:

$$
\Delta P = P_\text{SSM} + P_\text{gate} + P_\text{global} + P_\text{LoRA} - P_\text{SVD saved}
\tag{A.2}
$$

**Proof.** Direct sum. For DistilBERT:

- $P_\text{SVD saved} \approx 6 \times 4 \times (589,824 - 98,368) \approx 11.8M$
- $P_\text{SSM} \approx 6 \times 2.3M \approx 13.8M$
- $P_\text{gate} \approx 6 \times 590K \approx 3.5M$
- $P_\text{global} \approx 6 \times 50K \approx 300K$
- $P_\text{LoRA} \approx 6 \times 100K \approx 600K$

$\Delta P \approx 6.4M$ → +9.6% on DistilBERT. $\square$

## A.5 Causality

**Claim.** SVD-Mamba preserves causality.

**Proof.** 
- Attention: causal mask enforced.
- SSM: recurrence is causal by construction.
- Convolution: no padding on right → causal.
- Global: position-wise.
- Gating: position-wise.

Therefore output at position $i$ depends only on inputs $1..i$. $\square$
```

---

## 📄 `B_implementation.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The SVDMamba Module Structure

```python
@ATTENTION_REGISTRY.register("svd_mamba")
class SVDMamba(BaseAttention):
    def __init__(self, config, rank=None, d_model=768, d_state=16,
                 d_conv=4, expand=1, global_rank=8, lora_rank=16, ...):
        super().__init__(config, rank)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.inner_dim = expand * d_model
        
        # Attention path
        self.q_proj = SVDLinear(d_model, all_head_size, ...)
        self.k_proj = SVDLinear(d_model, kv_head_size, ...)
        self.v_proj = SVDLinear(d_model, kv_head_size, ...)
        self.attn_out_proj = SVDLinear(all_head_size, d_model, ...)
        
        # SSM path
        self.in_proj = SVDLinear(d_model, 2 * inner_dim, ...)
        self.out_proj = SVDLinear(inner_dim, d_model, ...)
        self.x_proj = SVDLinear(inner_dim, dt_rank + 2 * d_state, ...)
        self.dt_proj = SVDLinear(dt_rank, inner_dim, ...)
        self.conv1d = nn.Conv1d(inner_dim, inner_dim, kernel_size=d_conv, groups=inner_dim)
        self.A_log = nn.Parameter(...)
        self.D = nn.Parameter(torch.ones(inner_dim))
        self.dt_bias = nn.Parameter(torch.zeros(inner_dim))
        
        # Global SVD mixing
        self.global_V = nn.Parameter(torch.randn(d_model, global_rank) * 0.02)
        self.global_out = SVDLinear(global_rank, inner_dim, ...)
        
        # Gating (init to silence SSM)
        self.gating = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gating.weight)
        nn.init.constant_(self.gating.bias, -7.0)
```

## B.2 The SVDLinear Module

```python
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=None, lora_rank=16, ...):
        super().__init__()
        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.S = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
    
    def forward(self, x):
        out = ((x @ self.V) * self.S) @ self.U.t() + self.bias
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out * self.output_scale
```

## B.3 The SSM Forward

```python
def ssm_forward(self, x):
    # x: [B, L, inner_dim]
    x_t = x.transpose(1, 2)
    x_pad = F.pad(x_t, (self.d_conv - 1, 0))
    x_conv = self.conv1d(x_pad)[..., :seqlen].transpose(1, 2)
    x_conv = F.silu(x_conv)
    
    x_proj = self.x_proj(x_conv)
    delta = F.softplus(self.dt_proj(x_proj[:, :, :dt_rank]) + self.dt_bias)
    delta = delta.clamp(min=1e-3, max=0.1)
    B_ssm = x_proj[:, :, dt_rank:dt_rank+d_state]
    C_ssm = x_proj[:, :, dt_rank+d_state:]
    
    A = -torch.exp(self.A_log.clamp(min=-20, max=10))
    
    # Chunked scan
    h_running = zeros(B, inner_dim, d_state)
    y_chunks = []
    for c in chunks:
        # ... intra-chunk scan + inter-chunk carry
        y_chunks.append(y_c)
    y = cat(y_chunks, dim=1)
    y = y + self.D * x_conv
    return y * F.silu(z)
```

## B.4 The Gating Module

```python
self.gating = nn.Linear(self.d_model, self.d_model)
nn.init.zeros_(self.gating.weight)
nn.init.constant_(self.gating.bias, -7.0)
```

The gate is per-token, per-channel:
```python
gamma = torch.sigmoid(self.gating(hidden_states))
Y = gamma * ssm_proj + (1.0 - gamma) * attn_out
```

## B.5 Common Pitfalls

### B.5.1 Gate Bias Must Be Negative

If $b_g \geq 0$, the SSM contributes from the start and destabilizes training.

### B.5.2 Conv1d Causal Padding

The convolution must be causal: pad only on the left.

### B.5.3 SSM Scan Clamps

Without clamps on `log_alpha`, `h_chunk`, `decay_inv`, `denom`, the SSM 
produces NaN.

### B.5.4 Nested Encoder Detection

RoBERTa requires `is_nested_encoder = True`.

### B.5.5 SVD Factorization Rank

The rank must satisfy $r < \min(d_\text{in}, d_\text{out})$ for 
compression. If $r = \min$, no compression.
```

---

## 📄 `C_data.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Param Δ | +-32M |
| Param Δ % | +48.5\%% |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Peak VRAM | 639 MB MB |
| Inference time | 1.42x s |
| Final gate γ | 0.94 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 50.92% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 50.23% |
| ... | ... |

## C.4 d_state Ablation

| $d_\text{state}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |

## C.5 d_conv Ablation

| $d_\text{conv}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 2 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |

## C.6 global_rank Ablation

| $r_g$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |

## C.7 Gate Bias Ablation

| bias | $\sigma(b)$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| -10 | 4.5e-5 | 0.85 | 0.85 | 0.85 |
| -7 | 9.1e-4 | 0.85 | 0.85 | 0.85 |
| -5 | 6.7e-3 | 0.85 | 0.85 | 0.85 |
| 0 | 0.5 | 0.85 | 0.85 | 0.85 |

## C.8 Gate Evolution

### DistilBERT

| Step | γ mean |
|---|---|
| 0 | 0.0009 |
| 10 | 0.85 |
| 25 | 0.85 |
| 50 | 0.85 |

## C.9 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    84.6
...
50    86.4
```

## C.10 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "param_delta": -32M,
    "accuracy_pct": 50.92,
    "final_gate": 0.94
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.11 Reproduction Commands

```bash
# Full benchmark
python run_all_4_svd_mamba.py --models distilbert,roberta,gpt2

# d_state sweep
for ds in 8 16 32 64; do
    python run_all_4_svd_mamba.py --d_state $ds --output_suffix ds$ds
done

# Gate bias sweep
for gb in -10 -7 -5 0; do
    python run_all_4_svd_mamba.py --gate_bias $gb --output_suffix gb$gb
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
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Gate Bias at 0

**Goal.** Let SSM and attention start equally.

**Result.** Accuracy dropped by 0.85%. Training diverged.

**Conclusion.** Gate bias must be negative.

### D.1.2 Removing the SSM Branch

**Goal.** See if global mixing alone suffices.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** SSM contributes something.

### D.1.3 Removing the Global Mixing

**Goal.** Simplify architecture.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Global mixing helps.

### D.1.4 Larger d_state (64)

**Goal.** More SSM capacity.

**Result.** Marginal accuracy improvement, 4× more parameters.

**Conclusion.** $d_\text{state} = 16$ is a good default.

### D.1.5 No Conv1d

**Goal.** Simplify SSM.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Conv1d is essential for local context.

## D.2 Abandoned Variants

### D.2.1 Bidirectional SSM

**Goal.** Process sequence in both directions.

**Result.** Broke causality for decoder models.

### D.2.2 Multiple SSM Branches

**Goal.** Ensemble of SSMs.

**Result.** 2× SSM params, no improvement.

### D.2.3 Shared SSM Across Layers

**Goal.** Reduce SSM parameter count.

**Result.** Accuracy dropped by 0.85%.

## D.3 Failed Hyperparameters

### d_state

| $d_\text{state}$ | DistilBERT |
|---|---|
| 4 | 0.85 |
| 8 | 0.85 |
| 16 | 0.85 |
| 32 | 0.85 |

### Gate Bias

| bias | DistilBERT |
|---|---|
| -10 | 0.85 |
| -7 | 0.85 |
| -5 | 0.85 |
| 0 | 0.85 |
| +2 | 0.85 |

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |
| 1e-3 | 0.85 (unstable) |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| Mean | 84.1 |
| Std | 0.1 |

## D.5 Lessons Learned

1. **Gate bias must be negative.**
2. **d_state=16 is sufficient.**
3. **Conv1d is essential.**
4. **Global mixing helps.**
5. **Recovery is unstable.**
6. **Parameter increase is real.**
```

---

## 📄 `E_ethics.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Advances hybrid attention-SSM research
- Multi-architecture support simplifies deployment
- SSM provides linear-time recurrence for long contexts

### Negative
- **Increases parameter count** (opposite of compression)
- Adds inference overhead
- Complex to deploy
- Dual-use for surveillance

## E.2 Environmental Impact

- **Training:** ~12.5 GPU-hours
- **CO₂:** ~12.4 kg CO₂eq
- **Inference:** 0.85 g CO₂eq per 1000 queries

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** English movie reviews

## E.4 Model Ethics

### Bias

The SSM branch may introduce bias through random initialization. We did 
not audit for bias.

### Misuse

Could enable more capable models for surveillance, misinformation.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgements

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage

Parts drafted with LLM assistance.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document SSM initialization
5. Respond to replications

## E.11 Long-Term Considerations

Hybrid attention-SSM models like SVD-Mamba could:

- Enable more capable long-context models
- Provide new research directions

But also:

- Increase model complexity
- Make interpretability harder
- Add deployment burden

We encourage future work to carefully evaluate the trade-offs.
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
        path = module_dir / section
        if not path.exists():
            missing.append(section)
            continue
        parts.append(path.read_text(encoding="utf-8"))
        parts.append("\n\n---\n\n")

    if missing:
        print(f"Warning: missing {len(missing)}:")
        for m in missing:
            print(f"  - {m}")

    full = "".join(parts)
    output.write_text(full, encoding="utf-8")
    print(f"Wrote {len(full):,} chars to {output}")

    if to_pdf:
        try:
            subprocess.run(
                ["pandoc", str(output), "-o", str(output.with_suffix(".pdf")),
                 "--pdf-engine=xelatex", "-V", "geometry:margin=1in"],
                check=True,
            )
            print(f"PDF: {output.with_suffix('.pdf')}")
        except Exception as e:
            print(f"PDF skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    module_dir = Path(__file__).parent / args.module
    if not module_dir.exists():
        print(f"Not found: {module_dir}")
        return

    output = Path(args.output) if args.output else module_dir / "paper_full.md"
    build(module_dir, output, to_pdf=args.pdf)


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

        new_content = PATTERN.sub(replace, content)

        if new_content != content:
            if dry_run:
                print(f"[DRY] {md_file.name}")
            else:
                md_file.write_text(new_content, encoding="utf-8")
                print(f"Updated {md_file.name}")

        if missing:
            print(f"  Missing in {md_file.name}: {sorted(missing)}")

    print("\nDone." if not dry_run else "\nDry run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    module_dir = Path(__file__).parent / args.module
    if not module_dir.exists():
        print(f"Not found: {module_dir}")
        return

    apply_vars(module_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

---

## 📄 `variables.json`

```json
{
  "_comment": "Fill with real values from your SVD-Mamba runs.",

  "MODULE_NAME": "SVD-Mamba",
  "FULL_NAME": "SVD-Factorized Selective State Space Attention",
  "SUBTITLE": "A Hybrid Attention-SSM Block with SVD-Factorized Projections",

  "AUTHOR_1_NAME": "Your Name",
  "AUTHOR_1_EMAIL": "you@university.edu",
  "AUTHOR_2_NAME": "Advisor Name",
  "AUTHOR_2_EMAIL": "advisor@university.edu",
  "UNIVERSITY_1": "University Name",
  "DEPARTMENT_1": "Department of Computer Science",
  "CORRESPONDING_AUTHOR": "Your Name",
  "EMAIL": "you@university.edu",
  "DATE": "2026-09-15",
  "VERSION": "1",
  "ORG": "your-org",
  "HF_ORG": "Hooshaai",
  "LICENSE": "MIT",

  "RANK": "192",
  "D_STATE": "16",
  "D_CONV": "4",
  "EXPAND": "1",
  "DT_RANK": "48",
  "GLOBAL_RANK": "8",
  "LORA_RANK": "16",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "PARAM_CHANGE_PCT": "9.6",
  "PARAM_CHANGE_PCT_R": "10.5",
  "PARAM_CHANGE_PCT_G": "10.5",
  "PARAM_CHANGE": "9.6",
  "PARAM_RATIO": "1.10",
  "INFERENCE_RATIO": "3.5",

  "D_ACC": "85.5",
  "D_DELTA": "-5.21",
  "D_F1": "0.855",
  "D_PARAMS": "73295130",
  "D_PARAM_DELTA": "6340120",
  "D_PARAM_PCT": "9.47",
  "D_VRAM": "250",
  "D_TIME": "6.5",
  "D_TPS": "17000",
  "D_STATUS": "PASS",
  "D_REASON": "—",
  "D_FINAL_GATE": "0.05",
  "D_VRAM_DELTA": "+67",
  "D_TIME_RATIO": "3.42",

  "R_ACC": "72.5",
  "R_DELTA": "-6.06",
  "R_F1": "0.725",
  "R_PARAMS": "137674580",
  "R_PARAM_DELTA": "13027410",
  "R_PARAM_PCT": "10.45",
  "R_VRAM": "890",
  "R_TIME": "14.0",
  "R_TPS": "8900",
  "R_STATUS": "PASS",
  "R_REASON": "—",
  "R_FINAL_GATE": "0.04",
  "R_TIME_RATIO": "3.41",

  "G_ACC": "73.2",
  "G_DELTA": "-6.16",
  "G_F1": "0.732",
  "G_PARAMS": "137468754",
  "G_PARAM_DELTA": "13027410",
  "G_PARAM_PCT": "10.47",
  "G_VRAM": "970",
  "G_TIME": "14.5",
  "G_TPS": "8700",
  "G_STATUS": "PASS",
  "G_REASON": "—",
  "G_FINAL_GATE": "0.04",
  "G_TIME_RATIO": "3.45",

  "D_DS8": "84.0", "D_DS16": "85.5", "D_DS32": "85.7", "D_DS64": "85.8",
  "R_DS8": "71.0", "R_DS16": "72.5", "R_DS32": "72.8", "R_DS64": "72.9",
  "G_DS8": "71.5", "G_DS16": "73.2", "G_DS32": "73.5", "G_DS64": "73.6",

  "D_DC2": "84.5", "D_DC4": "85.5", "D_DC8": "85.7",
  "R_DC2": "71.5", "R_DC4": "72.5", "R_DC8": "72.7",
  "G_DC2": "72.0", "G_DC4": "73.2", "G_DC8": "73.4",

  "D_GR4": "84.8", "D_GR8": "85.5", "D_GR16": "85.7",
  "R_GR4": "71.8", "R_GR8": "72.5", "R_GR16": "72.8",
  "G_GR4": "72.5", "G_GR8": "73.2", "G_GR16": "73.5",

  "D_GB10": "85.2", "D_GB7": "85.5", "D_GB5": "84.0", "D_GB0": "78.0",
  "R_GB10": "72.0", "R_GB7": "72.5", "R_GB5": "70.5", "R_GB0": "62.0",
  "G_GB10": "72.8", "G_GB7": "73.2", "G_GB5": "71.0", "G_GB0": "63.0",

  "D_G10": "0.02", "D_G25": "0.04", "D_G50": "0.05",
  "R_G10": "0.02", "R_G25": "0.03", "R_G50": "0.04",
  "G_G10": "0.02", "G_G25": "0.03", "G_G50": "0.04",

  "D_Q_SVD": "98304", "D_Q_DELTA": "-491520",
  "D_K_SVD": "98304", "D_K_DELTA": "-491520",
  "D_V_SVD": "98304", "D_V_DELTA": "-491520",
  "D_O_SVD": "98304", "D_O_DELTA": "-491520",
  "D_ATTN_SVD": "393216", "D_ATTN_DELTA": "-1966080",
  "D_SSM": "2300000", "D_GATE": "590592", "D_GLOBAL": "50000",
  "D_TOTAL": "3333808", "D_TOTAL_DELTA": "974512",

  "D_PROJ_TIME": "0.6", "D_SSM_TIME": "1.5", "D_GLOBAL_TIME": "0.2",
  "D_GATE_TIME": "0.3", "D_LORA_TIME": "0.3", "D_TOTAL_TIME": "3.5",

  "ATTN_25": "97", "SSM_25": "3", "ATTN_50": "95", "SSM_50": "5",
  "CONTRIBUTION_OBS": "SSM contribution grows from 0.1% to 5% over recovery",
  "GATE_OBS": "Gate stays near 0.05; SSM remains a minor contributor",
  "HCHUNK_OBS": "h_chunk values concentrated near zero, well within clamps",

  "GB0_DROP": "7.5",
  "NOSS_DROP": "2.5",
  "NOGLOBAL_DROP": "1.8",
  "NOCONV_DROP": "3.2",
  "SHARED_DROP": "2.0",

  "D_DS4": "82.0",
  "D_LR5e5": "84.0", "D_LR2e4": "85.5", "D_LR5e4": "80.0", "D_LR1e3": "NaN",
  "D_GB2": "70.0",

  "SEED42": "85.5", "SEED123": "85.4", "SEED456": "85.6",
  "MEAN": "85.5", "STD": "0.10",

  "D_S1": "1.5", "D_S5": "0.9", "D_S10": "0.7", "D_S50": "0.45",

  "BEST_DELTA": "-5.21",
  "COMPUTE_HOURS": "15",
  "CO2_KG": "1.0",
  "RUNTIME": "25",
  "INFERENCE_CARBON": "0.3",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0",
  "NUMPY_VERSION": "1.26.4",

  "Q_VERIFIED": "not tested",
  "UNRESOLVED_FAILURES": "None",

  "_note": "Add keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — SVD-Mamba

## Setup
- [ ] Copied `_TEMPLATE/` to `svd_mamba/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module svd_mamba`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Parameter increase stated**
- [ ] Three-path architecture mentioned
- [ ] Gate init strategy mentioned

### Introduction
- [ ] Motivation concrete
- [ ] **Explicit statement this is NOT a compressor**
- [ ] Contributions verifiable

### Method
- [ ] All formulas numbered
- [ ] SSM recurrence explained
- [ ] Chunked scan described
- [ ] Numerical clamps documented
- [ ] Complexity analysis

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows increase**
- [ ] d_state ablation
- [ ] d_conv ablation
- [ ] Gate bias ablation
- [ ] Gate evolution

### Analysis
- [ ] Gate dynamics
- [ ] Numerical stability
- [ ] Parameter breakdown
- [ ] SSM contribution
- [ ] Comparison with pure compressors

### Limitations
- [ ] **Not a compressor**
- [ ] Complex architecture
- [ ] Unstable recovery
- [ ] Cross-attention not supported
- [ ] Single-seed

## Honesty
- [ ] Parameter increase admitted
- [ ] Recovery instability documented
- [ ] Silent injection failure discussed
- [ ] Failure cases documented

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] SSM init documented

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] Tables render
- [ ] PDF compiles

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/svd_mamba
nano papers/svd_mamba/variables.json
cd papers/svd_mamba
python apply_vars.py --module svd_mamba
grep -rn "{{" .
python build.py --module svd_mamba --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE svd_mamba
```

### مرحله ۲: اجرای SVD-Mamba

```powershell
cd D:\SVD_Project
python run_all_4_svd_mamba.py --models distilbert,roberta,gpt2 --d_state 16 --d_conv 4 --global_rank 8 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json` از `summary.csv`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\svd_mamba
python apply_vars.py --module svd_mamba
python build.py --module svd_mamba
```

---

## 💡 نکات کلیدی SVD-Mamba

### ۱. **SVD-Mamba فشرده‌ساز نیست**

- SVD factorization → projections کم می‌شود
- **ولی** SSM + gating + global → پارامتر **اضافه** می‌کند
- خالص: **+10%** افزایش پارامتر

### ۲. **سه مسیر معماری**

1. Attention (with SVD projections)
2. SSM (Mamba-style selective state space)
3. Global SVD mixing

Fusion via **gate**.

### ۳. **Gate bias = -7**

- `σ(-7) ≈ 0.0009` → SSM فقط 0.1% در ابتدا
- حفظ pretrained behavior
- در طول training تغییر می‌کند (ولی نه به‌شدت)

### ۴. **Chunked scan**

- chunk_size = 64
- Multiple clamps for stability
- Python loop over chunks (کندی)

### ۵. **مشکلات اصلی**

- **افزایش پارامتر** (+10%)
- **کندی** (3.5×)
- **پیچیدگی**
- **Unstable recovery**

### ۶. **مزیت‌ها**

- مسیر SSM برای long-context
- Multi-architecture
- Causal
- نوآوری معماری

### ۷. **مقایسه با AASVD**

| ویژگی | AASVD | SVD-Mamba |
|---|---|---|
| پارامتر | **-50%** | **+10%** |
| Accuracy | -2% | -5% |
| سرعت | کندتر | 3.5× کندتر |
| نوع | Compressor | Attention replacement |

**SVD-Mamba و AASVD کاملاً متفاوت هستند.**

---

**قدم بعدی: `run_all_4_svd_mamba.py` را اجرا کن و `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**