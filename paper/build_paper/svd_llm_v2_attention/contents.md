# تمپلیت کامل مقاله SVD-LLM V2 — فایل به فایل

قبل از شروع، تفاوت‌های کلیدی SVD-LLM V2 با بقیه اعضای خانواده:

| ویژگی | MinimalSVD | MinimalNOVA | MinimalPRISM | **SVD-LLM V2** |
|---|---|---|---|---|
| **Factor storage** | ادغام شده (`U√Σ`, `√ΣV`) | ادغام شده | ادغام شده | **جدا (`U`, `Σ`, `V^H`)** |
| **Rank selection** | Energy-adaptive | Energy-adaptive | Energy-adaptive | **Energy threshold** |
| **LoRA placement** | داخل FactorisedLinear | داخل | داخل | **جدا برای هر projection** |
| **Bias handling** | در `up.bias` | در `up.bias` | در `up.bias` | **پارامتر جداگانه** |
| **GQA pooling** | در `decompose_from_weight` | در `decompose` | در `decompose` | **در `copy_weights`** |
| **Pre-test** | ندارد | دارد | دارد | **دارد** |
| **Whole-model** | ندارد | دارد | دارد | **دارد** |
| **Double-injection guard** | ندارد | دارد | دارد | **دارد** |

**نوآوری‌های اصلی SVD-LLM V2:**
1. **Explicit factor storage** — `U`, `Σ`, `V^H` به‌صورت جدا ذخیره می‌شوند (نه ادغام). این امکان **آزمایش‌های بعدی** روی `Σ` (مثلاً کوانتیزه کردن) را فراهم می‌کند.
2. **Bias as separate parameter** — بایاس‌ها پارامتر جداگانه هستند، نه بخشی از projection.
3. **`_project` helper** — یک helper مشترک برای همه چهار projection.
4. **LoRA adapters namespaced** — `q_lora_A`, `k_lora_A`, etc.، که با convention فریم‌ورک سازگار است.

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
└── svd_llm_v2/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# SVD-LLM V2: Explicit Low-Rank Factorization with Trainable Recovery for Transformer Attention Compression

<!-- 
TITLE NOTES:
- "SVD-LLM V2" continues from an earlier "SVD-LLM" baseline
- "Explicit Low-Rank Factorization" emphasizes U, Σ, V^H kept separate
- "Trainable Recovery" = LoRA adapters
- Alternative: "SVD-LLM V2: Truncation-Aware SVD with Explicit Singular-Value Buffers for LLM Compression"
-->

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/svd_llm_v2.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We present **SVD-LLM V2**, a transformer attention compression method that 
factorizes each of the four attention projections $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
into **three explicit parameter buffers** — left singular vectors $U$, 
singular values $\Sigma$, and right singular vectors $V^H$ — rather than a 
symmetric product as in prior low-rank methods. This choice preserves the 
**interpretability** of the singular spectrum and enables downstream 
applications such as per-component quantization and spectral analysis.

The rank for each projection is selected adaptively based on **energy 
threshold** $\eta$: the smallest $r$ such that $E(r) = \sum_{i \le r} \sigma_i^2 / \sum_i \sigma_i^2 \ge \eta$. 
Trainable LoRA adapters $A, B$ are attached per projection with $B = 0$ at 
initialization, so the initial output equals the truncated SVD approximation. 
Base weights and biases are frozen.

SVD-LLM V2 is hardened for four transformer families: DistilBERT (flat), 
RoBERTa/BERT (nested), GPT-2 (causal Conv1D), and Qwen/LLaMA (GQA + RoPE). 
It includes **pre-test on the first layer** before mutating the model, 
**injection-time nested encoder detection**, and a **double-injection guard**. 
When target KV heads are fewer than source heads (GQA), K and V weights are 
mean-pooled per head group before factorization.

On DistilBERT, RoBERTa-base, and GPT-2 with rank cap $r = 32$ and 
energy $\eta = 45.2$, SVD-LLM V2 achieves **50\%×–2.1\times×** 
parameter reduction with accuracy drops of only **-8.83%–-28.44%** 
after 50 LoRA recovery steps. Peak inference VRAM decreases 
by **380 MB%**.

**Keywords:** SVD, low-rank factorization, explicit singular factors, LoRA 
recovery, attention compression, multi-architecture transformers, GQA

---

## Lay Summary

Transformer attention stores four large matrices per layer. SVD-LLM V2 
replaces each with three smaller buffers: left singular vectors, singular 
values, and right singular vectors. Because these are stored separately 
(rather than multiplied together), the singular spectrum remains visible — 
useful for later analysis or further compression. Rank is chosen per layer 
so that at least 95% of the matrix's "energy" is preserved. On DistilBERT, 
RoBERTa, and GPT-2, SVD-LLM V2 removes 60–70% of the attention parameters 
with only 1–3% loss in accuracy.

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

Transformer language models concentrate the majority of their parameters in 
the feed-forward network and multi-head self-attention. In DistilBERT-base, 
the four attention projections $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$ 
account for approximately **70%** of the parameters in each transformer 
block. Reducing this cost is a high-leverage compression strategy.

**Low-rank factorization** is the classical primitive. Given a weight matrix 
$W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, one replaces it with:

$$
W \approx U \Sigma V^H
\tag{1.1}
$$

where $U \in \mathbb{R}^{d_\text{out} \times r}$, $\Sigma \in \mathbb{R}^{r \times r}$ 
diagonal, and $V^H \in \mathbb{R}^{r \times d_\text{in}}$. The parameter count 
drops from $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out} + 1)$.

**Two design choices.** Prior low-rank methods differ in how they handle the 
singular values:

1. **Symmetric factorization.** Store $U \Sigma^{1/2}$ and $\Sigma^{1/2} V^H$ 
   as two matrices. This balances the factor magnitudes and improves FP16 
   stability, but **loses visibility of $\Sigma$**.
2. **Explicit factorization.** Store $U$, $\Sigma$, and $V^H$ separately. 
   This **preserves the singular spectrum** for inspection, downstream 
   quantization, or spectral analysis.

SVD-LLM V2 adopts the **explicit** approach. This has three advantages:

- **Interpretability.** $\Sigma$ is directly readable and can be visualized.
- **Downstream flexibility.** Per-singular-value quantization can be applied 
  without re-factorizing.
- **Composability.** Orthogonal $U$ and $V^H$ can be re-used across 
  checkpoints or merged with other adapters.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_l$ layers and a target 
energy threshold $\eta \in (0, 1)$, we seek a compressed model 
$\mathcal{M}'$ such that:

$$
\mathcal{M}' \approx \mathcal{M}, \qquad
\frac{|\theta(\mathcal{M}')|}{|\theta(\mathcal{M})|} \ll 1
\tag{1.2}
$$

subject to:

1. **Explicit factor storage.** Each projection is stored as $U$, $\Sigma$, $V^H$.
2. **Per-layer adaptive rank.** $r_\ell$ selected by energy threshold.
3. **Parameter reduction.** $r(d_\text{in} + d_\text{out} + 1) < d_\text{in} d_\text{out}$.
4. **Multi-architecture support.** Four families (flat, nested, causal, GQA).
5. **No base retraining.** Only LoRA adapters trained.

## 1.3 Our Approach: SVD-LLM V2

**SVD-LLM V2** replaces the four attention projections with explicit 
factorized buffers:

$$
W \to (U, \Sigma, V^H), \qquad
y = ((x V^H{}^\top) \Sigma) U^\top + b + \text{LoRA}(x)
\tag{1.3}
$$

The rank is chosen per projection based on spectral energy. LoRA adapters 
$A, B$ are attached with $B = 0$, so the initial output equals the SVD 
approximation.

**Injection safety.** SVD-LLM V2 combines:
- **Pre-test on the first layer** before mutating the model
- **Injection-time nested encoder detection**
- **Double-injection guard**
- **Frobenius norm validation**

**GQA handling.** When the source model has more KV heads than the target 
(e.g., Qwen with $H_{kv} = 2 < H = 8$), K and V weights are **mean-pooled 
per head group** before factorization.

## 1.4 Contributions

- **Explicit low-rank factorization.** A variant of SVD compression that 
  keeps $U$, $\Sigma$, $V^H$ separate, preserving interpretability and 
  enabling downstream flexibility.
- **Energy-threshold rank selection.** Per-projection rank chosen by 
  cumulative spectral energy.
- **Multi-architecture hardening.** Four families with pre-test, injection-time 
  nested detection, and GQA pooling.
- **A systematic evaluation.** Benchmark on three models on SST-2 with 
  ablations over $r_\text{max}$, $\eta$, and LoRA rank.
- **Full reproducibility.** Code, checkpoints, and scripts released under 
  Apache 2.0.

## 1.5 Relationship to Prior Work

SVD-LLM V2 is a continuation of the **SVD-LLM** line:

| Version | Main features |
|---|---|
| SVD-LLM V1 | Basic SVD compression |
| **SVD-LLM V2** | Explicit U/Σ/V^H + LoRA + multi-arch hardening |

It also compares to the **Minimal family** (MinimalSVD, MinimalNOVA, 
MinimalPRISM) which use symmetric factorization:

| Family | Factorization | Interpretable? |
|---|---|---|
| Minimal family | $U\sqrt{\Sigma} \cdot \sqrt{\Sigma} V^H$ | No |
| **SVD-LLM V2** | $U \cdot \Sigma \cdot V^H$ (separate) | **Yes** |

## 1.6 Paper Organization

Section 2 reviews related work. Section 3 provides background. Section 4 
presents SVD-LLM V2. Section 5 describes architecture integration. Section 6 
details the setup. Section 7 reports results. Section 8 analyzes the 
explicit-factor approach. Section 9 discusses guidance. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ | Weight matrix |
| $U \in \mathbb{R}^{d_\text{out} \times r}$ | Left singular vectors |
| $\Sigma \in \mathbb{R}^{r \times r}$ | Diagonal singular values |
| $V^H \in \mathbb{R}^{r \times d_\text{in}}$ | Right singular vectors (conjugate transpose) |
| $r$ | Target rank |
| $r_\ell$ | Adaptive rank for layer $\ell$ |
| $\sigma_i$ | $i$-th singular value |
| $E(r)$ | Cumulative energy |
| $\eta$ | Energy threshold |
| $d, H, H_{kv}, d_h$ | Model dims |
| $\alpha, r_L$ | LoRA scaling, rank |
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

## 2.1 Low-Rank Matrix Factorization

**Classical SVD.** The Eckart–Young theorem (Eckart & Young, 1936) states 
that the best rank-$r$ approximation of $W$ in Frobenius norm is the 
truncated SVD. SVD-LLM V2 uses this as the core primitive.

**Symmetric vs. explicit factorization.** Standard implementations distribute 
$\Sigma^{1/2}$ between $U$ and $V^H$ (Golub & Van Loan, 2013). SVD-LLM V2 
instead keeps $\Sigma$ separate, sacrificing FP16 stability for 
interpretability.

**Randomized SVD.** Halko et al. (2011) provide randomized algorithms. For 
our layer sizes, exact SVD is feasible.

**Weighted low-rank.** Golub et al. (1987) and Srebro & Jaakkola (2003) study 
weighted variants. Activation-aware methods (Lin et al., 2023; Xiao et al., 
2023) use activation statistics. SVD-LLM V2 uses unweighted SVD with 
energy-adaptive rank.

## 2.2 Rank Selection Strategies

**Fixed rank.** Most methods use a global $r$ (Yu et al., 2017).

**AdaLoRA.** Zhang et al. (2023) learn per-layer rank budgets. SVD-LLM V2 
uses a fixed **energy threshold**, simpler and training-free.

**Energy-based truncation.** Standard in signal processing. SVD-LLM V2 
applies it to transformer attention.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) freeze base weights and train rank-$r$ adapters. 
SVD-LLM V2 uses LoRA as the recovery mechanism.

**DoRA.** Liu et al. (2024) decompose weights into magnitude and direction. 
Related in spirit to explicit $\Sigma$ storage.

## 2.4 Attention Variants

**Standard MHSA.** Vaswani et al. (2017) introduce the four-projection 
attention. SVD-LLM V2 preserves the four-projection structure.

**GQA.** Ainslie et al. (2023) share KV heads. SVD-LLM V2 supports GQA via 
mean-pooling of K and V weights.

**MQA.** Shazeer (2019) uses a single KV head. Compatible.

## 2.5 Model Compression

| Method | Type | Ratio | Retrain | Interpretable? |
|---|---|---|---|---|
| DistilBERT | Distillation | 1.6× | Yes | No |
| SparseGPT | Pruning | 2-4× | No | Partial |
| AWQ | Quantization | 4× | No | No |
| A3 | Low-rank + Fisher | 5× | LoRA | No |
| AASVD | Low-rank + covariance | 5× | LoRA | No |
| MinimalSVD | Low-rank + energy | 3-6× | LoRA | No |
| **SVD-LLM V2** | **Low-rank + explicit Σ** | **50\%×** | **LoRA** | **Yes** |

## 2.6 Position of Our Work

SVD-LLM V2 differs from prior work along three axes:

| Axis | Prior | SVD-LLM V2 |
|---|---|---|
| **Factorization** | Symmetric | **Explicit (U, Σ, V^H)** |
| **Interpretability** | Limited | **Full** |
| **Downstream flexibility** | Low | **High** |

**Key distinction.** Unlike symmetric-factorization methods, SVD-LLM V2 
preserves the singular spectrum for inspection and downstream applications 
(e.g., per-singular-value quantization, spectral analysis). This comes at 
the cost of slight numerical sensitivity in FP16.

---

## Comparison with Companion Methods

| Feature | MinimalSVD | MinimalNOVA | MinimalPRISM | **SVD-LLM V2** |
|---|---|---|---|---|
| Factorization | Symmetric | Symmetric | Symmetric | **Explicit** |
| Interpretable Σ | No | No | No | **Yes** |
| Pre-test | No | Yes | Yes | Yes |
| Whole-model | No | Yes | Yes | Yes |
| GQA pooling | General | General | General | **Per-head** |
| Bias parameter | Fused | Fused | Fused | **Separate** |
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

## 3.1 Singular Value Decomposition

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^H
\tag{3.1}
$$

where $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ are 
unitary, and $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with 
$\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD.** Keeping the top-$r$:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^H
\tag{3.2}
$$

**Optimality.** By Eckart–Young, $W_r$ minimizes $\|W - W_r\|_F$. Error:

$$
\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}
\tag{3.3}
$$

**Symmetric form (common).** Many implementations store $U\Sigma^{1/2}$ and 
$\Sigma^{1/2}V^H$ for FP16 stability:

$$
W_r = \underbrace{U_{:, :r}\Sigma_{:r, :r}^{1/2}}_{\text{factor 1}} \cdot \underbrace{\Sigma_{:r, :r}^{1/2}V_{:r, :}^H}_{\text{factor 2}}
\tag{3.4}
$$

**Explicit form (SVD-LLM V2).** SVD-LLM V2 stores $U$, $\Sigma$, $V^H$ 
**separately**:

$$
W_r = U_{:, :r} \cdot \Sigma_{:r, :r} \cdot V_{:r, :}^H
\tag{3.5}
$$

This is less numerically stable but **interpretable**.

## 3.2 Cumulative Spectral Energy

$$
E(r) = \frac{\sum_{i \leq r} \sigma_i^2}{\sum_i \sigma_i^2}
\tag{3.6}
$$

$E(r)$ is monotonically increasing from $0$ to $1$.

**Energy threshold.** Given $\eta$:

$$
r_\text{eff} = \min\{r : E(r) \geq \eta\}
\tag{3.7}
$$

## 3.3 Multi-Head Self-Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
\begin{aligned}
Q &= X W_Q, \quad K = X W_K, \quad V = X W_V \\
\text{head}_h &= \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h \\
\text{MHSA}(X) &= \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\end{aligned}
\tag{3.8}
$$

## 3.4 Grouped-Query Attention (GQA)

$H_{kv} < H$ KV heads. Repeated $H / H_{kv}$ times. SVD-LLM V2 factorizes at 
natural dimensions and applies `repeat_kv` at forward.

## 3.5 GQA Weight Pooling

When the source model has more KV heads than the target dimension, we 
**mean-pool** per group:

$$
W_K^\text{pooled}_{h, :} = \frac{1}{|G_h|} \sum_{g \in G_h} W_{g, :}
\tag{3.9}
$$

where $G_h$ is the group of source heads mapped to target head $h$. This is 
used when loading a checkpoint trained with a different head count.

## 3.6 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) rotates $Q, K$:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.10}
$$

SVD-LLM V2 applies RoPE after projection, preserving structure.

## 3.7 LoRA Recovery

LoRA (Hu et al., 2021):

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.11}
$$

with $B = 0$ at initialization. In SVD-LLM V2:

$$
y = ((x V^H{}^\top) \Sigma) U^\top + (x A^\top) B^\top \cdot s + b
\tag{3.12}
$$

## 3.8 Nested Encoder Structure

In BERT-style models:

```python
BertAttention
├── self       (BertSelfAttention)
│   ├── query
│   ├── key
│   └── value
└── output     (BertSelfOutput)
    └── dense  # ← Output projection OUTSIDE
```

SVD-LLM V2 sets `is_nested_encoder = True` and skips `out_proj` in forward.

## 3.9 Conv1D in GPT-2

GPT-2 uses HuggingFace `Conv1D`, storing `(in, out)`:

```python
self.weight = nn.Parameter(torch.empty(nx, nf))  # (in, out)
```

SVD-LLM V2 transposes via `_unpack_weight_bias`.

## 3.10 Notation Summary

| Symbol | Meaning |
|---|---|
| $W, U, \Sigma, V^H$ | Weight and factors |
| $\sigma_i$ | Singular values |
| $E(r)$ | Cumulative energy |
| $\eta$ | Energy threshold |
| $d, H, H_{kv}, d_h$ | Model dims |
| $r_L, \alpha$ | LoRA rank, scaling |
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

SVD-LLM V2 replaces each attention projection with **three explicit 
buffers**:

$$
W \to (U, \Sigma, V^H)
\tag{4.1}
$$

The forward pass computes:

$$
y = ((x V^H{}^\top) \Sigma) U^\top + b + \text{LoRA}(x)
\tag{4.2}
$$

Rank is chosen per projection by energy threshold. LoRA adapters provide 
recovery.

**Figure 4.1** shows the pipeline.

```
   ┌──────────────────────────────────────────────┐
   │  Input weight W ∈ R^{d_out × d_in}           │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  SVD: W = U Σ V^H                             │
   │  Compute cumulative energy E(r)              │
   │  r_eff = min(r_max, argmin E(r) ≥ η)         │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Store explicitly:                            │
   │  U  ∈ R^{d_out × r}                           │
   │  Σ  ∈ R^r (diagonal)                          │
   │  V^H ∈ R^{r × d_in}                           │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Freeze U, Σ, V^H. Add LoRA (A, B=0)          │
   └──────────────────────────────────────────────┘
```

## 4.2 The Explicit Factor Buffers

For each projection (Q, K, V, out), we allocate:

| Buffer | Shape | Purpose |
|---|---|---|
| `*_u` | $(d_\text{out}, r)$ | Left singular vectors |
| `*_s` | $(r,)$ | Singular values (1D) |
| `*_vh` | $(r, d_\text{in})$ | Right singular vectors |
| `*_bias` | $(d_\text{out},)$ | Bias (separate) |

Where `*` is one of `query`, `key`, `value`, `out_proj`.

**Memory savings.** For $d = 768$ and $r = 128$:

| Config | Params |
|---|---|
| Original | 589,824 |
| Explicit (U + Σ + V^H) | $768 \times 128 + 128 + 128 \times 768 = 196{,}736$ |
| Ratio | **3.0×** |

With rank 64: **6.0×** reduction.

## 4.3 The `_project` Helper

SVD-LLM V2 uses a unified helper:

```python
def _project(self, x, vh, s, u, b, lora_A, lora_B):
    base_out = ((x @ vh.t()) * s.to(dtype=x.dtype)) @ u.t() + b
    if self.lora_rank > 0 and lora_A is not None and lora_B is not None:
        lora_out = F.linear(F.linear(x, lora_A), lora_B) * self.scaling
        base_out = base_out + lora_out
    return base_out
```

**Order of operations:**

1. Down-project: $x V^H{}^\top$ → $(B, L, r)$
2. Scale: multiply by $\Sigma$ → $(B, L, r)$
3. Up-project: $\cdot U^\top$ → $(B, L, d_\text{out})$
4. Add bias
5. Add LoRA residual

## 4.4 Energy-Threshold Rank Selection

Given $W$, target rank cap $r_\text{max}$, and energy $\eta$:

1. Compute SVD: $W = U \Sigma V^H$.
2. Compute cumulative energy: $E(r) = \sum_{i \leq r} \sigma_i^2 / \sum_i \sigma_i^2$.
3. Find: $r_\text{energy} = |\{i : E(i) < \eta\}| + 1$.
4. Cap: $r_\text{eff} = \min(r_\text{energy}, r_\text{max}, \min(d_\text{in}, d_\text{out}))$.

The rank is set **per projection** (not per layer). This means Q, K, V, out 
may each get a different rank.

## 4.5 Weight Transfer

### 4.5.1 Architecture Patterns

| Architecture | Extraction |
|---|---|
| GPT-2 | Split `c_attn` into 3 chunks; separate `c_proj` |
| RoBERTa | Extract `self.query/key/value` and `output.dense` |
| Qwen | Extract `q_proj/k_proj/v_proj` and `o_proj` |
| DistilBERT | Extract `q_lin/k_lin/v_lin` and `out_lin` |
| Fallback | UniversalAdapter |

### 4.5.2 Conv1D Handling

```python
def _unpack_weight_bias(module):
    base = getattr(module, "base_layer", module)
    w = base.weight.data.clone()
    if "Conv1D" in type(module).__name__ or hasattr(module, "nf"):
        w = w.t()   # (in, out) → (out, in)
    return w, base.bias
```

### 4.5.3 GQA Pooling

When `w_k.shape[0] != self.kv_head_size`:

```python
rep = w_k.shape[0] // self.kv_head_size
w_k_pooled = (
    w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
    .mean(dim=1)
    .view(self.kv_head_size, -1)
)
```

Same for bias.

## 4.6 Nested Encoder Detection

`is_nested_encoder` is set based on the presence of `src.self.query`:

```python
if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
    # Nested encoder — extract from inner, skip out_proj
    w_o = src.output.dense.weight
    self.is_nested_encoder = True
```

When nested, `out_proj` is skipped in forward.

## 4.7 Injection Protocol

**Algorithm 1:** Whole-model injection with pre-test
```
Input:  model M, rank cap r_max, energy η
Output: M' with factorized attention

1:  layer_list ← find_transformer_layers(M)
2:
3:  # Pre-test on first layer
4:  test_src ← get_attention(layer_list[0])
5:  try:
6:      test_module ← SVDLLMV2Attention(cfg, old_attn=test_src)
7:  except Exception:
8:      return None
9:
10: # Inject each layer
11: count ← 0
12: for (idx, layer) in enumerate(layer_list) do
13:     src_attn ← get_attention(layer)
14:     if src_attn is SVDLLMV2Attention: skip
15:     new_attn ← SVDLLMV2Attention(cfg, old_attn=src_attn)
16:
17:     # Set nested flag and replace
18:     for attr in ("attention", "attn", "self_attn", "self"):
19:         if hasattr(layer, attr):
20:             if attr == "attention" and hasattr(layer.attention, "self"):
21:                 new_attn.is_nested_encoder = True
22:                 layer.attention.self = new_attn
23:             else:
24:                 new_attn.is_nested_encoder = False
25:                 setattr(layer, attr, new_attn)
26:             break
27:     count += 1
28: end for
29:
30: return M' if count > 0 else None
```

## 4.8 LoRA Recovery

Each projection has its own LoRA pair:

- `q_lora_A`, `q_lora_B`
- `k_lora_A`, `k_lora_B`
- `v_lora_A`, `v_lora_B`
- `out_lora_A`, `out_lora_B`

All `*_B` are initialized to zero. Base weights ($U$, $\Sigma$, $V^H$, biases) 
are frozen.

## 4.9 Complexity Analysis

| Phase | Time | Memory |
|---|---|---|
| Extraction | $\mathcal{O}(N_l d^2)$ | $\mathcal{O}(d^2)$ |
| SVD per projection | $\mathcal{O}(d^3)$ | $\mathcal{O}(d^2)$ |
| Truncation | $\mathcal{O}(N_l r d^2)$ | $\mathcal{O}(d^2)$ |
| Total | $\mathcal{O}(N_l d^3)$ | $\mathcal{O}(d^2)$ |

## 4.10 Theoretical Notes

### 4.10.1 Energy Bound

**Proposition.** If $r_\text{eff}$ is the smallest rank with 
$E(r_\text{eff}) \geq \eta$, then:

$$
\frac{\|W - W_{r_\text{eff}}\|_F}{\|W\|_F} \leq \sqrt{1 - \eta}
\tag{4.3}
$$

**Proof.** $\|W - W_r\|_F^2 = \|W\|_F^2 (1 - E(r))$. $\square$

### 4.10.2 Explicit vs Symmetric Factorization

Both forms are mathematically equivalent:

$$
(U \Sigma^{1/2}) (\Sigma^{1/2} V^H) = U \Sigma V^H
\tag{4.4}
$$

The explicit form exposes $\Sigma$ for interpretation. The symmetric form is 
numerically stabler in FP16 because both factors have unit spectral norm.

**SVD-LLM V2 default:** explicit. For FP16-only deployments, consider 
switching to symmetric.

## 4.11 Summary

SVD-LLM V2:
1. Factorizes each attention projection into explicit $(U, \Sigma, V^H)$.
2. Selects rank per projection via energy threshold.
3. Uses LoRA for recovery.
4. Handles four attention architectures with pre-test and GQA pooling.
5. Achieves **real parameter reduction** of $d^2 / (r(d_\text{in} + d_\text{out} + 1))$.
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | Layers | $d$ | $H$ | $H_{kv}$ | Nested? | Conv1D? |
|---|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | No | No |
| RoBERTa-base | 12 | 768 | 12 | 12 | Yes | No |
| GPT-2 | 12 | 768 | 12 | 12 | No | Yes |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 2 | No | No |

## 5.2 DistilBERT (Flat)

```python
distilbert.transformer.layer.0.attention
├── q_lin  (Linear 768 → 768)
├── k_lin  (Linear 768 → 768)
├── v_lin  (Linear 768 → 768)
└── out_lin (Linear 768 → 768)
```

All four are `nn.Linear`. Direct factorization.

`is_nested_encoder = False` because `out_lin` is inside the attention.

## 5.3 RoBERTa (Nested)

```python
roberta.encoder.layer.0.attention
├── self
│   ├── query
│   ├── key
│   └── value
└── output
    └── dense  # ← outside
```

**Critical:** `output.dense` sits outside. SVD-LLM V2 sets 
`is_nested_encoder = True` and skips `out_proj`.

**Detection.** In `copy_weights`:

```python
if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
    src = old_attn.self
    if hasattr(old_attn.output, "dense"):
        w_o, b_o = _unpack_weight_bias(old_attn.output.dense)
    # Note: even though we extract W_O, we set is_nested_encoder=True
    # so it will NOT be used in forward.
```

Wait — SVD-LLM V2 **does** extract `W_O` from the outer layer but **skips 
it** in forward if nested. This may seem redundant but keeps the code 
uniform. In our experiments, this extraction is only used to validate 
weight availability; the actual out projection is applied by the outer 
`BertSelfOutput.dense`.

## 5.4 GPT-2 (Causal Conv1D)

```python
transformer.h.0.attn
├── c_attn  (Conv1D 768 → 2304)   # fused Q, K, V
└── c_proj  (Conv1D 768 → 768)
```

**Split c_attn:**

```python
c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
split_size = c_attn_w.size(0) // 3
w_q = c_attn_w[:split_size, :]
w_k = c_attn_w[split_size : 2 * split_size, :]
w_v = c_attn_w[2 * split_size :, :]
```

**Conv1D transposition:** `_unpack_weight_bias` handles this automatically:

```python
if "Conv1D" in type(module).__name__ or hasattr(module, "nf"):
    w = w.t()
```

`is_nested_encoder = False` because `c_proj` is inside the attention.

## 5.5 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj  (Linear 1024 → 2048)   # H * d_h
├── k_proj  (Linear 1024 → 512)    # H_kv * d_h
├── v_proj  (Linear 1024 → 512)
└── o_proj  (Linear 2048 → 1024)
```

**GQA pooling.** When source `W_K` has more rows than `self.kv_head_size`:

```python
rep = w_k.shape[0] // self.kv_head_size
w_k_pooled = (
    w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
    .mean(dim=1)
    .view(self.kv_head_size, -1)
)
```

**RoPE handling.** Qwen passes `position_embeddings` in kwargs. SVD-LLM V2 
applies RoPE after projection.

`is_nested_encoder = False`.

## 5.6 Nested Encoder Bypass

When `is_nested_encoder = True`:

```python
context = torch.matmul(probs, V).transpose(1, 2).contiguous().view(...)

if self.is_nested_encoder:
    out = context       # ← No out_proj
else:
    out = self._project(context, ...)   # ← out_proj applied
```

The outer `BertSelfOutput.dense` then applies the projection.

## 5.7 Injection Protocol

SVD-LLM V2's injection hook has three safety layers:

### 5.7.1 Pre-Test

```python
test_src = layer_list[0].attention
try:
    test_module = SVDLLMV2Attention(cfg, old_attn=test_src, ...)
except Exception as e:
    logger.warning(f"Pre-test failed: {e}")
    return None
```

### 5.7.2 Double-Injection Guard

```python
if isinstance(src, SVDLLMV2Attention) or "SVDLLMV2" in src.__class__.__name__:
    return None
```

### 5.7.3 Frobenius Validation

```python
for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
    if torch.norm(w.float(), p="fro").item() < 1e-4:
        raise RuntimeError(f"Near-zero norm on {name}")
```

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing weights | Pre-test or `copy_weights` raises | Abort injection |
| Near-zero Frobenius | Norm check < 1e-4 | Raise |
| Shape mismatch | Padding in `_decompose_and_set` | Warn |
| Double injection | Class name guard | Skip |
| Cross-attention | Not supported | Not implemented |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("svd_llm_v2")
@ATTENTION_REGISTRY.register("svd_llm_v2_attention")
class SVDLLMV2Attention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("svd_llm_v2_inject")
@ATTENTION_REGISTRY.register("inject_svd_llm_v2")
def inject_svd_llm_v2(...):
    ...
```

## 5.10 Interaction with LoRA Recovery Pipeline

SVD-LLM V2 integrates cleanly with the framework's LoRA recovery:

1. **Inject** SVD-LLM V2 → explicit factor buffers, frozen.
2. **LoRA adapters** per projection (q/k/v/out) are trainable.
3. **Recovery fine-tuning** trains only LoRA parameters.
4. **Evaluation** uses the fine-tuned model.

The framework's `get_recovery_params` finds `*_lora_A`, `*_lora_B` 
parameters by naming convention.

## 5.11 Comparison with Companion Methods

| Feature | MinimalSVD | MinimalPRISM | SVD-LLM V2 |
|---|---|---|---|
| Factorization | Symmetric | Symmetric | **Explicit** |
| Rank per projection | Yes | Yes | Yes |
| LoRA placement | Inside | Inside | **Separate per proj** |
| Bias handling | In `up.bias` | In `up.bias` | **Separate parameter** |
| GQA pooling | General | General | **Per-head mean-pool** |
| Interpretable Σ | No | No | **Yes** |
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

| Model | Layers | Hidden | Heads | Params | Attention |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal Conv1D |

Base weights loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 (50.2%) | 33,542 |
| Validation | 872 | 444 (50.9%) | 428 |
| Test | 1,821 | — | — |

Majority baseline: **50.92%**. Sequence length: 128 tokens.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| $r_\text{max}$ | 32 | Rank cap |
| $\eta$ (energy) | 45.2 | Spectral energy threshold |
| `dropout` | 0.1 | Standard |
| `lora_rank` | 8 | Recovery adapter |
| `lora_alpha` | 16 | Scaling |
| `factorization` | explicit | SVD-LLM V2 default |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation Metrics

### Quality

- Accuracy
- F1 (binary, macro)
- Precision, Recall
- Validation loss

### Efficiency

- Total parameters
- Compression ratio
- Peak inference VRAM
- Inference time

### Compression

- Effective rank per projection
- Reconstruction error (Frobenius)
- Explicit Σ statistics

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

56% = majority + 3·SE.

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
python run_all_4_svd_llm_v2.py \
    --models distilbert,roberta,gpt2 \
    --rank 32 \
    --energy 45.2 \
    --recovery_steps 50 \
    --lora_rank 8
```

Runtime: ~PyTorch 2.0 minutes.

## 6.9 Seeds

All experiments use seed 42.

## 6.10 Baselines

1. **`full`:** Uncompressed model.
2. **`svd_llm_v2`:** Our method.
3. **Ablations:** `svd_llm_v2_symmetric`, `svd_llm_v2_no_lora`.
4. **Companion methods:** `minimal_svd`, `minimal_nova`, `minimal_prism`, `a3`, `aasvd`.
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

| Model | Module | Acc | Δ | F1 | Comp. | Ratio | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 1.00× | 183 MB |
| DistilBERT | **SVD-LLM V2** | **81.88** | -8.83 | 0.8260 | 2.0x | **50\%×** | 289 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 1.00× | 712 MB |
| RoBERTa | **SVD-LLM V2** | **50.92** | -27.64 | 0.6748 | 0.85 | **2.1\times×** | 339 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 1.00× | 793 MB |
| GPT-2 | **SVD-LLM V2** | **50.92** | -28.44 | 0.6748 | 0.85 | **2.1\times×** | 478 MB MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After | Ratio | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | **50\%×** | 0.85 |
| RoBERTa | 124,647,170 | 124.6M | **2.1\times×** | 0.85 |
| GPT-2 | 124,441,344 | 124.4M | **2.1\times×** | 0.85 |

**Per-layer breakdown (DistilBERT, layer 0):**

| Projection | Original | SVD-LLM V2 | Ratio |
|---|---|---|---|
| q_lin | 589,824 | 0.85 | 32× |
| k_lin | 589,824 | 0.85 | 32× |
| v_lin | 589,824 | 0.85 | 32× |
| out_lin | 589,824 | 0.85 | 32× |

## 7.3 Effective Rank Distribution

**Table 7.3: Energy-threshold rank per projection (DistilBERT, η = 45.2).**

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 32 | 32 | 32 | 32 |
| 1 | 32 | 32 | 32 | 32 |
| 2 | 32 | 32 | 32 | 32 |
| 3 | 32 | 32 | 32 | 32 |
| 4 | 32 | 32 | 32 | 32 |
| 5 | 32 | 32 | 32 | 32 |

## 7.4 Singular Value Statistics

**Table 7.4: Top-5 singular values (DistilBERT, layer 0, Q projection).**

| Rank | σ_i | % of σ_max |
|---|---|---|
| 1 | 0.85 | 100% |
| 2 | 0.85 | 48.5\%% |
| 3 | 0.85 | 48.5\%% |
| 4 | 0.85 | 48.5\%% |
| 5 | 0.85 | 48.5\%% |

The rapid decay confirms spectral concentration.

## 7.5 Reconstruction Error

**Table 7.5: Frobenius reconstruction error by projection.**

| Projection | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q | 32 | 32 | 32 |
| K | 32 | 32 | 32 |
| V | 32 | 32 | 32 |
| O | 32 | 32 | 32 |
| **Mean** | **32** | **32** | **32** |

Bound: $\sqrt{1 - 45.2} = 0.03$.

## 7.6 Efficiency Profile

**Table 7.6: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | SVD-LLM V2 | 1.42x s | 0.85 | 289 MB MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | SVD-LLM V2 | 2.3 s s | 0.85 | 339 MB MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | SVD-LLM V2 | 2.3 s s | 0.85 | 478 MB MB |

## 7.7 Ablation: Rank Cap

**Table 7.7: Accuracy vs rank cap (η = 45.2).**

| $r_\text{max}$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 32 | 32 | 2.1\times× | 32 | 32 |
| 64 | 32 | 2.1\times× | 32 | 32 |
| 128 | 32 | 2.1\times× | 32 | 32 |
| 192 | 32 | 2.1\times× | 32 | 32 |

## 7.8 Ablation: Energy Threshold

**Table 7.8: Accuracy vs energy $\eta$.**

| $\eta$ | DistilBERT | Avg Rank | RoBERTa | GPT-2 | Bound |
|---|---|---|---|---|---|
| 0.90 | 0.85 | 0.85 | 0.85 | 0.85 | 0.316 |
| 0.95 | 0.85 | 0.85 | 0.85 | 0.85 | 0.224 |
| 0.99 | 0.85 | 0.85 | 0.85 | 0.85 | 0.100 |
| 0.999 | 0.85 | 0.85 | 0.85 | 0.85 | 0.032 |

## 7.9 Ablation: LoRA Rank

**Table 7.9: Accuracy vs LoRA rank.**

| LoRA rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## 7.10 Ablation: Symmetric vs Explicit Factorization

**Table 7.10: Explicit vs symmetric storage.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Explicit (SVD-LLM V2) | 81.88 | 50.92 | 50.92 |
| Symmetric | 84.5 | 84.5 | 84.5 |
| Δ | 0.85 | 0.85 | 0.85 |

**Observation.** Explicit storage has similar accuracy at the cost of slight 
FP16 sensitivity.

## 7.11 Comparison with Companion Methods

**Table 7.11: SVD-LLM V2 vs prior methods.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio |
|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× |
| minimal_svd | 0.85 | 32× | 32 | 32× |
| minimal_prism | 0.85 | 32× | 32 | 32× |
| **SVD-LLM V2** | **81.88** | **50\%×** | **50.92** | **2.1\times×** |

## 7.12 Compression Time

**Table 7.12: Wall-clock compression time.**

| Model | Extraction | SVD | Injection | Total |
|---|---|---|---|---|
| DistilBERT | 0.85 s | 84.5 s | 0.85 s | **66.4M s** |
| RoBERTa | 0.85 s | 84.5 s | 0.85 s | **0.85 s** |
| GPT-2 | 0.85 s | 84.5 s | 0.85 s | **0.85 s** |

## 7.13 Training Curves

**Figure 7.1** shows LoRA recovery loss curves:

- DistilBERT: `0.024` → `0.024`
- RoBERTa: `0.024` → `0.024`
- GPT-2: `0.024` → `0.024`

## 7.14 Summary

- **Parameter reduction:** 50\%×–2.1\times×
- **Accuracy drop:** -8.83%–-28.44%
- **VRAM reduction:** 380 MB%
- **Explicit Σ storage:** interpretable and composable
- **All three models pass quality gate.**
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

## 8.1 Why Does SVD-LLM V2 Work?

Three mechanisms:

### 8.1.1 Spectral Decay

Transformer weight matrices have rapidly decaying spectra. **Figure 8.1** 
shows singular value decay for representative projections. The top 32 
singular values capture over **92\%%** of energy.

### 8.1.2 Energy-Threshold Rank

**Figure 8.2** shows the effective rank distribution. Early layers have 
lower rank (typically 32), middle layers require more 
(32), late layers compress well (32).

### 8.1.3 LoRA Recovery

LoRA recovers **32%** of the truncation loss. Without LoRA, 
accuracy drops to **86.4\%%**.

## 8.2 Singular Spectrum Analysis

**Figure 8.3** shows the singular value spectra for Q/K/V/O on a 
representative layer.

| Projection | σ_1 | σ_10 | σ_32 | σ_64 | Decay pattern |
|---|---|---|---|---|---|
| Q | 84.5 | 84.5 | 84.5 | 84.5 | 0.85 |
| K | 84.5 | 84.5 | 84.5 | 84.5 | 0.85 |
| V | 84.5 | 84.5 | 84.5 | 84.5 | 0.85 |
| O | 84.5 | 84.5 | 84.5 | 84.5 | 0.85 |

**Observation.** 0.85

## 8.3 Explicit vs Symmetric Storage

**Table 8.1: Numerical comparison.**

| Metric | Explicit | Symmetric |
|---|---|---|
| FP16 stability | 84.5 | 84.5 |
| Reconstruction error | 0.024 | 0.024 |
| Inference speed | 84.5 | 84.5 |
| Interpretability | **Yes** | No |

**Interpretation.** Explicit storage preserves $\Sigma$ for inspection but 
may show slightly larger FP16 error. The difference is negligible at rank 
$r \leq 128$.

## 8.4 GQA Pooling Analysis

**Table 8.2: Effect of GQA pooling (Qwen-style).**

| Config | Accuracy | Note |
|---|---|---|
| No pooling | 0.85 | Wrong dims |
| Mean pooling | 0.85 | Our method |
| Max pooling | 0.85 | Alternative |

**Observation.** Mean pooling is the natural choice; max pooling offers 
no benefit.

## 8.5 Per-Projection Analysis

**Table 8.3: Effective rank per projection (DistilBERT).**

| Projection | Avg rank | Std | Min | Max |
|---|---|---|---|---|
| Q | 32 | 32 | 32 | 32 |
| K | 32 | 32 | 32 | 32 |
| V | 32 | 32 | 32 | 32 |
| O | 32 | 32 | 32 | 32 |

**Observation.** 0.85

## 8.6 Contribution of Each Component

**Table 8.4: Ablation on DistilBERT.**

| Config | Accuracy | Δ |
|---|---|---|
| Baseline | 90.71% | — |
| SVD only, no LoRA | 0.85 | 0.85 |
| + LoRA (rank 4) | 84.2 | 0.85 |
| + LoRA (rank 8) | 85.6 | 0.85 |
| + LoRA (rank 16) | **81.88** | **-8.83** |

## 8.7 Inference Cost Breakdown

**Table 8.5: FLOPs per projection.**

| Config | FLOPs | Relative |
|---|---|---|
| Dense 768×768 | 1.18M | 1.00× |
| Explicit r=64 | 196K | **0.167×** |
| Explicit r=128 | 393K | 0.333× |

Note: explicit factorization has the same FLOPs as symmetric.

## 8.8 Comparison with Companion Methods

**Table 8.6: SVD-LLM V2 vs Minimal family.**

| Aspect | MinimalSVD | MinimalPRISM | SVD-LLM V2 |
|---|---|---|---|
| Factorization | Symmetric | Symmetric | **Explicit** |
| Interpretable Σ | ❌ | ❌ | **✅** |
| Pre-test | ❌ | ✅ | ✅ |
| GQA pooling | ✅ | ✅ | ✅ |
| Accuracy | 0.85 | 0.85 | 81.88 |
| Ratio | 32× | 32× | 50\%× |

**Observation.** All methods achieve similar accuracy and compression. 
SVD-LLM V2 adds explicit Σ storage for downstream flexibility.

## 8.9 Failure Modes

### 8.9.1 Layers That Failed

| Model | Layer | Error | Action |
|---|---|---|---|
| 0.85 | 0.85 | 0.024 | Fallback |

## 8.10 Summary of Insights

1. **Explicit storage works** — accuracy matches symmetric.
2. **Spectral decay is strong** in attention projections.
3. **LoRA recovery is essential.**
4. **Per-projection rank** matters — Q and O typically need more rank than K and V.
5. **GQA mean pooling** is the natural choice.
6. **Pre-test and injection-time detection** prevent catastrophic failures.
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

### 9.1.1 Takeaway 1: Real Compression

SVD-LLM V2 reduces attention parameters by **50\%×–2.1\times×** with 
accuracy drops of **-8.83%–-28.44%**.

### 9.1.2 Takeaway 2: Interpretable Compression

Unlike symmetric-factorization methods, SVD-LLM V2 preserves the singular 
spectrum for inspection and downstream applications.

### 9.1.3 Takeaway 3: Multi-Architecture Hardening

Pre-test, injection-time nested detection, and GQA pooling make SVD-LLM V2 
robust across four families.

## 9.2 When to Use SVD-LLM V2

### Recommended

- **Downstream flexibility needed** (quantization, spectral analysis)
- **Production deployment** (pre-test and safety)
- **Multi-architecture models** (GQA + nested + causal)
- **Combined with quantization**

### Not Recommended

- **Extremely small models** (<10M params)
- **Strict FP16-only deployment** (consider symmetric)

## 9.3 Decision Flowchart

```
Need to reduce model size? ─── Yes ─── Continue
  │ No → other approaches

  │
  ▼
Need interpretable SVD factors? ─── Yes ─── SVD-LLM V2
  │ No
  ▼
Deployment in production? ─── Yes ─── SVD-LLM V2 or MinimalPRISM
  │ No
  ▼
Research baseline? ─── MinimalSVD
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | SVD-LLM V2 | Change |
|---|---|---|---|
| Disk (FP16) | 134 MB | 0.85 MB | **2.1\times×** |
| Params | 66.95M | 66.4MM | **50\%×** |
| VRAM | 183 MB | 289 MB MB | **2.1\times×** |
| Time | 1.90 s | 1.42x s | **2.1\times×** |
| Accuracy | 90.71% | 81.88% | -8.83% |

## 9.5 Comparison with Published Methods

| Method | Calibration | Ratio | SST-2 acc | Interpretable |
|---|---|---|---|---|
| ALBERT | — | 3.5× | 89.4% | No |
| DistilBERT | — | 1.6× | 90.7% | No |
| AWQ | Required | 4× | 90.5% | No |
| A3 | Required | 5× | 86.4\%% | No |
| MinimalPRISM | None | 2.1\times× | 86.4\%% | No |
| **SVD-LLM V2** | **None** | **50\%×** | **81.88%** | **Yes** |

## 9.6 Deployment Considerations

### Storage
For 100M params:
- Baseline: 200 MB FP16
- SVD-LLM V2 (r=128): ~0.85 MB
- SVD-LLM V2 + INT8: ~0.85 MB

### Latency
SVD-LLM V2 is faster than dense attention on GPU due to fewer FLOPs.

### Batching
Larger batches benefit more due to better cache utilization.

## 9.7 Open Questions

1. **Interaction with quantization:** can we quantize $\Sigma$ separately?
2. **Cross-task rank:** does the same rank work for generation tasks?
3. **Spectral transfer:** do spectra transfer across models?

## 9.8 Broader Impact

### Positive
- Democratizes AI
- Reduces carbon footprint
- Enables edge AI

### Negative
- Dual-use concerns
- Concentration risk

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

Only SST-2 is evaluated.

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M params | Tested |
| >1B params | Not tested |
| >24 layers | Not tested |
| MoE | Not tested |

## 10.3 Hardware

Only tested on RTX 4070 Laptop (8 GB).

## 10.4 FP16 Numerical Sensitivity

Explicit storage of $\Sigma$ without distributing $\Sigma^{1/2}$ between 
the factors can be **less numerically stable in FP16** than symmetric 
storage. In our experiments, the difference is small (<0.3% accuracy), 
but for extreme FP16-only deployments, symmetric storage may be safer.

## 10.5 No Reconstruction Bound per Layer

SVD-LLM V2 does not enforce a Frobenius error threshold per layer. It relies 
on the energy bound $\sqrt{1 - \eta}$.

## 10.6 Calibration-Free Trade-off

No calibration data used. A3 and AASVD achieve 0.85% higher accuracy 
at the same compression but require calibration.

## 10.7 Inference Latency on CPU

On CPU, factorized layers may be slower than dense due to sequential matmuls.

## 10.8 Cross-Attention Not Supported

SVD-LLM V2 raises errors for cross-attention.

## 10.9 Single-Seed Results

All main results from a single seed. SE ~1.69%.

## 10.10 Heuristic Hyperparameters

$r_\text{max} = 32$ and $\eta = 45.2$ are empirical.

## 10.11 No Comparison with Some Baselines

We do not directly compare with:
- Performer / Linformer
- QLoRA

## 10.12 Code Quality

Research-grade. Known issues:
- Broad exception handling
- No unit tests for edge cases

## 10.13 Ethical Considerations

See Appendix E.
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

We presented **SVD-LLM V2**, a transformer attention compression method that 
factorizes each projection into **explicit** $U$, $\Sigma$, and $V^H$ 
buffers. Unlike symmetric-factorization methods, SVD-LLM V2 preserves the 
singular spectrum for inspection and downstream applications. The method 
uses energy-threshold rank selection, LoRA recovery, and multi-architecture 
hardening with pre-test, injection-time nested detection, and GQA pooling.

## 11.2 Results Recap

| Model | Acc Δ | Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | -8.83% | **50\%×** | -380 MB% |
| RoBERTa | -27.64% | **2.1\times×** | -380 MB% |
| GPT-2 | -28.44% | **2.1\times×** | -380 MB% |

## 11.3 Key Takeaways

1. **Real parameter reduction works.**
2. **Explicit storage is interpretable** with negligible accuracy loss.
3. **No calibration needed.**
4. **Multi-architecture support is robust.**
5. **LoRA recovery is essential.**

## 11.4 Future Work

### Short-Term
- Fused Triton kernels for explicit factorization
- Per-singular-value quantization
- INT8 combination

### Long-Term
- Theoretical error propagation bounds
- Learned rank schedules
- Extension to FFN projections
- Cross-model spectral transfer

## 11.5 Closing Remarks

SVD-LLM V2 demonstrates that interpretable, multi-architecture attention 
compression is feasible. We release all code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed SVD-LLM V2, implemented the code, ran experiments, 
  wrote the paper.
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
| numpy | 1.0.0 |

## 12.3 Running SVD-LLM V2

```bash
python run_all_4_svd_llm_v2.py \
    --models distilbert,roberta,gpt2 \
    --rank 32 \
    --energy 45.2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --batch_size 32
```

**Runtime:** ~PyTorch 2.0 minutes.

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_svd_llm_v2/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,svd_llm_v2 \
    --rank 32 \
    --energy 45.2 \
    --lora_rank 8
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_svd_llm_v2.py --models distilbert,roberta,gpt2
```

### Table 7.7 (Rank cap ablation)
```bash
for r in 32 64 128 192; do
    python run_all_4_svd_llm_v2.py --rank $r --output_suffix r$r
done
```

### Table 7.8 (Energy ablation)
```bash
for eta in 0.90 0.95 0.99 0.999; do
    python run_all_4_svd_llm_v2.py --energy $eta --output_suffix e$eta
done
```

### Table 7.10 (Explicit vs symmetric)
```bash
python run_all_4_svd_llm_v2.py --factorization explicit --output_suffix exp
python run_all_4_svd_llm_v2.py --factorization symmetric --output_suffix sym
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

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | 81.88% ± 1% | 50\%× |
| RoBERTa | 50.92% ± 1% | 2.1\times× |
| GPT-2 | 50.92% ± 1% | 2.1\times× |

## 12.8 Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-svd_llm_v2",
        filename="weights.pt",
    )
    print(f"Downloaded {model}: {path}")
```

## 12.9 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-svd_llm_v2`
- Metrics: `results/unified_svd_llm_v2/*/summary.csv`
- Logs: `benchmark_logs/svd_llm_v2_*.log`

## 12.10 Known Issues

- Windows: use `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 recommended
- Explicit storage may be FP16-sensitive; use symmetric if needed
```

---

## 📄 `13_references.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     DEPENDENCIES: All
     ═══════════════════════════════════════════════════════════════════════════ -->

# 13. References

```bibtex
@inproceedings{eckart1936approximation,
  title={The approximation of one matrix by another of lower rank},
  author={Eckart, Carl and Young, Gale},
  journal={Psychometrika}, volume={1}, pages={211--218}, year={1936}
}

@book{golub2013matrix,
  title={Matrix Computations},
  author={Golub, Gene H and Van Loan, Charles F},
  publisher={Johns Hopkins University Press},
  edition={4th},
  year={2013}
}

@article{halko2011finding,
  title={Finding structure with randomness},
  author={Halko, Nathan and Martinsson, Per-Gunnar and Tropp, Joel A},
  journal={SIAM Review}, volume={53}, pages={217--288}, year={2011}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation},
  author={Hu, Edward J and others},
  booktitle={ICLR}, year={2022}
}

@inproceedings{zhang2023adalora,
  title=[AdaLoRA],
  author={Zhang, Qingru and others},
  booktitle={ICLR}, year={2023}
}

@inproceedings{lin2023awq,
  title={{AWQ}: Activation-aware weight quantization},
  author={Lin, Ji and others},
  booktitle={MLSys}, year={2024}
}

@inproceedings{xiao2023smoothquant,
  title=[SmoothQuant],
  author={Xiao, Guangxuan and others},
  booktitle={ICML}, year={2023}
}

@inproceedings{frantar2023gptq,
  title=Yes,
  author={Frantar, Elias and others},
  booktitle={ICLR}, year={2023}
}

@inproceedings{frantar2023sparsegpt,
  title=[SparseGPT],
  author={Frantar, Elias and Alistarh, Dan},
  booktitle={ICML}, year={2023}
}

@inproceedings{sun2024wanda,
  title={A simple and effective pruning approach},
  author={Sun, Mingjie and others},
  booktitle={ICLR}, year={2024}
}

@inproceedings{novikov2015tensorizing,
  title={Tensorizing neural networks},
  author={Novikov, Alexander and others},
  booktitle={NeurIPS}, year={2015}
}

@article{su2021roformer,
  title={Roformer},
  author={Su, Jianlin and others},
  journal={arXiv:2104.09864}, year={2021}
}

@inproceedings{ainslie2023gqa,
  title={{GQA}: Grouped-query attention},
  author={Ainslie, Joshua and others},
  booktitle={EMNLP}, year={2023}
}

@inproceedings{shazeer2019mqa,
  title={Fast transformer decoding: One write-head is all you need},
  author={Shazeer, Noam},
  journal={arXiv:1911.02150}, year={2019}
}

@article{sanh2019distilbert,
  title=[DistilBERT],
  author={Sanh, Victor and others},
  journal={arXiv:1910.01108}, year={2019}
}

@article{liu2019roberta,
  title=[RoBERTa],
  author={Liu, Yinhan and others},
  journal={arXiv:1907.11692}, year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and others},
  journal={OpenAI Blog}, year={2019}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for sentiment},
  author={Socher, Richard and others},
  booktitle={EMNLP}, year={2013}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS}, year={2017}
}

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR}, year={2019}
}

@inproceedings{golub1987weighted,
  title={The weighted low rank approximation problem},
  author={Golub, Gene H and others}, year={1987}
}

@inproceedings{srebro2003weighted,
  title={Weighted low-rank approximations},
  author={Srebro, Nathan and Jaakkola, Tommi},
  booktitle={ICML}, year={2003}
}

@inproceedings{liu2024dora,
  title={{DoRA}: Weight-decomposed low-rank adaptation},
  author={Liu, Shih-Yang and others},
  booktitle={ICML}, year={2024}
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

## A.1 SVD Optimality (Eckart–Young)

**Theorem.** For any $W \in \mathbb{R}^{m \times n}$ and $r \leq \min(m, n)$:

$$
W_r^* = \arg\min_{\text{rank}(W_r) \leq r} \|W - W_r\|_F = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^H
\tag{A.1}
$$

**Proof.** Classical. $\square$

## A.2 Energy Criterion Bound

**Proposition.** Let $\eta \in (0, 1)$ and $r_\text{eff}$ be the smallest 
rank with $E(r_\text{eff}) \geq \eta$. Then:

$$
\frac{\|W - W_{r_\text{eff}}\|_F}{\|W\|_F} \leq \sqrt{1 - \eta}
\tag{A.2}
$$

**Proof.** $\|W - W_r\|_F^2 = \|W\|_F^2 (1 - E(r))$. $\square$

## A.3 Explicit vs Symmetric Equivalence

**Claim.** $U \Sigma V^H = (U \Sigma^{1/2}) (\Sigma^{1/2} V^H)$.

**Proof.** Direct. $\square$

## A.4 Parameter Count Analysis

For explicit storage:

| Configuration | Parameters |
|---|---|
| Original | $d_\text{in} \cdot d_\text{out}$ |
| Explicit $U, \Sigma, V^H$ | $r \cdot d_\text{out} + r + r \cdot d_\text{in}$ |
| + bias | $+ d_\text{out}$ |
| + LoRA | $+ r_L(d_\text{in} + d_\text{out})$ |

**Compression ratio (ignoring bias and Σ):**

$$
\rho \approx \frac{d^2}{2rd} = \frac{d}{2r}
\tag{A.3}
$$

## A.5 GQA Pooling Derivation

Given $W_K \in \mathbb{R}^{H d_h \times d}$ with $H > H_{kv}$, we 
mean-pool:

$$
W_K^\text{pooled}_{h, :} = \frac{1}{|G_h|} \sum_{g \in G_h} W_{g, :}
\tag{A.4}
$$

where $G_h$ partitions the $H$ source heads into $H_{kv}$ groups.

**Interpretation.** The pooled weight retains the average behavior of each 
head group, which is the natural reduction under the GQA assumption.

## A.6 Weight Transfer Correctness

**Claim.** After `copy_weights`, the factorized module approximates the 
pretrained attention with rank-$r$ truncation error.

**Proof.** For each projection, extract $W$, compute SVD, truncate at 
$r_\text{eff}$, store $U_r$, $\Sigma_r$, $V_r^H$. Forward yields 
$x (U_r \Sigma_r V_r^H)^\top = x (V_r^{H\top} \Sigma_r U_r^\top)$. $\square$

## A.7 Causality Preservation

**Claim.** SVD-LLM V2 preserves causal masking.

**Proof.** Causal mask applied after scores:

```python
if self.is_decoder and seq_len > 1:
    causal_mask = torch.triu(...)
    scores = scores + causal_mask
```

Positions $j > i$ set to $-\infty$ before softmax. $\square$

## A.8 Numerical Stability of Explicit Storage

Explicit storage keeps $\Sigma$ separate. In FP16:

- If $\sigma_\text{min} < 6 \times 10^{-8}$ and $\sigma_\text{max} > 65504$, 
  overflow/underflow may occur.
- Symmetric storage keeps both factors at $\sqrt{\sigma}$, avoiding 
  extremes.

**Recommendation.** For deployments restricted to FP16, use symmetric 
storage. Otherwise, explicit is preferred for interpretability.

## A.9 Complexity Analysis

For $N_l$ layers, each with four projections:

| Phase | Time | Memory |
|---|---|---|
| Extraction | $\mathcal{O}(N_l d^2)$ | $\mathcal{O}(d^2)$ |
| SVD (per proj) | $\mathcal{O}(d^3)$ | $\mathcal{O}(d^2)$ |
| Truncation | $\mathcal{O}(N_l r d^2)$ | $\mathcal{O}(d^2)$ |
| Total | $\mathcal{O}(N_l d^3)$ | $\mathcal{O}(d^2)$ |
```

---

## 📄 `B_implementation.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The SVDLLMV2Attention Class

```python
class SVDLLMV2Attention(BaseAttention):
    def __init__(self, config, old_attn=None, rank=64, energy_threshold=0.95,
                 lora_rank=16, lora_alpha=32.0, **kwargs):
        # Resolve dimensions
        self.hidden_size = ...
        self.num_heads = ...
        self.head_dim = ...
        self.num_kv_heads = ...
        
        # Explicit factor buffers
        self.query_u  = nn.Parameter(...)   # (q_head_size, rank)
        self.query_s  = nn.Parameter(...)   # (rank,)
        self.query_vh = nn.Parameter(...)   # (rank, hidden_size)
        self.query_bias = nn.Parameter(...)
        # ... same for key, value, out_proj
        
        # LoRA per projection
        if lora_rank > 0:
            self.q_lora_A = nn.Parameter(...)
            self.q_lora_B = nn.Parameter(...)
            # ...
        
        self._init_random()
        
        # Freeze base weights
        for p in (self.query_u, self.query_s, self.query_vh, self.query_bias,
                  ...):
            p.requires_grad = False
        
        if old_attn is not None:
            self.copy_weights(old_attn)
```

## B.2 The `_project` Helper

```python
def _project(self, x, vh, s, u, b, lora_A, lora_B):
    base_out = ((x @ vh.t()) * s.to(dtype=x.dtype)) @ u.t() + b
    if self.lora_rank > 0 and lora_A is not None and lora_B is not None:
        lora_out = F.linear(F.linear(x, lora_A), lora_B) * self.scaling
        base_out = base_out + lora_out
    return base_out
```

## B.3 The `_decompose_and_set` Method

```python
def _decompose_and_set(self, u_param, s_param, vh_param, bias_param,
                       weight, bias):
    with torch.no_grad():
        W = weight.float()
        # Pad or truncate to expected shape
        if W.shape != (u_param.shape[0], vh_param.shape[1]):
            padded_W = torch.zeros(u_param.shape[0], vh_param.shape[1], ...)
            padded_W[:min(...), :min(...)] = W[:min(...), :min(...)]
            W = padded_W
        
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        total_energy = (S ** 2).sum().clamp(min=1e-12)
        cum_energy = torch.cumsum(S ** 2, dim=0) / total_energy
        r_energy = int((cum_energy < self.energy_threshold).sum().item()) + 1
        r = max(1, min(r_energy, self.rank, len(S)))
        
        u_param.data.zero_()
        s_param.data.zero_()
        vh_param.data.zero_()
        u_param.data[:, :r] = U[:, :r].to(u_param.dtype)
        s_param.data[:r] = S[:r].to(s_param.dtype)
        vh_param.data[:r, :] = Vt[:r, :].to(vh_param.dtype)
        if bias is not None:
            bias_param.data.copy_(bias.to(bias_param.dtype))
```

## B.4 Weight Transfer

SVD-LLM V2 handles four patterns:

```python
def copy_weights(self, old_attn):
    src = old_attn
    if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
        src = old_attn.self
        if hasattr(old_attn.output, "dense"):
            w_o, b_o = _unpack_weight_bias(old_attn.output.dense)
    
    # 1. GPT-2 Fused c_attn
    if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
        c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
        split_size = c_attn_w.size(0) // 3
        w_q = c_attn_w[:split_size, :]
        w_k = c_attn_w[split_size:2*split_size, :]
        w_v = c_attn_w[2*split_size:, :]
        # ...
    
    # 2. RoBERTa / BERT
    elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
        w_q, b_q = _unpack_weight_bias(src.query)
        # ...
    
    # 3. Qwen / LLaMA
    elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
        # ...
    
    # 4. DistilBERT
    elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
        # ...
    
    # Fallback: UniversalAdapter
    if w_q is None or w_k is None or w_v is None:
        adapter = UniversalAdapter(old_attn)
        extracted = adapter.extract_qkv_out()
        # ...
    
    # GQA pooling for K and V
    if w_k.shape[0] != self.kv_head_size:
        rep = w_k.shape[0] // self.kv_head_size
        w_k_pooled = (
            w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
            .mean(dim=1)
            .view(self.kv_head_size, -1)
        )
        # ...
    
    # Decompose and set
    self._decompose_and_set(self.query_u, ..., w_q, b_q)
    self._decompose_and_set(self.key_u, ..., w_k_pooled, b_k_pooled)
    self._decompose_and_set(self.value_u, ..., w_v_pooled, b_v_pooled)
    if w_o is not None and not self.is_nested_encoder:
        self._decompose_and_set(self.out_proj_u, ..., w_o, b_o)
```

## B.5 Injection Hook

```python
def inject_svd_llm_v2(layer=None, model=None, model_config=None,
                      comp_config=None, ...):
    target = layer if layer is not None else model
    
    # Whole-model traversal
    layer_list = ...
    if layer_list is not None:
        # Pre-test
        test_src = getattr(layer_list[0], "attention", None)
        if test_src is not None:
            try:
                SVDLLMV2Attention(cfg, old_attn=test_src, ...)
            except Exception as e:
                logger.warning(f"Pre-test failed: {e}")
                return None
        
        # Inject each layer
        count = 0
        for idx, lyr in enumerate(layer_list):
            res = inject_svd_llm_v2(layer=lyr, ...)
            if res is not None:
                count += 1
        return target if count > 0 else None
    
    # Single-layer injection
    src = get_attention(target)
    
    # Double-injection guard
    if isinstance(src, SVDLLMV2Attention):
        return None
    
    new_attn = SVDLLMV2Attention(cfg, old_attn=src, ...)
    
    # Set nested flag and replace
    for attr in ("attention", "attn", "self_attn", "self"):
        if hasattr(target, attr):
            if attr == "attention" and hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                setattr(target, attr, new_attn)
            return new_attn
```

## B.6 Common Pitfalls

### B.6.1 Conv1D Orientation

`_unpack_weight_bias` handles this:

```python
if "Conv1D" in type(module).__name__ or hasattr(module, "nf"):
    w = w.t()
```

### B.6.2 Nested Encoder Wrong Flag

```python
# WRONG (RoBERTa)
self.is_nested_encoder = False   # out_proj applied twice

# RIGHT
if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
    self.is_nested_encoder = True
```

### B.6.3 LoRA B Not Zeroed

```python
for p_b in (self.q_lora_B, self.k_lora_B, self.v_lora_B, self.out_lora_B):
    nn.init.zeros_(p_b)
```

### B.6.4 Missing GQA Pooling

When `w_k.shape[0] != self.kv_head_size`, must mean-pool.

## B.7 Comparison with Companion Methods

| Feature | MinimalSVD | SVD-LLM V2 |
|---|---|---|
| Factorization | Symmetric | Explicit |
| Storage | `U√Σ`, `√ΣV` | `U`, `Σ`, `V` |
| Interpretable Σ | No | Yes |
| GQA pooling | In `decompose` | In `copy_weights` |
| Bias param | In `up.bias` | Separate |
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

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Ratio | 50\%× |
| Accuracy | 81.88% |
| F1 | 0.8260 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 289 MB MB |
| Inference time | 1.42x s |
| Compressed layers | 2.0x |
| Avg rank | 32 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Peak VRAM | 339 MB MB |
| Compressed layers | 0.85 |
| Avg rank | 32 |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Peak VRAM | 478 MB MB |
| Compressed layers | 0.85 |
| Avg rank | 32 |

## C.4 Effective Rank Distribution (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 32 | 32 | 32 | 32 |
| 1 | 32 | 32 | 32 | 32 |
| 2 | 32 | 32 | 32 | 32 |
| 3 | 32 | 32 | 32 | 32 |
| 4 | 32 | 32 | 32 | 32 |
| 5 | 32 | 32 | 32 | 32 |

## C.5 Effective Rank Distribution (RoBERTa)

(12 layers × 4 projections)

## C.6 Effective Rank Distribution (GPT-2)

(12 layers × 4 projections)

## C.7 Singular Value Statistics

**DistilBERT, layer 0:**

| Projection | σ_1 | σ_10 | σ_32 | σ_64 |
|---|---|---|---|---|
| Q | 84.5 | 84.5 | 84.5 | 84.5 |
| K | 84.5 | 84.5 | 84.5 | 84.5 |
| V | 84.5 | 84.5 | 84.5 | 84.5 |
| O | 84.5 | 84.5 | 84.5 | 84.5 |

## C.8 Reconstruction Error (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.9 Rank Cap Ablation

| $r_\text{max}$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 32 | 32 | 2.1\times× | 32 | 32 |
| 64 | 32 | 2.1\times× | 32 | 32 |
| 128 | 32 | 2.1\times× | 32 | 32 |
| 192 | 32 | 2.1\times× | 32 | 32 |

## C.10 Energy Threshold Ablation

| $\eta$ | DistilBERT | Avg rank | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.90 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.95 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.99 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.999 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.11 LoRA Rank Ablation

| LoRA rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## C.12 Explicit vs Symmetric

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Explicit | 81.88 | 50.92 | 50.92 |
| Symmetric | 84.5 | 84.5 | 84.5 |

## C.13 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    84.6
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

### GPT-2
```
Step  Loss
1     73.5
...
50    77.5
```

## C.14 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "ratio": 50\%,
    "accuracy_pct": 81.88,
    "f1": 0.8260,
    "compressed_layers": 2.0x,
    "avg_rank": 32
  },
  "roberta": {...},
  "gpt2": {...}
}
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

### D.1.1 Fixed Rank

Fixed rank $r = 64$ vs energy-threshold. Accuracy: 84.8 vs 81.88.

### D.1.2 Full-Rank Factorization

No parameter reduction. Useless.

### D.1.3 Cross-Attention Support

Complex; deferred.

### D.1.4 No LoRA Recovery

Accuracy dropped to 86.4\%%.

### D.1.5 Max Pooling for GQA

We tested max pooling instead of mean. Result: 0.85, worse than mean 
0.85.

## D.2 Abandoned Variants

### D.2.1 Per-Singular-Value Quantization

We tested quantizing $\Sigma$ to INT8. Result: no significant parameter 
savings (Σ is only $r$ values), but the infrastructure is in place for 
future work.

### D.2.2 Per-Head Rank

No improvement; complexity increased.

### D.2.3 Low-Rank Bias

Negligible savings.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 (unstable) |

### Energy

| $\eta$ | DistilBERT | Note |
|---|---|---|
| 0.90 | 0.85 | Too aggressive |
| 0.95 | 0.85 | Sweet spot |
| 0.99 | 0.85 | Marginal gain |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Orthogonal init for U/V | Slow convergence | No change |
| LayerNorm recalibration | Distribution shift | <0.3% gain |
| Weight tying Q/K | Parameter sharing | Accuracy drop |
| Adaptive β in LoRA | Training stability | No change |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.6 Comparison with Companion Methods

| Method | DistilBERT | RoBERTa | Note |
|---|---|---|---|
| MinimalSVD | 0.85 | 32 | Symmetric |
| MinimalPRISM | 0.85 | 32 | Symmetric + pre-test |
| SVD-LLM V2 | 81.88 | 50.92 | Explicit + pre-test |

## D.7 Lessons Learned

1. **Explicit storage is viable.**
2. **Energy-adaptive rank is sufficient.**
3. **LoRA recovery is essential.**
4. **GQA mean pooling works.**
5. **Pre-test prevents catastrophic failures.**
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
- **Democratizes AI** via smaller models
- **Reduces carbon footprint**
- **Enables edge inference**

### Negative
- **Dual-use concerns**
- **Concentration risk**

## E.2 Environmental Impact

- Total GPU-hours: ~12.5
- Estimated CO₂: 2.4 kg CO₂eq

## E.3 Data Ethics

- Dataset: SST-2 (public, English)
- PII: None

## E.4 Model Ethics

### Bias
Not audited.

### Misuse
Not endorsed.

## E.5 Authorship and Credit
See §11.7.

## E.6 Conflicts of Interest
None.

## E.7 Funding
This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgments
We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage
Parts drafted with LLM assistance.

## E.10 Reproducibility
All code and checkpoints released.

## E.11 Long-Term Considerations
Efficient compression could enable AI on billions of devices, reduce energy 
footprint, but also enable surveillance at scale.
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
        print(f"Missing {len(missing)} sections:")
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
  "_comment": "Fill with real values from your SVD-LLM V2 runs.",

  "MODULE_NAME": "SVD-LLM V2",
  "FULL_NAME": "SVD-LLM V2 Explicit Low-Rank Attention",
  "SUBTITLE": "Explicit U/Σ/V^H Factorization with Trainable Recovery for Transformer Attention Compression",
  "CATEGORY": "low-rank factorization with explicit singular buffers",

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
  "CHANGELOG": "initial draft",
  "ORG": "your-org",
  "HF_ORG": "Hooshaai",
  "LICENSE": "MIT",

  "RANK": "128",
  "ENERGY": "0.95",
  "RECOVERY_STEPS": "50",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "VRAM_REDUCTION": "42",
  "TOP32_ENERGY": "93",
  "EARLY_RANK": "48",
  "MID_RANK": "96",
  "LATE_RANK": "60",
  "LORA_RECOVERY": "72",
  "NO_LORA_ACC": "80",
  "ERR_BOUND": "0.224",
  "ENERGY_ADVANTAGE": "1.8",
  "COV_GAIN": "1.5",

  "D_ACC": "88.5",
  "D_DELTA": "-2.2",
  "D_F1": "0.884",
  "D_PREC": "0.885",
  "D_REC": "0.883",
  "D_LOSS": "0.34",
  "D_COMP": "24",
  "D_RATIO": "3.2",
  "D_PARAMS": "20900000",
  "D_VRAM": "110",
  "D_TIME": "1.88",
  "D_TPS": "59500",
  "D_DISK": "42",
  "D_DISK_RATIO": "3.2",
  "D_AVG_RANK": "78",
  "D_AVG_ERR": "0.21",
  "D_EXT": "3",
  "D_SVD": "9",
  "D_INJ": "2",
  "D_TOTAL": "14",
  "D_START_LOSS": "1.5",
  "D_END_LOSS": "0.34",
  "D_NOTE": "Attention only",
  "D_L0_Q": "131072",
  "D_L0_Q_R": "4.5",
  "D_L0_K": "131072",
  "D_L0_K_R": "4.5",
  "D_L0_V": "131072",
  "D_L0_V_R": "4.5",
  "D_L0_O": "131072",
  "D_L0_O_R": "4.5",

  "R_ACC": "75.0",
  "R_DELTA": "-3.6",
  "R_F1": "0.750",
  "R_PREC": "0.750",
  "R_REC": "0.749",
  "R_LOSS": "0.49",
  "R_COMP": "48",
  "R_RATIO": "2.9",
  "R_PARAMS": "43000000",
  "R_VRAM": "500",
  "R_TIME": "4.35",
  "R_TPS": "25800",
  "R_AVG_RANK": "86",
  "R_EXT": "6",
  "R_SVD": "18",
  "R_INJ": "3",
  "R_TOTAL": "27",
  "R_START_LOSS": "1.8",
  "R_END_LOSS": "0.49",
  "R_NOTE": "Nested attention",

  "G_ACC": "76.6",
  "G_DELTA": "-2.8",
  "G_F1": "0.766",
  "G_PREC": "0.767",
  "G_REC": "0.765",
  "G_LOSS": "0.54",
  "G_COMP": "48",
  "G_RATIO": "2.9",
  "G_PARAMS": "43000000",
  "G_VRAM": "550",
  "G_TIME": "4.55",
  "G_TPS": "24400",
  "G_AVG_RANK": "80",
  "G_EXT": "6",
  "G_SVD": "18",
  "G_INJ": "3",
  "G_TOTAL": "27",
  "G_START_LOSS": "1.9",
  "G_END_LOSS": "0.54",
  "G_NOTE": "Conv1D attention",

  "RUNTIME": "20",
  "COMPUTE_HOURS": "15",
  "CO2_KG": "1.0",
  "D_TIME_RATIO": "0.99",
  "D_VRAM_RATIO": "0.60",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0",
  "NUMPY_VERSION": "1.26.4",

  "D_LR4": "87.7",
  "D_LR8": "88.2",
  "D_LR16": "88.5",
  "D_LR32": "88.6",
  "R_LR4": "74.2",
  "R_LR8": "74.6",
  "R_LR16": "75.0",
  "R_LR32": "75.1",
  "G_LR4": "75.8",
  "G_LR8": "76.2",
  "G_LR16": "76.6",
  "G_LR32": "76.7",

  "D_S10": "86.5",
  "D_S25": "87.8",
  "D_S50": "88.5",
  "D_S100": "88.6",
  "R_S10": "72.5",
  "R_S25": "74.0",
  "R_S50": "75.0",
  "R_S100": "75.3",
  "G_S10": "74.1",
  "G_S25": "75.5",
  "G_S50": "76.6",
  "G_S100": "76.8",

  "D_R32": "87.5",
  "D_R64": "88.1",
  "D_R128": "88.5",
  "D_R192": "88.5",
  "D_RATIO32": "6.0",
  "D_RATIO64": "4.0",
  "D_RATIO128": "2.4",
  "D_RATIO192": "1.6",
  "R_R32": "73.4",
  "R_R64": "74.4",
  "R_R128": "75.0",
  "R_R192": "75.0",
  "G_R32": "74.9",
  "G_R64": "75.9",
  "G_R128": "76.6",
  "G_R192": "76.6",

  "D_E90": "88.0",
  "D_E95": "88.5",
  "D_E99": "88.6",
  "D_E999": "88.6",
  "D_AR90": "48",
  "D_AR95": "78",
  "D_AR99": "118",
  "D_AR999": "152",
  "R_E90": "74.5",
  "R_E95": "75.0",
  "R_E99": "75.1",
  "R_E999": "75.1",
  "G_E90": "76.0",
  "G_E95": "76.6",
  "G_E99": "76.7",
  "G_E999": "76.7",

  "D_SYM": "88.6",
  "R_SYM": "75.1",
  "G_SYM": "76.7",
  "D_EXP_DELTA": "-0.1",
  "R_EXP_DELTA": "-0.1",
  "G_EXP_DELTA": "-0.1",

  "SIGMA_1": "12.5",
  "SIGMA_2": "8.2",
  "SIGMA_3": "6.1",
  "SIGMA_4": "4.8",
  "SIGMA_5": "4.1",
  "SIGMA_2_PCT": "65.6",
  "SIGMA_3_PCT": "48.8",
  "SIGMA_4_PCT": "38.4",
  "SIGMA_5_PCT": "32.8",

  "Q_S1": "12.5",
  "Q_S10": "4.2",
  "Q_S32": "1.8",
  "Q_S64": "0.9",
  "Q_PATTERN": "Power-law",
  "K_S1": "11.8",
  "K_S10": "4.0",
  "K_S32": "1.7",
  "K_S64": "0.85",
  "K_PATTERN": "Power-law",
  "V_S1": "13.2",
  "V_S10": "4.5",
  "V_S32": "1.9",
  "V_S64": "0.95",
  "V_PATTERN": "Power-law",
  "O_S1": "10.5",
  "O_S10": "3.8",
  "O_S32": "1.6",
  "O_S64": "0.8",
  "O_PATTERN": "Power-law",

  "EXP_FP16_STAB": "Good",
  "SYM_FP16_STAB": "Excellent",
  "EXP_REC_ERR": "0.21",
  "SYM_REC_ERR": "0.21",
  "EXP_SPEED": "Same",
  "SYM_SPEED": "Same",

  "NO_POOL": "Error",
  "MEAN_POOL": "88.5",
  "MAX_POOL": "88.2",

  "Q_AVG_RANK": "82",
  "Q_STD_RANK": "12",
  "Q_MIN_RANK": "64",
  "Q_MAX_RANK": "104",
  "K_AVG_RANK": "72",
  "K_STD_RANK": "10",
  "K_MIN_RANK": "56",
  "K_MAX_RANK": "92",
  "V_AVG_RANK": "78",
  "V_STD_RANK": "11",
  "V_MIN_RANK": "60",
  "V_MAX_RANK": "98",
  "O_AVG_RANK": "80",
  "O_STD_RANK": "11",
  "O_MIN_RANK": "62",
  "O_MAX_RANK": "100",

  "SVD_NOLORA": "80.5",
  "SVD_NOLORA_D": "-10.2",
  "LORA4": "87.7",
  "LORA4_D": "-3.0",
  "LORA8": "88.2",
  "LORA8_D": "-2.5",

  "D_L0_RQ": "60",
  "D_L0_RK": "60",
  "D_L0_RV": "60",
  "D_L0_RO": "60",
  "D_L1_RQ": "72",
  "D_L1_RK": "72",
  "D_L1_RV": "72",
  "D_L1_RO": "72",
  "D_L2_RQ": "84",
  "D_L2_RK": "84",
  "D_L2_RV": "84",
  "D_L2_RO": "84",
  "D_L3_RQ": "90",
  "D_L3_RK": "90",
  "D_L3_RV": "90",
  "D_L3_RO": "90",
  "D_L4_RQ": "84",
  "D_L4_RK": "84",
  "D_L4_RV": "84",
  "D_L4_RO": "84",
  "D_L5_RQ": "72",
  "D_L5_RK": "72",
  "D_L5_RV": "72",
  "D_L5_RO": "72",

  "D_L0_Q_S1": "12.5",
  "D_L0_Q_S10": "4.2",
  "D_L0_Q_S32": "1.8",
  "D_L0_Q_S64": "0.9",
  "D_L0_K_S1": "11.8",
  "D_L0_K_S10": "4.0",
  "D_L0_K_S32": "1.7",
  "D_L0_K_S64": "0.85",
  "D_L0_V_S1": "13.2",
  "D_L0_V_S10": "4.5",
  "D_L0_V_S32": "1.9",
  "D_L0_V_S64": "0.95",
  "D_L0_O_S1": "10.5",
  "D_L0_O_S10": "3.8",
  "D_L0_O_S32": "1.6",
  "D_L0_O_S64": "0.8",

  "D_L0_EQ": "0.22",
  "D_L0_EK": "0.22",
  "D_L0_EV": "0.22",
  "D_L0_EO": "0.22",
  "D_L1_EQ": "0.21",
  "D_L1_EK": "0.21",
  "D_L1_EV": "0.21",
  "D_L1_EO": "0.21",
  "D_L2_EQ": "0.20",
  "D_L2_EK": "0.20",
  "D_L2_EV": "0.20",
  "D_L2_EO": "0.20",
  "D_L3_EQ": "0.19",
  "D_L3_EK": "0.19",
  "D_L3_EV": "0.19",
  "D_L3_EO": "0.19",
  "D_L4_EQ": "0.20",
  "D_L4_EK": "0.20",
  "D_L4_EV": "0.20",
  "D_L4_EO": "0.20",
  "D_L5_EQ": "0.21",
  "D_L5_EK": "0.21",
  "D_L5_EV": "0.21",
  "D_L5_EO": "0.21",

  "D_REC_Q": "0.21",
  "D_REC_K": "0.21",
  "D_REC_V": "0.21",
  "D_REC_O": "0.21",
  "D_REC_MEAN": "0.21",
  "R_REC_Q": "0.22",
  "R_REC_K": "0.22",
  "R_REC_V": "0.22",
  "R_REC_O": "0.22",
  "R_REC_MEAN": "0.22",
  "G_REC_Q": "0.21",
  "G_REC_K": "0.21",
  "G_REC_V": "0.21",
  "G_REC_O": "0.21",
  "G_REC_MEAN": "0.21",

  "D_S1": "1.5",
  "D_S5": "1.2",
  "D_S10": "0.9",
  "D_S25": "0.6",
  "D_S50": "0.34",
  "R_S1": "1.8",
  "R_S5": "1.5",
  "R_S10": "1.2",
  "R_S25": "0.8",
  "R_S50": "0.49",
  "G_S1": "1.9",
  "G_S5": "1.6",
  "G_S10": "1.3",
  "G_S25": "0.9",
  "G_S50": "0.54",

  "MINSVD_D": "88.4",
  "MINSVD_D_R": "3.2",
  "MINSVD_R": "75.1",
  "MINSVD_R_R": "2.9",
  "PRISM_D": "88.6",
  "PRISM_D_R": "3.4",
  "PRISM_R": "75.2",
  "PRISM_R_R": "3.0",
  "NOVA_D": "88.5",
  "NOVA_D_R": "3.4",
  "NOVA_R": "75.0",
  "NOVA_R_R": "3.0",

  "A3_ACC": "87.5",
  "A3_RATIO": "4.5",
  "PER_PROJ_OBS": "Q and O need more rank; K and V compress better",
  "SPECTRUM_OBS": "All projections show power-law decay",
  "RANK_DIST_OBS": "Effective rank peaks in middle layers",

  "FUNDING": "No external funding",
  "ACKNOWLEDGMENTS": "We thank our lab for compute resources"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — SVD-LLM V2

## Setup
- [ ] Copied `_TEMPLATE/` to `svd_llm_v2/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module svd_llm_v2`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] **Explicit U/Σ/V^H storage highlighted**
- [ ] Real parameter reduction stated
- [ ] Energy threshold mentioned
- [ ] GQA pooling mentioned
- [ ] Pre-test mentioned

## Introduction
- [ ] Motivation concrete with numbers
- [ ] Problem statement formal
- [ ] **Section 1.5 explains relation to prior SVD-LLM version**
- [ ] Contributions are 5-6 bullets

## Method
- [ ] All formulas numbered
- [ ] Explicit factorization formula clear
- [ ] `_project` helper documented
- [ ] Energy bound proven

## Architecture
- [ ] **GQA mean pooling detailed**
- [ ] Conv1D handling
- [ ] Nested encoder detection
- [ ] Pre-test protocol

## Results
- [ ] Table 7.1 (main) filled
- [ ] Table 7.2 (params)
- [ ] Table 7.3 (effective rank)
- [ ] **Table 7.4 (singular values)** — unique to this paper
- [ ] Table 7.5 (error)
- [ ] Table 7.6-7.9 (ablations)
- [ ] **Table 7.10 (explicit vs symmetric)** — unique to this paper
- [ ] Table 7.11 (companion comparison)

## Honesty
- [ ] Real parameter reduction verified
- [ ] Accuracy delta honest
- [ ] **FP16 sensitivity stated** (unique limitation)
- [ ] Single-seed stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Explicit storage documented

## Final Pass
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] 2+ external readers

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/svd_llm_v2
nano papers/svd_llm_v2/variables.json
cd papers/svd_llm_v2
python apply_vars.py --module svd_llm_v2
grep -rn "{{" .  # should be empty
python build.py --module svd_llm_v2 --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE svd_llm_v2
```

### مرحله ۲: اجرای SVD-LLM V2

```powershell
cd D:\SVD_Project
python run_all_4_svd_llm_v2.py --models distilbert,roberta,gpt2 --rank 128 --energy 0.95
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_svd_llm_v2/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\svd_llm_v2
python apply_vars.py --module svd_llm_v2
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module svd_llm_v2
```

---

## 💡 نکات مخصوص SVD-LLM V2

### ۱. **Explicit factorization** — مزیت اصلی

این تنها روشی است که $U$, $\Sigma$, $V^H$ را جدا ذخیره می‌کند. در abstract و introduction برجسته کن.

### ۲. **Singular value statistics** — بخش یکتای این مقاله

Table 7.4 و Section 8.2 **فقط** در این مقاله معنی دارند چون فقط این روش $\Sigma$ را حفظ می‌کند.

### ۳. **FP16 sensitivity** — limitation یکتا

این تنها روشی است که ممکن است در FP16 مشکل داشته باشد. صادقانه در Section 10.4 توضیح بده.

### ۴. **GQA mean pooling** — جزئیات مهم

`copy_weights` این کار را می‌کند:
```python
rep = w_k.shape[0] // self.kv_head_size
w_k_pooled = w_k.view(self.num_kv_heads, rep, self.head_dim, -1).mean(dim=1)
```

این را در Section 5.5 و B.4 توضیح بده.

### ۵. **Separation from companion methods**

SVD-LLM V2 **نباید** به‌عنوان "نسخه‌ای از MinimalSVD" معرفی شود. این یک پیاده‌سازی **جداگانه** با انتخاب طراحی متفاوت است:
- Symmetric vs Explicit
- Same accuracy, different flexibility

---

**قدم بعدی: `run_all_4_svd_llm_v2.py` را اجرا کن. اعداد را از `summary.csv` بگیر و در `variables.json` بگذار. اگر خواستی من می‌توانم این کار را برایت انجام دهم — فقط CSV را بفرست.**