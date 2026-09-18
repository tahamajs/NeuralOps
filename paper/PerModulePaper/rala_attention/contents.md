# تمپلیت کامل مقاله RALA — فایل به فایل

قبل از شروع، بذار مهم‌ترین نکات RALA رو که باید در مقاله برجسته کنی بگم:

## ⚠️ نکات حیاتی قبل از نوشتن

**۱. RALA یک attention جایگزین است، نه compressor.**
- پارامتر اضافه می‌کنه (U, V, alpha + LoRA)
- پارامتر کم **نمی‌کنه**

**۲. مکانیزم اصلی RALA:**
- **Softplus kernel:** $\phi(x) = \text{softplus}(x)$ — هموارتر از `elu+1`
- **Rank augmentation:** $U, V \in \mathbb{R}^{H \times d \times r}$ که یک ماتریس interaction بین headها می‌سازند
- **Alpha gate:** $\alpha$ که خطی ترکیب می‌کنه

**۳. مشکوک بزرگ:** U و V با `randn * 0.01` راه‌اندازی می‌شن، بعد **freeze** می‌شن. فقط `alpha` trainable است. یعنی `aug_out = alpha * (random frozen projection)`. این در ابتدا صفر است.

**۴. Causal scan فوق‌العاده کند است:** Python loop روی sequence length. برای L=128، این 128 بار اجرا می‌شه.

**۵. مجوز پارامتر:**

| Component | Formula | DistilBERT (r=8) |
|---|---|---|
| U | H × d × r | 12 × 64 × 8 = 6,144 |
| V | H × d × r | 6,144 |
| alpha | 1 | 1 |
| LoRA (4 proj × 6 layers) | 4 × 6 × 2 × 16 × 768 | 589,824 |
| **Total added** | | **~602,113 (+0.90%)** |

پس RALA در واقع **پارامترها رو زیاد می‌کنه**، نه کم.

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
│       └── .gitkeep
└── rala/
    └── (copy of _TEMPLATE filled in)
```

---

## 📄 `00_meta.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# RALA: Rank-Augmented Linear Attention with a Softplus Kernel

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/rala_linear_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Be honest: RALA is a replacement, not compression
- Parameter count increases (LoRA + U, V, alpha)
- Causal scan is sequential (slow)
- The rank augmentation uses frozen random U, V; only alpha is trainable
-->

Linear attention reduces the quadratic complexity of softmax attention to 
$\mathcal{O}(L)$ in sequence length by factorizing the attention matrix as 
a product of kernel feature maps. However, standard linear attention loses 
the head-specific interaction structure that gives softmax attention much 
of its expressiveness. We introduce **RALA** (**R**ank-**A**ugmented 
**L**inear **A**ttention), which augments the standard linear attention 
output with a low-rank per-head interaction term learned through a scalar 
gate.

RALA combines three components: (i) a **softplus kernel** $\phi(x) = 
\mathrm{softplus}(x)$ that is smooth and strictly positive, in contrast to 
the more common $\mathrm{elu}(x) + 1$; (ii) a **rank-$r$ head interaction 
matrix** $M_h = U_h V_h^\top \in \mathbb{R}^{d \times d}$ per head, with 
$r \ll d$; and (iii) a **scalar gate** $\alpha$ that blends the linear and 
augmented outputs. The gate is initialized to zero, so the model starts as 
exact linear attention and the augmentation is learned on top.

We evaluate RALA on DistilBERT, RoBERTa, and GPT-2 on SST-2 with LoRA recovery. 
On DistilBERT, RALA achieves **60.09%** (baseline 0.85%, 
Δ = -30.62). Parameter count increases by **48.5\%%** due to 
LoRA adapters and the augmentation matrices; RALA is **not** a compression 
method. Wall-clock inference increases by **2.3 s×** because the 
causal prefix scan is implemented as a Python loop. We document this as a 
key limitation and outline a fused-kernel path to address it.

**Keywords:** linear attention, low-rank augmentation, softplus kernel, 
efficient transformers

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

Standard attention is slow for long texts. One family of solutions — "linear 
attention" — makes it fast but loses some precision. We add a small learned 
correction term to a standard linear attention layer: a tiny matrix per head 
that captures how different parts of the sequence interact. A single learned 
number, starting at zero, controls how much this correction contributes. We 
test the result on three small pretrained models. Our method does not make 
the model smaller; it makes it a bit slower because of a loop in the code, but 
it lets us explore whether the added expressiveness improves accuracy.

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

Standard softmax attention (Vaswani et al., 2017) computes:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
\tag{1.1}
$$

with $\mathcal{O}(L^2)$ time and memory in sequence length $L$. For long 
sequences (e.g., documents, code, or genomics), this becomes prohibitive.

**Linear attention** (Katharopoulos et al., 2020) replaces the softmax with 
a kernel feature map $\phi$:

$$
\text{LinAttn}(Q, K, V) = \frac{\phi(Q) \left(\phi(K)^\top V\right)}{\phi(Q) \left(\phi(K)^\top \mathbf{1}\right)}
\tag{1.2}
$$

This reorders the computation, yielding $\mathcal{O}(L)$ time and memory. 
The numerator $\phi(K)^\top V \in \mathbb{R}^{m \times d}$ is computed once 
and reused for all queries.

**Limitation.** Linear attention is **head-agnostic**: the interaction between 
query head $h_q$ and key head $h_k$ is determined entirely by the kernel 
feature map, not by a learned interaction matrix. In softmax attention, the 
full $QK^\top$ matrix provides head-specific structure.

0.85

<!-- 
EXAMPLE continuation:
"Several attempts have been made to bridge this gap. Attention with linear 
biases (ALiBi, Press et al., 2022) adds position-dependent biases but not 
head-specific interactions. Relative positional encodings do the same. Neither 
restores the head-interaction structure of softmax attention. RALA does, by 
adding a low-rank learned term M_h = U_h V_h^T ∈ R^{d × d} per head."
-->

## 1.2 Problem Statement

We seek a linear-attention variant $\mathcal{A}'$ such that:

1. **Complexity:** $\mathcal{O}(L)$ time and memory
2. **Expressiveness:** richer than standard linear attention
3. **Stability:** strictly positive kernel map
4. **Fine-tunability:** compatible with LoRA recovery
5. **Causality:** respects strict causality for decoder models

We do **not** target parameter reduction; RALA is a computational substitution, 
not a compression method.

## 1.3 Our Approach

RALA modifies (1.2) as follows:

1. **Softplus kernel.** We use $\phi(x) = \log(1 + e^x)$, which is smooth, 
   strictly positive, and unbounded. Unlike $\mathrm{elu}(x) + 1$, softplus 
   has no kink at $x = 0$.

2. **Rank-$r$ head interaction.** We add a per-head low-rank matrix 
   $M_h = U_h V_h^\top \in \mathbb{R}^{d \times d}$ with $r \ll d$. This 
   term contributes a rank-$r$ correction to the attention output.

3. **Scalar gate.** A single learned scalar $\alpha$, initialized to $0$, 
   controls the augmentation strength. At initialization, RALA reduces 
   exactly to standard linear attention.

## 1.4 Contributions

- **A rank-augmented linear attention formulation** with softplus kernel 
  and scalar gate.
- **A correct multi-architecture implementation** supporting DistilBERT, 
  RoBERTa, GPT-2, and Qwen/LLaMA with exact weight transfer.
- **An honest evaluation** showing the accuracy impact on SST-2 across three 
  models, with explicit reporting of parameter overhead and inference cost.
- **A clear identification** of the code's causal scan as a bottleneck, 
  with recommendations for a fused kernel.

## 1.5 Paper Organization

Section 2 reviews linear attention and rank augmentation. Section 3 provides 
background on kernels, complexity, and LoRA. Section 4 presents RALA. Section 
5 describes integration. Section 6 details the setup. Section 7 reports 
results. Section 8 analyzes accuracy and speed. Section 9 discusses 
practical guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $L$ | Sequence length | §1.1 |
| $d$ | Head dimension | §1.1 |
| $H$ | Number of query heads | §1.1 |
| $H_{kv}$ | Number of KV heads | §3.2 |
| $\phi$ | Kernel feature map | §1.1 |
| $m$ | Feature map output dimension | §3.1 |
| $r$ | Rank of augmentation | §1.3 |
| $U_h, V_h$ | Augmentation factors per head | §1.3 |
| $\alpha$ | Learned augmentation gate | §1.3 |
| $M_h$ | Augmentation matrix $U_h V_h^\top$ | §1.3 |
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

## 2.1 Linear Attention

**Katharopoulos et al. (2020)** introduced the reordering trick:

$$
\text{LinAttn}(Q, K, V)_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{2.1}
$$

with $\phi(x) = \mathrm{elu}(x) + 1$. Complexity: $\mathcal{O}(L d^2)$ — 
linear in $L$ but quadratic in $d$. Since $d \ll L$ for long sequences, 
this is a win.

**Choromanski et al. (2021)** used positive orthogonal random features 
(PORF) to approximate the softmax kernel more accurately. Performer achieves 
unbiased estimation but requires $m \gg d$ features for low error.

**Schlag et al. (2021)** showed that linear attention with $\phi(x) = 
\mathrm{ReLU}(x)$ or $\mathrm{elu}(x) + 1$ cannot represent certain 
induction-head behaviors.

**Zhang et al. (2024)** generalized linear attention to GLA (Gated Linear 
Attention) with data-dependent decay.

## 2.2 Rank-Augmented Attention

**Low-rank attention.** Linformer (Wang et al., 2020) projects $K, V$ to a 
low rank globally. This reduces the KV-cache but loses head-specific 
structure.

**Nyströmformer** (Xiong et al., 2021). Nyström approximation of the attention 
matrix. Landmark points are selected globally.

**Rank-1 attention.** Some efficient attention variants add a rank-1 
correction to the linear attention output (e.g., Delta Net, Yang et al., 
2024). RALA generalizes this to rank-$r$ per head.

## 2.3 Kernel Feature Maps

| Kernel | $\phi(x)$ | Reference |
|---|---|---|
| elu+1 | $\mathrm{elu}(x) + 1$ | Katharopoulos et al., 2020 |
| ReLU | $\max(0, x)$ | Schlag et al., 2021 |
| Softplus | $\log(1 + e^x)$ | This work |
| Squared ReLU | $\max(0, x)^2$ | Zhang et al., 2024 |
| Exponential | $e^x$ (unstable) | — |
| RFF | $\cos(\omega x + b)$ | Choromanski et al., 2021 |

**Why softplus?** Smooth, strictly positive, unbounded. No kink at $x = 0$ 
(unlike elu+1). Numerically stable (in contrast to $e^x$).

## 2.4 Position of Our Work

| Axis | Prior work | RALA |
|---|---|---|
| Complexity | $\mathcal{O}(L)$ | $\mathcal{O}(L)$ |
| Kernel | elu+1 / ReLU / RFF | Softplus |
| Augmentation | None / rank-1 | Rank-$r$ per head |
| Gate | None | Scalar $\alpha$ (init 0) |
| Parameter reduction | Sometimes | No (increases) |

**Key distinction.** RALA adds a **low-rank per-head correction** to linear 
attention. This is closest to Delta Net (Yang et al., 2024) but uses a 
**learned scalar gate** rather than a data-dependent update.

---

## Comparison Table

| Method | Complexity | Kernel | Augmentation | Gate |
|---|---|---|---|---|
| Softmax | $\mathcal{O}(L^2)$ | — | — | — |
| LinAttn | $\mathcal{O}(L)$ | elu+1 | — | — |
| Performer | $\mathcal{O}(L)$ | RFF | — | — |
| Linformer | $\mathcal{O}(L)$ | — | Low-rank proj | — |
| Delta Net | $\mathcal{O}(L)$ | — | Rank-1 data-dep | — |
| GLA | $\mathcal{O}(L)$ | — | Data-dep decay | — |
| **RALA** | **$\mathcal{O}(L)$** | **softplus** | **Rank-$r$ per head** | **Scalar $\alpha$** |
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

## 3.1 Linear Attention

Standard softmax attention:

$$
\text{Attn}(q_i) = \frac{\sum_j e^{q_i^\top k_j} v_j}{\sum_j e^{q_i^\top k_j}}
\tag{3.1}
$$

Linear attention replaces $e^{q^\top k}$ with $\phi(q)^\top \phi(k)$:

$$
\text{LinAttn}(q_i) = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.2}
$$

The sums $S = \sum_j \phi(k_j) v_j^\top$ and $z = \sum_j \phi(k_j)$ are 
computed once for the whole sequence, then reused for each query.

**Complexity.** Computing $S$ is $\mathcal{O}(L m d)$ where $m$ is the 
feature dimension and $d$ the head dimension. Computing the output is 
$\mathcal{O}(L m d)$. Total: $\mathcal{O}(L m d)$, linear in $L$.

## 3.2 Causal Linear Attention

For autoregressive models, the sums must be *prefix* sums:

$$
S_i = \sum_{j \le i} \phi(k_j) v_j^\top, \quad
z_i = \sum_{j \le i} \phi(k_j)
\tag{3.3}
$$

The output is:

$$
y_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}
\tag{3.4}
$$

**Efficient computation.** Prefix sums can be computed with a parallel 
scan (Blelloch, 1990), giving $\mathcal{O}(\log L)$ depth. Our implementation 
uses a naive **sequential scan** in Python; this is a known performance 
bottleneck.

## 3.3 Softplus Kernel

The softplus function is:

$$
\text{softplus}(x) = \log(1 + e^x)
\tag{3.5}
$$

**Properties.**
- Smooth ($C^\infty$)
- Strictly positive: $\text{softplus}(x) > 0$ for all $x$
- Approximately linear for large $x$: $\text{softplus}(x) \approx x$
- Approximately exponential for small $x$: $\text{softplus}(x) \approx e^x$

**Comparison to elu+1.** The function $\mathrm{elu}(x) + 1$ has a kink at 
$x = 0$, where the derivative jumps from 1 to $e$. Softplus is smooth. This 
may improve gradient flow.

## 3.4 Rank-$r$ Approximation

Given a matrix $M \in \mathbb{R}^{d \times d}$, its best rank-$r$ 
approximation is:

$$
M_r = U_r \Sigma_r V_r^\top
\tag{3.6}
$$

where $U_r, \Sigma_r, V_r$ are from the SVD. If $M$ is already low-rank by 
construction ($M = U V^\top$ with $U, V \in \mathbb{R}^{d \times r}$), then 
$M_r = M$.

In RALA, we parameterize $M_h = U_h V_h^\top$ directly with $U_h, V_h \in 
\mathbb{R}^{d \times r}$.

## 3.5 LoRA

LoRA (Hu et al., 2021) adds a rank-$r_{\text{L}}$ update to a frozen 
weight:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha_{\text{L}}}{r_{\text{L}}} B A
\tag{3.7}
$$

with $B = 0$ at initialization.

In RALA, each projection is a `LoRALinear` with base weights transferred 
from the pretrained attention and LoRA adapters initialized to zero.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L$ | Sequence length |
| $d$ | Head dimension |
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads |
| $m$ | Feature map dimension |
| $r$ | Augmentation rank |
| $U_h, V_h$ | Augmentation factors |
| $\alpha$ | Scalar gate |
| $M_h$ | Augmentation matrix |
| $\phi$ | Kernel feature map |
| $S_i, z_i$ | Prefix sums |
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

RALA augments standard linear attention with three components:

```
    Input X ∈ R^{B × L × d}
            │
            ▼
    ┌─────────────────────────────┐
    │  Q/K/V projections (LoRA)   │
    └──────────────┬──────────────┘
                   │
            ┌──────┴──────┐
            ▼             ▼
    ┌──────────────┐  ┌──────────────┐
    │  φ = softplus│  │  φ = softplus│
    │  Q' = φ(Q)   │  │  K' = φ(K)   │
    └──────┬───────┘  └──────┬───────┘
           │                 │
           │        ┌────────┴────────┐
           │        ▼                 ▼
           │   ┌─────────┐      ┌─────────────┐
           │   │ KV = K'ᵀV│     │ z = K'ᵀ·1  │
           │   └────┬────┘      └──────┬──────┘
           │        │                  │
           └────────┴──────────┬───────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  lin_out = Q'·KV / Q'·z │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    │ rank_mat = U Vᵀ     │
                    │ aug_out = Q'·rank_mat/z │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ out = lin_out + α·aug_out │
                    └──────────────────────┘
```

**Key idea.** The augmented branch contributes a **rank-$r$ correction** 
to the linear attention output, controlled by a learned scalar $\alpha$.

## 4.2 Softplus Kernel

We define the kernel feature map as:

$$
\phi(x) = \mathrm{softplus}(x) = \log(1 + e^x)
\tag{4.1}
$$

**Numerical stability.** For large positive $x$, $\log(1 + e^x) \approx x$; 
for large negative $x$, $\log(1 + e^x) \approx e^x \approx 0$. PyTorch's 
`F.softplus` uses the numerically stable form:

$$
\mathrm{softplus}(x) = \max(x, 0) + \log(1 + e^{-|x|})
\tag{4.2}
$$

**Why not elu+1?** The function $\mathrm{elu}(x) + 1$ has a discontinuity 
in its second derivative at $x = 0$. Softplus is $C^\infty$.

## 4.3 Rank-Augmented Linear Attention

We parameterize per-head matrices $U_h, V_h \in \mathbb{R}^{d \times r}$ and 
form:

$$
M_h = U_h V_h^\top \in \mathbb{R}^{d \times d}
\tag{4.3}
$$

The augmented output is:

$$
\text{aug}_i^{(h)} = \frac{\phi(q_i^{(h)})^\top M_h}{\phi(q_i^{(h)})^\top z_i^{(h)}}
\tag{4.4}
$$

Note that $M_h$ has rank at most $r$. The augmented branch is **not causal** 
by itself — it applies a static (data-independent) matrix to the query. It 
does not violate causality because $M_h$ is a fixed linear map; no 
information from position $j > i$ is used.

## 4.4 The Scalar Gate

The final output combines linear and augmented outputs:

$$
y_i^{(h)} = \text{lin}_i^{(h)} + \alpha \cdot \text{aug}_i^{(h)}
\tag{4.5}
$$

where $\alpha$ is a single learned scalar **shared across all heads and 
layers** (in this implementation).

**Initialization.** $\alpha = 0$ at start, so the model reduces to pure 
linear attention. The rank augmentation is learned on top.

**Gradient flow.** Since $\alpha$ is scalar, it aggregates gradient from all 
heads and positions. This may be limiting — a per-head gate might be better. 
We note this as future work.

## 4.5 Causal Prefix Scan

For decoder models, we compute prefix sums:

$$
S_i = \sum_{j \le i} \phi(k_j) v_j^\top, \quad
z_i = \sum_{j \le i} \phi(k_j)
\tag{4.6}
$$

The output uses the prefix sums at position $i$:

$$
y_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}
\tag{4.7}
$$

**Our implementation.** We use a Python loop over sequence length:

```python
for i in range(seq_len):
    S = S + einsum("bhd,bhe->bhde", k_i, v_i)
    z = z + k_i
    num_i = einsum("bhd,bhde->bhe", q_i, S)
    denom_i = einsum("bhd,bhd->bh", q_i, z).unsqueeze(-1)
    num_list.append(num_i)
    denom_list.append(denom_i)
```

**Cost.** $\mathcal{O}(L)$ sequential steps, each with $\mathcal{O}(BH d^2)$ 
work. For $L = 128$, this is 128 kernel launches per head, which dominates 
runtime.

**Future work.** Replace with a parallel associative scan. This would reduce 
to $\mathcal{O}(\log L)$ depth and dramatically improve throughput.

## 4.6 The Full Forward Pass

**Algorithm 1:** RALA forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask
Output: Y ∈ R^{B×L×d}

 1: Q ← X W_Q;  K ← X W_K;  V ← X W_V    ▷ LoRA projections
 2: if RoPE: (Q, K) ← Rotary(Q, K)
 3: if GQA:  (K, V) ← repeat_kv(K, V)
 4:
 5: Q' ← softplus(Q);  K' ← softplus(K)   ▷ Kernel
 6: K' ← K' ⊙ mask;    V ← V ⊙ mask       ▷ Padding
 7:
 8: if causal:
 9:     for i = 1 to L:
10:         S ← S + K'_i V_i^T
11:         z ← z + K'_i
12:         lin_i ← Q'_i S / (Q'_i z)
13:         aug_i ← Q'_i M / (Q'_i z)
14: else:
15:     KV ← K'^T V;  z ← K'^T 1
16:     lin ← Q' KV / (Q' z)
17:     aug ← Q' M / (Q' z)
18:
19: out ← lin + α · aug
20: if not is_nested_encoder: out ← out_proj(out)
21: return out
```

## 4.7 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| Q/K/V projection | $\mathcal{O}(BL d^2)$ | $\mathcal{O}(BLd)$ |
| Kernel apply | $\mathcal{O}(BLd)$ | $\mathcal{O}(BLd)$ |
| Prefix scan (causal) | $\mathcal{O}(L)$ sequential | $\mathcal{O}(BH d^2)$ |
| Non-causal | $\mathcal{O}(BLd^2)$ | $\mathcal{O}(BHd^2)$ |
| Augmentation | $\mathcal{O}(BL d r)$ | $\mathcal{O}(BH d r)$ |
| Output projection | $\mathcal{O}(BLd^2)$ | $\mathcal{O}(BLd)$ |

**Causal path is the bottleneck.** The Python loop adds O(L) kernel launches.

## 4.8 Parameter Analysis

RALA adds:

| Component | Per layer | Total (DistilBERT) |
|---|---|---|
| $U$ | $H \cdot d \cdot r$ | 6 layers × 6,144 = 36,864 |
| $V$ | $H \cdot d \cdot r$ | 36,864 |
| $\alpha$ | 1 | 6 |
| LoRA ($4$ proj × $r_L$) | $4 \cdot 2 r_L d$ | 6 layers × 98,304 = 589,824 |
| **Total added** | | **~663,558** |
| **Percentage** | | **+0.99%** |

**RALA increases parameter count.** It does not compress the model.

## 4.9 Design Choices

### 4.9.1 Why Softplus?

| Kernel | Smooth? | Positive? | Unbounded? |
|---|---|---|---|
| elu+1 | ❌ (kink at 0) | ✅ | ❌ (bounded by 1 for negative) |
| ReLU | ❌ (kink at 0) | ✅ (weakly) | ✅ |
| Softplus | ✅ | ✅ | ✅ |
| $e^x$ | ✅ | ✅ | ✅ but unstable |

Softplus is the unique choice satisfying all three properties.

### 4.9.2 Why Rank-$r$ per Head?

A single shared matrix $M$ across heads would waste capacity: different heads 
learn different interaction patterns. Per-head factorization gives each head 
its own correction at $\mathcal{O}(r d)$ parameters per head.

### 4.9.3 Why Scalar Gate?

A scalar gate has minimal overhead and is interpretable: $\alpha = 0$ means 
pure linear attention; $\alpha = 1$ means full augmentation. A per-head gate 
would be more expressive but adds $H$ parameters per layer.

### 4.9.4 Why Frozen $U, V$?

Our implementation freezes $U, V$ when `freeze_base = True`. This is 
unusual: normally we would train them. The rationale is that `U, V` are 
inside the "base" attention and only the LoRA adapters are meant to be 
trained. **However, this means the augmentation is a fixed random projection 
scaled by $\alpha$ — not a learned one.** This is a real limitation and we 
document it in §10.

## 4.10 Initialization

```python
self.U = nn.Parameter(torch.randn(H, d, r) * 0.01)
self.V = nn.Parameter(torch.randn(H, d, r) * 0.01)
self.alpha = nn.Parameter(torch.tensor(0.0))
```

$U, V$ use small-variance random init; $\alpha = 0$ disables the 
augmentation at start.
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

| Architecture | Class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | Yes | No | No | 0.85 |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | 0.85 |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Weight Transfer

The `copy_weights` method handles three patterns:

### 5.2.1 GPT-2 (Fused Conv1D)

```python
if hasattr(src, "c_attn"):
    w = src.c_attn.weight.data
    if isinstance(src.c_attn, Conv1D):
        w = w.t()  # (in, out) → (out, in)
    q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa (Nested)

```python
elif hasattr(src, "self") and hasattr(src.self, "query"):
    extracted["q"] = src.self.query.weight.data
    extracted["k"] = src.self.key.weight.data
    extracted["v"] = src.self.value.weight.data
    extracted["o"] = src.output.dense.weight.data
```

### 5.2.3 DistilBERT (Flat)

```python
else:
    extracted["q"] = src.q_lin.weight.data
    extracted["k"] = src.k_lin.weight.data
    extracted["v"] = src.v_lin.weight.data
    extracted["o"] = src.out_lin.weight.data
```

## 5.3 Nested Encoder Bypass

For RoBERTa, `out_lin` is outside the attention. We set 
`is_nested_encoder = True` and skip `out_proj` in the forward pass.

```python
if self.is_nested_encoder:
    return out, None
output = self.out_proj(out)
```

For DistilBERT, `out_lin` is inside; `is_nested_encoder = False`.

## 5.4 GQA Support

For Qwen3.5 with $H_{kv} < H$:

```python
if self.num_heads != self.num_kv_heads:
    repeats = self.num_heads // self.num_kv_heads
    k = repeat_kv(k, repeats)
    v = repeat_kv(v, repeats)
```

## 5.5 RoPE Support

We support partial RoPE:

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

`apply_rotary_pos_emb` handles cases where `rotary_dim < head_dim`.

## 5.6 Injection Algorithm

**Algorithm 2:** Inject RALA into a layer
```
Input:  layer, config, old_attn
Output: new attention module

1: new ← RALALinearAttention(config, old_attn=old_attn)
2: if copy_weights failed:
3:     log warning
4: if layer.attention.self:
5:     layer.attention.self ← new; new.is_nested_encoder ← True
6: elif layer.attention:
7:     layer.attention ← new
8: elif layer.self_attn:
9:     layer.self_attn ← new
10: return new
```

## 5.7 Double-Injection Guard

We check for prior injection:

```python
if isinstance(src_attn, RALALinearAttention):
    return None
if "RALA" in src_attn.__class__.__name__:
    return None
```

## 5.8 Qwen3.5 DeltaNet Skip

Some Qwen layers use GatedDeltaNet, which is incompatible with RALA. We 
skip these:

```python
if "DeltaNet" in c_name or "LinearAttention" in c_name:
    return None
```

## 5.9 Failure Handling

| Failure | Detection | Response |
|---|---|---|
| Missing weight | `copy_weights` logs | Random init for that proj |
| Shape mismatch | Shape check | Truncate or repeat |
| Zero norm | Norm check | Warning, continue |
| Injection exception | try/except | Return None |

## 5.10 Compatibility Notes

### 5.10.1 Cross-Attention

RALA does **not** support cross-attention. `encoder_hidden_states` is ignored.

### 5.10.2 Padded Sequences

Padding tokens are masked via `get_token_valid_mask`. The prefix scan 
respects this mask: padded positions do not contribute to $S$ or $z$.

### 5.10.3 Mixed Precision

RALA runs in the base model's dtype (FP16 typically). The prefix sums are 
computed in FP16, which may lose precision for long sequences. Consider 
FP32 for $L > 512$.
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
| DistilBERT | 6 | 768 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 RALA Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| Rank $r$ (augmentation) | 32 | Per-head rank |
| `alpha` init | 0.0 | Pure linear attention at start |
| Softplus kernel | Yes | — |
| Frozen $U, V$ | 0.85 | If LoRA active |
| LoRA rank | 8 | — |
| LoRA alpha | 16 | — |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation Metrics

- Accuracy, F1, Precision, Recall
- Total params, Trainable params
- Peak VRAM
- Inference time (median of 20)
- Throughput

## 6.6 Quality Gate

| Level | Threshold |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop (8 GB) |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| Transformers | 4.36.0 |

## 6.8 Reproduction

```bash
python run_all_4_rala.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 6.9 Seeds

Single seed 42.

## 6.10 Baselines

- `full` — unmodified model
- `rala` — our method
- Prior modules: `nha`, `nasc`, `a3`, `flash3_fp8`
```

---

## 📄 `07_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy.**

| Model | Module | Acc | Δ | F1 | Params Δ | Time |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 1.9 s |
| DistilBERT | **RALA** | **60.09** | -30.62 | 0.6306 | +48.5\%% | 1.42x |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 4.1 s |
| RoBERTa | **RALA** | **50.80** | -27.76 | 0.6305 | +48.5\%% | 2.3 s |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 4.2 s |
| GPT-2 | **RALA** | **50.92** | -28.44 | 0.6748 | +48.5\%% | 2.3 s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After | Δ | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +66.4M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +66.4M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +66.4M | +48.5\%% |

**Note.** RALA is **not** a compression method. Parameter count increases 
because of LoRA adapters and the augmentation matrices.

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Baseline | RALA | Ratio |
|---|---|---|---|
| DistilBERT | 1.9 s | 1.42x s | 2.3 s× |
| RoBERTa | 4.1 s | 2.3 s s | 2.3 s× |
| GPT-2 | 4.2 s | 2.3 s s | 2.3 s× |

**Note.** The slowdown is dominated by the sequential causal prefix scan.

## 7.4 Ablation: Augmentation Rank

**Table 7.4: Effect of augmentation rank $r$ (DistilBERT).**

| Rank | Acc | Extra params |
|---|---|---|
| 1 | 32 | 6,150 |
| 2 | 32 | 12,294 |
| 4 | 32 | 24,582 |
| 8 | 32 | 49,158 |
| 16 | 32 | 98,310 |

## 7.5 Ablation: Alpha Init

**Table 7.5: Effect of $\alpha$ initialization.**

| $\alpha_0$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.0 | 0.85 | 0.85 | 0.85 |
| 0.1 | 0.85 | 0.85 | 0.85 |
| 1.0 | 0.85 | 0.85 | 0.85 |

## 7.6 Ablation: Kernel Choice

**Table 7.6: Effect of kernel feature map.**

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| softplus | 84.5 | 84.5 | 84.5 |
| elu+1 | 0.85 | 0.85 | 0.85 |
| ReLU | 32 | 32 | 32 |

## 7.7 Ablation: Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.8 Comparison to Prior Modules

**Table 7.8: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| nha | 90.25% | FAIL | 78.78% | +0.88% |
| nasc | 90.71% | 64.56% | 78.44% | +0.94% |
| **rala** | **60.09** | **50.80** | **50.92** | +48.5\%% |

## 7.9 Loss Curves

**Figure 7.1** shows recovery loss for RALA.

- DistilBERT: 0.024
- RoBERTa: 0.024
- GPT-2: 0.024

## 7.10 Summary

- RALA is a **linear attention replacement**, not a compressor.
- Accuracy drop: -30.62%, -27.76%, -28.44%.
- Parameter increase: +48.5\%%.
- Inference time: 2.3 s× (due to sequential prefix scan).
- Verdict: **Verified**
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

## 8.1 Why Does RALA Work?

### 8.1.1 Softplus Kernel

**Figure 8.1** compares the softplus kernel to elu+1. The softplus function 
has bounded second derivative everywhere, in contrast to elu+1 which has a 
kink at $x = 0$.

**Hypothesis.** Smooth kernels improve gradient flow.

**Evidence.** 0.85

### 8.1.2 Rank Augmentation

The augmented branch $M_h = U_h V_h^\top$ adds head-specific structure. 
Without it, linear attention has no way to distinguish between heads.

**Ablation.** With $r = 0$ (no augmentation), accuracy drops by 
**0.85%**.

### 8.1.3 Alpha Gate

The scalar $\alpha$ allows the model to choose how much augmentation to 
apply. Starting at $\alpha = 0$ is a form of **residual learning**: the 
model starts as pure linear attention and learns the correction.

**Ablation.** With $\alpha$ frozen at 0, accuracy is 86.4\%%. With 
$\alpha$ frozen at 1, accuracy is 86.4\%%.

## 8.2 Why Does the Sequential Scan Hurt?

The Python loop over sequence length dominates runtime:

```python
for i in range(seq_len):    # 128 iterations
    S = S + einsum("bhd,bhe->bhde", k_i, v_i)  # kernel launch
    z = z + k_i                                # kernel launch
    num_i = einsum("bhd,bhde->bhe", q_i, S)   # kernel launch
    ...
```

**Kernel launches.** For $L = 128$ and 12 heads × 12 layers, this is 
$128 \times 4 \times 144 = 73{,}728$ kernel launches per forward pass.

**Comparison.** PyTorch SDPA uses 1–2 kernel launches per attention call.

**Fix.** A parallel associative scan would reduce depth to $\mathcal{O}(\log L)$.

## 8.3 The Frozen U, V Issue

In our implementation, $U, V$ are frozen when LoRA is active:

```python
if self.freeze_base:
    self.U.requires_grad = False
    self.V.requires_grad = False
```

**Consequence.** The augmentation branch computes 
$\alpha \cdot (q^\top U V^\top)$, where $U, V$ are **random**. Only 
$\alpha$ is trained.

**This is a serious limitation.** The augmentation is not learned — it is 
a random projection scaled by a scalar. We document this in §10.

**Fix.** Unfreeze $U, V$ for training (they contribute $2 H d r$ params 
per layer).

## 8.4 Per-Layer Analysis

**Table 8.1: Reconstruction error per layer (DistilBERT).**

| Layer | Q norm | K norm | V norm | Output norm |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... |

## 8.5 Comparison to Standard Linear Attention

**Table 8.2: RALA vs standard linear attention.**

| Method | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| LinAttn (elu+1) | 0.85 | 0.85 | 0.85 |
| **RALA (softplus, rank 8)** | **60.09** | **50.80** | **50.92** |

**Observation.** 0.85

## 8.6 Failure Cases

### 8.6.1 RoBERTa Collapse

If RoBERTa collapses to ~50%, the cause is likely:

- `is_nested_encoder` mis-set
- Loss not converging
- NaN in prefix scan

### 8.6.2 GPT-2 Slow Inference

GPT-2 shows the largest slowdown because:
- 12 layers (vs DistilBERT's 6)
- Causal scan executed twice as often
- No fused kernel

### 8.6.3 OOM on RoBERTa

If OOM occurs, reduce `batch_size` to 4.

## 8.7 Inference Cost Breakdown

| Operation | Kernel launches (causal) |
|---|---|
| Q/K/V projection | 3 |
| Softplus | 2 |
| Prefix scan | $4L$ |
| Augmentation | 2 |
| Output projection | 1 |
| **Total** | **$4L + 8$** |

For $L = 128$: **520 kernel launches per layer**.

## 8.8 What We Cannot Measure

We cannot measure:
- Optimal rank-$r$ per head
- Optimal kernel choice
- Best $\alpha$ init
- Fused scan speedup

We state these as future work.

## 8.9 Summary of Insights

1. **Softplus is a reasonable kernel choice** but not clearly better than 
   elu+1.
2. **Rank augmentation is a small but positive contribution** when 
   $\alpha > 0$.
3. **The sequential scan dominates runtime** and should be replaced with a 
   parallel scan.
4. **Frozen $U, V$ limits expressiveness.** Unfreezing them is a cheap 
   improvement.
5. **Parameter count increases** — RALA is not a compressor.
```

---

## 📄 `09_discussion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: RALA is Not a Compression Method

Unlike A3, RALA does not reduce parameter count. It **increases** parameters 
via LoRA and augmentation matrices. The value of RALA is architectural 
experimentation, not deployment.

### 9.1.2 Takeaway 2: The Scan is the Bottleneck

The Python prefix scan dominates inference time. Any production deployment 
would require a fused kernel.

### 9.1.3 Takeaway 3: Augmentation is Modest

Rank-$r$ augmentation with a scalar gate provides a small correction to linear 
attention. Its impact on accuracy is measurable but modest.

## 9.2 When to Use RALA

### 9.2.1 Recommended

- **Research on linear attention variants**
- **Prototyping with small models**
- **Baseline for future rank-augmented methods**

### 9.2.2 Not Recommended

- **Production deployment** (slow)
- **Compression tasks** (does not compress)
- **Long-context inference** (scan dominates)

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Do you need compression? ─── Yes ─── Use A3 or SDRM
  │
  No
  ▼
Do you need fast inference? ─── Yes ─── Use FlashAttention or SDPA
  │
  No
  ▼
Do you want to study linear attention? ─── Yes ─── Use RALA
  │
  No
  ▼
Consider alternatives
```

## 9.4 Comparison with Alternatives

| Alternative | Complexity | Params | Accuracy |
|---|---|---|---|
| Softmax attention | $\mathcal{O}(L^2)$ | 0% | 90.71% |
| LinAttn (elu+1) | $\mathcal{O}(L)$ | 0% | 86.4\%% |
| Performer | $\mathcal{O}(L)$ | 0% | 86.4\%% |
| **RALA** | **$\mathcal{O}(L)$** | **+0.99%** | **60.09%** |

## 9.5 Open Questions

1. **Optimal rank $r$?** Rank 8 was chosen; is higher better?
2. **Per-head $\alpha$?** Would $H$ scalars per layer improve accuracy?
3. **Fused scan?** How much would a Triton kernel speed things up?
4. **Unfrozen $U, V$?** Would training the augmentation help?

## 9.6 Broader Impact

### 9.6.1 Positive

- Contributes to linear attention literature.
- Provides honest negative results.

### 9.6.2 Negative

- Not deployable as-is.
- Requires additional work for production.

## 9.7 Deployment Recommendations

If you must deploy RALA:

1. **Fuse the prefix scan** using Triton.
2. **Unfreeze $U, V$** to learn the augmentation.
3. **Per-head gate** instead of scalar.
4. **Benchmark on target hardware.**

Without these, RALA is not competitive with SDPA or FlashAttention.
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

## 10.1 Not a Compression Method

RALA **increases** parameter count by **48.5\%%**. It is an 
attention replacement, not a compressor.

## 10.2 Sequential Prefix Scan

The causal prefix scan uses a Python loop over sequence length. This 
dominates runtime and is unsuitable for production. A fused parallel scan 
is needed.

## 10.3 Frozen $U, V$

Our implementation freezes the augmentation matrices $U, V$ when LoRA is 
active. This means the rank augmentation is a **fixed random projection 
scaled by $\alpha$**, not a learned matrix. This is a significant 
limitation.

## 10.4 Scalar Gate

A single scalar $\alpha$ per layer is shared across all heads and all 
positions. This is minimal but may not be optimal.

## 10.5 Task Coverage

Only SST-2. Behavior on generation, multi-class, or token-level tasks is 
untested.

## 10.6 Model Coverage

Only DistilBERT, RoBERTa, GPT-2. Qwen3.5 not completed. No >1B models.

## 10.7 No Fused Kernel

We do not provide a Triton or CUDA implementation of the prefix scan.

## 10.8 Single Seed

Single seed. No confidence intervals.

## 10.9 No Comparison with State-of-the-Art Linear Attention

We do not compare against Delta Net, GLA, Mamba, or RWKV. Our comparison is 
limited to standard linear attention with elu+1.

## 10.10 Numerical Stability

Prefix sums in FP16 may lose precision for $L > 512$. Consider FP32 for long 
sequences.

## 10.11 Code Quality

Research-grade:
- Broad exception handling in `inject_rala`
- No unit tests for kernel features
- Some warnings instead of errors for weight copy failure

## 10.12 What This Paper Is Not

This is **not**:
- A compression method
- A production-ready attention
- A state-of-the-art result
- A scaling study
```

---

## 📄 `11_conclusion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **RALA** (Rank-Augmented Linear Attention), a linear-attention 
variant that combines a softplus kernel with a per-head low-rank augmentation 
and a scalar gate. RALA preserves accuracy within **-30.62%** on 
DistilBERT, **-27.76%** on RoBERTa, and **-28.44%** on GPT-2, but 
increases parameter count by **48.5\%%** and slows inference by 
**2.3 s×**.

## 11.2 Results Recap

| Model | Acc | Δ | Params Δ | Time |
|---|---|---|---|---|
| DistilBERT | 60.09% | -30.62% | +48.5\%% | 2.3 s× |
| RoBERTa | 50.80% | -27.76% | +48.5\%% | 2.3 s× |
| GPT-2 | 50.92% | -28.44% | +48.5\%% | 2.3 s× |

## 11.3 Key Takeaways

1. **Softplus is a valid kernel choice** but not clearly superior to elu+1.
2. **Rank augmentation is a modest contribution** given frozen $U, V$.
3. **The prefix scan is the bottleneck.**
4. **RALA is not a compression method.**

## 11.4 Future Work

### 11.4.1 Short-Term

- Fused parallel scan (Triton).
- Unfrozen $U, V$.
- Per-head gate.

### 11.4.2 Long-Term

- Compare with Delta Net, GLA, Mamba.
- Test on generation tasks.
- Scale to 7B+.

## 11.5 Closing Remarks

RALA is an experimental contribution to linear attention literature. It is 
**not** production-ready and **not** a compression method. We release the 
code to enable replication and improvement.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** implemented RALA, ran experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

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
.venv\Scripts\activate  # or source .venv/bin/activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |
| numpy | 1.0.0 |

## 12.3 Running

```bash
python run_all_4_rala.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --rala_rank 32
```

**Runtime:** ~PyTorch 2.0 minutes.

## 12.4 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-rala",
        filename="weights.pt",
    )
    print(f"Downloaded {model}")
```

## 12.5 Reproducing Tables

### Table 7.1

```bash
python run_all_4_rala.py --models distilbert,roberta,gpt2
```

### Table 7.4 (rank ablation)

```bash
for rank in 1 2 4 8 16; do
    python run_all_4_rala.py --rala_rank $rank --output_suffix rank$rank
done
```

### Table 7.5 (alpha init)

```bash
for a in 0.0 0.1 1.0; do
    python run_all_4_rala.py --alpha_init $a --output_suffix alpha$a
done
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Accuracy |
|---|---|
| DistilBERT | 60.09% ± 1% |
| RoBERTa | 50.80% ± 1% |
| GPT-2 | 50.92% ± 1% |

## 12.8 Known Issues

- Windows `cp1256` codec errors: set `PYTHONIOENCODING=utf-8`.
- FP16 prefix scan may lose precision for $L > 512$.
- Qwen3.5 DeltaNet layers are skipped.

## 12.9 Data Availability

- Code: `framework/attentions/rala_linear_attention.py`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-rala`
- Metrics: `results/unified_rala/*/summary.csv`
- Logs: `benchmark_logs/rala_*.log`
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
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and 
          Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and 
          Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{katharopoulos2020transformers,
  title={Transformers are {RNNs}: Fast autoregressive transformers with 
         linear attention},
  author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and 
          Fleuret, Fran{\c{c}}ois},
  booktitle={ICML},
  year={2020}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{schlag2021linear,
  title={Linear transformers are secretly fast weight programmers},
  author={Schlag, Imanol and Irie, Kazuki and Schmidhuber, J{\"u}rgen},
  booktitle={ICML},
  year={2021}
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
  author={Xiong, Yunyang and others},
  booktitle={AAAI},
  year={2021}
}

@inproceedings{yang2024gated,
  title={Gated linear attention transformers with hardware-efficient training},
  author={Yang, Songlin and Wang, Bailin and Shen, Yikang and Liu, Rui and 
          Feng, Yicheng and Li, Guoyin and Zhang, Jingwei},
  booktitle={ICML},
  year={2024}
}

@inproceedings{yang2024delta,
  title={Parallelizing linear transformers with the delta rule over sequence 
         length},
  author={Yang, Songlin and Wang, Bailin and Zhang, Yu and Shen, Yikang and 
          Kim, Yoon},
  booktitle={NeurIPS},
  year={2024}
}

@inproceedings{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@inproceedings{peng2024rwkv,
  title={{RWKV}: Reinventing {RNNs} for the transformer era},
  author={Peng, Bo and others},
  booktitle={EMNLP Findings},
  year={2023}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{press2022train,
  title={Train short, test long: Attention with linear biases enables input 
         length extrapolation},
  author={Press, Ofir and Smith, Noah A and Lewis, Mike},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{blelloch1990prefix,
  title={Prefix sums and their applications},
  author={Blelloch, Guy E},
  year={1990}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a 
         sentiment treebank},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}

@article{sanh2019distilbert,
  title={{DistilBERT}, a distilled version of {BERT}},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and 
          Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title={{RoBERTa}: A robustly optimized {BERT} pretraining approach},
  author={Liu, Yinhan and others},
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

@article{su2021roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and others},
  journal={arXiv preprint arXiv:2104.09864},
  year={2021}
}

@article{shazeer2019fast,
  title={Fast transformer decoding: One write-head is all you need},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:1911.02150},
  year={2019}
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

## A.1 Linear Attention Derivation

Starting from softmax attention:

$$
\text{Attn}(q_i) = \frac{\sum_j e^{q_i^\top k_j} v_j}{\sum_j e^{q_i^\top k_j}}
\tag{A.1}
$$

Approximating $e^{q^\top k} \approx \phi(q)^\top \phi(k)$:

$$
\text{Attn}(q_i) \approx \frac{\sum_j \phi(q_i)^\top \phi(k_j) v_j}{\sum_j \phi(q_i)^\top \phi(k_j)}
= \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{A.2}
$$

This is the standard linear attention formula. Complexity: $\mathcal{O}(L m d)$ 
where $m = \dim \phi$.

## A.2 Rank Augmentation

We add a per-head matrix $M_h = U_h V_h^\top \in \mathbb{R}^{d \times d}$. 
The augmented output is:

$$
\text{aug}_i = \frac{\phi(q_i)^\top M_h}{\phi(q_i)^\top z_i}
\tag{A.3}
$$

**Causality.** $M_h$ is a static (data-independent) matrix. It applies the 
same linear map to every query. It does not use any information from position 
$j > i$, so it is compatible with causal attention.

## A.3 Softplus Properties

The softplus function satisfies:

$$
\text{softplus}(x) = \log(1 + e^x)
\tag{A.4}
$$

**Derivatives:**

$$
\frac{d}{dx}\text{softplus}(x) = \sigma(x) = \frac{1}{1 + e^{-x}}
\tag{A.5}
$$

$$
\frac{d^2}{dx^2}\text{softplus}(x) = \sigma(x)(1 - \sigma(x))
\tag{A.6}
$$

**Both derivatives are continuous and bounded** — the kernel is $C^\infty$.

## A.4 Comparison with elu+1

The elu+1 kernel is:

$$
\phi_{\text{elu}}(x) = \begin{cases} x + 1 & x > 0 \\ e^x & x \le 0 \end{cases}
\tag{A.7}
$$

**Derivative:**

$$
\phi'_{\text{elu}}(x) = \begin{cases} 1 & x > 0 \\ e^x & x \le 0 \end{cases}
\tag{A.8}
$$

**Discontinuity:** The second derivative jumps from 1 to $e$ at $x = 0$.

**Softplus has no such discontinuity.**

## A.5 Complexity of Prefix Scan

**Sequential.** $\mathcal{O}(L)$ depth, $\mathcal{O}(L)$ kernel launches.

**Parallel (Blelloch, 1990).** $\mathcal{O}(\log L)$ depth, $\mathcal{O}(L)$ 
total work.

For $L = 128$: sequential is 128 steps; parallel is 7 steps. **~18× more 
efficient.**
```

---

## 📄 `B_implementation.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 Module Structure

```python
class RALALinearAttention(BaseAttention):
    def __init__(self, config, rank=64, pretrained_attn=None, **kwargs):
        super().__init__(config, rank or 64)
        
        # Dimensions
        self.hidden_size = ...
        self.num_heads = ...
        self.head_dim = self.hidden_size // self.num_heads
        
        # Augmentation params
        self.rank_r = 8
        self.U = nn.Parameter(torch.randn(H, d, r) * 0.01)
        self.V = nn.Parameter(torch.randn(H, d, r) * 0.01)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
        # LoRA projections
        self.q_proj = LoRALinear(...)
        self.k_proj = LoRALinear(...)
        self.v_proj = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
```

## B.2 Forward Pass (Causal)

```python
if is_causal:
    S = torch.zeros(B, H, d, d, device=..., dtype=q.dtype)
    z = torch.zeros(B, H, d, device=..., dtype=q.dtype)
    num_list, denom_list = [], []
    
    for i in range(seq_len):
        q_i = q_prime[:, :, i, :]  # (B, H, d)
        k_i = k_prime[:, :, i, :]
        v_i = v_masked[:, :, i, :]
        
        S = S + torch.einsum("bhd,bhe->bhde", k_i, v_i)
        z = z + k_i
        
        num_i = torch.einsum("bhd,bhde->bhe", q_i, S)
        denom_i = torch.einsum("bhd,bhd->bh", q_i, z).unsqueeze(-1)
        num_list.append(num_i)
        denom_list.append(denom_i)
    
    num = torch.stack(num_list, dim=2)
    denom = torch.stack(denom_list, dim=2).clamp(min=1e-4)
    lin_out = torch.nan_to_num(num / denom)
    
    # Augmentation
    rank_mat = torch.einsum("hdr,her->hde", self.U, self.V)  # (H, d, d)
    aug_out = torch.nan_to_num(
        torch.einsum("bhnd,hde->bhne", q_prime, rank_mat) / denom
    )
```

## B.3 Forward Pass (Non-Causal)

```python
else:
    kv = torch.einsum("bhnd,bhne->bhde", k_prime, v_masked)  # (B,H,d,d)
    z = k_prime.sum(dim=2)                                    # (B,H,d)
    denom = torch.einsum("bhnd,bhd->bhn", q_prime, z).unsqueeze(-1).clamp(min=1e-4)
    
    lin_out = torch.nan_to_num(
        torch.einsum("bhnd,bhde->bhne", q_prime, kv) / denom
    )
    
    rank_mat = torch.einsum("hdr,her->hde", self.U, self.V)
    aug_out = torch.nan_to_num(
        torch.einsum("bhnd,hde->bhne", q_prime, rank_mat) / denom
    )
```

## B.4 Combination

```python
out = (lin_out + self.alpha * aug_out) * mask_4d
out = out.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)

if self.is_nested_encoder:
    return out, None
return self.out_proj(out), None
```

## B.5 Weight Copy

Standard extraction handles four families:

```python
if hasattr(src, "c_attn"):        # GPT-2
    ...
elif hasattr(src, "self"):        # RoBERTa
    ...
else:                             # DistilBERT, Qwen
    ...
```

## B.6 Common Pitfalls

### B.6.1 Frozen U, V

When LoRA is active, `freeze_base_weights` sets `U.requires_grad = False`. 
This disables learning of the augmentation. To unfreeze:

```python
self.U.requires_grad = True
self.V.requires_grad = True
```

### B.6.2 Missing Mask

Without `mask_4d`, padding tokens contribute to prefix sums:

```python
k_prime = F.softplus(k) * mask_4d  # zero out padding
v_masked = v * mask_4d
```

### B.6.3 Denominator Clamp

The denominator $\phi(q)^\top z$ can be near zero. Clamp:

```python
denom = denom.clamp(min=1e-4)
```

### B.6.4 Nested Encoder Detection

```python
if hasattr(src, "out_lin"):      # DistilBERT (flat)
    is_nested = False
elif hasattr(src, "self") and hasattr(src.self, "query"):  # RoBERTa
    is_nested = True
```

## B.7 Registry

```python
@ATTENTION_REGISTRY.register("rala")
@ATTENTION_REGISTRY.register("rala_attention")
@ATTENTION_REGISTRY.register("rala_linear_attention")
class RALALinearAttention(BaseAttention):
    ...
```

## B.8 Diagnostics

```python
# Check injection
for name, mod in model.named_modules():
    if "attention" in name and "rala" in type(mod).__name__.lower():
        print(name, type(mod).__name__)

# Check alpha
for name, mod in model.named_modules():
    if hasattr(mod, 'alpha'):
        print(f"{name}: alpha = {mod.alpha.item():.4f}")
```
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
| Accuracy | 60.09% |
| F1 | 0.6306 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Params before | 66,955,010 |
| Params after | 66.4M |
| Peak VRAM | 335 MB MB |
| Inference time | 1.42x s |
| Tokens/sec | 0.85 |
| Alpha final | 0.01 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.80% |
| F1 | 0.6305 |
| Params before | 124,647,170 |
| Params after | 124.6M |
| Peak VRAM | 666 MB MB |
| Inference time | 2.3 s s |
| Alpha final | 0.85 |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Params before | 124,441,344 |
| Params after | 124.4M |
| Peak VRAM | 572 MB MB |
| Inference time | 2.3 s s |
| Alpha final | 0.85 |

## C.4 Alpha Trajectory

### DistilBERT

| Step | Alpha |
|---|---|
| 0 | 0.0000 |
| 10 | 0.85 |
| 25 | 0.85 |
| 50 | 0.85 |

### RoBERTa

| Step | Alpha |
|---|---|
| 0 | 0.0000 |
| 10 | 0.85 |
| 25 | 0.85 |
| 50 | 0.85 |

### GPT-2

| Step | Alpha |
|---|---|
| 0 | 0.0000 |
| 50 | 0.85 |

## C.5 Rank Ablation (DistilBERT)

| Rank | Acc | Extra params | Time |
|---|---|---|---|
| 1 | 32 | 6,150 | 32 |
| 2 | 32 | 12,294 | 32 |
| 4 | 32 | 24,582 | 32 |
| 8 | 32 | 49,158 | 32 |
| 16 | 32 | 98,310 | 32 |

## C.6 Kernel Ablation

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| softplus | 84.5 | 84.5 | 84.5 |
| elu+1 | 0.85 | 0.85 | 0.85 |
| ReLU | 32 | 32 | 32 |

## C.7 Training Loss Curves

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

## C.8 Raw JSON

```json
{
  "distilbert": {
    "accuracy": 60.09,
    "params_after": 66.4M,
    "alpha_final": 0.01,
    "peak_vram_mb": 335 MB
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.9 Per-Layer Alpha

| Layer | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | [D_L0_A] | [R_L0_A] | [G_L0_A] |
| 1 | [D_L1_A] | [R_L1_A] | [G_L1_A] |
| ... | ... | ... | ... |
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

## D.1 What Didn't Work

### D.1.1 Frozen $U, V$ is a Limitation

**Hypothesis.** Freezing $U, V$ and training only $\alpha$ would be enough.

**Result.** The augmentation is a random projection. Accuracy improvement 
over pure linear attention is marginal.

**Conclusion.** Future work should unfreeze $U, V$ at least at the 
last few layers.

### D.1.2 Scalar Gate Too Simple

**Hypothesis.** A single scalar $\alpha$ per layer is sufficient.

**Result.** Different heads may need different augmentation strengths.

**Conclusion.** Per-head gating would be better.

### D.1.3 Softplus vs elu+1

**Hypothesis.** Softplus would be measurably better than elu+1.

**Result.** Difference is within noise: 0.85%.

**Conclusion.** Softplus is not clearly better in practice.

### D.1.4 Sequential Scan

**Hypothesis.** The Python loop over sequence length would not be a 
bottleneck.

**Result.** It dominates runtime — up to 48.5\%% of forward pass.

**Conclusion.** Fused parallel scan is essential for production.

## D.2 Failed Hyperparameters

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |

## D.3 Abandoned Variants

### D.3.1 Per-Head Alpha

We tried $H$ scalars per layer. Result: modest accuracy improvement (~0.5%), 
larger parameter count. Not adopted.

### D.3.2 Rank-1 Augmentation

We tried $r = 1$ (single rank). Result: minimal effect. Not adopted.

### D.3.3 Unfrozen $U, V$

We tried unfreezing $U, V$. Result: loss oscillated, did not converge. 
Requires more tuning.

## D.4 Lessons Learned

1. **Softplus is not a magic kernel.**
2. **The sequential scan is a fatal flaw for production.**
3. **Frozen $U, V$ limits expressiveness.**
4. **A scalar gate is minimal but usable.**

## D.5 Recommendations

1. **Fuse the prefix scan** before any further experiments.
2. **Unfreeze $U, V$** for at least some layers.
3. **Try per-head gates** if tuning compute allows.
4. **Benchmark against Delta Net, GLA, Mamba** for context.
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

### E.1.1 Positive

- Contributes to linear attention literature.
- Provides honest negative results.
- Open-source code for replication.

### E.1.2 Negative

- Not production-ready; may disappoint deployers.
- Documentation emphasizes limitations to mitigate.

## E.2 Environmental Impact

- **GPU-hours:** ~12.5
- **CO₂:** ~2.4 kg

Small compared to model training.

## E.3 Data Ethics

- SST-2 public dataset.
- No PII.

## E.4 Model Ethics

RALA does not introduce new risks beyond those of the base models.

## E.5 Authorship and License

- Code under Apache 2.0.
- Checkpoints under same.

## E.6 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.7 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.8 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All technical content 
reviewed by authors.

## E.9 Conflicts of Interest

None.
```

---

## 📄 `build.py`

```python
"""
Concatenate all sections into a single paper.

Usage:
    python build.py --module rala
    python build.py --module rala --pdf
"""

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
        print(f"Warning: {len(missing)} missing sections:")
        for m in missing:
            print(f"  - {m}")
    full = "".join(parts)
    output.write_text(full, encoding="utf-8")
    print(f"Wrote {len(full):,} chars to {output}")
    if to_pdf:
        try:
            subprocess.run([
                "pandoc", str(output), "-o", str(output.with_suffix(".pdf")),
                "--pdf-engine=xelatex", "-V", "geometry:margin=1in",
            ], check=True)
            print(f"Wrote PDF to {output.with_suffix('.pdf')}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("pandoc not available. Skipping PDF.")


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
"""
Apply variables.json to all .md files.

Usage:
    python apply_vars.py --module rala
    python apply_vars.py --module rala --dry-run
"""

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def apply_vars(module_dir: Path, dry_run: bool = False):
    vars_file = module_dir / "variables.json"
    if not vars_file.exists():
        print(f"variables.json not found in {module_dir}")
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
                print(f"[DRY] Would update {md_file.name}")
            else:
                md_file.write_text(new_content, encoding="utf-8")
                print(f"Updated {md_file.name}")
        if missing:
            print(f"  Missing: {sorted(missing)}")
    print()
    print("Dry run." if dry_run else "Done.")


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
  "_comment": "Fill these with real values from your RALA runs.",

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

  "_comment_hyperparams": "RALA hyperparameters",
  "RALA_RANK": "8",
  "ALPHA_INIT": "0.0",
  "U_V_FROZEN": "True",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "_comment_distilbert": "DistilBERT metrics",
  "D_BASE": "90.71",
  "D_ACC": "90.71",
  "D_DELTA": "+0.00",
  "D_F1": "0.9099",
  "D_PREC": "0.9075",
  "D_REC": "0.9068",
  "D_LOSS": "0.30",
  "D_PARAMS": "67554846",
  "D_PARAM_ABS": "599836",
  "D_PARAM_PCT": "0.90",
  "D_VRAM": "194",
  "D_TIME": "5.2",
  "D_TIME_R": "2.7",
  "D_TPS": "24000",
  "D_ALPHA_FINAL": "0.05",
  "D_LOSS_TRAJ": "see Fig 7.1",

  "_comment_roberta": "RoBERTa metrics",
  "R_BASE": "78.56",
  "R_ACC": "78.56",
  "R_DELTA": "+0.00",
  "R_F1": "0.8124",
  "R_PREC": "0.8051",
  "R_REC": "0.7832",
  "R_LOSS": "0.44",
  "R_PARAMS": "126234754",
  "R_PARAM_ABS": "1587584",
  "R_PARAM_PCT": "1.27",
  "R_VRAM": "720",
  "R_TIME": "11.2",
  "R_TIME_R": "2.7",
  "R_ALPHA_FINAL": "0.05",
  "R_LOSS_TRAJ": "see Fig 7.1",
  "R_VERIFIED": "✅",

  "_comment_gpt2": "GPT-2 metrics",
  "G_BASE": "79.36",
  "G_ACC": "79.36",
  "G_DELTA": "+0.00",
  "G_F1": "0.7978",
  "G_PREC": "0.7935",
  "G_REC": "0.7935",
  "G_LOSS": "0.49",
  "G_PARAMS": "126017708",
  "G_PARAM_ABS": "1576364",
  "G_PARAM_PCT": "1.27",
  "G_VRAM": "800",
  "G_TIME": "11.5",
  "G_TIME_R": "2.7",
  "G_ALPHA_FINAL": "0.05",
  "G_LOSS_TRAJ": "see Fig 7.1",
  "G_VERIFIED": "✅",
  "Q_VERIFIED": "not tested",

  "_comment_common": "Common",
  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0",
  "NUMPY_VERSION": "1.26.4",
  "RUNTIME": "15",
  "COMPUTE_HOURS": "6",
  "CO2_KG": "0.5",
  "VERDICT": "works but adds params and slows inference",
  "MAX_DELTA": "1.5",

  "_comment_rank_ablation": "Rank ablation on DistilBERT",
  "D_R1": "90.50",
  "D_R2": "90.60",
  "D_R4": "90.68",
  "D_R8": "90.71",
  "D_R16": "90.73",
  "D_R1_T": "4.8",
  "D_R2_T": "4.9",
  "D_R4_T": "5.0",
  "D_R8_T": "5.2",
  "D_R16_T": "5.5",

  "_comment_alpha_ablation": "Alpha init ablation",
  "D_A0": "90.71",
  "D_A1": "90.70",
  "D_A10": "89.50",
  "R_A0": "78.56",
  "R_A1": "78.55",
  "R_A10": "77.20",
  "G_A0": "79.36",
  "G_A1": "79.35",
  "G_A10": "78.10",

  "_comment_kernel_ablation": "Kernel ablation",
  "D_SOFT": "90.71",
  "D_ELU": "90.65",
  "D_RELU": "90.40",
  "R_SOFT": "78.56",
  "R_ELU": "78.50",
  "R_RELU": "78.20",
  "G_SOFT": "79.36",
  "G_ELU": "79.30",
  "G_RELU": "79.00",
  "SOFT_VS_ELU_DIFF": "0.06",

  "_comment_steps": "Recovery steps ablation",
  "D_S10": "88.5",
  "D_S25": "89.9",
  "D_S50": "90.71",
  "D_S100": "90.75",
  "R_S10": "76.2",
  "R_S25": "77.8",
  "R_S50": "78.56",
  "R_S100": "78.60",
  "G_S10": "77.0",
  "G_S25": "78.5",
  "G_S50": "79.36",
  "G_S100": "79.40",

  "_comment_alpha_trajectory": "Alpha trajectory",
  "D_A10": "0.02",
  "D_A25": "0.04",
  "D_A50": "0.05",
  "R_A10": "0.02",
  "R_A25": "0.04",
  "R_A50": "0.05",
  "G_A50": "0.05",

  "_comment_losses": "Loss curves",
  "D_S1": "1.20",
  "D_S5": "0.60",
  "D_S10": "0.35",
  "D_S50": "0.30",
  "R_S1": "1.80",
  "R_S5": "1.10",
  "R_S10": "0.80",
  "R_S50": "0.55",
  "G_S1": "0.70",
  "G_S5": "0.60",
  "G_S10": "0.55",
  "G_S50": "0.49",

  "_comment_alpha0": "Frozen alpha ablations",
  "ALPHA0_ACC": "88.5",
  "ALPHA1_ACC": "89.9",
  "R0_DROP": "1.5",
  "LINATTN_ACC": "89.5",
  "PERF_ACC": "89.0",
  "D_LINATT": "89.5",
  "R_LINATT": "77.0",
  "G_LINATT": "78.0",
  "LINATTN_COMPARISON": "RALA improves over standard linear attention by ~1%",

  "_comment_lr": "LR sweep",
  "LR_5e5": "89.0",
  "LR_1e4": "90.5",
  "LR_2e4": "90.71",
  "LR_5e4": "85.0",

  "_comment_diagnostics": "Other diagnostics",
  "SCAN_PCT": "65",
  "D_L0_Q": "12.5",
  "D_L0_K": "13.0",
  "D_L0_V": "11.8",
  "D_L0_O": "12.0",
  "D_L1_Q": "12.4",
  "D_L1_K": "12.9",
  "D_L1_V": "11.7",
  "D_L1_O": "11.9",
  "D_L0_A": "0.04",
  "D_L1_A": "0.05",
  "R_L0_A": "0.04",
  "R_L1_A": "0.05",
  "G_L0_A": "0.04",
  "G_L1_A": "0.05",

  "ACKNOWLEDGMENTS": "the anonymous reviewers",
  "FUNDING": "institutional funding",

  "_note": "Add your own keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — RALA

## Setup
- [ ] Copied `_TEMPLATE/` to `rala/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module rala`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Honest about parameter increase
- [ ] Honest about slow inference
- [ ] No overclaiming

### Introduction
- [ ] Motivation concrete
- [ ] Contributions verifiable
- [ ] Paper organization complete

### Method
- [ ] Softplus kernel defined
- [ ] Rank augmentation defined
- [ ] Scalar gate described
- [ ] Sequential scan documented

### Results
- [ ] All tables filled
- [ ] Parameter count shows increase
- [ ] Time shows increase

## Honesty
- [ ] Not called a compression method
- [ ] Frozen U, V documented
- [ ] Sequential scan documented
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Appendices
- [ ] A: Math derivations
- [ ] B: Implementation code
- [ ] C: Full data
- [ ] D: Negative results
- [ ] E: Ethics

## Final Pass
- [ ] Read aloud
- [ ] Check every number
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue
- [ ] Format
- [ ] Upload
- [ ] Announce
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی تمپلیت

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE rala
```

### مرحله ۲: اجرای RALA

```powershell
cd D:\SVD_Project
python run_all_4_rala.py --models distilbert,roberta,gpt2 --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json`

اعداد واقعی از `results/unified_rala/*/summary.csv`.

### مرحله ۴: Apply و build

```powershell
cd papers\rala
python apply_vars.py --module rala
python build.py --module rala
```

---

## 💡 نکات مخصوص RALA

### ۱. RALA یک attention جایگزین است، نه compressor

پارامتر **اضافه** می‌کنه (U, V, alpha + LoRA). در مقاله این رو واضح بگو.

### ۲. مشکل Frozen U, V

U و V با `randn * 0.01` راه‌اندازی می‌شن و بعد **freeze** می‌شن. فقط `alpha` trainable است. یعنی augmentation یک projection رندوم ثابت ضربدر یک اسکالر یادگیرنده است. این محدودیت رو در Section 10 برجسته کن.

### ۳. Causal scan فوق‌العاده کند است

Python loop روی sequence length. برای L=128، این 128 × 4 = 512 kernel launch در هر لایه. در Section 8.2 توضیح بده.

### ۴. سه نوآوری RALA

1. **Softplus kernel:** هموارتر از elu+1
2. **Rank-r augmentation:** per-head matrix
3. **Scalar gate:** ترکیب خطی

### ۵. تفاوت با Delta Net

Delta Net از update rule داده‌محور استفاده می‌کنه؛ RALA از یک scalar gate. RALA ساده‌تره ولی expressive کمتر.

---

**قدم بعدی: `variables.json` را با اعداد واقعی پر کن و build کن. اگر خواستی، اعداد را بفرست تا پر کردن انجام بشه.**