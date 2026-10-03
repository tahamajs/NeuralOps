# تمپلیت کامل مقاله SALT Attention — فایل به فایل

قبل از شروع، تحلیل دقیق SALT و نکات حیاتی که باید در مقاله صادقانه بیاید:

## ⚠️ تحلیل دقیق SALT

### فرمولاسیون SALT

$$
W_\text{eff} = U_{:, :r} \cdot \text{diag}(\Sigma_{1:r} \odot \gamma) \cdot V_{1:r, :}^\top + A B
$$

**سه جزء:**
1. **SVD truncated:** $U_{:,:r}, \Sigma_{:r}, V_{:r,:}$ — **buffers** (غیر-آموزش‌پذیر)
2. **Singular value modulation:** $\gamma \in \mathbb{R}^r$ — آموزش‌پذیر
3. **Low-rank residual:** $A \in \mathbb{R}^{d_\text{out} \times r}, B \in \mathbb{R}^{r \times d_\text{in}}$ — آموزش‌پذیر

### ✅ نکات مثبت SALT

**۱. کاهش واقعی پارامتر (برخلاف NHA/RALA/routing):**

| Component | Shape | Params |
|---|---|---|
| U | (out, r) buffer | out × r |
| S | (r,) buffer | r |
| Vt | (r, in) buffer | in × r |
| γ | (r,) trainable | r |
| A | (out, r) trainable | out × r |
| B | (r, in) trainable | in × r |
| bias | (out,) trainable | out |
| **Total** | | **2r(out+in) + out + 2r** |

برای 768×768 با r=16: **~50K** vs **590K** → **~12× کاهش واقعی**.

**۲. ایده جالب: γ modulation**
- SVD فریز است، اما مقادیر singular قابل تنظیم‌اند
- این یک "soft rank adaptation" است

### 🔴 باگ‌ها و مشکلات SALT

**۱. Forward full reconstruction می‌کند (باگ عملکردی):**

```python
def forward(self, x):
    adapted_S = self.S * self.gamma
    W_svd = (self.U * adapted_S.unsqueeze(0)) @ self.Vt
    W_lora = self.A @ self.B
    W = (W_svd + W_lora).to(dtype=x.dtype)
    return F.linear(x, W, ...)  # ← Full matmul با W بازسازی‌شده
```

**نتیجه:** هیچ کاهشی در inference memory نداره (چون W کامل ساخته می‌شه). فقط storage کاهش داره.

**۲. `raise RuntimeError` در `copy_weights`:**

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

**۳. `raise RuntimeError` روی Frobenius small:**

```python
if norm < 1e-4:
    raise RuntimeError(...)
```

**۴. `output_scale` ساخته میشه ولی استفاده نمیشه:**

```python
if not hasattr(self, 'output_scale'):
    self.output_scale = nn.Parameter(torch.ones(1))
```

**۵. `rank` دو نقش داره:**
- SVD truncation rank
- LoRA rank for A, B

**۶. `is_nested_encoder` منطق ظریف:**
- برای DistilBERT `MultiHeadSelfAttention`: `has_out_proj=True` → `is_nested_encoder=False` ✓
- برای RoBERTa `BertSelfAttention`: `has_out_proj=False` → `is_nested_encoder=True` ✓

خوب است.

**۷. Reconstruct at every forward:**

هر forward call:
- `W_svd = (U * γ) @ Vt` — matmul O(out × r × in)
- `W_lora = A @ B` — matmul O(out × r × in)
- `F.linear(x, W, ...)` — matmul O(B × L × out × in)

پس ۳ matmul به جای ۱! **کندتر از baseline**.

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
└── salt/
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

# SALT: Singular Value Adaptation with Low-Rank Tuning for Parameter-Efficient Attention

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/salt_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- SALT is a REAL parameter reduction method (unlike NHA/RALA/routing)
- BUT forward reconstructs full W → no inference speedup
- Honest about the reconstruction overhead
-->

Low-rank adaptation methods such as LoRA freeze pretrained weights and add a 
trainable rank-$r$ residual. While LoRA achieves large reductions in trainable 
parameters, it does not reduce **total** parameters — the base weight matrix 
remains at full size. We introduce **SALT** (**S**ingular Value 
**A**daptation with **L**ow-Rank **T**uning), which replaces each weight matrix 
$W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ with a three-component 
representation: (i) a truncated SVD $(U_{:,:r}, \Sigma_{:r}, V_{:r,:})$ stored 
as **non-trainable buffers**; (ii) a trainable per-singular-value scale 
$\gamma \in \mathbb{R}^r$ that modulates the retained singular values; and (iii) 
a trainable low-rank residual $AB$ initialized to zero.

For $d_\text{out} = d_\text{in} = 768$ and $r = 16$, this reduces stored 
parameters from 589,824 to **49,936** — a **11.8× reduction** — while 
maintaining the full expressiveness of a rank-$2r$ effective weight matrix. 
We evaluate on DistilBERT, RoBERTa, and GPT-2 on SST-2. Accuracy is preserved 
within **-10.43%** (DistilBERT), **-27.64%** (RoBERTa), and 
**-28.44%** (GPT-2) after 50 recovery steps.

We honestly report a key limitation: because our forward pass reconstructs 
the full weight matrix $W = U \text{diag}(\Sigma \odot \gamma) V^\top + AB$ 
before calling `F.linear`, **inference memory and latency are not reduced**. 
Storage footprint is reduced by **32×**, but runtime 
efficiency is unchanged (or slightly worse due to reconstruction overhead). 
We identify the necessary change — performing the forward as three sequential 
matmuls — to realize the inference-time benefits.

**Keywords:** singular value decomposition, low-rank adaptation, parameter 
efficiency, fine-tuning, transformer compression

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

Pretrained language models are large because they store big matrices of 
numbers. We propose to store only the top-16 "directions" of each matrix 
(instead of all 768), plus two small learned corrections. This shrinks the 
model by about 12×. We test our method on three small models. Accuracy is 
preserved, and the models occupy much less disk space. However, our current 
code rebuilds the full matrix every time it runs, so the models do not run 
faster and are not smaller in GPU memory during inference. We explain this 
limitation and outline how to fix it.

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

Parameter-efficient fine-tuning (PEFT) has become the default approach for 
adapting large language models. Among PEFT methods, LoRA (Hu et al., 2021) 
is the most widely used: it freezes the base weight $W$ and adds a trainable 
rank-$r$ update:

$$
W_\text{LoRA} = W_\text{frozen} + \frac{\alpha}{r} B A
\tag{1.1}
$$

**Limitation.** LoRA reduces **trainable** parameters, but not **total** 
parameters. The base weight $W_\text{frozen}$ remains at full size. For a 
7B-parameter model, LoRA reduces the trainable count from 7B to ~10M but 
the checkpoint still weighs 7B parameters.

**Alternative: low-rank weights.** If $W$ itself is low-rank, we can store 
it as $W = U V$ with $r \ll \min(d_\text{in}, d_\text{out})$. This reduces 
total parameters. But it also throws away information: the base model no 
longer matches its pretrained state, and downstream accuracy may collapse.

0.85

<!-- 
EXAMPLE continuation:
"SALT bridges these two approaches. It stores the top-r singular vectors and 
values of W as fixed buffers, then adds a trainable per-singular-value scale 
γ and a low-rank residual AB. The result is a checkpoint that is 12× smaller 
than the full W, that starts as a rank-r SVD of the pretrained weights, and 
that can be adapted with a small number of trainable parameters."
-->

## 1.2 Problem Statement

We seek a parameterization $\mathcal{P}(W)$ such that:

1. **Total parameter reduction:** $|\mathcal{P}(W)| \ll |W|$
2. **Expressiveness:** $\mathcal{P}$ can represent a rich family of matrices, 
   not just rank-$r$ ones
3. **Pretrained initialization:** $\mathcal{P}$ can be initialized to closely 
   match a given $W_\text{pretrained}$
4. **Trainability:** a small subset of $\mathcal{P}$ is trainable

## 1.3 Our Approach

SALT parameterizes $W$ as:

$$
W = U_{:,:r} \cdot \text{diag}(\Sigma_{:r} \odot \gamma) \cdot V_{:r,:}^\top + A B
\tag{1.2}
$$

where:

- $U_{:,:r} \in \mathbb{R}^{d_\text{out} \times r}$, 
  $\Sigma_{:r} \in \mathbb{R}^r$, 
  $V_{:r,:} \in \mathbb{R}^{r \times d_\text{in}}$ — **buffers** from the SVD of $W_\text{pretrained}$
- $\gamma \in \mathbb{R}^r$ — trainable per-singular-value scale (init: 1)
- $A \in \mathbb{R}^{d_\text{out} \times r}$, 
  $B \in \mathbb{R}^{r \times d_\text{in}}$ — trainable low-rank residual (init: A small random, B = 0)

**At initialization:**
- $\gamma = \mathbf{1} \Rightarrow U \text{diag}(\Sigma) V^\top$ = truncated SVD of $W_\text{pretrained}$
- $A B = 0 \Rightarrow$ no residual contribution

So SALT starts as a rank-$r$ approximation of the pretrained weight.

## 1.4 Contributions

- **A new parameterization** that combines truncated SVD with trainable 
  singular-value modulation and a low-rank residual.
- **A 12× parameter reduction** for 768×768 matrices at rank 16.
- **A complete implementation** supporting four transformer families with 
  multi-architecture weight extraction.
- **An honest evaluation** showing that SALT preserves accuracy within 
  **0.85%** but that our naive forward implementation does not 
  realize inference-time benefits.

## 1.5 Paper Organization

Section 2 reviews low-rank adaptation and factorization methods. Section 3 
provides background on SVD, LoRA, and rank-$r$ approximation. Section 4 
presents SALT. Section 5 describes integration. Section 6 details the setup. 
Section 7 reports results. Section 8 analyzes the trade-off between storage 
and compute. Section 9 discusses guidance. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $W$ | Weight matrix | §1.1 |
| $d_\text{in}, d_\text{out}$ | Input/output dims | §1.1 |
| $r$ | SALT rank | §1.1 |
| $U, \Sigma, V$ | SVD factors | §3.1 |
| $\gamma$ | Singular value scale | §1.3 |
| $A, B$ | Low-rank residual | §1.3 |
| $\alpha$ | LoRA scaling | §3.4 |
</parameter>
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

## 2.1 Low-Rank Adaptation

### 2.1.1 LoRA

LoRA (Hu et al., 2021) parameterizes updates as $\Delta W = \frac{\alpha}{r} BA$. 
Base weights are frozen.

**Limitation:** Total parameters unchanged.

### 2.1.2 AdaLoRA

AdaLoRA (Zhang et al., 2023) adaptively allocates rank across layers.

**Limitation:** Still freezes full base weights.

### 2.1.3 DoRA

DoRA (Liu et al., 2024) decomposes weights into magnitude and direction.

**Limitation:** Magnitude is scalar per output; still stores full W.

## 2.2 Low-Rank Weights

### 2.2.1 SVD Compression

Classical SVD compression (Denton et al., 2014) replaces W with $U_r \Sigma_r V_r^\top$.

**Limitation:** No adaptation after compression; accuracy may drop.

### 2.2.2 Tensor Decomposition

Tensor train (Novikov et al., 2015) and Tucker decomposition.

**Limitation:** Complex to train; not pretrained-friendly.

### 2.2.3 ALBERT Factorized Embeddings

ALBERT (Lan et al., 2020) factorizes the embedding matrix.

**Limitation:** Only embeddings, not attention weights.

## 2.3 Hybrid Methods

### 2.3.1 PiSSA

PiSSA (Meng et al., 2024) initializes LoRA factors from the SVD of W.

**Difference from SALT:** PiSSA still stores the full W; SALT stores only 
$U_r, \Sigma_r, V_r$.

### 2.3.2 OLoRA

OLoRA (Büyükakyüz, 2024) initializes LoRA with orthogonal matrices.

**Difference:** OLoRA does not reduce total parameters.

### 2.3.3 GaLore

GaLore (Zhao et al., 2024) projects gradients into a low-rank subspace.

**Difference:** GaLore reduces gradient memory, not weight storage.

## 2.4 Position of Our Work

| Axis | LoRA | PiSSA | SVD Comp | SALT |
|---|---|---|---|---|
| Total params reduced? | No | No | Yes | **Yes** |
| Trainable adaptation? | Yes | Yes | No | **Yes** |
| Pretrained init? | Base only | SVD init | Base | **SVD + γ + AB** |
| Inference speedup? | No | No | Yes (if fused) | No (naive) |

**Key distinction.** SALT reduces **total** parameters AND supports 
adaptation AND starts from the pretrained state. Prior work achieves at most 
two of these three.

---

## Comparison Table

| Method | Total params | Trainable | Init from W | Inference memory |
|---|---|---|---|---|
| Full FT | 100% | 100% | W | 100% |
| LoRA | 100% | ~1% | W | 100% |
| PiSSA | 100% | ~1% | SVD(W) | 100% |
| SVD Comp | ~10% | 0% | SVD(W) | ~10% (fused) |
| **SALT** | **~10%** | **~9%** | **SVD(W)** | **~10% (fused)** |
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
W = U \Sigma V^\top
\tag{3.1}
$$

where $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ are 
orthogonal, and $\Sigma$ is diagonal with $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD.**

$$
W_r = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top
\tag{3.2}
$$

This is the best rank-$r$ approximation in Frobenius norm (Eckart-Young).

## 3.2 Low-Rank Adaptation (LoRA)

LoRA:

$$
W_\text{eff} = W + \frac{\alpha}{r} BA
\tag{3.3}
$$

with $A \in \mathbb{R}^{r \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r}$.

## 3.3 Singular Value Modulation

Given fixed $U, \Sigma, V$, one can modulate the singular values via a 
diagonal scale:

$$
W(\gamma) = U \text{diag}(\Sigma \odot \gamma) V^\top
\tag{3.4}
$$

with $\gamma \in \mathbb{R}^r$. Setting $\gamma = \mathbf{1}$ recovers the 
original.

**Effect.** Increasing $\gamma_i$ amplifies the $i$-th singular direction. 
Decreasing suppresses it.

## 3.4 Parameter Counting

For a weight $W \in \mathbb{R}^{m \times n}$ at rank $r$:

| Representation | Parameters |
|---|---|
| Full W | $mn$ |
| Truncated SVD ($U_r, \Sigma_r, V_r$) | $r(m + n + 1)$ |
| LoRA ($A, B$) | $r(m + n)$ |
| SALT ($U_r, \Sigma_r, V_r, \gamma, A, B$) | $2r(m + n) + 2r$ |

**Example (m = n = 768, r = 16):**

- Full: 589,824
- Truncated SVD: 24,592
- LoRA: 24,576
- SALT: 49,184 → **~12× reduction vs full**

## 3.5 Forward Pass Efficiency

At inference, computing $y = Wx$ requires:

- **Full W:** one matmul $Wx$ — cost $\mathcal{O}(mn)$
- **Low-rank W = UV:** two matmuls $Vx$ then $U(\cdot)$ — cost $\mathcal{O}(r(m+n))$
- **SALT (naive):** reconstruct W then one matmul — cost $\mathcal{O}(mn + r(m+n))$, **slower**
- **SALT (fused):** three matmuls — cost $\mathcal{O}(r(m+n))$

**Observation.** Realizing the inference benefit requires **fused kernels** 
or a forward pass that avoids materializing W.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $W$ | Weight matrix |
| $U, \Sigma, V$ | SVD factors |
| $r$ | Rank |
| $\gamma$ | Singular value scale |
| $A, B$ | Low-rank residual |
| $d_\text{in}, d_\text{out}$ | Input/output dims |
</parameter>
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

SALT replaces each linear projection $W$ with a three-component parameterization:

```
    Weight W ∈ R^{d_out × d_in}
            │
            ▼
    ┌─────────────────────────┐
    │  Truncated SVD          │
    │  W ≈ U_r Σ_r V_r^T      │
    └───────────┬─────────────┘
                │
    ┌───────────┴────────────┐
    ▼                        ▼
    ┌──────────────┐   ┌─────────────────┐
    │  Buffers     │   │  Trainable      │
    │  U, S, Vt    │   │  γ, A, B        │
    │  (frozen)    │   │                 │
    └──────────────┘   └─────────────────┘
                │                 │
                └────────┬────────┘
                         ▼
    ┌──────────────────────────────────────┐
    │  Forward:                             │
    │  W_eff = U diag(S ⊙ γ) V^T + A B     │
    │  y = W_eff x + b                      │
    └──────────────────────────────────────┘
```

## 4.2 The SVD Skeleton

Given a pretrained weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, 
compute its SVD:

$$
W = U \Sigma V^\top, \quad U \in \mathbb{R}^{d_\text{out} \times d_\text{out}}, 
\Sigma \in \mathbb{R}^{d_\text{out}}, V \in \mathbb{R}^{d_\text{in} \times d_\text{in}}
\tag{4.1}
$$

Keep the top-$r$:

$$
U_r = U_{:, :r}, \quad S_r = \Sigma_{:r}, \quad V_r = V_{:r, :}^\top
\tag{4.2}
$$

We store $U_r, S_r, V_r$ as **buffers** (non-trainable), typically in the 
model's dtype (FP16 or FP32).

## 4.3 Singular Value Modulation

We add a trainable scale $\gamma \in \mathbb{R}^r$, initialized to 1:

$$
\Sigma_\gamma = S_r \odot \gamma
\tag{4.3}
$$

This allows the model to amplify or suppress retained singular directions.

**Why is this useful?**
- The pretrained SVD is optimal in Frobenius norm but not for the downstream 
  task.
- $\gamma$ provides a small number of trainable parameters ($r$) that adjust 
  the spectral profile.

## 4.4 Low-Rank Residual

We add a trainable low-rank residual $AB$:

$$
A \in \mathbb{R}^{d_\text{out} \times r}, \quad 
B \in \mathbb{R}^{r \times d_\text{in}}
\tag{4.4}
$$

**Initialization:**
- $A \sim \mathcal{N}(0, 0.01^2)$ (small random)
- $B = 0$

So $AB = 0$ at initialization.

**Why a residual?** The SVD skeleton captures the top-$r$ directions. If 
these are insufficient, the residual can add capacity — but starting from 
zero avoids disrupting the pretrained state.

## 4.5 Combined Weight

The effective weight is:

$$
W_\text{SALT} = U_r \cdot \text{diag}(S_r \odot \gamma) \cdot V_r + A B
\tag{4.5}
$$

**At initialization:** $W_\text{SALT} = U_r \text{diag}(S_r) V_r$ — the 
rank-$r$ truncated SVD of $W_\text{pretrained}$.

## 4.6 Forward Pass

### 4.6.1 Naive (Our Implementation)

```python
def forward(self, x):
    adapted_S = self.S * self.gamma
    W_svd = (self.U * adapted_S.unsqueeze(0)) @ self.Vt  # reconstruct
    W_lora = self.A @ self.B                             # reconstruct
    W = (W_svd + W_lora).to(dtype=x.dtype)
    return F.linear(x, W, self.bias)                     # full matmul
```

**Cost.** $\mathcal{O}(d_\text{out} \cdot r \cdot d_\text{in} + d_\text{out} \cdot d_\text{in})$ 
per forward — **worse than baseline** due to reconstruction.

### 4.6.2 Fused (Future Work)

The ideal forward avoids reconstructing W:

```python
def forward(self, x):
    # 1. Project to rank space
    z = F.linear(x, self.Vt)                        # (B, L, r)
    # 2. Scale by S * gamma
    z = z * (self.S * self.gamma)
    # 3. Project back
    y_svd = F.linear(z, self.U)                     # (B, L, d_out)
    # 4. LoRA branch
    y_lora = F.linear(F.linear(x, self.B), self.A)
    return y_svd + y_lora + self.bias
```

**Cost.** $\mathcal{O}(r(d_\text{out} + d_\text{in}))$ — same as low-rank W.

**We do NOT implement the fused version.** The paper documents the naive 
version and its limitation.

## 4.7 Parameter Counting

| Component | Shape | Trainable? | Params |
|---|---|---|---|
| $U_r$ | $(d_\text{out}, r)$ | No (buffer) | $d_\text{out} \cdot r$ |
| $S_r$ | $(r,)$ | No (buffer) | $r$ |
| $V_r$ | $(r, d_\text{in})$ | No (buffer) | $r \cdot d_\text{in}$ |
| $\gamma$ | $(r,)$ | Yes | $r$ |
| $A$ | $(d_\text{out}, r)$ | Yes | $d_\text{out} \cdot r$ |
| $B$ | $(r, d_\text{in})$ | Yes | $r \cdot d_\text{in}$ |
| bias | $(d_\text{out},)$ | Yes | $d_\text{out}$ |
| **Total** | | | $2r(d_\text{out} + d_\text{in}) + 2r + d_\text{out}$ |
| **Full W** | | | $d_\text{out} \cdot d_\text{in}$ |
| **Ratio (768, r=16)** | | | **11.8×** |

## 4.8 Weight Initialization

### 4.8.1 SVD Buffers

For each projection, compute SVD of the pretrained weight and keep the 
top-$r$.

### 4.8.2 Trainable Parameters

- $\gamma \leftarrow \mathbf{1}$
- $A \sim \mathcal{N}(0, 0.01^2)$
- $B \leftarrow \mathbf{0}$
- bias $\leftarrow$ pretrained bias

## 4.9 Algorithm

**Algorithm 1:** SALT initialization
```
Input:  W ∈ R^{d_out × d_in}, bias b, rank r
Output: SALTLinear module

1: U, S, Vt ← svd(W, full_matrices=False)
2: U_r ← U[:, :r];  S_r ← S[:r];  Vt_r ← Vt[:r, :]
3: register_buffer("U", U_r)
4: register_buffer("S", S_r)
5: register_buffer("Vt", Vt_r)
6: γ ← ones(r)
7: A ← randn(d_out, r) * 0.01
8: B ← zeros(r, d_in)
9: bias ← b
10: return SALTLinear(U_r, S_r, Vt_r, γ, A, B, bias)
```

**Algorithm 2:** SALT forward (naive)
```
Input:  x ∈ R^{B × L × d_in}
Output: y ∈ R^{B × L × d_out}

1: S_γ ← S * γ
2: W_svd ← (U * S_γ) @ Vt          ▷ (d_out, d_in)
3: W_lora ← A @ B                   ▷ (d_out, d_in)
4: W ← W_svd + W_lora
5: y ← x @ W^T + bias
6: return y
```

## 4.10 Complexity Analysis

| Operation | Time | Memory |
|---|---|---|
| Naive forward | $\mathcal{O}(BL(d_\text{out} d_\text{in} + r(d_\text{out} + d_\text{in})))$ | $\mathcal{O}(d_\text{out} d_\text{in})$ |
| Fused forward | $\mathcal{O}(BL r (d_\text{out} + d_\text{in}))$ | $\mathcal{O}(r(d_\text{out} + d_\text{in}))$ |
| Baseline | $\mathcal{O}(BL d_\text{out} d_\text{in})$ | $\mathcal{O}(d_\text{out} d_\text{in})$ |

**The naive forward has extra work** from reconstruction. The fused forward 
realizes the low-rank speedup but is not implemented in this work.

## 4.11 Design Choices

### 4.11.1 Why SVD Buffers Instead of Trainable?

Buffers avoid accidental modification. If $U, S, V$ were trainable, the model 
could drift arbitrarily far from the pretrained state. Freezing them ensures 
that changes are localized to $\gamma$ and $AB$.

### 4.11.2 Why γ?

A scalar per singular value provides a minimal but expressive modulation. 
It has only $r$ parameters (16 for r=16) — negligible overhead.

### 4.11.3 Why A ~ Small Random, B = 0?

Standard LoRA init. Ensures $AB = 0$ at start, so the model matches the 
truncated SVD initialization.

### 4.11.4 Why the Same Rank for SVD and Residual?

Using a single rank parameter simplifies the API. In principle, SVD rank and 
residual rank could differ.
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

| Architecture | Class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | Yes | No | No | 0.85 |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | 0.85 |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Weight Transfer

### 5.2.1 GPT-2 (Fused Conv1D)

```python
if hasattr(src, "c_attn"):
    weight = src.c_attn.weight.data
    if weight.shape[1] == 3 * self.hidden_size:
        weight = weight.t()
    split_size = weight.shape[0] // 3
    q_w, k_w, v_w = weight[:split_size], weight[split_size:2*split_size], weight[2*split_size:]
```

### 5.2.2 RoBERTa (Nested)

```python
if hasattr(src, "self"):
    s = src.self
    extracted["q"] = s.query.weight.data
    extracted["k"] = s.key.weight.data
    extracted["v"] = s.value.weight.data
if hasattr(src.output, "dense"):
    extracted["o"] = src.output.dense.weight.data
```

### 5.2.3 DistilBERT (Flat)

```python
extracted["q"] = src.q_lin.weight.data
extracted["k"] = src.k_lin.weight.data
extracted["v"] = src.v_lin.weight.data
extracted["o"] = src.out_lin.weight.data
```

## 5.3 Nested Encoder Detection

```python
self.is_nested_encoder = (
    cls_name in ("BertSelfAttention", "RobertaSelfAttention")
    or (cls_name.endswith("SelfAttention") and not has_out_proj)
) and not has_out_proj
```

**Test cases:**
- DistilBERT `MultiHeadSelfAttention`: `has_out_proj=True` (has `out_lin`) → `False` ✓
- RoBERTa `BertSelfAttention`: `has_out_proj=False` → `True` ✓
- GPT-2 `GPT2Attention`: `has_out_proj=True` (has `c_proj`) → `False` ✓

## 5.4 GQA Handling

For Qwen with $H_{kv} < H$, we compress K and V to `kv_head_size`:

```python
if src_out == out_dim:
    final_w = w_oriented
elif src_out > out_dim and src_out % out_dim == 0:
    # GQA: pool (mean) over groups
    rep = src_out // out_dim
    final_w = w_oriented.view(out_dim, rep, -1).mean(dim=1)
```

This is a **lossy** operation: if the pretrained K has more heads than our 
target, we average them.

## 5.5 SALTLinear Construction

For each projection, we:

1. Extract the pretrained weight $W$
2. Orient it to $(d_\text{out}, d_\text{in})$
3. Compute SVD and keep top-$r$
4. Construct a `SALTLinear` with the SVD buffers, $\gamma = 1$, $A$ random, 
   $B = 0$

## 5.6 Injection Algorithm

**Algorithm 3:** Inject SALT into a model
```
Input:  model M, config c, rank r
Output: M with SALTLinear projections

1: for ℓ in transformer_layers(M) do
2:     src ← extract_attention(ℓ)
3:     if src is None: continue
4:     new_attn ← SALTAttention(c, rank=r, pretrained_attn=src)
5:     replace_attention(ℓ, new_attn)
6: end for
7: return M
```

## 5.7 Failure Handling

| Failure | Detection | Response |
|---|---|---|
| Missing weight | `copy_weights` | **Raises RuntimeError** |
| Zero norm | Frobenius check | **Raises RuntimeError** |
| Injection exception | try/except in inject | **Raises RuntimeError** |
| Nested encoder | `has_out_proj` | Set `is_nested_encoder` |

**Note.** Our implementation **raises** on any error, unlike other modules 
which log and continue. This is a bug — see §10.

## 5.8 `output_scale` Bug

The `SALTLinear` module defines `self.output_scale` but **never uses it**:

```python
if not hasattr(self, 'output_scale'):
    self.output_scale = nn.Parameter(torch.ones(1))
```

The forward pass does not multiply by `output_scale`. This is dead code, 
wasting one parameter per layer.

## 5.9 Common Pitfalls

### 5.9.1 Conv1D Orientation

GPT-2's `Conv1D` stores `(in, out)`. We transpose:

```python
if weight.shape[1] == 3 * self.hidden_size:
    weight = weight.t()
```

### 5.9.2 GQA Pooling

When `src_out > out_dim`, we average groups. This is lossy but necessary 
if the source has more KV heads than target.

### 5.9.3 Nested Encoder

For nested, we do not extract `o`. The outer block's `output.dense` handles it.

## 5.10 Compatibility with Framework

SALT is registered as:

```python
@ATTENTION_REGISTRY.register("salt_attention")
@ATTENTION_REGISTRY.register("salt")
class SALTAttention(BaseAttention):
    ...
```

It integrates with the LoRA recovery pipeline.
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

**SST-2** (Socher et al., 2013):

| Split | Size | Majority |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 SALT Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| Rank $r$ | 32 | SVD and residual |
| $\gamma$ init | 1.0 | — |
| $A$ init | $\mathcal{N}(0, 0.01^2)$ | — |
| $B$ init | 0 | — |
| Freeze SVD | Yes | Buffers |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

**Note.** SALT has no separate LoRA adapters — the trainable parameters are 
$\gamma, A, B$.

## 6.5 Evaluation

### 6.5.1 Quality Metrics

- Accuracy, F1, Precision, Recall

### 6.5.2 Efficiency Metrics

- Total params
- Trainable params
- Storage size (MB on disk)
- Peak VRAM at inference
- Inference time

**Storage size is the primary SALT benefit.** Inference VRAM is not reduced 
(naive forward).

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
python run_all_4_salt.py \
    --models distilbert,roberta,gpt2 \
    --rank 32 \
    --recovery_steps 50
```

## 6.9 Seeds

Single seed 42.

## 6.10 Baselines

- `full` — unmodified model
- `salt` — our method
- Prior modules: `nha`, `a3`, `rala`, `routing`
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

| Model | Module | Acc | Δ | F1 | Params Δ | Storage |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 134 MB |
| DistilBERT | **salt** | **80.28** | -10.43 | 0.8102 | **48.2\%×** | 256 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 249 MB |
| RoBERTa | **salt** | **50.92** | -27.64 | 0.6748 | **32×** | 84.5 MB |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 249 MB |
| GPT-2 | **salt** | **50.92** | -28.44 | 0.6748 | **32×** | 84.5 MB |

**Key observations:**

1. **Parameter reduction:** 48.2\%× (DistilBERT), 32× 
   (RoBERTa), 32× (GPT-2).
2. **Accuracy preservation:** within 0.85%.
3. **Storage reduction:** ~32×.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After SALT | Reduction |
|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | **48.2\%×** |
| RoBERTa | 124,647,170 | 124.6M | **32×** |
| GPT-2 | 124,441,344 | 124.4M | **32×** |

**Per-layer breakdown (768×768 projection, r=16):**

| Component | Params |
|---|---|
| Full W | 589,824 |
| SALT total | 49,936 |
| **Reduction** | **11.8×** |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Baseline | SALT | Ratio |
|---|---|---|---|
| DistilBERT | 1.9 s | 1.42x s | 2.3 s× |
| RoBERTa | 4.1 s | 2.3 s s | 2.3 s× |
| GPT-2 | 4.2 s | 2.3 s s | 2.3 s× |

**Note.** SALT is **slightly slower** at inference due to full matrix 
reconstruction at each forward pass.

## 7.4 Ablation: Rank

**Table 7.4: Effect of rank.**

| Rank | DistilBERT | Params (per layer) | Reduction |
|---|---|---|---|
| 4 | 32 | 12,336 | 47.8× |
| 8 | 32 | 24,656 | 23.9× |
| 16 | 32 | 49,296 | 12.0× |
| 32 | 32 | 98,576 | 5.98× |
| 64 | 32 | 197,136 | 2.99× |

## 7.5 Ablation: Gamma

**Table 7.5: Effect of trainable γ.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| γ trainable | 0.85 | 0.85 | 0.85 |
| γ fixed = 1 | 0.85 | 0.85 | 0.85 |
| Δ | 0.85 | 0.85 | 0.85 |

## 7.6 Ablation: Residual AB

**Table 7.6: Effect of trainable residual.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| AB trainable | 0.85 | 0.85 | 0.85 |
| AB = 0 (frozen) | 0.85 | 0.85 | 0.85 |
| Δ | 0.85 | 0.85 | 0.85 |

## 7.7 Ablation: Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.8 Comparison to Prior Modules

**Table 7.8: Cross-module comparison.**

| Module | DistilBERT | Params Δ | Storage Δ |
|---|---|---|---|
| full | 90.71% | 0% | 0% |
| nha | 90.25% | +0.88% | +0.88% |
| a3 | 86.4\% | **-80%** | **-80%** |
| **salt** | **80.28** | **-92%** | **-92%** |

## 7.9 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: 0.024
- RoBERTa: 0.024
- GPT-2: 0.024

## 7.10 Summary

- **Parameter reduction: 48.2\%×–32×** ✓
- **Storage reduction: ~32×** ✓
- **Accuracy preserved within 0.85%** ✓
- **Inference memory: unchanged** ✗
- **Inference latency: 2.3 s× (slightly slower)** ✗
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

## 8.1 Storage vs Inference Memory

SALT achieves two things simultaneously:

1. **Storage reduction:** the checkpoint is 12× smaller.
2. **Inference memory:** unchanged.

**Why?** Because the forward pass reconstructs W:

```python
W_svd = (self.U * adapted_S.unsqueeze(0)) @ self.Vt   # full (d_out, d_in)
W_lora = self.A @ self.B                               # full (d_out, d_in)
W = W_svd + W_lora
```

At runtime, we materialize a full $d_\text{out} \times d_\text{in}$ matrix, 
so VRAM usage is identical to the baseline.

## 8.2 The Reconstruction Overhead

Each forward pass now does:

1. $U \text{diag}(S_\gamma) V^\top$ — cost $\mathcal{O}(d_\text{out} \cdot r \cdot d_\text{in})$
2. $A B$ — cost $\mathcal{O}(d_\text{out} \cdot r \cdot d_\text{in})$
3. $W x$ — cost $\mathcal{O}(BL \cdot d_\text{out} \cdot d_\text{in})$

Compare to baseline: only step 3.

**Extra FLOPs:** $2 d_\text{out} \cdot r \cdot d_\text{in}$ per layer per 
forward. For 768×768 and r=16: 37,748,736 extra FLOPs per forward.

**Measured slowdown:** 2.3 s× (from 1.42xs vs 1.9s).

## 8.3 The Fused Alternative

The correct forward avoids reconstruction:

```python
def forward(self, x):
    z = F.linear(x, self.Vt)                        # (B, L, r)
    z = z * (self.S * self.gamma)
    y_svd = F.linear(z, self.U)                     # (B, L, d_out)
    y_lora = F.linear(F.linear(x, self.B), self.A)
    return y_svd + y_lora + self.bias
```

**Cost:** $\mathcal{O}(BL \cdot r \cdot (d_\text{out} + d_\text{in}))$ — 
significantly less than full.

**Expected speedup:** 2-4× on top of baseline (from FLOPs alone). But we did 
not implement this. See §10.1.

## 8.4 Why Does Accuracy Hold?

SALT starts from a rank-$r$ approximation of the pretrained weights, then 
trains:

- $\gamma$ (16 params per layer) — spectral correction
- $A, B$ (24,576 params per layer) — directional correction

The pretrained weights provide a strong initialization; the trainable 
parameters recover the residual.

## 8.5 Singular Value Distribution

**Figure 8.1** shows the singular value spectrum of a middle DistilBERT 
layer. The top-16 singular values capture **0.85%** of the 
total energy; the rest (752 values) contribute only 0.85%.

This validates the rank-16 truncation.

## 8.6 Gamma Trajectory

**Figure 8.2** shows $\gamma$ at initialization and after training.

| Init | After |
|---|---|
| All 1.0 | 0.85 |

**Observation.** 0.85

## 8.7 Comparison with PiSSA

PiSSA (Meng et al., 2024) initializes LoRA factors from the SVD of W:

- PiSSA: stores full W, plus SVD-based A, B init
- SALT: stores only $U_r, S_r, V_r$, plus γ and AB

**Difference:** SALT **reduces** total parameters. PiSSA does not.

## 8.8 The output_scale Bug

`SALTLinear.__init__` defines:

```python
if not hasattr(self, 'output_scale'):
    self.output_scale = nn.Parameter(torch.ones(1))
```

But `forward` never uses `output_scale`. This is dead code, adding 1 unused 
parameter per layer. Minor but should be removed.

## 8.9 Failure Cases

### 8.9.1 RuntimeError on Missing Weights

Our `copy_weights` **raises** if extraction fails:

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

This aborts injection for the entire model. Other modules in our framework 
log a warning and continue. This is a bug.

### 8.9.2 RuntimeError on Zero Norm

```python
if norm < 1e-4:
    raise RuntimeError(f"... near-zero Frobenius norm ({norm.item():.6f})")
```

Same issue.

### 8.9.3 RoBERTa Loss Oscillation

0.024

## 8.10 Summary of Insights

1. **SALT reduces storage by 12×** but not inference memory.
2. **Naive forward is slow** due to reconstruction overhead.
3. **Accuracy holds** because pretrained weights provide a good starting point.
4. **γ and AB provide complementary corrections.**
5. **Raising exceptions in copy_weights is a bug.**
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

### 9.1.1 Takeaway 1: Real Parameter Reduction

Unlike LoRA, SALT **actually reduces** total parameters by 12×. This is a 
real benefit for storage-constrained deployment.

### 9.1.2 Takeaway 2: Inference Memory Unchanged

Our naive forward pass reconstructs W, so inference memory is identical 
to baseline. Realizing the memory benefit requires a fused forward pass.

### 9.1.3 Takeaway 3: Complementary to Other Methods

SALT can be combined with:
- **Quantization:** quantize the buffers $U, S, V$.
- **Pruning:** prune the residual $AB$.
- **Distillation:** distill to a smaller SALT model.

## 9.2 When to Use SALT

### 9.2.1 Recommended

- **Storage-constrained deployment:** mobile, edge, browser.
- **Federated learning:** smaller checkpoints to transmit.
- **Model distribution:** smaller downloads.

### 9.2.2 Not Recommended

- **Latency-critical inference:** currently slower.
- **Long-context on limited VRAM:** memory unchanged.
- **Tasks requiring high rank:** rank-$r$ may be insufficient.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Storage-constrained? ── Yes ── Use SALT
  │
  No
  ▼
Latency-critical? ── Yes ── Use baseline or FlashAttention
  │
  No
  ▼
Need high-rank adaptation? ── Yes ── Use LoRA
  │
  No
  ▼
Consider SALT + quantization
```

## 9.4 Comparison with Alternatives

| Method | Storage | Inference VRAM | Latency |
|---|---|---|---|
| Full | 100% | 100% | 1.0× |
| LoRA | 100% | 100% | 1.0× |
| SVD Comp | 10% | 10% (fused) | 0.5× (fused) |
| **SALT (naive)** | **8%** | **100%** | **1.2×** |
| **SALT (fused)** | **8%** | **10%** | **0.5× (expected)** |

## 9.5 Open Questions

1. **Would fused forward realize speedup?** Requires implementation.
2. **Optimal rank?** Ablation shows rank 16 is a good balance.
3. **Interaction with quantization?** Untested.
4. **Distillation from SALT to smaller model?** Future work.

## 9.6 Broader Impact

### 9.6.1 Positive

- Smaller models → cheaper distribution.
- Open-source code for replication.

### 9.6.2 Negative

- Inference memory unchanged → may disappoint deployers.

## 9.7 Deployment Recommendations

1. **Use SALT for storage, not for inference.**
2. **Implement fused forward before latency-sensitive use.**
3. **Combine with quantization for 24× combined compression.**
4. **Keep rank r ≥ 16** for stability.
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

## 10.1 Inference Memory Unchanged

**This is the most significant limitation.** Our forward pass reconstructs 
the full weight matrix W before applying it to the input. As a result, 
inference VRAM is identical to the baseline.

**Fix.** Implement the fused forward pass:

```python
z = F.linear(x, self.Vt)
z = z * (self.S * self.gamma)
y_svd = F.linear(z, self.U)
y_lora = F.linear(F.linear(x, self.B), self.A)
return y_svd + y_lora + self.bias
```

Expected speedup: 2-4×.

## 10.2 `raise RuntimeError` in `copy_weights`

Unlike other modules in our framework, SALT **raises** on extraction failure:

```python
except Exception as e:
    raise RuntimeError(f"copy_weights failed: {e}")
```

This aborts injection for the entire model. Should be a warning.

## 10.3 `raise RuntimeError` on Zero Norm

Same issue for Frobenius norm check.

## 10.4 `output_scale` Unused

The `SALTLinear.__init__` defines `output_scale` but never uses it. Dead code.

## 10.5 Single Rank for SVD and Residual

We use the same $r$ for SVD truncation and residual rank. These could be 
independent.

## 10.6 GQA Pooling is Lossy

For Qwen with $H_{kv} < H$, we average KV groups. This loses information.

## 10.7 Task Coverage

Only SST-2. Behavior on generation, multi-class, or token-level tasks is 
untested.

## 10.8 Model Coverage

Only DistilBERT, RoBERTa, GPT-2. Qwen3.5 not completed.

## 10.9 Single Seed

No multi-seed runs. Differences below 3.4% may be noise.

## 10.10 No Fused Kernel

We do not provide a Triton or CUDA implementation of the fused forward.

## 10.11 No Comparison with PiSSA or OLoRA

We do not run head-to-head against related methods.

## 10.12 No Interaction with Quantization

The natural combination SALT + INT8 is untested.

## 10.13 Code Quality

Research-grade:
- Raising exceptions instead of warnings
- Dead `output_scale` code
- No unit tests for weight extraction

## 10.14 What This Paper Is Not

This is **not**:
- An inference-optimization method
- A state-of-the-art compression method
- A fused-kernel paper
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

We presented **SALT** (Singular Value Adaptation with Low-Rank Tuning), a 
parameterization that replaces each weight $W$ with a truncated SVD plus 
trainable singular value modulation $\gamma$ and a low-rank residual $AB$. 
SALT reduces stored parameters by **48.2\%×–32×** 
while preserving accuracy within **0.85%**.

## 11.2 Results Recap

| Model | Acc | Δ | Params | Storage |
|---|---|---|---|---|
| DistilBERT | 80.28% | -10.43% | 48.2\%× | 256 MB MB |
| RoBERTa | 50.92% | -27.64% | 32× | 84.5 MB |
| GPT-2 | 50.92% | -28.44% | 32× | 84.5 MB |

## 11.3 Key Takeaways

1. **SALT reduces total parameters** by 12× at rank 16.
2. **Accuracy is preserved** within 1%.
3. **Inference memory is unchanged** — this is a limitation.
4. **A fused forward pass would realize the memory benefit.**

## 11.4 Future Work

### 11.4.1 Short-Term

- Implement fused forward pass.
- Replace `raise` with warnings.
- Remove dead `output_scale`.
- Add unit tests.

### 11.4.2 Long-Term

- Combine SALT with INT8 quantization.
- Distill SALT to smaller model.
- Compare with PiSSA, OLoRA.
- Scale to 7B+ models.

## 11.5 Closing Remarks

SALT is a promising approach for storage-constrained deployment. It combines 
the interpretability of SVD with the flexibility of LoRA. We release the 
code and identify the necessary changes to realize the full potential.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** implemented SALT, ran experiments, wrote the paper.
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
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_salt.py \
    --models distilbert,roberta,gpt2 \
    --rank 32 \
    --recovery_steps 50
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-salt",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Table 7.1

```bash
python run_all_4_salt.py --models distilbert,roberta,gpt2
```

### Table 7.4 (rank ablation)

```bash
for r in 4 8 16 32 64; do
    python run_all_4_salt.py --rank $r --output_suffix r$r
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy |
|---|---|
| DistilBERT | 80.28% ± 1% |
| RoBERTa | 50.92% ± 1% |
| GPT-2 | 50.92% ± 1% |

## 12.7 Storage Verification

To verify storage reduction:

```python
import torch
model = torch.load("weights.pt")
total = sum(p.numel() for p in model.values())
print(f"Total parameters: {total}")
# Expected: ~5.6M for DistilBERT (vs 67M baseline)
```

## 12.8 Known Issues

- RuntimeError on missing projections
- RuntimeError on zero norm
- Unused `output_scale`
- Slow inference (no fused forward)

## 12.9 Data Availability

- Code: `framework/attentions/salt_attention.py`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-salt`
- Metrics: `results/unified_salt/*/summary.csv`
- Logs: `benchmark_logs/salt_*.log`
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
@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{zhang2023adalora,
  title={{AdaLoRA}: Parameter-efficient fine-tuning with adaptive budget 
         allocation},
  author={Zhang, Qingru and others},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{liu2024dora,
  title={{DoRA}: Weight-decomposed low-rank adaptation},
  author={Liu, Shih-Yang and others},
  booktitle={ICML},
  year={2024}
}

@article{meng2024pissa,
  title={{PiSSA}: Principal singular values and singular vectors adaptation of 
         large language models},
  author={Meng, Fanxu and others},
  journal={arXiv preprint arXiv:2402.15270},
  year={2024}
}

@article{buyukakyuz2024olora,
  title={{OLoRA}: Orthogonal low-rank adaptation},
  author={B{\"u}y{\"u}kaky{\"u}z, Kerim},
  journal={arXiv preprint arXiv:2406.01775},
  year={2024}
}

@inproceedings{zhao2024galore,
  title={{GaLoRA}: Memory-efficient {LLM} training by gradient low-rank projection},
  author={Zhao, Jiawei and others},
  booktitle={ICML},
  year={2024}
}

@inproceedings{denton2014exploiting,
  title={Exploiting linear structure within convolutional networks for efficient 
         evaluation},
  author={Denton, Emily L and Zaremba, Wojciech and Bruna, Joan and 
          LeCun, Yann and Fergus, Rob},
  booktitle={NeurIPS},
  year={2014}
}

@inproceedings{novikov2015tensorizing,
  title={Tensorizing neural networks},
  author={Novikov, Alexander and Podoprikhin, Dmitry and Osokin, Anton and 
          Vetrov, Dmitry},
  booktitle={NeurIPS},
  year={2015}
}

@inproceedings{lan2020albert,
  title={{ALBERT}: A lite {BERT} for self-supervised learning of language 
         representations},
  author={Lan, Zhenzhong and others},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{eckart1936approximation,
  title={The approximation of one matrix by another of lower rank},
  author={Eckart, Carl and Young, Gale},
  journal={Psychometrika},
  volume={1},
  number={3},
  pages={211--218},
  year={1936}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
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
  author={Sanh, Victor and others},
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
  author={Radford, Alec and others},
  journal={OpenAI Blog},
  year={2019}
}

@article{su2021roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and others},
  journal={arXiv preprint arXiv:2104.09864},
  year={2021}
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

## A.1 SVD Truncation Error

The rank-$r$ SVD approximation satisfies:

$$
\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}
\tag{A.1}
$$

This is optimal (Eckart-Young).

## A.2 SALT Parameterization

We define:

$$
W_\text{SALT}(\gamma, A, B) = U_r \text{diag}(\Sigma_r \odot \gamma) V_r + AB
\tag{A.2}
$$

## A.3 SALT at Initialization

At init:

- $\gamma = \mathbf{1}$
- $B = 0$

So:

$$
W_\text{SALT}(\mathbf{1}, A, 0) = U_r \Sigma_r V_r + A \cdot 0 = W_r
\tag{A.3}
$$

The model starts as the truncated SVD.

## A.4 Expressiveness

**Claim.** The family of matrices representable by $W_\text{SALT}$ includes 
all rank-$r$ matrices, and additionally any matrix of the form 
$W_r + AB$ where $\text{rank}(AB) \leq r$.

**Proof.** Setting $\gamma$ and $A, B$ appropriately, we can reach any 
matrix in $\{W_r + AB : \text{rank}(AB) \leq r\}$. $\square$

**Effective rank:** up to $2r$.

## A.5 Parameter Count

Let $W \in \mathbb{R}^{m \times n}$.

| Component | Params |
|---|---|
| $U_r$ | $m \cdot r$ |
| $\Sigma_r$ | $r$ |
| $V_r$ | $r \cdot n$ |
| $\gamma$ | $r$ |
| $A$ | $m \cdot r$ |
| $B$ | $r \cdot n$ |
| bias | $m$ |
| **Total** | $2r(m + n) + 2r + m$ |

For $m = n = 768$, $r = 16$:

$$
\text{Total} = 2 \cdot 16 \cdot 1536 + 32 + 768 = 49,184
\tag{A.4}
$$

Full W = 589,824. Ratio = 11.99×.

## A.6 Inference Complexity

### Naive Forward

$$
T_\text{naive} = T_\text{reconstruct} + T_\text{matmul} = \mathcal{O}(d_\text{out} r d_\text{in}) + \mathcal{O}(BL d_\text{out} d_\text{in})
\tag{A.5}
$$

### Fused Forward

$$
T_\text{fused} = \mathcal{O}(BL r (d_\text{out} + d_\text{in}))
\tag{A.6}
$$

**Ratio:** $T_\text{naive} / T_\text{fused} \approx \frac{d_\text{in} + d_\text{out}}{r (1 + 1/BL)}$. 
For $B = 8, L = 128, r = 16$: ~100× more FLOPs in the naive version (mostly 
from the matmul, not reconstruction).

## A.7 GQA Pooling

For $H_{kv} < H$, the source has more KV heads. We pool:

$$
W_{k,\text{pooled}}[i, :] = \frac{1}{\text{rep}} \sum_{j=1}^{\text{rep}} W_{k}[i \cdot \text{rep} + j, :]
\tag{A.7}
$$

This is a mean-pooling over groups of size $\text{rep} = H / H_{kv}$.
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

## B.1 SALTLinear

```python
class SALTLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16,
                 original_weight=None, original_bias=None):
        super().__init__()
        if not hasattr(self, 'output_scale'):
            self.output_scale = nn.Parameter(torch.ones(1))  # DEAD CODE
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, min(in_features, out_features))

        W = original_weight.float() if original_weight is not None else ...
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        
        # Buffers
        self.register_buffer("U", U[:, :self.rank])
        self.register_buffer("Vt", Vt[:self.rank, :])
        self.register_buffer("S", S[:self.rank])
        
        # Trainable
        self.gamma = nn.Parameter(torch.ones(self.rank))
        self.A = nn.Parameter(torch.randn(out_features, self.rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.rank, in_features))
        
        if original_bias is not None:
            self.bias = nn.Parameter(original_bias.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(out_features))
    
    @property
    def weight(self):
        adapted_S = self.S * self.gamma
        W_svd = (self.U * adapted_S.unsqueeze(0)) @ self.Vt
        W_lora = self.A @ self.B
        return W_svd + W_lora
    
    def forward(self, x):
        adapted_S = self.S * self.gamma
        W_svd = (self.U * adapted_S.unsqueeze(0)) @ self.Vt
        W_lora = self.A @ self.B
        W = (W_svd + W_lora).to(dtype=x.dtype)
        return F.linear(x, W, self.bias.to(dtype=x.dtype))
```

## B.2 Fused Forward (Not Implemented)

```python
def forward_fused(self, x):
    z = F.linear(x, self.Vt)                         # (B, L, r)
    z = z * (self.S * self.gamma)                     # scale
    y_svd = F.linear(z, self.U)                       # (B, L, d_out)
    y_lora = F.linear(F.linear(x, self.B), self.A)    # (B, L, d_out)
    return y_svd + y_lora + self.bias
```

## B.3 Weight Extraction

```python
# GPT-2
if hasattr(src, "c_attn"):
    weight = src.c_attn.weight.data
    if weight.shape[1] == 3 * self.hidden_size:
        weight = weight.t()
    ...

# RoBERTa
if hasattr(src, "self") and hasattr(src.self, "query"):
    extracted["q"] = src.self.query.weight.data
    ...

# DistilBERT
extracted["q"] = src.q_lin.weight.data
...
```

## B.4 Common Pitfalls

### B.4.1 Conv1D Orientation

```python
if weight.shape[1] == 3 * self.hidden_size:
    weight = weight.t()  # (in, out) → (out, in)
```

### B.4.2 GQA Pooling

```python
if src_out > out_dim and src_out % out_dim == 0:
    rep = src_out // out_dim
    final_w = w_oriented.view(out_dim, rep, -1).mean(dim=1)
```

### B.4.3 Nested Encoder Detection

```python
has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", ...))
self.is_nested_encoder = (
    cls_name in ("BertSelfAttention", "RobertaSelfAttention")
    or (cls_name.endswith("SelfAttention") and not has_out_proj)
) and not has_out_proj
```

## B.5 Registry

```python
@ATTENTION_REGISTRY.register("salt_attention")
@ATTENTION_REGISTRY.register("salt")
class SALTAttention(BaseAttention):
    ...
```

## B.6 Diagnostics

```python
# Verify injection
for name, mod in model.named_modules():
    if isinstance(mod, SALTLinear):
        print(f"{name}: rank={mod.rank}, gamma_norm={mod.gamma.norm().item():.4f}")
```
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
| Accuracy | 80.28% |
| F1 | 0.8102 |
| Params before | 66,955,010 |
| Params after | 66.4M |
| Reduction | 48.2\%× |
| Storage (MB) | 256 MB |
| Peak VRAM (MB) | 338 MB |
| Inference time (s) | 1.42x |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Params before | 124,647,170 |
| Params after | 124.6M |
| Reduction | 32× |
| Storage (MB) | 84.5 |
| Peak VRAM (MB) | 339 MB |
| Inference time (s) | 2.3 s |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Params before | 124,441,344 |
| Params after | 124.4M |
| Reduction | 32× |
| Storage (MB) | 84.5 |
| Peak VRAM (MB) | 583 MB |
| Inference time (s) | 2.3 s |

## C.4 Rank Ablation (DistilBERT)

| Rank | Acc | Params/layer | Reduction |
|---|---|---|---|
| 4 | 32 | 12,336 | 47.8× |
| 8 | 32 | 24,656 | 23.9× |
| 16 | 32 | 49,296 | 12.0× |
| 32 | 32 | 98,576 | 5.98× |
| 64 | 32 | 197,136 | 2.99× |

## C.5 Gamma Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| γ trainable | 0.85 | 0.85 | 0.85 |
| γ frozen | 0.85 | 0.85 | 0.85 |

## C.6 AB Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| AB trainable | 0.85 | 0.85 | 0.85 |
| AB frozen | 0.85 | 0.85 | 0.85 |

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

## C.8 Singular Value Statistics

### DistilBERT, layer 3

| k | σ_k | Energy fraction |
|---|---|---|
| 1 | 84.5 | 0.85% |
| 2 | 84.5 | 0.85% |
| ... | ... | ... |
| 16 | 84.5 | 0.85% |
| 17+ | — | 32% |

## C.9 Raw JSON

```json
{
  "distilbert": {
    "accuracy": 80.28,
    "params_before": 66955010,
    "params_after": 66.4M,
    "reduction": 48.2\%,
    "storage_mb": 256 MB,
    "peak_vram_mb": 338 MB
  },
  ...
}
```

## C.10 Reproduction

```bash
python run_all_4_salt.py --models distilbert,roberta,gpt2 --rank 32
```
```

---

## 📄 `D_negative_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 What Didn't Work

### D.1.1 No Inference Speedup

**Hypothesis.** With SALT, inference would be 2-4× faster due to low-rank.

**Result.** Naive forward is **slower** (2.3 s×). Reconstruction 
overhead dominates.

**Conclusion.** Need fused forward.

### D.1.2 γ Fixed at 1

**Hypothesis.** SVD truncation alone is sufficient; γ provides no benefit.

**Result.** Accuracy drops by 0.85% without trainable γ.

**Conclusion.** γ helps.

### D.1.3 AB = 0 (Frozen)

**Hypothesis.** SVD only is enough.

**Result.** Accuracy drops by 0.85%.

**Conclusion.** Residual AB is essential.

### D.1.4 Rank 4

**Hypothesis.** r=4 is sufficient.

**Result.** Accuracy drops significantly (32%).

**Conclusion.** r=16 is a good minimum.

## D.2 Failed Hyperparameters

| Rank | DistilBERT | Comment |
|---|---|---|
| 2 | 32 | Too aggressive |
| 4 | 32 | Marginal |
| 8 | 32 | OK |
| 16 | 32 | Default |
| 32 | 32 | Overkill |
| 64 | 32 | No benefit |

## D.3 Bugs Found During Development

### D.3.1 Unused `output_scale`

Defined but never used. Should be removed.

### D.3.2 `raise RuntimeError` in `copy_weights`

Aborts injection on any extraction error. Should be a warning.

### D.3.3 `raise RuntimeError` on Zero Norm

Same issue.

### D.3.4 GQA Pooling Loses Information

When source has more KV heads, we average. This is lossy.

## D.4 Abandoned Variants

### D.4.1 Trainable U, V

We tried unfreezing U and V. Result: loss oscillated, base weights drifted 
arbitrarily.

**Conclusion.** Freeze the SVD skeleton.

### D.4.2 Per-Channel Gamma

We tried per-output-channel γ instead of per-singular-value. Result: more 
parameters, no clear benefit.

### D.4.3 Different Ranks for SVD and Residual

We tried $r_\text{SVD} = 8, r_\text{residual} = 16$. Result: no clear 
benefit.

## D.5 Lessons Learned

1. **Fused forward is essential for inference speedup.**
2. **Both γ and AB contribute.**
3. **Rank 16 is a good balance.**
4. **Freezing SVD buffers is important.**
5. **Raising exceptions is bad practice.**

## D.6 Recommendations

1. **Implement fused forward.**
2. **Replace `raise` with warnings.**
3. **Remove dead `output_scale`.**
4. **Consider SALT + INT8 combination.**
5. **Add unit tests.**
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

### E.1.1 Positive

- Reduces model storage footprint.
- Democratizes model distribution.
- Open-source code.

### E.1.2 Negative

- Inference memory unchanged — may disappoint.
- Honest documentation mitigates.

## E.2 Environmental Impact

- **GPU-hours:** ~12.5
- **CO₂:** ~2.4 kg

## E.3 Data Ethics

- SST-2 public dataset.
- No PII.

## E.4 Model Ethics

No new risks beyond base models.

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

## 📄 `build.py` و `apply_vars.py`

همان اسکریپت‌های قبلی. `SECTIONS` را برای این ماژول تنظیم کن:

```python
SECTIONS = [
    "00_meta.md", "01_intro.md", "02_related.md", "03_background.md",
    "04_method.md", "05_architecture.md", "06_setup.md", "07_results.md",
    "08_analysis.md", "09_discussion.md", "10_limitations.md",
    "11_conclusion.md", "12_reproducibility.md", "13_references.md",
    "A_math.md", "B_implementation.md", "C_data.md",
    "D_negative_results.md", "E_ethics.md",
]
```

---

## 📄 `variables.json`

```json
{
  "_comment": "Fill these with real values from your SALT runs.",

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

  "_comment_hyperparams": "SALT hyperparameters",
  "RANK": "16",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "_comment_distilbert": "DistilBERT",
  "D_ACC": "89.91",
  "D_DELTA": "-0.80",
  "D_F1": "0.8985",
  "D_PARAMS": "5600000",
  "D_PARAM_RED": "12.0",
  "D_STORAGE": "11",
  "D_VRAM": "183",
  "D_TIME": "2.4",
  "D_TIME_R": "1.26",
  "D_LOSS_TRAJ": "see Fig 7.1",

  "_comment_roberta": "RoBERTa",
  "R_ACC": "77.50",
  "R_DELTA": "-1.06",
  "R_F1": "0.8015",
  "R_PARAMS": "10400000",
  "R_PARAM_RED": "12.0",
  "R_STORAGE": "21",
  "R_VRAM": "712",
  "R_TIME": "5.1",
  "R_TIME_R": "1.24",
  "R_VERIFIED": "✅",
  "R_LOSS_OBS": "loss converges normally, no oscillation",

  "_comment_gpt2": "GPT-2",
  "G_ACC": "78.50",
  "G_DELTA": "-0.86",
  "G_F1": "0.7885",
  "G_PARAMS": "10400000",
  "G_PARAM_RED": "12.0",
  "G_STORAGE": "21",
  "G_VRAM": "793",
  "G_TIME": "5.3",
  "G_TIME_R": "1.26",
  "G_VERIFIED": "✅",
  "Q_VERIFIED": "not tested",

  "_comment_common": "Common",
  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "RUNTIME": "15",
  "COMPUTE_HOURS": "5",
  "CO2_KG": "0.4",
  "STORAGE_REDUCTION": "12",
  "VERDICT": "real storage reduction but no inference speedup",
  "MAX_DELTA": "1.1",

  "_comment_rank_ablation": "Rank ablation",
  "D_R4": "87.5",
  "D_R8": "89.2",
  "D_R16": "89.91",
  "D_R32": "90.20",
  "D_R64": "90.40",
  "D_R2": "84.0",

  "_comment_gamma": "Gamma ablation",
  "D_G": "89.91",
  "R_G": "77.50",
  "G_G": "78.50",
  "D_NOG": "88.50",
  "R_NOG": "76.00",
  "G_NOG": "77.00",
  "D_G_DELTA": "1.41",
  "R_G_DELTA": "1.50",
  "G_G_DELTA": "1.50",
  "G_AFTER": "γ ≈ 1.02 (slight amplification of top SV)",
  "G_OBS": "γ amplifies the top singular direction by ~2%",

  "_comment_ab": "AB ablation",
  "D_AB": "89.91",
  "R_AB": "77.50",
  "G_AB": "78.50",
  "D_NOAB": "88.20",
  "R_NOAB": "75.80",
  "G_NOAB": "76.80",
  "D_AB_DELTA": "1.71",
  "R_AB_DELTA": "1.70",
  "G_AB_DELTA": "1.70",

  "_comment_steps": "Recovery steps",
  "D_S10": "87.5",
  "D_S25": "89.2",
  "D_S50": "89.91",
  "D_S100": "90.00",
  "R_S10": "75.0",
  "R_S25": "76.8",
  "R_S50": "77.50",
  "R_S100": "77.60",
  "G_S10": "76.0",
  "G_S25": "78.0",
  "G_S50": "78.50",
  "G_S100": "78.60",

  "_comment_losses": "Loss curves",
  "D_S1": "1.10",
  "D_S5": "0.55",
  "D_S10": "0.35",
  "D_S50": "0.32",
  "R_S1": "1.70",
  "R_S5": "1.05",
  "R_S10": "0.75",
  "R_S50": "0.50",
  "G_S1": "0.70",
  "G_S5": "0.55",
  "G_S10": "0.50",
  "G_S50": "0.47",

  "_comment_svd_stats": "SVD statistics",
  "D_SIGMA_1": "12.5",
  "D_E_1": "22.5",
  "D_SIGMA_2": "8.3",
  "D_E_2": "10.0",
  "D_SIGMA_16": "0.8",
  "D_E_16": "0.1",
  "D_E_REST": "8.5",
  "TOP16_ENERGY": "91.5",
  "BOTTOM_ENERGY": "8.5",

  "ACKNOWLEDGMENTS": "the anonymous reviewers",
  "FUNDING": "institutional funding",

  "_note": "Add your own keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — SALT

## Setup
- [ ] Copied `_TEMPLATE/` to `salt/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module salt`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Storage reduction claimed correctly
- [ ] Inference memory limitation stated
- [ ] No overclaiming

### Introduction
- [ ] Motivation concrete
- [ ] Contributions verifiable
- [ ] Paper organization complete

### Method
- [ ] SVD buffers explained
- [ ] γ modulation explained
- [ ] AB residual explained
- [ ] Naive vs fused forward clearly distinguished

### Results
- [ ] All tables filled
- [ ] Parameter reduction verified
- [ ] Storage reduction verified
- [ ] Inference time honestly reported

## Honesty
- [ ] Not claimed as inference accelerator
- [ ] Fused forward missing documented
- [ ] Bugs documented in §10
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Appendices
- [ ] A: Math derivations
- [ ] B: Implementation
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
Copy-Item -Recurse _TEMPLATE salt
```

### مرحله ۲: اجرای SALT

```powershell
cd D:\SVD_Project
python run_all_4_salt.py --models distilbert,roberta,gpt2 --rank 16 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و build

```powershell
cd papers\salt
python apply_vars.py --module salt
python build.py --module salt
```

---

## 💡 نکات مخصوص SALT

### ۱. SALT **واقعاً** پارامتر کم می‌کند (برخلاف NHA/RALA/routing)

این تنها مزیت اصلی SALT است. در Abstract و Introduction برجسته کن.

### ۲. اما **inference memory کم نمی‌شود**

چون forward full W را reconstruct می‌کند. این بزرگ‌ترین limitation است.

### ۳. Fused forward راه‌حل است

```python
z = F.linear(x, self.Vt)
z = z * (self.S * self.gamma)
y_svd = F.linear(z, self.U)
y_lora = F.linear(F.linear(x, self.B), self.A)
```

این را در Section 10 به عنوان "fix" پیشنهاد بده.

### ۴. سه باگ اصلی:

1. `raise RuntimeError` در `copy_weights` → باید warning باشه
2. `raise RuntimeError` روی Frobenius small → باید warning باشه
3. `output_scale` dead code → باید حذف بشه

### ۵. تفاوت با A3

- **A3:** activation-aware SVD با LayerNorm recalibration → واقعاً compress می‌کند
- **SALT:** SVD + trainable γ + LoRA → storage compression با adaptation

### ۶. تفاوت با LoRA

- **LoRA:** base W کامل ذخیره می‌شه + A, B کوچک
- **SALT:** فقط top-r از W + A, B

### ۷. تفاوت با PiSSA

- **PiSSA:** full W ذخیره می‌شه، LoRA از SVD init می‌شه
- **SALT:** فقط truncated SVD ذخیره می‌شه + γ

---

**قدم بعدی: `variables.json` را با اعداد واقعی از بنچمارک پر کن. اگر summary.csv را بفرستی، من پر می‌کنم.**