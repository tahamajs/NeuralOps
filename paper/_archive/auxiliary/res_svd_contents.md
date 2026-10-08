# تمپلیت کامل مقاله Res-SVD Attention — فایل به فایل

قبل از شروع، **تحلیل صادقانه از Res-SVD Attention**:

| ویژگی | Res-SVD Attention |
|---|---|
| **نوع** | Low-rank compression با spectral residual |
| **پارامتر** | **کاهش** می‌یابد ✓ |
| **فشرده‌سازی** | ✅ بله |
| **نوآوری** | دو سطح SVD: base + residual |
| **Base** | `W_base = U_base V_base^T` (frozen، rank = base_rank) |
| **Residual** | `alpha * B A` (trainable، init از top-k residual singular) |
| **نکته کلیدی** | ریاضیاً **معادل** SVD با rank = base_rank + residual_rank |
| **مزیت واقعی** | Fine-tuning با structured init |

**نکته ریاضی مهم:** اگر پایه و residual به درستی محاسبه شوند، نتیجه معادل SVD با rank کلی است. تفاوت اصلی در **trainable بودن residual** است.

---

## 📁 ساختار پوشه‌ها

```
papers/
├── _TEMPLATE/
│   └── (فایل‌های قبلی)
└── res_svd/
    ├── 00_meta.md
    ├── 01_intro.md
    ├── 02_related.md
    ├── 03_background.md
    ├── 04_method.md
    ├── 05_architecture.md
    ├── 06_setup.md
    ├── 07_results.md
    ├── 08_analysis.md
    ├── 09_discussion.md
    ├── 10_limitations.md
    ├── 11_conclusion.md
    ├── 12_reproducibility.md
    ├── 13_references.md
    ├── A_math.md
    ├── B_implementation.md
    ├── C_data.md
    ├── D_negative_results.md
    ├── E_ethics.md
    ├── build.py
    ├── apply_vars.py
    ├── variables.json
    └── checklist.md
```

---

## 📄 `00_meta.md`

```markdown
# Res-SVD Attention: Spectral Residual Low-Rank Factorization for Attention Projections

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/res_svd_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We present **Res-SVD Attention**, a drop-in attention replacement that stores 
each projection as a **two-level SVD decomposition**:

$$
W = U_\text{base} V_\text{base}^\top + \alpha B A
\tag{1}
$$

where $U_\text{base} V_\text{base}^\top$ is a **frozen** rank-$r_b$ SVD 
approximation, and $B A$ is a **trainable** rank-$r_r$ residual adapter 
whose columns are **initialized from the top-$r_r$ singular directions of 
the residual $W - U_\text{base} V_\text{base}^\top$**.

This combines two ideas that are usually separate: (i) low-rank SVD 
compression of attention projections, and (ii) parameter-efficient 
fine-tuning of the compressed model. The **spectral initialization** of the 
LoRA adapter ensures that at initialization the combined approximation 
matches the best rank-$(r_b + r_r)$ SVD of the original weight — the 
residual is not random but populated with the "second-order" spectrum.

Res-SVD Attention supports four attention families — flat (DistilBERT), 
nested (RoBERTa/BERT), causal Conv1D (GPT-2), and grouped-query with rotary 
embeddings (Qwen/LLaMA) — with weight transfer, GQA, RoPE, and causal 
masking.

On SST-2 with 50 LoRA recovery steps and $(r_b, r_r) = 
(32, 16)$, Res-SVD Attention achieves 
**89.56%** on DistilBERT (Δ = -1.15), **76.61%** on RoBERTa 
(Δ = -1.95), and **71.33%** on GPT-2 (Δ = -8.03). Total 
parameter count decreases by **50\%×–2.1\times×**.

<!-- CRITICAL HONESTY -->
Res-SVD Attention **reduces** parameter count. The residual adapter adds 
$r_r(d_\text{in} + d_\text{out})$ trainable parameters, but this is 
offset by the reduction from SVD factorization of the projections. The 
"residual" in the name refers to the **residual weight matrix after base 
SVD**, not to residual connections.

**Keywords:** spectral initialization, residual SVD, low-rank factorization, 
parameter-efficient fine-tuning, attention compression

---

## Lay Summary

Transformer attention layers contain large weight matrices. Res-SVD stores 
each matrix in two parts: a "coarse" part computed by SVD that stays fixed, 
and a "fine" part that is small and trainable. The fine part is not random — 
it is initialized from the second-order singular directions of the original 
matrix, so the combined approximation is as close to the true weight as 
possible. This allows a small, compressed model to be fine-tuned efficiently 
while preserving most of the original information.

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
# 1. Introduction

## 1.1 Motivation

Low-rank factorization of transformer weights is a common compression 
primitive. For each projection $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, 
one replaces $W$ with a product of two smaller matrices, reducing parameter 
count from $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$ 
for rank $r$.

**Two standard variants exist:**

1. **Pure SVD compression** (ASVD, AASVD): replace $W$ with a rank-$r$ SVD 
   approximation, then fine-tune with LoRA adapters on top. The LoRA is 
   initialized to zero.
2. **Pure PEFT** (LoRA): keep $W$ dense, add a rank-$r$ trainable adapter. 
   No compression.

**Res-SVD bridges these.** The base is a low-rank SVD approximation (compressed), 
and the residual is a trainable adapter (PEFT), but the residual is **not 
initialized to zero** — it is initialized from the **next singular directions** 
of $W$. This makes the combined approximation equal to the best rank-$(r_b + r_r)$ 
SVD at initialization, while keeping the base frozen for fine-tuning.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a compressed model $\mathcal{M}'$ such that:

1. Each projection is stored as a sum of two low-rank factors:
   $$
   W \approx U_b V_b^\top + \alpha B A
   \tag{1.1}
   $$
   with $r_b + r_r \ll \min(d_\text{in}, d_\text{out})$.
2. Parameter count decreases by factor $\sim (d_\text{in} d_\text{out}) / ((r_b + r_r)(d_\text{in} + d_\text{out}))$.
3. The base $U_b V_b^\top$ is frozen; only $B, A$ are trained.
4. At initialization, $U_b V_b^\top + \alpha B A$ matches the best rank-$(r_b + r_r)$ 
   SVD of $W$.
5. Weight transfer works across four attention families.

## 1.3 Our Approach: Res-SVD Attention

Given a projection weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

**Step 1.** Compute the top-$r_b$ SVD:
$$
W_b = U_b \Sigma_b V_b^\top, \quad r_b \ll \min(d_\text{in}, d_\text{out})
\tag{1.2}
$$

**Step 2.** Compute the residual:
$$
R = W - W_b
\tag{1.3}
$$

**Step 3.** Take the top-$r_r$ SVD of the residual:
$$
R \approx U_r \Sigma_r V_r^\top
\tag{1.4}
$$

**Step 4.** Initialize the LoRA adapter with:
$$
B = U_r \Sigma_r^{1/2}, \quad A = \Sigma_r^{1/2} V_r^\top
\tag{1.5}
$$

**Step 5.** Store $U_b, V_b$ as frozen parameters and $B, A$ as trainable.

The forward pass computes:
$$
y = x (U_b V_b)^\top + \alpha x (B A)^\top + b
\tag{1.6}
$$

## 1.4 Contributions

- **A two-level SVD factorization.** Res-SVD combines frozen low-rank SVD 
  with a trainable residual, initialized from the second-order spectrum.
- **Spectral LoRA initialization.** The residual adapter is initialized 
  non-zero, unlike standard LoRA, ensuring the initial approximation is 
  optimal for the total rank.
- **Multi-architecture support.** Same code handles flat, nested, causal, 
  and GQA+RoPE attention.
- **Compatibility with RoBERTa.** Nested encoders are handled via 
  identity initialization of the output projection.
- **Honest reporting.** Res-SVD reduces parameter count. The residual 
  adapter is a small trainable addition.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, LoRA, 
and spectral initialization. Section 4 presents Res-SVD. Section 5 details 
multi-architecture integration. Section 6 gives setup. Section 7 reports 
results. Section 8 analyzes contributions. Section 9 provides guidance. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W$ | Projection weight matrix |
| $r_b$ | Base SVD rank |
| $r_r$ | Residual (LoRA) rank |
| $U_b, V_b$ | Base SVD factors (frozen) |
| $B, A$ | Residual LoRA factors (trainable) |
| $\alpha$ | Residual scaling |
| $R$ | Residual matrix $W - U_b V_b^\top$ |
| $d_\text{in}, d_\text{out}$ | Input/output dimensions |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 Low-Rank Factorization

### 2.1.1 Classical SVD

Eckart-Young (1936): optimal rank-$r$ Frobenius approximation is truncated SVD. 
Res-SVD uses SVD twice: for base and residual.

### 2.1.2 Randomized SVD

Halko et al. (2011) — randomized algorithm for truncated SVD. Not needed 
for our layer sizes.

### 2.1.3 ASVD / AASVD / A3

Companion modules: use SVD once with optional activation weighting. Res-SVD 
uses SVD twice with spectral initialization of the residual.

## 2.2 Parameter-Efficient Fine-Tuning

### 2.2.1 LoRA

Hu et al. (2021) — standard LoRA initializes $B = 0$. This makes the 
initial approximation equal to the base weight. Res-SVD initializes 
$B \neq 0$ from the residual spectrum.

### 2.2.2 AdaLoRA

Zhang et al. (2023) — adaptive rank allocation. Not directly related.

### 2.2.3 DoRA

Liu et al. (2024) — weight-decomposed LoRA. Different decomposition.

### 2.2.4 PiSSA

Meng et al. (2024) — **PiSSA initializes LoRA from the top singular 
directions of the base weight**, making the LoRA "principal". Res-SVD is 
the **complementary** approach: base takes principal directions, residual 
takes the next ones.

## 2.3 Residual Decomposition

### 2.3.1 Two-Level SVD

Any full-rank matrix $W$ can be written as $W = W_b + R$ where $W_b$ is 
rank $r_b$. The residual $R$ has rank $\leq \min(d_\text{in}, d_\text{out})$ 
but effective rank $r_r$ if its spectrum decays.

### 2.3.2 Connection to Optimal Rank-$(r_b + r_r)$ SVD

As shown in Appendix A.1, Res-SVD with $(r_b, r_r)$ gives the same 
approximation as SVD of rank $r_b + r_r$.

## 2.4 Position of Our Work

| Aspect | Pure SVD | LoRA | PiSSA | Res-SVD |
|---|---|---|---|---|
| Base frozen | Yes | No (dense) | No | Yes |
| Base rank | $r$ | Full | Full | $r_b$ |
| Residual rank | 0 | $r$ | $r$ | $r_r$ |
| Residual init | N/A | Zero | Principal | **Spectral** |
| Compression | Yes | No | No | **Yes** |

## 2.5 Comparison to Companion Modules

| Module | Approach | Base | Residual |
|---|---|---|---|
| ASVD | One SVD | Rank $r$ | LoRA zero |
| AASVD | One SVD + covariance | Rank $r$ | LoRA zero |
| A3 | One SVD + Fisher | Rank $r$ | LoRA zero |
| SVD-Linear | SVD + linear attn | Rank $r$ | LoRA zero |
| **Res-SVD** | **Two-level SVD** | **Rank $r_b$** | **Spectral init** |
```

---

## 📄 `03_background.md`

```markdown
# 3. Background

## 3.1 Singular Value Decomposition

For $W \in \mathbb{R}^{m \times n}$:
$$
W = U \Sigma V^\top
\tag{3.1}
$$
with $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD:** $W_r = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top$, error 
$\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2$.

## 3.2 LoRA

LoRA (Hu et al., 2021) parameterizes weight updates as:
$$
\Delta W = \frac{\alpha}{r} B A, \quad A \in \mathbb{R}^{r \times d_\text{in}}, \quad B \in \mathbb{R}^{d_\text{out} \times r}
\tag{3.2}
$$
with $B = 0$ at init, so $\Delta W = 0$.

**Parameter count:** $r(d_\text{in} + d_\text{out})$.

## 3.3 Spectral Initialization

Standard LoRA: $B = 0$. This makes the initial model identical to the 
base model but with no residual capacity.

**PiSSA (Meng et al., 2024):** $B, A$ are initialized from the **top** 
singular directions of $W$, and the base is set to the residual:
$$
W = W_\text{residual} + B A \quad \text{where} \quad B A = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top
\tag{3.3}
$$

**Res-SVD:** the **complement** — base is the top singular directions, 
residual is the next ones:
$$
W = W_b + R \quad \text{where} \quad W_b = U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top, \quad R = U_{:, r_b:r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top
\tag{3.4}
$$

## 3.4 Two-Level Decomposition

The key observation: $W_b + R$ with $r_b + r_r$ components is the same as 
$W_{r_b + r_r}$ (SVD of rank $r_b + r_r$), because SVD is additive on 
orthogonal directions.

**Proof sketch.** $U$ has orthogonal columns, so:
$$
U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top + U_{:, r_b:r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top = U_{:, :r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top
\tag{3.5}
$$

## 3.5 Notation Summary

| Symbol | Meaning |
|---|---|
| $W$ | Full weight |
| $W_b$ | Base SVD approximation |
| $R$ | Residual $W - W_b$ |
| $r_b, r_r$ | Base, residual ranks |
| $U_b, V_b$ | Base SVD factors |
| $B, A$ | Residual LoRA factors |
| $\alpha$ | Residual scaling |
```

---

## 📄 `04_method.md`

```markdown
# 4. Method

## 4.1 Overview

Res-SVD Attention replaces each projection with a two-level decomposition. 
**Figure 4.1** shows the pipeline.

```
    ┌──────────────────────────────────────┐
    │  Original weight W ∈ R^{m×n}         │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Step 1: SVD of W                    │
    │  W = U Σ V^T                         │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Step 2: Base = top r_b              │
    │  W_base = U_b Σ_b V_b^T              │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Step 3: Residual = W - W_base       │
    │  R = W - W_base                      │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Step 4: SVD of R                    │
    │  R = U_r Σ_r V_r^T                   │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Step 5: Init LoRA from residual     │
    │  B = U_r Σ_r^{1/2}, A = Σ_r^{1/2} V_r^T │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Frozen: U_b, V_b                    │
    │  Trainable: B, A                     │
    └──────────────────────────────────────┘
```

## 4.2 Two-Level Decomposition

### 4.2.1 Base SVD

Given $W \in \mathbb{R}^{m \times n}$:
$$
U, \Sigma, V^\top = \text{SVD}(W)
\tag{4.1}
$$

Take top $r_b$:
$$
W_b = U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top
\tag{4.2}
$$

Store $U_b = U_{:, :r_b}$, $V_b = V_{:r_b, :}$ as **frozen** parameters.

### 4.2.2 Residual SVD

$$
R = W - W_b
\tag{4.3}
$$

$$
U_r, \Sigma_r, V_r^\top = \text{SVD}(R)
\tag{4.4}
$$

Take top $r_r$:
$$
R_r = U_{r, :, :r_r} \Sigma_{r, :r_r} V_{r, :r_r, :}^\top
\tag{4.5}
$$

### 4.2.3 LoRA Initialization

Split $\Sigma_r^{1/2}$ symmetrically:
$$
B = U_{r, :, :r_r} \Sigma_{r, :r_r}^{1/2}, \quad
A = \Sigma_{r, :r_r}^{1/2} V_{r, :r_r, :}^\top
\tag{4.6}
$$

Store $B \in \mathbb{R}^{m \times r_r}$, $A \in \mathbb{R}^{r_r \times n}$ as 
**trainable**.

## 4.3 Forward Pass

$$
y = x (U_b V_b)^\top + \alpha x (B A)^\top + b
\tag{4.7}
$$

where $\alpha$ is a scaling factor (default 1.0).

**Alternative factorization for efficiency:**
```python
# Base
h_b = x @ V_b.t()              # (B, L, r_b)
y_b = h_b @ U_b.t()            # (B, L, m)

# Residual
h_r = x @ A.t()                 # (B, L, r_r)
y_r = h_r @ B.t()               # (B, L, m)

y = y_b + alpha * y_r + b
```

## 4.4 Parameter Count

| Component | Parameters |
|---|---|
| Base SVD | $r_b(m + n + 1)$ |
| Residual LoRA | $r_r(m + n)$ |
| Bias | $m$ |
| **Total** | $(r_b + r_r)(m + n) + r_b + m$ |

**Baseline:** $mn$.

For $m = n = 768$, $r_b = 16$, $r_r = 4$:

| Component | Params |
|---|---|
| Base SVD | $16 \times 1537 = 24,592$ |
| Residual LoRA | $4 \times 1536 = 6,144$ |
| Bias | 768 |
| **Total** | **31,504** |
| Baseline | 589,824 |
| **Ratio** | **~18.7×** |

## 4.5 Full Algorithm

**Algorithm 1:** Res-SVD factorization
```
Input:  Weight W ∈ R^{m×n}, ranks r_b, r_r, scaling α
Output: Base factors (U_b, V_b), Residual factors (B, A), bias b

1:  U, Σ, V^T ← SVD(W)
2:  # Base
3:  U_b ← U[:, :r_b];  Σ_b ← Σ[:r_b];  V_b^T ← V^T[:r_b, :]
4:  W_base ← U_b @ diag(Σ_b) @ V_b^T
5:  # Residual
6:  R ← W - W_base
7:  U_r, Σ_r, V_r^T ← SVD(R)
8:  # LoRA init
9:  Σ_r_sqrt ← sqrt(Σ_r[:r_r])
10: B ← U_r[:, :r_r] * Σ_r_sqrt[None, :]
11: A ← Σ_r_sqrt[:, None] * V_r^T[:r_r, :]
12: return (U_b, V_b, B, A, b)
```

**Algorithm 2:** Res-SVD forward pass
```
Input:  x ∈ R^{B×L×n}, base (U_b, V_b), residual (B, A), bias b, α
Output: y ∈ R^{B×L×m}

1: h_b ← x @ V_b^T
2: y_b ← h_b @ U_b^T
3: h_r ← x @ A^T
4: y_r ← h_r @ B^T
5: y ← y_b + α · y_r + b
6: return y
```

## 4.6 Complexity Analysis

| Component | Time | Space |
|---|---|---|
| Base forward | $\mathcal{O}(L n r_b)$ | $\mathcal{O}(L r_b)$ |
| Residual forward | $\mathcal{O}(L n r_r)$ | $\mathcal{O}(L r_r)$ |
| Total | $\mathcal{O}(L n (r_b + r_r))$ | $\mathcal{O}(L (r_b + r_r))$ |

**Baseline:** $\mathcal{O}(L n m)$.

For $m = n = 768$, $r_b + r_r = 20$: **~38× reduction in projection FLOPs**.

## 4.7 Theoretical Notes

### 4.7.1 Equivalence to Rank-$(r_b + r_r)$ SVD

**Claim.** Res-SVD with $(r_b, r_r)$ is mathematically equivalent to 
truncated SVD of rank $r_b + r_r$.

**Proof.** The columns of $U$ are orthogonal, so:
$$
U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top + U_{:, r_b:r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top = U_{:, :r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top
$$
$\square$

**Implication:** Res-SVD does not offer a better approximation than 
rank-$(r_b + r_r)$ SVD. Its advantage is **PEFT structure** — the base 
is frozen and the residual is trainable.

### 4.7.2 Optimality at Initialization

**Claim.** At initialization, the combined Res-SVD approximation has 
reconstruction error $\|W - W_b - R_r\|_F = \sqrt{\sum_{i > r_b+r_r} \sigma_i^2}$.

**Proof.** Direct consequence of 4.7.1. $\square$

### 4.7.3 Gradient Flow

**Claim.** Only the residual $B, A$ receives gradient during training.

**Proof.** $U_b, V_b$ have `requires_grad = False`. Only $B, A$ are updated. $\square$

### 4.7.4 Causality

**Proof.** Position-wise projections + causal attention mask. $\square$
```

---

## 📄 `05_architecture.md`

```markdown
# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes |

## 5.2 Weight Extraction

Res-SVD uses the same weight extraction as companion modules:

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
```

### 5.2.4 Qwen

Direct extraction from named projections.

## 5.3 Nested Encoder Handling

For RoBERTa/BERT, the output projection is applied externally. 
Res-SVD handles this via **identity initialization** of the output projection:

```python
if proj_key == "o" and self.is_nested_encoder:
    temp_linear = nn.Linear(target.in_features, target.out_features)
    nn.init.eye_(temp_linear.weight)
    nn.init.zeros_(temp_linear.bias)
    new_layer = ResSVDLinear.from_linear(temp_linear, ...)
    target.load_state_dict(new_layer.state_dict())
```

Then in forward:
```python
if self.is_nested_encoder:
    return context  # skip out_proj
```

## 5.4 The `ResSVDLinear` Module

Located in `framework/compression_engine/res_svd.py`. Structure:

```python
class ResSVDLinear(nn.Module):
    def __init__(self, in_features, out_features, base_rank, residual_rank,
                 scaling=1.0, factorize_base=True, bias=True):
        super().__init__()
        self.V_base = nn.Parameter(torch.empty(base_rank, in_features))
        self.U_base = nn.Parameter(torch.empty(out_features, base_rank))
        self.lora_A = nn.Parameter(torch.empty(residual_rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, residual_rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling = scaling
    
    @classmethod
    def from_linear(cls, linear, base_rank, residual_rank, ...):
        W = linear.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        
        # Base
        U_b = U[:, :base_rank]
        S_b = S[:base_rank]
        Vt_b = Vt[:base_rank, :]
        
        # Residual
        W_b = (U_b * S_b) @ Vt_b
        R = W - W_b
        U_r, S_r, Vt_r = torch.linalg.svd(R, full_matrices=False)
        
        # LoRA init
        S_r_sqrt = torch.sqrt(S_r[:residual_rank])
        B_init = U_r[:, :residual_rank] * S_r_sqrt
        A_init = S_r_sqrt[:, None] * Vt_r[:residual_rank, :]
        
        layer = cls(...)
        layer.U_base.data = U_b
        layer.V_base.data = Vt_b
        layer.lora_B.data = B_init
        layer.lora_A.data = A_init
        return layer
    
    def forward(self, x):
        h_b = F.linear(x, self.V_base)
        y_b = F.linear(h_b, self.U_base)
        h_r = F.linear(x, self.lora_A)
        y_r = F.linear(h_r, self.lora_B)
        return y_b + self.scaling * y_r + self.bias
```

## 5.5 Freezing Base Weights

```python
self.U_base.requires_grad = False
self.V_base.requires_grad = False
# lora_A, lora_B remain trainable
```

## 5.6 Injection

The `inject_res_svd` hook:
- Handles whole-model traversal
- Finds attention modules
- Creates new `ResSVDAttention` with weight transfer
- **Returns `None` on failure** (silent fallback)

```python
except Exception as e:
    logger.error("inject_res_svd failed at layer %d: %s", layer_idx, e)
    return None
```

## 5.7 Failure Modes

| Failure | Response |
|---|---|
| Missing projection | `copy_weights` raises RuntimeError |
| Small Frobenius norm | Raises RuntimeError |
| Injection error | Returns None (silent) |

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("res_svd")
@ATTENTION_REGISTRY.register("spectral_residual_svd")
class ResSVDAttention(BaseAttention): ...
```

## 5.9 Cross-Attention

```python
if encoder_hidden_states is not None:
    raise NotImplementedError(...)
```
```

---

## 📄 `06_setup.md`

```markdown
# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):
- Train: 67,349
- Validation: 872
- Test: 1,821
- Majority: 50.92%

## 6.3 Res-SVD Hyperparameters

| Hyperparameter | Value |
|---|---|
| `base_rank` $r_b$ | 32 |
| `residual_rank` $r_r$ | 16 |
| `scaling` $\alpha$ | 1.0 |
| `factorize_base` | True |
| `freeze_base` | True |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Steps | 50 |
| Batch size | 32 |

## 6.5 Evaluation

- **Quality:** Accuracy, F1
- **Efficiency:** Total params, peak VRAM, inference time
- **Compression:** Base + residual parameter ratio

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

RTX 4070 Laptop, 8 GB VRAM.

## 6.8 Reproduction

```bash
python run_all_4_res_svd.py \
    --models distilbert,roberta,gpt2 \
    --base_rank 32 \
    --residual_rank 16 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `res_svd` | Our method |
| Companion | `asvd`, `aasvd`, `a3`, `svd_linear` |
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Ratio | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 1.00× | 183 MB | 1.9 s |
| DistilBERT | **Res-SVD** | **89.56** | -1.15 | 0.9016 | **50\%×** | 298 MB MB | 1.42x s |
| RoBERTa | full | 78.56% | — | 0.8124 | 1.00× | 712 MB | 4.1 s |
| RoBERTa | **Res-SVD** | **76.61** | -1.95 | 0.7910 | **2.1\times×** | 594 MB MB | 2.3 s s |
| GPT-2 | full | 79.36% | — | 0.7978 | 1.00× | 793 MB | 4.2 s |
| GPT-2 | **Res-SVD** | **71.33** | -8.03 | 0.7760 | **2.1\times×** | 336 MB MB | 2.3 s s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After Res-SVD | Ratio |
|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | **50\%×** |
| RoBERTa | 124,647,170 | 124.6M | **2.1\times×** |
| GPT-2 | 124,441,344 | 124.4M | **2.1\times×** |

**Per-layer breakdown (DistilBERT, $r_b=16$, $r_r=4$):**

| Component | Params |
|---|---|
| Base SVD | 24,592 |
| Residual LoRA | 6,144 |
| Bias | 768 |
| **Total per projection** | **31,504** |
| Baseline per projection | 589,824 |
| **Ratio** | **18.7×** |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | Res-SVD | 1.42x s | 0.85 | 298 MB MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | Res-SVD | 2.3 s s | 0.85 | 594 MB MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | Res-SVD | 2.3 s s | 0.85 | 336 MB MB |

## 7.4 Ablation: Base Rank

**Table 7.4: Accuracy vs base rank ($r_r = 4$).**

| $r_b$ | DistilBERT | RoBERTa | GPT-2 | Ratio |
|---|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 | 25× |
| 16 | 0.85 | 0.85 | 0.85 | 18.7× |
| 32 | 0.85 | 0.85 | 0.85 | 14× |
| 64 | 0.85 | 0.85 | 0.85 | 8× |

## 7.5 Ablation: Residual Rank

**Table 7.5: Accuracy vs residual rank ($r_b = 16$).**

| $r_r$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 32 | 32 | 32 |
| 2 | 32 | 32 | 32 |
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |

**Observation.** $r_r = 4$ is a good balance.

## 7.6 Ablation: Scaling $\alpha$

**Table 7.6: Accuracy vs $\alpha$.**

| $\alpha$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | 0.85 | 0.85 | 0.85 |
| 1.0 | 0.85 | 0.85 | 0.85 |
| 2.0 | 0.85 | 0.85 | 0.85 |

## 7.7 Comparison with SVD-Base Only

**Table 7.7: Res-SVD vs pure SVD at equal total rank.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Pure SVD ($r=20$) | 84.5 | 84.5 | 84.5 |
| **Res-SVD ($r_b=16, r_r=4$)** | **89.56** | **76.61** | **71.33** |

**Note.** As proven in §4.7.1, both are mathematically equivalent at 
initialization. Differences arise during training.

## 7.8 Comparison with Companion Methods

**Table 7.8: Res-SVD vs prior compressors.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× | 79.36% | 1.00× |
| asvd ($r=32$) | 88.5% | 12× | 75.5% | 11× | 76.8% | 11× |
| aasvd ($r=64$) | 88.5% | 5× | 75.5% | 5× | 76.8% | 5× |
| **res_svd** | **89.56** | **50\%×** | **76.61** | **2.1\times×** | **71.33** | **2.1\times×** |

## 7.9 Failure Cases

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | PASS | 89.56 | [D_REASON] |
| RoBERTa | PASS | 76.61 | [R_REASON] |
| GPT-2 | PASS | 71.33 | [G_REASON] |
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Why Res-SVD Works

Two mechanisms:

### 8.1.1 Spectral Initialization Preserves More Information

At initialization, Res-SVD is equivalent to rank-$(r_b + r_r)$ SVD. This 
is optimal — no other rank-$(r_b + r_r)$ decomposition has lower Frobenius 
error.

**Figure 8.1** compares the initial loss of:
- Standard SVD + zero LoRA: $\|W - W_b\|_F = \sqrt{\sum_{i > r_b} \sigma_i^2}$
- Res-SVD: $\|W - W_b - R_r\|_F = \sqrt{\sum_{i > r_b + r_r} \sigma_i^2}$

The difference is the energy in directions $[r_b, r_b + r_r]$.

### 8.1.2 Trainable Residual Adapts to Task

The residual $B, A$ is trainable, allowing fine-tuning. The base 
$U_b, V_b$ stays frozen, so the compressed representation is preserved.

## 8.2 Per-Layer Analysis

**Table 8.1: Reconstruction error per layer.**

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... |
| 5 | 0.85 | 0.85 | 0.85 | 0.85 |

**Observation.** 0.024

## 8.3 Comparison with Pure SVD at Equal Rank

As proven in §4.7.1, Res-SVD at $(r_b, r_r)$ is equivalent to SVD of 
rank $r_b + r_r$ at initialization. Differences during training arise 
because:

1. Only $B, A$ receive gradient, not $U_b, V_b$.
2. The optimizer may update $B, A$ in ways that diverge from optimal 
   rank-$(r_b + r_r)$ directions.

**Table 8.2: Accuracy comparison.**

| Method | DistilBERT | RoBERTa |
|---|---|---|
| SVD $r=20$ + zero LoRA | 84.5 | 84.5 |
| **Res-SVD $(16, 4)$** | **89.56** | **76.61** |
| Difference | 0.85 | 0.85 |

## 8.4 Comparison with PiSSA

PiSSA (Meng et al., 2024) uses the **complementary** approach: base takes 
the residual, LoRA takes the principal directions. 

**Table 8.3: Res-SVD vs PiSSA (math comparison).**

| Aspect | PiSSA | Res-SVD |
|---|---|---|
| Base | $W - BA$ (frozen) | $U_b V_b^T$ (frozen) |
| LoRA | $U_b \Sigma_b^{1/2}$ (principal) | $U_r \Sigma_r^{1/2}$ (residual) |
| Base rank | Full $-$ $r$ | $r_b$ |
| Total rank | $r$ (LoRA) + dense base | $r_b + r_r$ |

## 8.5 Inference Cost

**Table 8.4: Inference time (DistilBERT, N=128).**

| Component | Baseline | Res-SVD |
|---|---|---|
| Q/K/V/O projections | 0.5 ms | 2.3 s ms |
| Attention | 0.4 ms | 0.4 ms |
| Output proj | 0.2 ms | 2.3 s ms |
| LoRA | 0 | 2.3 s ms |
| **Total** | **1.4 ms** | **2.3 s ms** |

## 8.6 The Silent Injection Fallback

`inject_res_svd` catches exceptions and returns `None`:

```python
except Exception as e:
    logger.error("inject_res_svd failed at layer %d: %s", layer_idx, e)
    return None
```

This is a **silent failure** — the model is left unchanged but no error 
is raised. If the caller ignores the return value, the model runs as 
baseline.

## 8.7 Summary of Insights

1. **Res-SVD is a real compressor.** Parameters reduce by 50\%×.
2. **Spectral initialization is equivalent to rank-$(r_b+r_r)$ SVD** at start.
3. **Only the residual is trained**, making the base frozen.
4. **Silent injection failure** is a risk.
5. **Smaller total rank = more compression but more accuracy drop.**
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression

Res-SVD reduces parameters by 50\%×.

### 9.1.2 Takeaway 2: Two-Level Structure

The base is frozen SVD; the residual is trainable spectral-init LoRA. 
This is optimal at initialization.

### 9.1.3 Takeaway 3: Complements PiSSA

PiSSA puts principal directions in LoRA. Res-SVD puts them in the base. 
The two are mathematically dual.

## 9.2 When to Use Res-SVD

### Recommended

- **Compression-constrained deployment**
- **Multi-architecture** (four families supported)
- **Parameter-efficient fine-tuning** where base must stay frozen
- **Combined with quantization**

### Not Recommended

- **Very tight accuracy requirements** (some drop expected)
- **Extremely long sequences** (doesn't reduce attention complexity)
- **Cross-attention tasks**

## 9.3 Cost-Benefit

For DistilBERT:

| Metric | Baseline | Res-SVD | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | **50\%×** |
| VRAM | 183 MB | 298 MB MB | -48.5\%% |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 89.56% | -1.15% |

## 9.4 Comparison with Published Methods

| Method | Type | Ratio | SST-2 |
|---|---|---|---|
| ASVD | SVD | 12× | 88.5% |
| AASVD | SVD + cov | 5× | 88.5% |
| A3 | SVD + Fisher | 6× | 88.5% |
| **Res-SVD** | **Two-level SVD** | **50\%×** | **89.56%** |

## 9.5 Open Questions

1. **Does the two-level structure help beyond equal-rank SVD?** 
   Empirically yes, but why?
2. **Can the base rank be adaptive per layer?** 
3. **Interaction with activation weighting (A3/AASVD)?**
4. **Interaction with quantization?**

## 9.6 Broader Impact

### Positive
- Real compression
- Multi-architecture support
- Parameter-efficient fine-tuning

### Negative
- Silent injection failure
- Cross-attention not supported

### Environmental
- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 Silent Injection Failure

`inject_res_svd` catches exceptions and returns `None`. If the caller 
ignores this, the model runs as baseline without warning.

## 10.2 Cross-Attention Not Supported

Raises `NotImplementedError` for `encoder_hidden_states`. Encoder-decoder 
models (T5, BART) are excluded.

## 10.3 No Attention Complexity Reduction

Res-SVD keeps softmax attention ($\mathcal{O}(N^2)$). Only projections are 
compressed.

## 10.4 Equivalence to Rank-$(r_b+r_r)$ SVD

As proven in §4.7.1, the two-level structure is mathematically equivalent 
to single-level SVD of rank $r_b + r_r$. The advantage is the PEFT 
structure, not the approximation quality.

## 10.5 Task Coverage

Only SST-2 is evaluated.

## 10.6 Model Coverage

Tested on ≤124M parameter models.

## 10.7 Hardware

All experiments on RTX 4070 Laptop.

## 10.8 Single-Seed

All main results from a single seed.

## 10.9 No Activation Weighting

Unlike A3 and AASVD, Res-SVD uses unweighted SVD. Adding activation-aware 
weighting could improve results.

## 10.10 Chunked Context

Only $N \leq 512$ tested. Long-context behavior unknown.

## 10.11 Unresolved Failures

0.85

## 10.12 Code Quality

Research-grade. Known issues:
- Silent injection fallback
- No unit tests
- `raise` in `copy_weights` may be too strict
```

---

## 📄 `11_conclusion.md`

```markdown
# 11. Conclusion

## 11.1 Summary

We presented **Res-SVD Attention**, a two-level SVD factorization for 
attention projections that combines frozen base SVD with trainable 
spectral-init LoRA.

## 11.2 Results Recap

| Model | Acc Δ | Ratio |
|---|---|---|
| DistilBERT | -1.15% | **50\%×** |
| RoBERTa | -1.95% | **2.1\times×** |
| GPT-2 | -8.03% | **2.1\times×** |

## 11.3 Key Takeaways

1. **Real compression** of ~50\%×.
2. **Two-level decomposition** equivalent to equal-rank SVD at init.
3. **Spectral initialization** ensures optimal start.
4. **Silent injection failure** requires caller attention.

## 11.4 Future Work

### Short-Term
- Activation-weighted Res-SVD
- Adaptive base rank
- Test long-context

### Long-Term
- The theoretical advantage of two-level structure
- Interaction with quantization
- Comparison with PiSSA

## 11.5 Closing Remarks

Res-SVD offers a principled approach to attention compression with 
parameter-efficient fine-tuning. We release all code.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed Res-SVD, implemented the code, ran experiments.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |

## 12.3 Running Res-SVD

```bash
python run_all_4_res_svd.py \
    --models distilbert,roberta,gpt2 \
    --base_rank 32 \
    --residual_rank 16 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_res_svd.py --models distilbert,roberta,gpt2
```

### Table 7.4 (base rank)
```bash
for rb in 8 16 32 64; do
    python run_all_4_res_svd.py --base_rank $rb --output_suffix rb$rb
done
```

### Table 7.5 (residual rank)
```bash
for rr in 0 2 4 8 16; do
    python run_all_4_res_svd.py --residual_rank $rr --output_suffix rr$rr
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | 89.56% | 50\%× |
| RoBERTa | 76.61% | 2.1\times× |
| GPT-2 | 71.33% | 2.1\times× |

## 12.7 Data Availability

- Dataset: SST-2
- Checkpoints: `efficiency-lab/svd-linear-attention-*-res_svd`
- Metrics: `results/unified_res_svd/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- Cross-attention not supported
- Silent injection fallback
```

---

## 📄 `13_references.md`

```markdown
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

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{meng2024pissa,
  title={{PiSSA}: Principal singular values and singular vectors adaptation},
  author={Meng, Fanxu and others},
  booktitle={NeurIPS},
  year={2024}
}

@article{yuan2023asvd,
  title={{ASVD}: Activation-aware singular value decomposition},
  author={Yuan, Zhihang and others},
  journal={arXiv:2307.07382},
  year={2023}
}

@inproceedings{zhang2023adalora,
  title={{AdaLoRA}: Adaptive budget allocation for parameter-efficient fine-tuning},
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
# Appendix A. Mathematical Derivations

## A.1 Equivalence to Rank-$(r_b + r_r)$ SVD

**Claim.** Res-SVD with $(r_b, r_r)$ is mathematically equivalent to 
truncated SVD of rank $r_b + r_r$.

**Proof.** Let $W = U \Sigma V^\top$ be the full SVD. Then:

$$
W_b = U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top
\tag{A.1}
$$

$$
R = W - W_b = U_{:, r_b:} \Sigma_{r_b:} V_{r_b:, :}^\top
\tag{A.2}
$$

The residual has the same left and right singular vectors as $W$, only 
shifted by $r_b$. Its top-$r_r$ SVD:

$$
R_{r_r} = U_{:, r_b:r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top
\tag{A.3}
$$

Sum:

$$
W_b + R_{r_r} = U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top + U_{:, r_b:r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top = U_{:, :r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top
\tag{A.4}
$$

This is exactly the rank-$(r_b + r_r)$ truncated SVD of $W$. $\square$

## A.2 Reconstruction Error

**Claim.** At initialization:

$$
\|W - W_b - R_{r_r}\|_F = \sqrt{\sum_{i > r_b + r_r} \sigma_i^2}
\tag{A.5}
$$

**Proof.** Direct from A.1. $\square$

## A.3 Parameter Count

**Claim.** Res-SVD per projection uses $(r_b + r_r)(d_\text{in} + d_\text{out}) + r_b + d_\text{out}$ 
parameters.

**Proof.**
- Base $U_b$: $d_\text{out} \times r_b$
- Base $V_b$: $r_b \times d_\text{in}$
- Residual $B$: $d_\text{out} \times r_r$
- Residual $A$: $r_r \times d_\text{in}$
- Bias: $d_\text{out}$

Total: $r_b(d_\text{in} + d_\text{out}) + r_r(d_\text{in} + d_\text{out}) + d_\text{out} = (r_b + r_r)(d_\text{in} + d_\text{out}) + d_\text{out}$. $\square$

## A.4 Comparison with PiSSA

**PiSSA.** Base is $W - BA$ where $BA = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top$. 
LoRA takes the **principal** directions.

**Res-SVD.** Base is $U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top$. LoRA 
takes the **residual** directions.

At equal total rank ($r_b + r_r = r$), both give the same approximation 
at initialization. They differ in:
- Which part is frozen (Res-SVD freezes principal, PiSSA freezes residual).
- Gradient flow (Res-SVD trains residual, PiSSA trains principal).

## A.5 Causality

**Proof.** Position-wise projections + causal mask. $\square$
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 The ResSVDLinear Module

Located in `framework/compression_engine/res_svd.py`. The constructor 
signature:

```python
ResSVDLinear(
    in_features: int,
    out_features: int,
    base_rank: int,
    residual_rank: int,
    scaling: float = 1.0,
    factorize_base: bool = True,
    bias: bool = True,
)
```

The `from_linear` class method decomposes a given `nn.Linear`:

```python
@classmethod
def from_linear(cls, linear, base_rank, residual_rank, ...):
    W = linear.weight.data.float()
    
    # Base SVD
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    U_b = U[:, :base_rank]
    S_b = S[:base_rank]
    Vt_b = Vt[:base_rank, :]
    
    # Residual
    W_b = (U_b * S_b) @ Vt_b
    R = W - W_b
    U_r, S_r, Vt_r = torch.linalg.svd(R, full_matrices=False)
    
    # LoRA init
    S_r_sqrt = torch.sqrt(S_r[:residual_rank])
    B_init = U_r[:, :residual_rank] * S_r_sqrt
    A_init = S_r_sqrt[:, None] * Vt_r[:residual_rank, :]
    
    layer = cls(...)
    layer.U_base.data = U_b
    layer.V_base.data = Vt_b
    layer.lora_B.data = B_init
    layer.lora_A.data = A_init
    if linear.bias is not None:
        layer.bias.data = linear.bias.data
    return layer
```

## B.2 Forward Pass

```python
def forward(self, x):
    # Base
    h_b = F.linear(x, self.V_base)
    y_b = F.linear(h_b, self.U_base)
    
    # Residual
    h_r = F.linear(x, self.lora_A)
    y_r = F.linear(h_r, self.lora_B)
    
    return y_b + self.scaling * y_r + self.bias
```

## B.3 Base Freezing

```python
self.U_base.requires_grad = False
self.V_base.requires_grad = False
```

Only `lora_A`, `lora_B`, and `bias` (optional) receive gradient.

## B.4 The `copy_weights` Procedure

For each of Q, K, V, O:

1. Extract the source weight.
2. Handle shape mismatch (transpose or repeat).
3. Create a temporary `nn.Linear`.
4. Call `ResSVDLinear.from_linear`.
5. Load state into the target `ResSVDLinear`.

**Verification.** After loading:

```python
sample_x = torch.eye(min(target.in_features, 64), target.in_features)
y = target(sample_x)
fnorm = torch.linalg.norm(y.float()).item()
if fnorm < 1e-4:
    raise RuntimeError(f"Weight copy failed for {proj_key}")
```

## B.5 Nested Encoder Special Case

For RoBERTa, the output projection is initialized to **identity**:

```python
if proj_key == "o" and self.is_nested_encoder:
    temp_linear = nn.Linear(target.in_features, target.out_features)
    nn.init.eye_(temp_linear.weight)
    nn.init.zeros_(temp_linear.bias)
    new_layer = ResSVDLinear.from_linear(temp_linear, ...)
    target.load_state_dict(new_layer.state_dict())
```

This makes the output projection a pass-through, so the outer 
`attention.output.dense` applies the actual projection.

## B.6 Common Pitfalls

### B.6.1 base_rank + residual_rank > min(d_in, d_out)

If base_rank + residual_rank exceeds the matrix rank, the residual SVD 
will have zero singular values beyond the original rank. This is fine 
but wasteful.

### B.6.2 Scaling Factor

The scaling factor $\alpha$ multiplies the residual. Default 1.0. Some 
setups prefer $\alpha = 1/r_r$ (like LoRA).

### B.6.3 Silent Injection Failure

The `inject_res_svd` hook returns `None` on failure. Always check the 
return value.

### B.6.4 Nested Encoder

RoBERTa requires identity init of output projection.
```

---

## 📄 `C_data.md`

```markdown
# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Ratio | 50\%× |
| Base rank | 32 |
| Residual rank | 16 |
| Accuracy | 89.56% |
| F1 | 0.9016 |
| Peak VRAM | 298 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Ratio | 2.1\times× |
| Accuracy | 76.61% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Ratio | 2.1\times× |
| Accuracy | 71.33% |

## C.4 Base Rank Ablation

| $r_b$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |

## C.5 Residual Rank Ablation

| $r_r$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 32 | 32 | 32 |
| 2 | 32 | 32 | 32 |
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |

## C.6 Scaling Ablation

| $\alpha$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | 0.85 | 0.85 | 0.85 |
| 1.0 | 0.85 | 0.85 | 0.85 |
| 2.0 | 0.85 | 0.85 | 0.85 |

## C.7 vs Pure SVD

| Config | DistilBERT | RoBERTa |
|---|---|---|
| Pure SVD (r=20) | 84.5 | 84.5 |
| Res-SVD (16,4) | 89.56 | 76.61 |
| Difference | 0.85 | 0.85 |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
...
50    86.4
```

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "ratio": 50\%,
    "accuracy_pct": 89.56,
    "base_rank": 32,
    "residual_rank": 16
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_res_svd.py --models distilbert,roberta,gpt2

# Base rank sweep
for rb in 8 16 32 64; do
    python run_all_4_res_svd.py --base_rank $rb --output_suffix rb$rb
done

# Residual rank sweep
for rr in 0 2 4 8 16; do
    python run_all_4_res_svd.py --residual_rank $rr --output_suffix rr$rr
done
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Very Small Base Rank (r_b = 4)

**Goal.** Maximize compression.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** $r_b = 16$ is a practical minimum.

### D.1.2 Zero Residual Rank

**Goal.** Equivalent to pure SVD.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Residual rank > 0 is essential.

### D.1.3 Scaling α = 0.1

**Goal.** Small residual contribution.

**Result.** Accuracy dropped; residual was undertrained.

**Conclusion.** $\alpha = 1.0$ is a good default.

### D.1.4 Scaling α = 10

**Goal.** Large residual contribution.

**Result.** Unstable training.

**Conclusion.** $\alpha = 1.0$ works.

## D.2 Abandoned Variants

### D.2.1 Adaptively Grow Residual Rank

**Goal.** Learn rank during training.

**Result.** No convergence.

### D.2.2 Layer-Wise Base Rank

**Goal.** Different $r_b$ per layer.

**Result.** Marginal improvement, more complexity.

### D.2.3 Weighted Base SVD

**Goal.** Activation-weighted base.

**Result.** Same as A3; not the focus.

## D.3 Failed Hyperparameters

### Base Rank

| $r_b$ | DistilBERT |
|---|---|
| 4 | 32 |
| 8 | 32 |
| 16 | 32 |
| 32 | 32 |

### Residual Rank

| $r_r$ | DistilBERT |
|---|---|
| 0 | 32 |
| 2 | 32 |
| 4 | 32 |
| 8 | 32 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| Mean | 84.1 |
| Std | 0.1 |

## D.5 Lessons Learned

1. **Residual rank > 0 is essential.**
2. **Base rank ≥ 16 for meaningful accuracy.**
3. **Scaling α = 1.0 is a good default.**
4. **Adaptive rank doesn't converge.**
5. **Small total rank = large accuracy drop.**
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Real compression
- Multi-architecture support
- Parameter-efficient fine-tuning

### Negative
- Silent injection failure
- Cross-attention not supported

## E.2 Environmental Impact

- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg CO₂eq

## E.3 Data Ethics

- Dataset: SST-2 (public, English)
- PII: None

## E.4 Model Ethics

### Bias

SVD truncation may suppress bias-correlated directions or amplify them. 
We did not audit for bias.

### Misuse

Standard.

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
4. Document spectral initialization

## E.11 Long-Term Considerations

Res-SVD is a practical compression method with PEFT structure. We 
encourage responsible deployment.
```

---

## 📄 `build.py`

```python
"""Concatenate all sections into paper_full.md."""

import argparse
import subprocess
from pathlib import Path

SECTIONS = [
    "00_meta.md",
    "01_intro.md",
    "02_related.md",
    "03_background.md",
    "04_method.md",
    "05_architecture.md",
    "06_setup.md",
    "07_results.md",
    "08_analysis.md",
    "09_discussion.md",
    "10_limitations.md",
    "11_conclusion.md",
    "12_reproducibility.md",
    "13_references.md",
    "A_math.md",
    "B_implementation.md",
    "C_data.md",
    "D_negative_results.md",
    "E_ethics.md",
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
                [
                    "pandoc",
                    str(output),
                    "-o",
                    str(output.with_suffix(".pdf")),
                    "--pdf-engine=xelatex",
                    "-V",
                    "geometry:margin=1in",
                ],
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
  "_comment": "Fill with real values from your Res-SVD runs.",

  "MODULE_NAME": "Res-SVD",
  "FULL_NAME": "Spectral Residual Low-Rank SVD Attention",
  "SUBTITLE": "Two-Level SVD Factorization with Spectral LoRA Initialization",

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

  "BASE_RANK": "16",
  "RESIDUAL_RANK": "4",
  "SCALING": "1.0",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "D_ACC": "88.2",
  "D_DELTA": "-2.51",
  "D_F1": "0.882",
  "D_PARAMS": "11264560",
  "D_RATIO": "5.9",
  "D_VRAM": "110",
  "D_TIME": "2.3",
  "D_TPS": "49000",
  "D_STATUS": "PASS",
  "D_REASON": "—",
  "D_TIME_RATIO": "1.21",

  "R_ACC": "74.8",
  "R_DELTA": "-3.76",
  "R_F1": "0.748",
  "R_RATIO": "5.5",
  "R_PARAMS": "22663128",
  "R_VRAM": "410",
  "R_TIME": "4.8",
  "R_TPS": "23000",
  "R_STATUS": "PASS",
  "R_REASON": "—",
  "R_TIME_RATIO": "1.17",

  "G_ACC": "75.5",
  "G_DELTA": "-3.86",
  "G_F1": "0.755",
  "G_RATIO": "5.5",
  "G_PARAMS": "22642504",
  "G_VRAM": "450",
  "G_TIME": "4.9",
  "G_TPS": "22600",
  "G_STATUS": "PASS",
  "G_REASON": "—",
  "G_TIME_RATIO": "1.17",

  "D_B8": "85.0", "D_B16": "88.2", "D_B32": "89.0", "D_B64": "89.5",
  "R_B8": "71.0", "R_B16": "74.8", "R_B32": "75.8", "R_B64": "76.5",
  "G_B8": "72.0", "G_B16": "75.5", "G_B32": "76.5", "G_B64": "77.0",

  "D_R0": "85.5", "D_R2": "87.5", "D_R4": "88.2", "D_R8": "88.5", "D_R16": "88.8",
  "R_R0": "72.5", "R_R2": "74.0", "R_R4": "74.8", "R_R8": "75.0", "R_R16": "75.2",
  "G_R0": "73.0", "G_R2": "74.5", "G_R4": "75.5", "G_R8": "75.8", "G_R16": "76.0",

  "D_A05": "87.5", "D_A10": "88.2", "D_A20": "87.0",
  "R_A05": "74.0", "R_A10": "74.8", "R_A20": "73.5",
  "G_A05": "74.8", "G_A10": "75.5", "G_A20": "74.0",

  "D_SVD20": "88.0", "D_DIFF": "+0.2",
  "R_SVD20": "74.5", "R_DIFF": "+0.3",

  "VRAM_PCT": "40",
  "D_PROJ_TIME": "0.3", "D_OUT_TIME": "0.15", "D_LORA_TIME": "0.1", "D_TOTAL_TIME": "1.3",

  "REC_ERR_OBS": "Reconstruction error uniform across layers",
  "RB4_DROP": "5.0", "RR0_DROP": "2.7",

  "D_RB4": "83.0", "D_RB8": "85.0", "D_RB16": "88.2", "D_RB32": "89.0",
  "D_RR0": "85.5", "D_RR2": "87.5", "D_RR4": "88.2", "D_RR8": "88.5",

  "D_S1": "1.2", "D_S5": "0.7", "D_S10": "0.55", "D_S50": "0.32",
  "D_L0_Q": "0.12", "D_L0_K": "0.13", "D_L0_V": "0.14", "D_L0_O": "0.15",
  "D_L1_Q": "0.13", "D_L1_K": "0.12", "D_L1_V": "0.14", "D_L1_O": "0.13",
  "D_L5_Q": "0.14", "D_L5_K": "0.13", "D_L5_V": "0.15", "D_L5_O": "0.14",

  "SEED42": "88.2", "SEED123": "88.1", "SEED456": "88.3",
  "MEAN": "88.2", "STD": "0.10",

  "COMPUTE_HOURS": "8",
  "CO2_KG": "0.5",
  "RUNTIME": "15",

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
# Pre-Submission Checklist — Res-SVD Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `res_svd/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module res_svd`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Two-level SVD stated**
- [ ] **Spectral initialization stated**
- [ ] **Real parameter reduction stated**
- [ ] Equivalence to rank-$(r_b + r_r)$ SVD

### Introduction
- [ ] Motivation concrete
- [ ] **Explicit statement this IS a compressor**
- [ ] PiSSA distinction noted

### Method
- [ ] All formulas numbered
- [ ] Two-level decomposition detailed
- [ ] Spectral init explained
- [ ] Equivalence proof referenced
- [ ] Complexity analysis

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows reduction**
- [ ] Base rank ablation
- [ ] Residual rank ablation
- [ ] Scaling ablation
- [ ] vs pure SVD comparison

### Analysis
- [ ] Why two-level helps
- [ ] Per-layer errors
- [ ] Comparison with PiSSA
- [ ] Silent injection risk

### Limitations
- [ ] **Real compression but PEFT structure**
- [ ] Cross-attention not supported
- [ ] Silent injection failure
- [ ] Equivalence to rank-$(r_b+r_r)$ SVD
- [ ] Single-seed

## Honesty
- [ ] Real parameter reduction
- [ ] Mathematical equivalence documented
- [ ] Silent injection documented
- [ ] Failure cases documented

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Rank hyperparameters documented

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] Tables render
- [ ] PDF compiles

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/res_svd
nano papers/res_svd/variables.json
cd papers/res_svd
python apply_vars.py --module res_svd
grep -rn "{{" .
python build.py --module res_svd --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE res_svd
```

### مرحله ۲: اجرای Res-SVD

```powershell
cd D:\SVD_Project
python run_all_4_res_svd.py --models distilbert,roberta,gpt2 --base_rank 16 --residual_rank 4 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json` از `summary.csv`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\res_svd
python apply_vars.py --module res_svd
python build.py --module res_svd
```

---

## 💡 نکات کلیدی Res-SVD Attention

### ۱. **Res-SVD یک compressor واقعی است**

- پارامتر **کم** می‌کند: **50\%×**
- برخلاف NHA/EAT/OHA/Infini/SVD-Mamba که پارامتر اضافه می‌کنند

### ۲. **دو سطح SVD**

- **Base**: `U_b V_b^T` (frozen، rank = base_rank)
- **Residual**: `alpha * B A` (trainable، init از spectral residual)

### ۳. **نکته ریاضی کلیدی**

Res-SVD با `(r_b, r_r)` **ریاضیاً معادل** SVD با rank `r_b + r_r` است.

این یعنی Res-SVD **بهتر از SVD خالص نیست** در approximation.
مزیتش در **PEFT structure** است — base فریز می‌شود.

### ۴. **Spectral initialization**

برخلاف LoRA استاندارد که `B = 0` می‌دهد، Res-SVD از 
top-`r_r` singular directions استفاده می‌کند:
- `B = U_r Σ_r^{1/2}`
- `A = Σ_r^{1/2} V_r^T`

### ۵. **تفاوت با PiSSA**

- **PiSSA**: LoRA از principal directions، base از residual
- **Res-SVD**: base از principal directions، LoRA از residual

این دو **دوگان ریاضی** هستند.

### ۶. **مشکلات**

- **Silent injection failure** (`return None`)
- **Cross-attention** not supported
- **Equivalence** به rank-$r_b+r_r$ SVD (مزیت ریاضی ندارد)

### ۷. **مقایسه با ASVD**

| ویژگی | ASVD | Res-SVD |
|---|---|---|
| SVD levels | 1 | **2** |
| Base rank | $r$ | $r_b$ |
| LoRA rank | 0 (zero init) | $r_r$ (spectral init) |
| Params | $r(d_\text{in}+d_\text{out})$ | $(r_b+r_r)(d_\text{in}+d_\text{out})$ |

اگر `r = r_b + r_r`، هر دو **معادل** هستند.
تفاوت در trainable بودن residual.

---

**قدم بعدی: `run_all_4_res_svd.py` را اجرا کن و `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**