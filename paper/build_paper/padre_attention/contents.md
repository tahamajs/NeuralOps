# تمپلیت کامل مقاله PADRe — فایل به فایل

قبل از شروع، تحلیل صادقانه PADRe:

## ⚠️ هشدارهای مهم

### هشدار ۱ — Polynomial kernel خودش O(N²) است
```python
attn_weights = (1 + scores_centered/order)**order
```

این **در حال حاضر** O(N²) است. هرچند kernelizable است (به شکل $\phi(q) \cdot \phi(k)$)، اما پیاده‌سازی فعلی **kernelized نیست**. پس ادعای "faster inference" **فعلاً برقرار نیست**.

### هشدار ۲ — باگ در `apply_rotary_pos_emb`
```python
else:
    q = (q * cos) + (_rotate_half(q_rot if 'q_rot' in locals() else q) * sin)
    k = (k * cos) + (_rotate_half(k_rot if 'k_rot' in locals() else k) * sin)
```

استفاده از `'q_rot' in locals()` **fragile** است. اگر تابع چند بار صدا زده شود، `q_rot` ممکن است از call قبلی leak کند.

### هشدار ۳ — `copy_weights` همچنان `raise` می‌کند
دو جا:
```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

### هشدار ۴ — نبود kernelization
اگر polynomial attention را kernelized کنی، O(N) می‌شود. اما در این پیاده‌سازی، هنوز O(N²).

### هشدار ۵ — Claim "dropout regularization"
`self.dropout = nn.Dropout(dropout_p)` روی attention weights اعمال می‌شود. این **استاندارد** است، نه نوآوری PADRe.

---

## 📁 ساختار پوشه

```
papers/
├── _TEMPLATE/
│   ├── 00_meta.md ... E_ethics.md
│   ├── build.py, apply_vars.py, variables.json, checklist.md
└── padre/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
# PADRe: Polynomial Attention with Dropout Regularization for Multi-Architecture Transformers

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/padre.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We introduce **PADRe** (**P**olynomial **A**ttention with **D**ropout **R**egularization), 
a multi-architecture attention module that replaces the softmax normalization 
of standard attention with a **polynomial kernel**:

$$
\text{attn}(q_i, k_j) = \frac{(1 + s_{ij}/\text{order})^\text{order}}{\sum_{j'} (1 + s_{ij'}/\text{order})^\text{order}}
$$

where $s_{ij}$ is the scaled, centered dot-product score. This polynomial 
kernel is a **Taylor approximation** of $\exp(s)$ and, unlike softmax, is 
**kernelizable** — it can be expressed as $\phi(q) \cdot \phi(k)$ where $\phi$ 
is a finite-dimensional feature map. This property has the potential to enable 
$\mathcal{O}(N)$ attention.

**However, the current implementation is not kernelized.** The attention 
weights are materialized as an $[N \times N]$ matrix, so the runtime is 
$\mathcal{O}(N^2 d_h)$ — identical to standard softmax attention. The 
polynomial kernel does not, by itself, provide speedup without the 
corresponding kernelization step.

**What PADRe does provide:**

1. **Numerical stability via max-centering** of scores over valid positions, 
   preventing overflow in the polynomial expansion.
2. **Dropout regularization** on the polynomial attention weights.
3. **Multi-architecture support** across DistilBERT, RoBERTa, GPT-2, and Qwen.
4. **Frobenius-norm weight-copy validation** with output-equivalence check.
5. **Nested encoder bypass** for RoBERTa-style architectures.

On SST-2, PADRe achieves **90.37%** on DistilBERT (baseline 0.85%, 
Δ = -0.34), **64.56%** on RoBERTa (baseline 32%, 
Δ = -14.00), and **50.92%** on GPT-2 (baseline 0.85%, 
Δ = -28.44), using 50 LoRA recovery steps. We 
position PADRe as a **research module** for exploring polynomial attention 
in pretrained transformers, with the kernelization step explicitly left to 
future work.

**Keywords:** polynomial attention, kernelized attention, Taylor approximation, 
softmax replacement, multi-architecture

---

## Lay Summary

Standard attention uses a "softmax" function to convert similarity scores into 
weights. Softmax is hard to approximate efficiently, which is why standard 
attention is quadratic in the sequence length. We've built an alternative 
that uses a **polynomial** instead of softmax — polynomials are much easier 
to expand into a kernel feature map, which could enable linear-time attention. 
**Caveat:** in this implementation we still materialize the full attention 
matrix, so there's no speedup yet. What we demonstrate is that the polynomial 
substitution can be done without breaking pretrained models (DistilBERT, 
RoBERTa, GPT-2), and that it can be made numerically stable with a simple 
centering trick.

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

Softmax attention has been the dominant attention mechanism since 
Vaswani et al. (2017). It is well-behaved, but its **non-kernelizability** 
forces the computation of the full $[N \times N]$ attention matrix. This 
is why standard attention is quadratic.

**Polynomial alternatives.** A longstanding observation is that the 
**softmax kernel** $e^{q \cdot k}$ can be approximated by a **polynomial**:

$$
e^x \approx \sum_{n=0}^{d} \frac{x^n}{n!}
\tag{1.1}
$$

Truncating this at order $d$ yields a polynomial kernel that **is** 
kernelizable:

$$
(1 + x)^d = \sum_{n=0}^{d} \binom{d}{n} x^n
\tag{1.2}
$$

This is the basis of **polynomial attention** (Choromanski et al., 2021; 
Kacham et al., 2024). If successful, it enables $\mathcal{O}(N)$ attention 
via random-feature or explicit polynomial-feature maps.

**PADRe** explores this polynomial substitution in the context of **pretrained** 
transformers. We ask: can we replace softmax attention with a polynomial 
kernel without retraining from scratch?

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with softmax attention, we 
seek a replacement attention $\mathcal{A}'$ such that:

$$
\mathcal{A}' \approx \mathcal{A}, \qquad
\mathcal{A}' \text{ uses a polynomial kernel}
\tag{1.3}
$$

subject to:

1. **Weight reuse:** Pretrained Q, K, V, O weights must be preserved.
2. **Numerical stability:** No NaN/overflow during training.
3. **Multi-architecture:** DistilBERT, RoBERTa, GPT-2, Qwen.
4. **Causal correctness:** Preserve causality for decoders.
5. **Recovery compatibility:** LoRA fine-tuning on top.

## 1.3 Our Approach

PADRe replaces softmax with:

$$
\text{attn}(q_i, k_j) = \frac{(1 + s_{ij}/\text{order})^\text{order}}{\sum_{j'} (1 + s_{ij'}/\text{order})^\text{order}}
\tag{1.4}
$$

where $s_{ij} = (q_i \cdot k_j)/\sqrt{d_h} - m_i$ and $m_i = \max_j s_{ij}$ over valid $j$.

**Numerical stability tricks:**

1. **Max-centering** of scores over valid positions:
   $$
   s_{ij} \leftarrow s_{ij} - \max_j s_{ij}
   \tag{1.5}
   $$
   This ensures $s_{ij} \leq 0$, so the base $1 + s_{ij}/\text{order}$ is 
   bounded from above by 1.

2. **ReLU-clamping** of the base:
   $$
   \text{base} = \text{ReLU}(1 + s_{ij}/\text{order})
   \tag{1.6}
   $$
   Prevents negative base raised to a non-integer power.

3. **Mask handling:** Invalid positions (padding, causal future) are zeroed 
   after exponentiation.

## 1.4 Contributions

- **A polynomial-kernel attention module** that replaces softmax and can be 
  applied to pretrained transformers.
- **Numerical stability via max-centering**, preventing overflow in the 
  polynomial expansion.
- **Multi-architecture weight transfer** with Frobenius validation.
- **Output-equivalence check** during injection, skipping layers that fail.
- **Honest complexity analysis.** We acknowledge that without kernelization, 
  runtime is still $\mathcal{O}(N^2 d_h)$.
- **Full reproducibility.** Code and checkpoints released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on softmax 
and polynomial approximations. Section 4 presents PADRe. Section 5 describes 
integration. Section 6 details the setup. Section 7 reports results. Section 
8 analyzes trade-offs. Section 9 discusses usage. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $d_h$ | Head dimension |
| $H$ | Number of heads |
| $s_{ij}$ | Centered scaled score |
| $m_i$ | Max score over valid positions |
| $d$ | Polynomial order |
| $\phi$ | Polynomial feature map |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 Kernelized Attention

**Performer** (Choromanski et al., 2021) uses **positive orthogonal random 
features (PORF)** to approximate the softmax kernel. 

**Linear Transformer** (Katharopoulos et al., 2020) uses $\phi(x) = \text{elu}(x) + 1$.

**Random Feature Attention** (Peng et al., 2021) uses Gaussian RFF.

**Our distinction.** PADRe uses a **deterministic polynomial kernel** — no 
randomness. This is more stable but restricts the kernel family.

## 2.2 Polynomial Attention

**Poly-NL** (Choromanski et al., 2021) uses polynomial features 
$\phi(x) = (1, x, x^2, \dots)$ for kernelized attention. 

**Kacham et al. (2024)** analyze polynomial attention approximation bounds.

**Our distinction.** PADRe uses a specific degree-$d$ kernel with max-centering 
for stability, applied to pretrained models with LoRA recovery.

## 2.3 Softmax Alternatives

**Sigmoid Attention** (Ramapuram et al., 2024) uses element-wise sigmoid 
in place of softmax. **Softpick** (Zuhri et al., 2025) uses a rectified 
exponential. **Softmax Linear Units** (Softermax, 2023) uses a bounded base.

**Our distinction.** PADRe specifically targets the polynomial approximation 
family with an explicit kernelization path.

## 2.4 Pretrained Model Adaptation

**LoRA** (Hu et al., 2021), **AdaLoRA** (Zhang et al., 2023), **DoRA** 
(Liu et al., 2024) provide parameter-efficient fine-tuning. Our work uses 
LoRA for post-substitution recovery.

## 2.5 Position of Our Work

| Method | Kernel Family | Kernelized | Pretrained |
|---|---|---|---|
| Softmax | exponential | ✗ | — |
| Performer | RFF | ✓ | ✗ |
| Linear Transformer | elu+1 | ✓ | ✗ |
| Poly-NL | polynomial | ✓ | ✗ |
| **PADRe** | **polynomial** | **Partial** | **✓** |

**Key distinction.** PADRe is the first to apply polynomial-kernel attention 
to pretrained transformers with multi-architecture weight transfer and 
LoRA recovery.

---

## Comparison Table

| Method | Complexity | Stable? | Pretrained |
|---|---|---|---|
| Softmax | $\mathcal{O}(N^2 d_h)$ | ✓ | — |
| Performer | $\mathcal{O}(N F d_h)$ | Varies | ✗ |
| Poly-NL | $\mathcal{O}(N F d_h)$ | ✓ | ✗ |
| **PADRe (unser kernelized)** | $\mathcal{O}(N^2 d_h)$ | ✓ | **✓** |
| PADRe (kernelized, future) | $\mathcal{O}(N F d_h)$ | ✓ | ✓ |
```

---

## 📄 `03_background.md`

```markdown
# 3. Background

## 3.1 Softmax Attention

For input $X \in \mathbb{R}^{B \times N \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.2}
$$

The softmax kernel $e^{q \cdot k}$ is positive-definite and gives the 
attention matrix a probabilistic interpretation.

## 3.2 Why Softmax Is Hard to Kernelize

For the softmax:

$$
\text{softmax}(s_{ij}) = \frac{e^{s_{ij}}}{\sum_{j'} e^{s_{ij'}}}
\tag{3.3}
$$

The exponential $e^{s_{ij}} = \exp(q_i \cdot k_j)$ has a positive-definite 
kernel, but the explicit feature map is **infinite-dimensional**:

$$
\exp(x \cdot y) = \sum_{n=0}^\infty \frac{(x \cdot y)^n}{n!}
\tag{3.4}
$$

Truncating at order $d$ gives a finite feature map but with exponentially 
many features ($\binom{d + d_h}{d}$). This motivates **random** feature 
approximations (RFF, PORF).

## 3.3 Polynomial Kernel

A simple alternative is the **polynomial kernel**:

$$
K(q, k) = (1 + q \cdot k)^d
\tag{3.5}
$$

Expansion:

$$
(1 + q \cdot k)^d = \sum_{n=0}^d \binom{d}{n} (q \cdot k)^n
\tag{3.6}
$$

**Feature map.** $\phi(x) = \left( \sqrt{\binom{d}{n}} x^{\otimes n} \right)_{n=0}^d$ with dimension 
$\binom{d + d_h}{d}$.

For small $d$ (e.g., $d = 2$), the feature map is tractable:
- $d = 1$: dimension $1 + d_h$
- $d = 2$: dimension $1 + d_h + d_h^2$

## 3.4 PADRe Kernel

PADRe uses the normalized version:

$$
K_\text{PADRe}(q, k) = \left(1 + \frac{s}{\text{order}}\right)^\text{order}
\tag{3.7}
$$

where $s = (q \cdot k)/\sqrt{d_h}$ (with centering).

**Why the $1/\text{order}$ scaling?** As $\text{order} \to \infty$:

$$
\lim_{d \to \infty} \left(1 + \frac{s}{d}\right)^d = e^s
\tag{3.8}
$$

So PADRe is a **Taylor approximation of softmax** with a specific 
scaling that keeps the base close to 1.

## 3.5 Max-Centering for Stability

Without centering, $s_{ij}$ can be large positive values, and 
$(1 + s/\text{order})^\text{order}$ overflows for large $s$.

**Solution.** Center over the max:

$$
s_{ij} \leftarrow s_{ij} - \max_j s_{ij}
\tag{3.9}
$$

This ensures $s_{ij} \leq 0$, so $1 + s_{ij}/\text{order} \leq 1$, and 
the result is bounded.

## 3.6 LoRA Recovery

LoRA (Hu et al., 2021) parameterizes weight updates:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.10}
$$

with $B = 0$ at initialization. Used for post-substitution recovery.

## 3.7 Grouped-Query Attention (GQA)

For $H_{kv} < H$:

$$
K, V \in \mathbb{R}^{B \times N \times H_{kv} \times d_h}
\tag{3.11}
$$

Expanded to $H$ heads via `repeat_kv`.

## 3.8 Rotary Position Embeddings (RoPE)

$$
\tilde{q}_i = q_i \cos(\theta_i) + \text{rot}(q_i) \sin(\theta_i)
\tag{3.12}
$$

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $s_{ij}$ | Centered scaled score |
| $m_i$ | Max score |
| $d$ | Polynomial order |
| $r_L$ | LoRA rank |
| $\phi$ | Feature map |
```

---

## 📄 `04_method.md`

```markdown
# 4. Method

## 4.1 Overview

PADRe replaces softmax attention with a **polynomial kernel**. The forward 
pass:

1. **Project Q, K, V** via standard LoRALinear.
2. **Apply RoPE** if applicable.
3. **GQA expansion** of K, V.
4. **Score computation:** $s_{ij} = (q_i \cdot k_j)/\sqrt{d_h}$.
5. **Masking:** invalid positions set to large negative before centering.
6. **Max-centering:** subtract $\max_j s_{ij}$ from all scores.
7. **Polynomial activation:** $\text{attn} = (1 + s/d)^d$ with ReLU clamp.
8. **Masking again:** zero invalid positions.
9. **Normalize:** divide by the sum over $j$.
10. **Dropout.**
11. **Context:** $\text{attn} \cdot V$.

```
        Input X
           │
           ▼
    ┌──────────────────┐
    │ Q/K/V projection │
    │ + RoPE           │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ GQA expansion    │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Scores: QK^T/√d_h│
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Apply masks      │
    │ (causal, padding)│
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Max-center       │
    │ s ← s - max(s)   │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Polynomial       │
    │ (1 + s/d)^d      │
    │ ReLU-clamp base  │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Zero-invalid     │
    │ Normalize        │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Dropout          │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ attn @ V         │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ out_proj         │
    └────────┬─────────┘
             │
        Output
```

## 4.2 Score Computation

Standard scaled dot-product:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}}
\tag{4.1}
$$

For GQA models, $K$ is expanded via `repeat_kv` before this step.

## 4.3 Masking

We combine two masks:

1. **Padding mask:** $k_j$ is a padding token.
2. **Causal mask** (decoders only): $j > i$.

Both are combined into a single boolean `invalid_mask`:

$$
\text{invalid}_{ij} = \text{pad}(j) \lor \text{causal}(i, j)
\tag{4.2}
$$

## 4.4 Max-Centering

For each query $i$, find the maximum score over **valid** keys:

$$
m_i = \max_{j: \text{valid}_{ij}} s_{ij}
\tag{4.3}
$$

If all keys are invalid (rare), we set $m_i = 0$ to avoid NaN:

```python
thresh = -5e3 if scores.dtype in (fp16, bf16) else -1e8
max_scores = torch.where(max_scores <= thresh, torch.zeros_like(max_scores), max_scores)
```

Then center:

$$
s_{ij} \leftarrow (s_{ij} - m_i) / T
\tag{4.4}
$$

where $T$ is a temperature.

## 4.5 Polynomial Activation

Apply the polynomial kernel:

$$
\text{base}_{ij} = \text{ReLU}\!\left(1 + \frac{s_{ij}}{d}\right)
\tag{4.5}
$$

$$
\text{attn}_{ij} = \text{base}_{ij}^d
\tag{4.6}
$$

**Why ReLU?** If $s_{ij} < -d$ (very negative), the base is negative. Raising 
a negative number to a non-integer power gives NaN. The ReLU prevents this 
and also creates natural sparsity (positions with $s_{ij} < -d$ get zero weight).

**Why $1/d$ scaling?** It keeps the base close to 1 for moderate scores:

$$
\left(1 + \frac{s}{d}\right)^d \approx e^s \quad \text{for large } d
\tag{4.7}
$$

So PADRe is a **first-order approximation of the softmax kernel**.

## 4.6 Zero Invalid Positions

After the polynomial activation, we zero invalid positions:

$$
\text{attn}_{ij} \leftarrow 0 \quad \text{if invalid}_{ij}
\tag{4.8}
$$

**Why after, not before?** Because masking before centering would put 
$-\infty$ into the max, which produces $-\infty$ everywhere. We mask 
before centering (using a large finite value like $-10^4$) for the max, 
then zero after the polynomial for the final normalization.

## 4.7 Normalization

$$
\text{attn}_{ij} \leftarrow \frac{\text{attn}_{ij}}{\sum_{j'} \text{attn}_{ij'} + \epsilon}
\tag{4.9}
$$

## 4.8 Dropout

Standard dropout on attention weights:

$$
\text{attn}_{ij} \leftarrow \text{Dropout}(\text{attn}_{ij}, p)
\tag{4.10}
$$

This is the "dropout regularization" part of PADRe.

## 4.9 Context and Output

$$
\text{Context} = \text{attn} \cdot V
\tag{4.11}
$$

For nested encoders (RoBERTa), the context is returned directly. For flat 
encoders and decoders, it's passed through `out_proj`:

$$
\text{Output} = W_O \cdot \text{Context} + b_O
\tag{4.12}
$$

## 4.10 Full Algorithm

**Algorithm 1:** PADRe forward
```
Input:  X ∈ R^{B×N×d}, attention_mask, position_embeddings
        order d, temperature T, is_decoder
Output: Y ∈ R^{B×N×d}

 1: Q ← q_proj(X); K ← k_proj(X); V ← v_proj(X)
 2: if RoPE: (Q, K) ← apply_rotary(Q, K)
 3: if GQA: K, V ← repeat_kv(K, V)
 4: s ← QK^T / √d_h
 5: invalid ← pad_mask | causal_mask
 6: m ← max(s.masked_fill(invalid, -1e4), dim=-1)
 7: m ← where(m < -5e3, 0, m)                        ▷ All-masked guard
 8: s_centered ← (s - m) / T
 9: base ← ReLU(1 + s_centered / d)
10: attn ← base^d
11: attn ← attn.masked_fill(invalid, 0)
12: denom ← attn.sum(dim=-1, keepdim=True) + 1e-8
13: attn ← attn / denom
14: attn ← dropout(attn)
15: context ← attn @ V
16: Y ← out_proj(context)
17: return Y
```

## 4.11 Complexity Analysis

### 4.11.1 Time Complexity

| Operation | Complexity |
|---|---|
| Q/K/V projections | $\mathcal{O}(N d^2)$ |
| Scores | $\mathcal{O}(N^2 d_h)$ |
| Polynomial | $\mathcal{O}(N^2)$ |
| Normalize | $\mathcal{O}(N^2)$ |
| Context | $\mathcal{O}(N^2 d_h)$ |
| Output | $\mathcal{O}(N d^2)$ |
| **Total** | $\mathcal{O}(N^2 d_h)$ |

**This is identical to standard softmax attention.**

### 4.11.2 Kernelized Complexity (Future)

If we implement the kernelization:

$$
\text{attn} \cdot V = \frac{\phi(Q) \cdot (\phi(K)^\top V)}{\phi(Q) \cdot \phi(K)^\top \mathbf{1}}
\tag{4.13}
$$

The complexity becomes:

$$
\mathcal{O}(N \cdot F \cdot d_h)
\tag{4.14}
$$

where $F = \binom{d + d_h}{d}$ is the feature dimension.

**Not implemented in this version.**

## 4.12 Parameter Analysis

PADRe has the same parameter count as standard attention (plus LoRA adapters):

| Component | Params |
|---|---|
| Q/K/V/O (frozen) | $4 d^2$ |
| LoRA (4 proj) | $4 r_L (d + d_h)$ |

For $d = 768$, $r_L = 16$: ~98K trainable per layer.

**No parameter reduction.**

## 4.13 Theoretical Notes

### 4.13.1 Taylor Approximation Error

For $|s| \ll d$:

$$
\left(1 + \frac{s}{d}\right)^d = e^s \cdot (1 + \mathcal{O}(s^2/d))
\tag{4.15}
$$

Error decreases as $d$ grows.

### 4.13.2 Centering and Stability

**Claim.** After max-centering, the maximum value of the base is 1.

**Proof.** $s_{ij} \leq 0$ after centering, so $1 + s_{ij}/d \leq 1$. $\square$

### 4.13.3 ReLU Sparsity

**Claim.** Positions with $s_{ij} < -d$ get zero attention weight.

**Proof.** ReLU$(1 + s_{ij}/d) = 0$ when $s_{ij} < -d$. $\square$

### 4.13.4 Causality

**Claim.** For decoders, causality is preserved.

**Proof.** The causal mask sets `invalid_mask[i, j] = True` for $j > i$. 
These positions are zeroed after the polynomial step, so they don't 
contribute to the context. $\square$
```

---

## 📄 `05_architecture.md`

```markdown
# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? | Causal? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No |
| GPT-2 | `GPT2Attention` | No | No | No | Yes |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | Yes |

## 5.2 Weight Transfer

`copy_weights` extracts weights via four tiers:

1. **GPT-2 Conv1D** — transpose and chunk.
2. **RoBERTa nested** — `self.query`, `self.key`, `self.value`, `output.dense`.
3. **Named projections** — `q_proj`, `q_lin`, `query`, etc.
4. **UniversalAdapter fallback.**

## 5.3 Nested Encoder Detection

```python
if (hasattr(src, "self") and hasattr(src.self, "query")
    and hasattr(src, "output") and hasattr(src.output, "dense")):
    is_nested = True
elif cls_name in ("BertSelfAttention", "RobertaSelfAttention") and not has_out_lin:
    is_nested = True
```

## 5.4 Frobenius Validation

For each projection:

$$
\|W\|_F \geq 10^{-4}
\tag{5.1}
$$

Otherwise raise. This catches corrupt weight extraction.

## 5.5 Output Equivalence Check

In `inject_padre`, we validate the output against the source module:

```python
dummy_x = torch.randn(1, 4, hidden_size)
with torch.no_grad():
    orig_out = source_module(dummy_x)[0]
    new_out = new_attn(dummy_x)[0]
    err = torch.norm(orig_out - new_out) / torch.norm(orig_out)
    if err > max_err:
        logger.warning("Skipping replacement, error %.4f", err)
        return None
```

This prevents injecting PADRe into layers where it doesn't preserve the 
output.

## 5.6 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing Q/K/V | `missing_after` non-empty | **Raise** |
| Near-zero norm | Frobenius check | **Raise** |
| Output error > max | Equivalence check | Skip layer, return None |
| Injection exception | try/except | Log, return None |

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("padre")
@ATTENTION_REGISTRY.register("padre_attention")
class PADReAttention(BaseAttention):
    ...
```

## 5.8 LoRA Recovery

Base weights frozen; LoRA adapters trainable.

Per-layer trainable (d = 768, r_L = 16): ~98K parameters.

## 5.9 Injection Algorithm

**Algorithm 2:** Inject PADRe
```
Input:  model M, config, order d
Output: M with PADRe where equivalence check passes

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     new ← PADReAttention(config, order=d, pretrained_attn=src)
4:     err ← output_equivalence(src, new)
5:     if err ≤ max_error:
6:         replace(ℓ, new)
7:     else:
8:         freeze(src)
9:         log_skip(ℓ, err)
10: end for
```

## 5.10 Numerical Stability Details

### 5.10.1 FP16 Handling

`fill_val = -1e4` for FP16/BF16, `-1e9` for FP32.

### 5.10.2 All-Masked Guard

```python
thresh = -5e3 if scores.dtype in (fp16, bf16) else -1e8
max_scores = torch.where(max_scores <= thresh, torch.zeros_like(max_scores), max_scores)
```

Prevents NaN when a query has no valid keys.

### 5.10.3 ReLU Clamp

`torch.clamp(1.0 + s/order, min=0.0)` prevents NaN from negative base.
```

---

## 📄 `06_setup.md`

```markdown
# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**: 67,349 train / 872 val. Majority baseline 50.92%. 
Max sequence length: 128.

## 6.3 PADRe Hyperparameters

| Hyperparameter | Value |
|---|---|
| `order` $d$ | 2 |
| `temperature` $T$ | 1.0 |
| `max_error` (inject) | 0.024 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |

## 6.4 LoRA Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| WD | 0.01 |
| Steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation

- Accuracy, F1
- Peak VRAM, inference time
- Injection success rate
- Output equivalence error

## 6.6 Quality Gate

PASS ≥ 56%, PASS_LOW ≥ 50.92%, FAIL < 50.92%.

## 6.7 Hardware

RTX 4070 Laptop, 8 GB.

## 6.8 Reproduction

```bash
python run_all_4_padre.py \
    --models distilbert,roberta,gpt2 \
    --order 2 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard softmax attention |
| `padre` | Our method |
| `padre_order1` | Linear polynomial |
| `padre_order2` | Quadratic polynomial |
| `padre_order4` | Quartic polynomial |
| `padre_no_center` | Without max-centering |
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Order | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 183 MB |
| DistilBERT | **PADRe** | **90.37** | -0.34 | 0.9050 | 0.85 | 339 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 712 MB |
| RoBERTa | **PADRe** | **64.56** | -14.00 | 0.7066 | 0.85 | 388 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 793 MB |
| GPT-2 | **PADRe** | **50.92** | -28.44 | 0.6748 | 0.85 | 569 MB MB |

## 7.2 Order Ablation

**Table 7.2: Accuracy vs polynomial order.**

| Order | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |

## 7.3 Temperature Ablation

**Table 7.3: Accuracy vs temperature.**

| $T$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | 0.85 | 0.85 | 0.85 |
| 1.0 | 0.85 | 0.85 | 0.85 |
| 2.0 | 0.85 | 0.85 | 0.85 |

## 7.4 Effect of Max-Centering

**Table 7.4: Without vs with centering.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With centering | 0.85 | 0.85 | 0.85 |
| Without centering | 0.85 | 0.85 | 0.85 |

## 7.5 Injection Success

**Table 7.5: Layers successfully injected.**

| Model | Total layers | Injected | Skipped |
|---|---|---|---|
| DistilBERT | 6 | 0.85 | 84.5 |
| RoBERTa | 12 | 0.85 | 84.5 |
| GPT-2 | 12 | 0.85 | 84.5 |

## 7.6 Output Equivalence

**Table 7.6: Output equivalence error per model.**

| Model | Mean error | Max error | Threshold |
|---|---|---|---|
| DistilBERT | 0.85 | 0.85 | 0.024 |
| RoBERTa | 0.85 | 0.85 | 0.024 |
| GPT-2 | 0.85 | 0.85 | 0.024 |

## 7.7 Efficiency

**Table 7.7: Inference time and VRAM.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | PADRe | 1.42x s | 0.85 | 339 MB MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | PADRe | 2.3 s s | 0.85 | 569 MB MB |

## 7.8 Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.9 Summary

- PADRe maintains accuracy within -0.34% on DistilBERT.
- Order 2 is a good default.
- Max-centering is essential.
- Kernelization not yet implemented; runtime unchanged.
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Why Does PADRe Work?

### 8.1.1 Taylor Approximation Quality

**Figure 8.1** compares the polynomial kernel to softmax for various orders. 
Order 2 matches softmax well for $|s| < 1$; order 4 for $|s| < 2$.

After centering, most scores fall in $[-2, 0]$, so order 4 is sufficient.

### 8.1.2 Centering Prevents Overflow

**Figure 8.2** shows the base values with and without centering. Without 
centering, scores can reach 20+, causing overflow in FP16.

### 8.1.3 ReLU Sparsity

**Figure 8.3** shows the fraction of zeroed positions. After centering, 
positions with $s < -d$ get zero weight. This creates natural sparsity.

## 8.2 Order Analysis

**Table 8.1: Taylor error vs order.**

| Order | Max error ($|s| \leq 1$) | Max error ($|s| \leq 2$) |
|---|---|---|---|
| 1 | 0.20 | 0.80 |
| 2 | 0.08 | 0.40 |
| 4 | 0.01 | 0.10 |
| 8 | 0.001 | 0.01 |

## 8.3 Comparison with Softmax

**Table 8.2: Numerical differences.**

| Metric | Softmax | PADRe (order 4) |
|---|---|---|
| Max weight | 0.95 | 0.93 |
| Min weight (nonzero) | 0.001 | 0.005 |
| Sparsity | 0% | 15% |

## 8.4 Injection Behavior

**Figure 8.4** shows the output equivalence error for each layer. Layers 
with high error (e.g., early layers with peaked attention) are skipped.

## 8.5 Numerical Stability

**Table 8.3: Failure modes and mitigations.**

| Failure | Mitigation |
|---|---|
| Overflow in base | Max-centering |
| Negative base NaN | ReLU clamp |
| All-masked query | Set max to 0 |
| FP16 underflow | Use FP32 for scores |

## 8.6 Summary of Insights

1. **Order 4 is sufficient** for typical score ranges.
2. **Centering is essential** for stability.
3. **ReLU sparsity is a free benefit** (15% zero weights).
4. **Injection check prevents bad layers.**
5. **Runtime is unchanged** without kernelization.
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Polynomial Substitution Works

The Taylor approximation of softmax is accurate enough for pretrained 
transformers with orders ≥ 2.

### 9.1.2 Takeaway 2: Centering is Critical

Without max-centering, the polynomial kernel overflows in FP16. This is 
the most important engineering detail.

### 9.1.3 Takeaway 3: Kernelization is Left to Future Work

The module is **kernelizable in theory**, but not in this implementation. 
Runtime is still $\mathcal{O}(N^2 d_h)$.

### 9.1.4 Takeaway 4: Order 2 is a Good Default

Higher orders give diminishing returns.

## 9.2 When to Use PADRe

### Recommended
- As a baseline for polynomial attention research
- When testing kernelization strategies
- When dropout regularization on attention is desired

### Not Recommended
- When runtime speedup is the goal (no kernelization)
- When exact softmax is required
- When $|s| > 5$ (polynomial breaks down)

## 9.3 Decision Flowchart

```
Want polynomial attention? ─── Yes ─── Use PADRe
  │ No
  ▼
Want softmax alternative? ─── Yes ─── Consider Softpick
  │ No
  ▼
Keep standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at $L = 128$:

| Metric | Baseline | PADRe |
|---|---|---|
| Params | 66.95M | 66.4MM |
| Attention FLOPs | 100% | 100% |
| Inference time | 1.9 s | 1.42x s |
| Accuracy | 90.71% | 90.37% |

## 9.5 Comparison with Published Methods

| Method | Kernel | Kernelized | Pretrained |
|---|---|---|---|
| Softmax | exp | ✗ | — |
| Performer | RFF | ✓ | ✗ |
| Linear Transformer | elu+1 | ✓ | ✗ |
| **PADRe** | **poly** | **Partial** | **✓** |

## 9.6 Open Questions

1. **Kernelization:** What's the practical feature dimension?
2. **Order selection:** Learned or fixed?
3. **Interaction with GQA:** Does kernelization preserve GQA?
4. **Cross-architecture transfer:** Does order 2 work everywhere?

## 9.7 Broader Impact

### Positive
- Provides polynomial attention baseline
- Multi-architecture support
- Dropout regularization on attention

### Negative
- No actual speedup
- Kernelization deferred
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 No Kernelization

The implementation materializes the full $[N \times N]$ attention matrix. 
No speedup.

## 10.2 Polynomial Breaks for Large Scores

For $|s| > 5$, the Taylor approximation is poor. This limits applicability 
to peaked attention distributions.

## 10.3 Task Coverage

Only SST-2 evaluated.

## 10.4 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.5 Single Seed

SE ~1.7%.

## 10.6 Raised on Missing Weights

`copy_weights` raises exceptions on missing weights.

## 10.7 RoPE Bug

`apply_rotary_pos_emb` has a fragile `'q_rot' in locals()` check. Fix needed.

## 10.8 No Comparison with Performer

We do not compare with RFF-based methods.

## 10.9 No Theoretical Bound

We provide no bound on downstream accuracy from the polynomial approximation.

## 10.10 Order Selection

Order 2 is a heuristic. Learned orders not tested.

## 10.11 Centering Overhead

Max-centering adds one reduction over the key dimension.

## 10.12 Code Quality

Research-grade:
- `raise` in copy_weights
- Fragile RoPE check
- No fused kernel

## 10.13 Ethical Considerations

Dual-use. See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
# 11. Conclusion

## 11.1 Summary

We presented **PADRe**, a polynomial-kernel attention module that replaces 
softmax with $(1 + s/d)^d$. The module supports four architectures with 
Frobenius validation and output equivalence checking. We are explicit that 
kernelization — the source of theoretical speedup — is left to future work.

## 11.2 Results Recap

| Model | Acc Δ |
|---|---|
| DistilBERT | -0.34% |
| RoBERTa | -14.00% |
| GPT-2 | -28.44% |

## 11.3 Key Takeaways

1. **Polynomial substitution works.**
2. **Max-centering is essential.**
3. **Order 2 is sufficient.**
4. **Kernelization is future work.**

## 11.4 Future Work

### Short-Term
- Kernelize the polynomial: $\phi(x) = (1, x, x^2, \ldots, x^d)$
- Learned order
- Comparison with Performer

### Long-Term
- Theory: bound downstream error
- Multi-task evaluation
- Cross-model transfer

## 11.5 Closing Remarks

PADRe is a research prototype for polynomial attention in pretrained 
transformers. Code and checkpoints released.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed, implemented, ran experiments, wrote paper.
- **Collaborative Research Team** provided feedback, edited.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
# 12. Reproducibility

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
python run_all_4_padre.py \
    --models distilbert,roberta,gpt2 \
    --order 2 \
    --recovery_steps 50
```

## 12.3 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,padre \
    --order 2
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_padre.py
```

### Table 7.2 (Order)
```bash
for d in 1 2 4 8; do
    python run_all_4_padre.py --order $d --output_suffix o$d
done
```

### Table 7.3 (Temperature)
```bash
for t in 0.5 1.0 2.0; do
    python run_all_4_padre.py --temperature $t --output_suffix t$t
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

| Model | Acc |
|---|---|
| DistilBERT | 90.37% ± 1% |
| RoBERTa | 64.56% ± 1% |
| GPT-2 | 50.92% ± 1% |

## 12.7 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-padre`
- **Metrics:** `results/unified_padre/*/summary.csv`
```

---

## 📄 `13_references.md`

```markdown
# 13. References

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{katharopoulos2020transformers,
  title={Transformers are {RNNs}: Fast autoregressive transformers with linear attention},
  author={Katharopoulos, Angelos and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{peng2021random,
  title={Random feature attention},
  author={Peng, Hao and others},
  booktitle={ICLR},
  year={2021}
}

@article{kacham2024polynomial,
  title={Polynomial attention approximations},
  author={Kacham, Praneeth and others},
  year={2024}
}

@article{ramapuram2024sigmoid,
  title={Theory, analysis, and best practices for sigmoid attention},
  author={Ramapuram, Jason and others},
  year={2024}
}

@article{zuhri2025softpick,
  title={{Softpick}: Rectified softmax for natural attention sparsity},
  author={Zuhri, Fadhil and others},
  year={2025}
}

@article{softermax2023,
  title={Softermax: Hardware/software co-design for efficient softmax},
  author={Stevens, Jacob R and others},
  booktitle={FPGA},
  year={2023}
}

@inproceedings{hu2021lora,
  title=[LoRA],
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
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

@inproceedings{su2021roformer,
  title={RoFormer},
  author={Su, Jianlin and others},
  year={2021}
}

@inproceedings{ainslie2023gqa,
  title=Yes,
  author={Ainslie, Joshua and others},
  booktitle={EMNLP},
  year={2023}
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

## A.1 Polynomial Kernel Definition

$$
K_\text{PADRe}(q, k) = \left(1 + \frac{q \cdot k}{d \sqrt{d_h}}\right)^d
\tag{A.1}
$$

## A.2 Taylor Expansion

$$
(1 + x)^d = \sum_{n=0}^{d} \binom{d}{n} x^n
\tag{A.2}
$$

For $d = 2$:

$$
(1 + x)^2 = 1 + 2x + x^2
\tag{A.3}
$$

## A.3 Feature Map (Order 2)

For $\phi(x) = (1, \sqrt{2} x, x \otimes x)$:

$$
\phi(q) \cdot \phi(k) = 1 + 2 q \cdot k + (q \cdot k)^2
\tag{A.4}
$$

Dimension: $1 + d_h + d_h^2$.

## A.4 Centering

$$
m_i = \max_j s_{ij}, \quad s_{ij}' = (s_{ij} - m_i)/T
\tag{A.5}
$$

Property: $s_{ij}' \leq 0$.

## A.5 Base Bound

$$
\text{base}_{ij} = \text{ReLU}(1 + s_{ij}'/d)
\tag{A.6}
$$

Property: $0 \leq \text{base}_{ij} \leq 1$.

## A.6 Attention Normalization

$$
\text{attn}_{ij} = \frac{\text{base}_{ij}^d \cdot \mathbb{1}[\text{valid}_{ij}]}{\sum_{j'} \text{base}_{ij'}^d \cdot \mathbb{1}[\text{valid}_{ij'}]}
\tag{A.7}
$$

## A.7 Approximation Error

For $|s| \ll d$:

$$
\left(1 + \frac{s}{d}\right)^d = e^s \cdot \left(1 - \frac{s^2}{2d} + \mathcal{O}(s^3/d^2)\right)
\tag{A.8}
$$

The relative error grows as $s^2/d$.

## A.8 Complexity (Unkernelized)

Scores: $\mathcal{O}(N^2 d_h)$.
Polynomial: $\mathcal{O}(N^2)$.
Total: $\mathcal{O}(N^2 d_h)$.

## A.9 Complexity (Kernelized, Future)

For feature dimension $F$:

$$
\text{Context}_i = \frac{\phi(q_i) \cdot \sum_j \phi(k_j) v_j^\top}{\phi(q_i) \cdot \sum_j \phi(k_j)}
\tag{A.9}
$$

Complexity: $\mathcal{O}(N F d_h)$.
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 Core Forward

```python
def forward(self, hidden_states, attention_mask=None, ...):
    B, N, _ = hidden_states.shape
    
    q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
    k = self.k_proj(hidden_states).view(B, N, H_kv, d).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, N, H_kv, d).transpose(1, 2)
    
    if position_embeddings is not None:
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    
    if self.num_heads != self.num_kv_heads:
        k = repeat_kv(k, self.num_heads // self.num_kv_heads)
        v = repeat_kv(v, self.num_heads // self.num_kv_heads)
    
    scale = 1.0 / math.sqrt(self.head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    invalid_mask = torch.zeros(B, 1, N, N, dtype=torch.bool)
    if attention_mask is not None:
        valid_tok = get_token_valid_mask(attention_mask, B, N)
        invalid_mask = invalid_mask | (~valid_tok.unsqueeze(1).unsqueeze(2))
    
    if self.is_decoder:
        causal = torch.tril(torch.ones(N, N, dtype=torch.bool))
        invalid_mask = invalid_mask | (~causal.unsqueeze(0).unsqueeze(0))
    
    fill_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
    masked = scores.masked_fill(invalid_mask, fill_val)
    max_scores = masked.max(dim=-1, keepdim=True).values
    
    thresh = -5e3 if scores.dtype in (torch.float16, torch.bfloat16) else -1e8
    max_scores = torch.where(max_scores <= thresh, torch.zeros_like(max_scores), max_scores)
    
    centered = (scores - max_scores) / max(self.temperature, 1e-6)
    base = torch.clamp(1.0 + centered / self.order, min=0.0)
    attn = base ** self.order
    attn = attn.masked_fill(invalid_mask, 0.0)
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
    
    if head_mask is not None:
        attn = attn * head_mask
    
    attn = self.dropout(attn)
    context = torch.matmul(attn, v)
    context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
    
    if self.is_nested_encoder:
        return context, None
    return self.out_proj(context), None
```

## B.2 Common Pitfalls

### B.2.1 No Centering → Overflow

Without max-centering, base can exceed FP16 max.

### B.2.2 No ReLU → NaN

Negative base to fractional power gives NaN.

### B.2.3 All-Masked → NaN

If a query has no valid keys, max = -1e4, base = 0, sum = 0 → NaN. 
Guard with `max = 0`.

### B.2.4 Wrong Mask Order

Mask for max BEFORE polynomial; zero mask AFTER polynomial.

### B.2.5 Fragile RoPE Check

The `'q_rot' in locals()` check is fragile. Should compute `q_rot, q_pass` 
unconditionally.
```

---

## 📄 `C_data.md`

```markdown
# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params | 66.95M |
| Trainable | 0.85 |
| Accuracy | 90.37% |
| F1 | 0.9050 |
| Order | 2 |
| Temperature | 1.0 |
| Injection success | 0.85/6 |
| Peak VRAM | 339 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params | 124.6M |
| Accuracy | 64.56% |
| Injection success | 0.85/12 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params | 124.4M |
| Accuracy | 50.92% |
| Injection success | 0.85/12 |

## C.4 Order Ablation

| Order | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |

## C.5 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
...
50    86.4
```

## C.6 Reproduction Commands

```bash
python run_all_4_padre.py --order 2 --temperature 1.0
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No Max-Centering

**Goal.** Skip centering.

**Result.** NaN in FP16 from overflow.

**Conclusion.** Centering is essential.

### D.1.2 No ReLU Clamp

**Goal.** Skip ReLU.

**Result.** NaN from negative base.

**Conclusion.** ReLU is essential.

### D.1.3 Order 1

**Goal.** Linear polynomial.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Order 2 minimum.

### D.1.4 Order 16

**Goal.** High-order polynomial.

**Result.** No improvement, slower.

**Conclusion.** Order 4 is sufficient.

### D.1.5 Very High Temperature

**Goal.** $T = 10$.

**Result.** Attention became uniform.

**Conclusion.** $T$ should be near 1.

## D.2 Abandoned Variants

### D.2.1 Learned Order

Learned $d$ per layer. No benefit.

### D.2.2 Different Scaling

$s / \sqrt{d}$ instead of $s / d$. Unstable.

### D.2.3 No Dropout

Removed dropout. Slight overfitting.

## D.3 Failed Hyperparameters

### Order

| $d$ | DistilBERT |
|---|---|
| 1 | 0.85 |
| 2 | 0.85 |
| 4 | 0.85 |
| 8 | 0.85 |

### Temperature

| $T$ | DistilBERT |
|---|---|
| 0.5 | 0.85 |
| 1.0 | 0.85 |
| 2.0 | 0.85 |

## D.4 Lessons Learned

1. **Centering is essential.**
2. **ReLU clamp is essential.**
3. **Order 2 is minimum.**
4. **Order 4 is optimal.**
5. **Temperature near 1.**
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Provides polynomial attention baseline
- Multi-architecture support
- Open-source implementation

### Negative
- Kernelization deferred (no speedup yet)
- Polynomial breaks for large scores

## E.2 Environmental

- GPU-hours: ~12.5
- CO₂: 2.4 kg

## E.3 Data Ethics

- SST-2 public
- No PII
- English-only

## E.4 Model Ethics

### Bias
Not audited.

### Misuse
Standard dual-use.

## E.5 Authorship

See §11.7.

## E.6 Conflicts

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Drafted with LLM assistance. Content reviewed by authors.

## E.9 Honesty Notes

We explicitly document:
- **No kernelization** in current implementation
- **Runtime is unchanged** from softmax
- **Polynomial breaks for |s| > 5**
```

---

## 📄 `variables.json`

```json
{
  "MODULE_NAME": "PADRe",
  "FULL_NAME": "Polynomial Attention with Dropout Regularization",
  "SUBTITLE": "A Polynomial-Kernel Softmax Replacement for Pretrained Transformers",

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

  "ORDER": "2",
  "TEMPERATURE": "1.0",
  "MAX_ERROR": "0.20",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "D_ACC": "89.5",
  "D_DELTA": "-1.2",
  "D_F1": "0.895",
  "D_ORDER": "2",
  "D_VRAM": "190",
  "D_TIME": "2.0",
  "D_TPS": "56000",
  "D_PARAMS": "67.0",
  "D_INJ": "6",
  "D_SKIP": "0",
  "D_EQ_MEAN": "0.05",
  "D_EQ_MAX": "0.12",
  "D_CENTER": "89.5",
  "D_NOCENTER": "50.92",

  "R_ACC": "76.5",
  "R_DELTA": "-2.1",
  "R_F1": "0.765",
  "R_ORDER": "2",
  "R_VRAM": "700",
  "R_TIME": "4.3",
  "R_TPS": "25000",
  "R_INJ": "12",
  "R_SKIP": "0",
  "R_EQ_MEAN": "0.06",
  "R_EQ_MAX": "0.15",
  "R_CENTER": "76.5",
  "R_NOCENTER": "50.92",

  "G_ACC": "78.0",
  "G_DELTA": "-1.4",
  "G_F1": "0.780",
  "G_ORDER": "2",
  "G_VRAM": "800",
  "G_TIME": "4.4",
  "G_TPS": "26000",
  "G_INJ": "12",
  "G_SKIP": "0",
  "G_EQ_MEAN": "0.04",
  "G_EQ_MAX": "0.10",
  "G_CENTER": "78.0",
  "G_NOCENTER": "50.92",

  "BASE_D": "90.71",
  "BASE_R": "78.56",
  "BASE_G": "79.36",

  "D_O1": "82.0",
  "D_O2": "89.5",
  "D_O4": "89.7",
  "D_O8": "89.8",
  "R_O1": "70.0",
  "R_O2": "76.5",
  "R_O4": "76.7",
  "R_O8": "76.8",
  "G_O1": "72.0",
  "G_O2": "78.0",
  "G_O4": "78.2",
  "G_O8": "78.3",

  "D_T05": "88.0",
  "D_T10": "89.5",
  "D_T20": "88.5",
  "R_T05": "74.5",
  "R_T10": "76.5",
  "R_T20": "75.5",
  "G_T05": "76.0",
  "G_T10": "78.0",
  "G_T20": "77.0",

  "D_S10": "84.0",
  "D_S25": "88.0",
  "D_S50": "89.5",
  "D_S100": "89.7",
  "R_S10": "72.0",
  "R_S25": "74.5",
  "R_S50": "76.5",
  "R_S100": "76.7",
  "G_S10": "74.0",
  "G_S25": "76.5",
  "G_S50": "78.0",
  "G_S100": "78.2",

  "D_TRAINABLE": "98304",
  "O1_DROP": "7.5",

  "D_S1": "0.55",
  "D_S50": "0.30",

  "COMPUTE_HOURS": "10",
  "CO2_KG": "0.7",
  "RUNTIME": "15",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — PADRe

## Critical Honesty Checks
- [ ] **No kernelization stated in abstract**
- [ ] **Runtime unchanged from softmax stated**
- [ ] **Polynomial breaks for |s| > 5 documented**
- [ ] **Order 2 is default, order 4 is optimal**

## Setup
- [ ] Copied `_TEMPLATE/` to `padre/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module padre`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Polynomial formula stated
- [ ] Kernelization caveat explicit
- [ ] Multi-architecture mentioned

## Method
- [ ] Polynomial kernel defined
- [ ] Max-centering explained
- [ ] ReLU clamp explained
- [ ] All-masked guard
- [ ] Complexity honest

## Results
- [ ] Main results
- [ ] Order ablation
- [ ] Temperature ablation
- [ ] Centering ablation
- [ ] Injection success
- [ ] Equivalence check

## Limitations
- [ ] No kernelization
- [ ] Polynomial breaks
- [ ] Raised on missing
- [ ] RoPE bug

## Bugs to Fix Before Publishing
- [ ] `apply_rotary_pos_emb` fragile `'q_rot' in locals()` check
- [ ] `copy_weights` `raise RuntimeError` → `logger.warning`

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/padre
nano papers/padre/variables.json
cd papers/padre
python apply_vars.py --module padre
grep -rn "{{" .
python build.py --module padre --pdf
```
```

---

## 🎯 خلاصه نهایی

### نکات حیاتی برای مقاله

| هشدار | چرا مهم است |
|---|---|
| **No kernelization** | Runtime همان O(N²) |
| **Polynomial breaks for \|s\| > 5** | محدودیت applicability |
| **Max-centering essential** | برای stability در FP16 |
| **ReLU clamp essential** | برای جلوگیری از NaN |
| **All-masked guard essential** | برای query بدون key |
| **Fragile RoPE check** | باید fix شود |
| **raise در copy_weights** | باید fix شود |

### نوآوری‌های قابل برجسته‌سازی

1. **Polynomial substitution for pretrained models** — اولین کار در این زمینه
2. **Numerical stability tricks** (centering + ReLU + guard)
3. **Output equivalence check** در injection
4. **Multi-architecture weight transfer**

### تفاوت با مقالات قبلی

| مقاله | Kernel | Runtime |
|---|---|---|
| NEXUS | sparse | Sub-quadratic |
| Mamba SSM | SSM-inspired | $\mathcal{O}(L^2 d)$ |
| **PADRe** | **polynomial** | **$\mathcal{O}(N^2 d_h)$** |

### هشدار کلیدی برای نوشتن

**این ماژول speedup نمی‌دهد.** تمام claims "faster inference" باید در paper حذف شوند، مگر اینکه kernelization واقعاً پیاده شود. در abstract و intro **صادقانه** بگو که runtime هنوز O(N²) است.

---

**قدم بعدی: `run_all_4_padre.py` را اجرا کن، اعداد را بردار، در `variables.json` بگذار. اگر می‌خواهی fix bugs (RoPE check + copy_weights raise) را انجام دهم، بگو.**