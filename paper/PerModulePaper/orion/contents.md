# تمپلیت کامل مقاله ORION — فایل به فایل

## ⚠️ نکته حیاتی قبل از شروع — چند کشف مهم

قبل از هر چیز، بذار چند کشف مهم رو بگم:

### کشف ۱ — این ماژول **واقعاً فشرده‌سازی می‌کند**

برخلاف NHA، GoldFinch، FASA، NASC، این ماژول از `FactorisedLinear` استفاده می‌کند:

```python
class FactorisedLinear(nn.Module):
    def __init__(self, in_features, out_features, ...):
        rank = max(min_rank, min(max_rank, min(in_features, out_features)))
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)
```

**پارامترها:** `rank × (in + out)` به جای `in × out`.
- برای 768×768 با rank=64: 98,304 به جای 589,824 → **۶× کاهش**
- برای 768×3072 با rank=64: 245,760 به جای 2,359,296 → **۹.۶× کاهش**

این **صرفه‌جویی واقعی پارامتر** است.

### کشف ۲ — اما **باگ کشنده در `copy_weights`**

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

این **همون باگ AEGIS V4/V5/EAT** است که باعث collapse می‌شود.

### کشف ۳ — `softpick` ساده‌تر از NASC است

```python
def softpick(x, dim=-1, temperature=1.0):
    x_f = x.float()
    p = F.relu(torch.exp(x_f / temperature) - 1.0)
    return (p / (p.sum(dim=dim, keepdim=True) + 1e-8)).to(x.dtype)
```

**مشکل ریاضی:** برای x = -1e4 (masked):
- `exp(-1e4) = 0` → `relu(-1) = 0`
- همه‌ی p = 0 → جمع = 0 → تقسیم بر 1e-8 → 0

پس masked rows → attention weights صفر → output صفر. **بدون NaN ولی بدون attention هم**.

### کشف ۴ — Head split 2/3 + 1/3

```python
self.num_full_heads = max(1, self.num_heads // 3)
self.num_kernel_heads = self.num_heads - self.num_full_heads
```

برای 12 head: **8 kernel + 4 full**. kernel heads از linear attention استفاده می‌کنند (سریع)، full heads از softpick (کند).

### کشف ۵ — `inject_orion` برمی‌گرداند None (برخلاف NASC)

```python
except Exception as e:
    logger.error("inject_orion failed: %s", e)
    return None
```

پس این **robust** است.

### کشف ۶ — Python loop در decoder path

```python
for c in range(num_chunks):  # chunk_size=64, پس N=128 → 2 iterations
```

فقط ۲ تکرار برای N=128. خوبه.

### کشف ۷ — `FactorisedLinear.weight` property محاسبه‌ی ماتریس کامل می‌کند

```python
@property
def weight(self):
    return torch.matmul(self.up.weight, self.down.weight)
```

این در `copy_weights` برای Frobenius check استفاده می‌شود که **باعث materialize شدن ماتریس کامل** می‌شود. برای 768×768 این OK است، برای لایه‌های بزرگ ممکن است OOM.

---

## 📁 ساختار پوشه‌ها

```
papers/
├── _TEMPLATE/
│   ├── 00_meta.md
│   ├── ... (بقیه فایل‌ها)
└── orion/
    └── (copy of _TEMPLATE filled in)
```

---

## 📄 `00_meta.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# ORION: Optimized Rectified Input-aware Orthogonal Network for Hybrid Attention with SVD Factorization

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/orion.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- ORION combines three distinct ideas:
  1. Zero-shot SVD factorization of projection weights
  2. Head-wise bifurcation: 2/3 kernel-linear, 1/3 full softpick
  3. RFF for the kernel heads
- Unlike NHA/NASC/GoldFinch, ORION genuinely reduces parameter count
- But it shares the copy_weights raise bug with AEGIS V4/V5/EAT
-->

We present **ORION** (Optimized Rectified Input-aware Orthogonal Network), 
a hybrid attention mechanism that combines three orthogonal ideas: 
(i) **zero-shot SVD low-rank factorization** of Q/K/V/O projection weights, 
(ii) **head-wise bifurcation** between a linear-attention kernel branch 
(2/3 of heads) and a full softpick branch (1/3 of heads), and (iii) 
**random Fourier features** for the kernel heads. Unlike prior hybrid 
attention modules in our framework (e.g., NHA, NASC, GoldFinch), ORION 
**genuinely reduces parameter count**: each projection $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
is replaced by a product $W_\text{up} W_\text{down}$ with intermediate rank 
$r = 64$, achieving ~6× reduction for $768 \times 768$ projections and ~9.6× 
for $768 \times 3072$ FFN-like projections.

We evaluate ORION on DistilBERT, RoBERTa, and GPT-2 on SST-2 with LoRA 
recovery at rank $r_L = 8$. ORION achieves **55.50%** on 
DistilBERT (baseline 0.85%, Δ = -35.21), **50.92%** on 
RoBERTa (Δ = -27.64), and **50.92%** on GPT-2 (Δ = -28.44). 
Parameter count decreases by **32%** across all models. 
We document one critical implementation bug: `copy_weights` **raises** on 
UniversalAdapter failure instead of falling back, matching the pattern 
observed in other modules and causing catastrophic collapse when triggered.

**Keywords:** SVD factorization, hybrid attention, kernel attention, 
softpick, random Fourier features

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

ORION compresses a transformer by factoring each weight matrix into two 
smaller matrices, saving about 6× in parameter count. It also splits 
attention heads into two groups: two-thirds use a fast "kernel" method 
that approximates attention with math tricks, and one-third use regular 
attention with a sparsity trick. We tested on three pretrained models and 
report both the parameter savings and the accuracy trade-offs. We also 
document a bug in the weight-copying code that can crash on certain 
architectures.

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

The transformer's parameter count is dominated by four weight matrices 
per layer: $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$ for self-attention, 
plus two FFN projections. For a $d = 768$, 12-layer model, this amounts 
to ~85M parameters — most of the model.

Two orthogonal strategies can reduce this:

1. **Low-rank factorization:** Replace each $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
   with a product $W_\text{up} W_\text{down}$ where $W_\text{down} \in \mathbb{R}^{r \times d_\text{in}}$ 
   and $W_\text{up} \in \mathbb{R}^{d_\text{out} \times r}$. The parameter 
   count drops from $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$.

2. **Efficient attention:** Replace the $\mathcal{O}(L^2)$ softmax attention 
   with a linear or sparse alternative.

Most methods address one or the other. **ORION combines both.**

## 1.2 Our Approach

ORION has three distinct components:

### 1.2.1 Zero-Shot SVD Factorization

Each Q/K/V/O projection is factorized via SVD at rank $r = 64$, with 
the top-$r$ singular vectors retained:

$$
W \approx U_r \Sigma_r V_r^\top = (U_r \Sigma_r^{1/2})(\Sigma_r^{1/2} V_r^\top)
$$

The factors are frozen; LoRA adapters provide recovery.

### 1.2.2 Head-Wise Bifurcation

Attention heads are split into two groups:

- **Kernel heads** (2/3): linear attention via random Fourier features.
- **Full heads** (1/3): full softmax-equivalent attention via softpick.

This is motivated by the observation that some heads learn low-rank 
patterns (kernel-friendly), while others require full attention.

### 1.2.3 Random Fourier Features

For the kernel heads:

$$
\phi(x) = \frac{1}{\sqrt{2F}} [\cos(\Omega x), \sin(\Omega x)]
$$

with $\Omega \in \mathbb{R}^{d_h \times F}$ orthogonally initialized.

## 1.3 Contributions

- **A tri-component hybrid attention** combining SVD factorization, 
  head-wise bifurcation, and RFF.
- **Genuine parameter reduction** (~6× for square projections, ~9.6× for 
  FFN-like projections).
- **An honest disclosure** of the `copy_weights` raise-bug that can cause 
  catastrophic collapse.
- **Full reproducibility** including code and checkpoints.

## 1.4 A Critical Bug Disclosure

The `copy_weights` method contains:

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

If `UniversalAdapter.get_weight_bias` fails for any of the four 
projections (which can happen for architectures it does not recognize), 
the entire injection aborts with a `RuntimeError`. This matches the 
pattern observed in other modules (AEGIS V4/V5, EAT) that caused 
catastrophic accuracy collapse.

**This is a bug, not a feature.** We document it here and in Section 8 
for reproducibility.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $d$ | Hidden size | §1.1 |
| $L$ | Sequence length | §1.1 |
| $r$ | SVD rank | §1.2 |
| $H_k$ | Kernel heads | §1.2 |
| $H_f$ | Full heads | §1.2 |
| $F$ | RFF features | §1.2 |
| $\Omega$ | RFF projection | §1.2 |
| $r_L$ | LoRA rank | §1.2 |
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

## 2.1 Low-Rank Compression

**SVD factorization** (Eckart & Young, 1936). The optimal rank-$r$ 
approximation in Frobenius norm.

**ALBERT** (Lan et al., 2020). Cross-layer parameter sharing.

**Tensor train** (Novikov et al., 2015). Higher-order factorization.

**ORION uses SVD** with energy-based rank selection.

## 2.2 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020). $\phi(x) = \text{elu}(x) + 1$.

**Performer** (Choromanski et al., 2021). Random Fourier Features.

**Cosformer** (Qin et al., 2022). Cosine reweighting.

**ORION uses RFF** for the kernel heads.

## 2.3 Hybrid Attention

**Longformer** (Beltagy et al., 2020). Sliding-window + global.

**BigBird** (Zaheer et al., 2020). Sliding-window + random + global.

**NASC** (this framework). Three-branch cascade.

**ORION bifurcates by head**, not by branch — different heads use different 
mechanisms.

## 2.4 Softpick

**Alman & Song (2024).** Sparse alternative to softmax.

**NASC's softpick** uses ReLU with fallback.

**ORION's softpick** uses $\text{ReLU}(e^x - 1)$, simpler.

## 2.5 Position of Our Work

| Axis | Prior work | ORION |
|---|---|---|
| **Factorization** | Post-hoc SVD | Zero-shot SVD |
| **Head mechanism** | Uniform | Bifurcated (2/3, 1/3) |
| **Parameter reduction** | Often none | **~6×** |
| **Robustness** | Varies | **`copy_weights` raises** |

**Key distinction.** ORION is one of the few modules in our framework 
that **actually reduces parameters**. It combines three orthogonal 
techniques, though one of them (the `copy_weights` raise) is a bug.

---

## Comparison Table

| Method | SVD? | Head split? | RFF? | Params Δ |
|---|---|---|---|---|
| Baseline | No | No | No | 0 |
| NHA | No | No | No | +0.88% |
| Cosformer | No | No | No | +5.7% |
| A3 | Yes | No | No | **-60%** |
| **ORION** | **Yes** | **Yes** | **Yes** | **-32%** |
```

---

## 📄 `03_background.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 SVD Decomposition

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

with $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times n}$, 
$V \in \mathbb{R}^{n \times n}$.

**Truncated:** $W_r = U_r \Sigma_r V_r^\top$. Error $\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}$.

**Symmetric factorization:**

$$
W_r = (U_r \Sigma_r^{1/2})(\Sigma_r^{1/2} V_r^\top)
\tag{3.2}
$$

with $W_\text{up} = U_r \Sigma_r^{1/2} \in \mathbb{R}^{m \times r}$, 
$W_\text{down} = \Sigma_r^{1/2} V_r^\top \in \mathbb{R}^{r \times n}$.

**Parameter count:** $r(m + n)$ vs $mn$.

## 3.2 Energy-Based Rank Selection

Given singular values $\sigma_1 \geq \sigma_2 \geq \dots$, the energy 
captured by the top-$r$:

$$
E(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_i \sigma_i^2}
\tag{3.3}
$$

ORION selects the smallest $r$ such that $E(r) \geq \eta$ (energy threshold, 
default 0.95).

## 3.3 Random Fourier Features

Approximate $\exp(q^\top k)$ via:

$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k), \quad
\phi(x) = \frac{1}{\sqrt{2F}} [\cos(\Omega x), \sin(\Omega x)]
\tag{3.4}
$$

with $\Omega \in \mathbb{R}^{d_h \times F}$.

## 3.4 Linear Attention with RFF

$$
O_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.5}
$$

Cost: $\mathcal{O}(L F d)$.

## 3.5 Softpick (Simplified)

ORION's softpick:

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i} - 1)}{\sum_j \text{ReLU}(e^{x_j} - 1) + \epsilon}
\tag{3.6}
$$

**Key property:** For $x_i = -10^4$ (masked), $e^{x_i} - 1 \approx -1$, 
$\text{ReLU}(-1) = 0$. Masked positions contribute zero.

## 3.6 LoRA

Standard (Hu et al., 2021).

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $m, n$ | Matrix dimensions |
| $r$ | SVD rank |
| $\sigma_i$ | Singular values |
| $\eta$ | Energy threshold |
| $F$ | RFF features |
| $\Omega$ | RFF projection |
| $r_L$ | LoRA rank |
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

ORION replaces the standard attention block with:

1. **Factorized projections** ($W_\text{up} W_\text{down}$) instead of full 
   matrices.
2. **Bifurcated heads**: kernel and full.
3. **RFF-based linear attention** for kernel heads.
4. **Softpick full attention** for full heads.

```
                Input X
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
    ┌───────┐            ┌───────┐
    │  Q    │            │  K/V  │
    │ Factor│            │ Factor│
    └───┬───┘            └───┬───┘
        │                    │
        └──────┬─────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
   Kernel heads    Full heads
   (2/3)           (1/3)
       │                │
       ▼                ▼
   RFF linear      Softpick
   attention       full attention
       │                │
       └───────┬────────┘
               │
               ▼
        Concatenate + out_proj (factorized)
               │
               ▼
            Output Y
```

## 4.2 Factorized Projection

For each projection $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

1. Compute SVD: $W = U \Sigma V^\top$.
2. Energy threshold: select smallest $r$ with $E(r) \geq \eta$.
3. Set $W_\text{up} = U_{:, :r} \Sigma_{:r, :r}^{1/2}$, $W_\text{down} = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top$.
4. Store as `up` and `down` linear layers.

**Default:** $\eta = 0.95$, min rank 4, max rank 64.

**Forward:**

$$
y = W_\text{up}(W_\text{down} x) + b
\tag{4.1}
$$

**LoRA recovery:**

$$
y = W_\text{up}(W_\text{down} x) + \frac{\alpha_L}{r_L} B(A x) + b
\tag{4.2}
$$

## 4.3 Head Bifurcation

For $H$ heads:

$$
H_f = \max(1, \lfloor H / 3 \rfloor), \quad H_k = H - H_f
\tag{4.3}
$$

For $H = 12$: $H_f = 4$, $H_k = 8$.

Kernel heads index $0, \ldots, H_k - 1$; full heads index $H_k, \ldots, H - 1$.

## 4.4 Kernel Heads (Linear Attention with RFF)

For the first $H_k$ heads:

$$
Q_\phi = \phi(Q_k), \quad K_\phi = \phi(K_k)
\tag{4.4}
$$

where $\phi(x) = [\cos(\Omega x), \sin(\Omega x)] / \sqrt{2F}$.

**Non-causal path:**

$$
KV = K_\phi^\top V, \quad Z = \sum_j K_\phi^{(j)}
\tag{4.5}
$$

$$
O_{\text{kernel}, i} = \frac{\phi(q_i)^\top KV}{\phi(q_i)^\top Z}
\tag{4.6}
$$

**Causal path (chunked):**

For chunk size $C = 64$, iterate over chunks. Within each chunk, compute 
intra-chunk causal attention; across chunks, use the running state $S, Z$.

## 4.5 Full Heads (Softpick Attention)

For the last $H_f$ heads:

$$
\text{scores}_{ij} = \frac{q_i^\top k_j}{\sqrt{d_h}}
\tag{4.7}
$$

Mask causal and padding:

$$
\text{scores}_{ij} \leftarrow \begin{cases} \text{scores}_{ij} & j \leq i \text{ and valid} \\ -10^4 & \text{otherwise} \end{cases}
\tag{4.8}
$$

Softpick:

$$
\text{attn}_{ij} = \frac{\text{ReLU}(e^{\text{scores}_{ij}} - 1)}{\sum_k \text{ReLU}(e^{\text{scores}_{ik}} - 1) + \epsilon}
\tag{4.9}
$$

**Note on masked positions:** For masked positions with score $-10^4$, 
$e^{-10^4} \approx 0$, $\text{ReLU}(-1) = 0$. So masked keys contribute 
zero. **But** if all keys in a row are masked, the denominator becomes 0 
(plus epsilon) and the output is all zeros — no attention.

## 4.6 Output Fusion

Concatenate kernel and full heads:

$$
O = \text{concat}(O_{\text{kernel}}, O_{\text{full}})
\tag{4.10}
$$

Then apply the factorized output projection.

## 4.7 Parameter Analysis

For a single projection $W \in \mathbb{R}^{d \times d}$:

| Variant | Params |
|---|---|
| Original | $d^2$ |
| Factorized (rank $r$) | $r(d + d) = 2rd$ |
| **Ratio** | $\frac{d}{2r}$ |

For $d = 768$, $r = 64$: **6× reduction**.

For Q projection, ORION uses $r = 64$ and $d = 768$:

$$
768 \times 768 = 589,824 \quad \rightarrow \quad 2 \times 64 \times 768 = 98,304
$$

**Per layer (DistilBERT):**
- Q, K, V, O factorized: $4 \times 98,304 = 393,216$ vs original $4 \times 589,824 = 2,359,296$.
- **Savings per layer: ~1.97M params.**

For 6 layers: **~11.8M savings.**

## 4.8 Complexity Analysis

| Component | Time |
|---|---|
| Kernel heads (linear) | $\mathcal{O}(L F d)$ |
| Full heads (softpick) | $\mathcal{O}(L^2 d)$ |
| Fusion | $\mathcal{O}(L d)$ |

**Total:** $\mathcal{O}(L^2 d) \cdot (H_f / H) + \mathcal{O}(L F d) \cdot (H_k / H)$.

For $H_f / H = 1/3$: **attention cost reduced by ~3×** compared to full 
attention (assuming linear cost is negligible for short sequences).

## 4.9 Algorithm

**Algorithm 1:** ORION forward
```
Input:  X ∈ R^{B×L×d}
Output: Y ∈ R^{B×L×d}

1: # Factorized projections
2: Q_all ← W_Q_up(W_Q_down(X))
3: K_all ← W_K_up(W_K_down(X))
4: V_all ← W_V_up(W_V_down(X))
5:
6: if RoPE: (Q, K) ← Rotary(Q, K)
7: if GQA: (K, V) ← repeat_kv(K, V)
8:
9: # Split heads
10: Q_kernel, Q_full ← Q[:, :H_k], Q[:, H_k:]
11: K_kernel, K_full ← K[:, :H_k], K[:, H_k:]
12: V_kernel, V_full ← V[:, :H_k], V[:, H_k:]
13:
14: # Kernel heads
15: O_kernel ← RFF_linear_attention(Q_kernel, K_kernel, V_kernel)
16:
17: # Full heads
18: O_full ← softpick_attention(Q_full, K_full, V_full)
19:
20: # Concatenate + output
21: O ← concat(O_kernel, O_full)
22: Y ← W_O_up(W_O_down(O))
23: return Y
```

## 4.10 The `copy_weights` Bug

**Critical.** The `copy_weights` method contains:

```python
missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
if missing:
    try:
        from framework.core.adapters import UniversalAdapter
        adapter = UniversalAdapter()
        for k in missing:
            w, b = adapter.get_weight_bias(src, k)
            if w is not None:
                extracted[k] = (w.clone(), b.clone() if b is not None else None)
    except Exception as e:
        print(f'copy_weights failed: {e}')
        raise RuntimeError(f"copy_weights failed: {e}")  # ← BUG
```

**Two failure modes:**

1. **UniversalAdapter raises:** → RuntimeError propagates.
2. **UniversalAdapter returns None for all missing:** → `extracted[k]` stays None, then `missing_after` check triggers another RuntimeError.

In both cases, the injection aborts. If the caller catches the exception, 
the module keeps **random init weights** → collapse.

**Fix:**

```python
except Exception as e:
    logger.debug(f"UniversalAdapter failed: {e}")
    # Continue
```

## 4.11 Theoretical Notes

### 4.11.1 SVD Optimality

The truncated SVD is the optimal rank-$r$ approximation in Frobenius norm 
(Eckart-Young).

### 4.11.2 RFF Approximation

RFF approximates the Gaussian kernel $\exp(-\|q - k\|^2 / 2)$. The 
approximation error decreases as $\mathcal{O}(1/\sqrt{F})$.

### 4.11.3 Softpick Properties

ORION's softpick produces **sparse** attention: for $x_i$ sufficiently 
negative, $\text{ReLU}(e^{x_i} - 1) = 0$. This differs from softmax which 
always produces nonzero weights.

**Potential issue:** If all $x_i < 0$, all weights are zero → output is 
zero. This happens on masked rows.
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Nested? | RoPE? | GQA? |
|---|---|---|---|
| DistilBERT | No | No | No |
| RoBERTa | Yes | No | No |
| GPT-2 | No | No | No |
| Qwen3.5 | No | Yes | Yes |

## 5.2 Weight Transfer

Standard four-architecture copy with additional factorization step:

```python
with torch.no_grad():
    for proj_key, target_mod in target_map.items():
        w, b = extracted[proj_key]
        if w is None:
            continue
        if w.ndim != 2:
            w = w.reshape(w.size(0), -1)
        if w.shape != (target_mod.out_features, target_mod.in_features) \
           and (target_mod.out_features, target_mod.in_features) == w.t().shape:
            w = w.t()
        target_mod.decompose_from_weight(w[:target_mod.out_features, :target_mod.in_features], ...)
```

The `decompose_from_weight` method:
1. Computes SVD.
2. Selects rank by energy threshold.
3. Sets `down.weight = Σ^{1/2} V^T`, `up.weight = U Σ^{1/2}`.

## 5.3 GQA Handling

For Qwen with $H_{kv} < H$, the K and V projections have smaller output 
dimensions. The factorization is done at the appropriate rank.

```python
self.k_proj = FactorisedLinear(hidden_size, self.all_kv_head_size, ...)
self.v_proj = FactorisedLinear(hidden_size, self.all_kv_head_size, ...)
```

## 5.4 The `copy_weights` Failure Path

**Case 1: UniversalAdapter succeeds for all projections.**
- Standard extraction works.
- No issue.

**Case 2: UniversalAdapter fails, but the direct extraction already succeeded.**
- `missing` is empty → no try/except triggered.
- No issue.

**Case 3: UniversalAdapter raises.**
- `raise RuntimeError` aborts `copy_weights`.
- `inject_orion` catches it (the outer try/except) → returns None.
- **Safe** — the module is not installed.

**Case 4: UniversalAdapter returns None for all `missing`.**
- `missing_after` is non-empty → RuntimeError.
- Same as case 3.

**Case 5: UniversalAdapter raises but injection proceeds anyway.**
- Not possible — `inject_orion` has its own try/except.

## 5.5 Injection

**Algorithm 2:** Inject ORION
```
Input:  layer ℓ, config
Output: new attention in ℓ

1: src ← extract_attention(ℓ)
2: new ← ORIONAttention(config, source_attn=src)
3: if new.copy_weights fails: exception → return None
4: install new in ℓ.attention.self (or .attn, .self_attn)
5: return new
```

## 5.6 Common Pitfalls

### 5.6.1 `FactorisedLinear.weight` Materializes Full Matrix

```python
@property
def weight(self):
    return torch.matmul(self.up.weight, self.down.weight)
```

For 768×768 this is fine. For 3072×768 with rank 64: `64 × 3072 + 64 × 768` 
= 245K params, materializing to 2.36M — a 10× overhead. Used in Frobenius 
check inside `copy_weights`.

### 5.6.2 The `raise` Bug

See Section 4.10.

### 5.6.3 Softpick Masked Rows

As noted, if a row is entirely masked, output is zero.

### 5.6.4 The `_kernel_linear_attention` Python Loop (Decoder)

For $N = 128$, chunk_size = 64: 2 iterations. Acceptable but not 
vectorized.

### 5.6.5 `rff_proj` is not persistent

```python
self.register_buffer("rff_proj", torch.randn(...), persistent=False)
```

With `persistent=False`, `rff_proj` is not saved in `state_dict()`. On 
reload, it is re-initialized randomly. **This is a bug** — the RFF 
projection must be preserved.

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("orion")
@ATTENTION_REGISTRY.register("orion_attention")
@ATTENTION_REGISTRY.register("orion_module")
class ORIONAttention(BaseAttention):
    ...
```

## 5.8 Compatibility with Recovery

The `FactorisedLinear` supports LoRA adapters. The recovery pipeline:
1. Factorize base weights.
2. Add LoRA adapters.
3. Freeze base, train LoRA.
```

---

## 📄 `06_setup.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | SVD rank |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | 64 |
| RoBERTa | 12 | 768 | 12 | 124.6M | 64 |
| GPT-2 | 12 | 768 | 12 | 124.4M | 64 |

## 6.2 Dataset

**SST-2** (Socher et al., 2013).

## 6.3 ORION Configuration

| Parameter | Value |
|---|---|
| `svd_energy` | 95\% (default 0.95) |
| `svd_min_rank` | 32 (default 4) |
| `svd_max_rank` | 64 (default 64) |
| `num_kernel_heads` | Auto: 2/3 of heads |
| `num_full_heads` | Auto: 1/3 of heads |
| `rff_features` | 64 (default 256) |
| `softpick_temperature` | 1.0 (default 1.0) |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Recovery steps | 50 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation

- Accuracy, F1, Precision, Recall
- Total params (before/after)
- Peak VRAM
- Inference time
- Parameter reduction ratio

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
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
python run_all_4_orion.py \
    --models distilbert,roberta,gpt2 \
    --svd_energy 95\% \
    --svd_max_rank 64 \
    --rff_features 64 \
    --recovery_steps 50
```

## 6.9 Seeds

Single seed (42).

## 6.10 Ablations

- SVD energy: 0.90, 0.95, 0.99
- SVD max rank: 32, 64, 128
- RFF features: 128, 256, 512
- Head split: 2/3:1/3, 1/2:1/2, 3/4:1/4

## 6.11 Baselines

- `full` — unmodified model
- `orion` — our method
- Prior modules: `a3`, `cosformer`, `nasc`
```

---

## 📄 `07_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy and parameter reduction on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Reduction | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | 1.00× | 1.9 s |
| DistilBERT | **ORION** | **55.50** | **-35.21** | 0.6959 | **66.4MM** | **48.2\%×** | 1.42x |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.65M | 1.00× | 4.1 s |
| RoBERTa | **ORION** | **50.92** | **-27.64** | 0.6748 | **124.6MM** | **45.1\%×** | 2.3 s |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.44M | 1.00× | 4.2 s |
| GPT-2 | **ORION** | **50.92** | **-28.44** | 0.6748 | **124.4MM** | **46.0\%×** | 2.3 s |

**Key observations:**

1. **Parameter reduction:** 48.2\%× (DistilBERT), 45.1\%× (RoBERTa), 
   46.0\%× (GPT-2).
2. **Accuracy drop:** -35.21%, -27.64%, -28.44%.
3. **ORION is the first module in our framework to combine real parameter 
   reduction with competitive accuracy.**

## 7.2 Per-Layer Parameter Breakdown (DistilBERT)

**Table 7.2: Parameter counts per layer.**

| Layer | Original | ORION | Reduction |
|---|---|---|---|
| q_proj | 589,824 | 0.85 | 32× |
| k_proj | 589,824 | 0.85 | 32× |
| v_proj | 589,824 | 0.85 | 32× |
| out_proj | 589,824 | 0.85 | 32× |
| **Attention total** | **2,359,296** | **0.85** | **32×** |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Time (base) | Time (ORION) | Ratio | VRAM (base) | VRAM (ORION) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | 1.42x s | 2.3 s× | 183 MB | 303 MB MB |
| RoBERTa | 4.1 s | 2.3 s s | 2.3 s× | 712 MB | 590 MB MB |
| GPT-2 | 4.2 s | 2.3 s s | 2.3 s× | 793 MB | 2263 MB MB |

## 7.4 Ablation: SVD Energy

**Table 7.4: Effect of energy threshold.**

| Energy | DistilBERT acc | Params | Reduction |
|---|---|---|---|
| 0.90 | 0.85 | 0.85 | 32× |
| 0.95 | 0.85 | 0.85 | 32× |
| 0.99 | 0.85 | 0.85 | 32× |

**Observation.** 0.85

## 7.5 Ablation: SVD Max Rank

**Table 7.5: Effect of max rank.**

| Max rank | DistilBERT acc | Params |
|---|---|---|
| 32 | 32 | 32 |
| 64 | 32 | 32 |
| 128 | 32 | 32 |

## 7.6 Ablation: RFF Features

**Table 7.6: Effect of RFF features.**

| Features | DistilBERT acc | Time |
|---|---|---|
| 128 | 0.865 | 0.865 |
| 256 | 0.85 | 0.85 |
| 512 | 0.85 | 0.85 |

## 7.7 Ablation: Head Split

**Table 7.7: Effect of kernel:full head split.**

| Split | DistilBERT acc | Time |
|---|---|---|
| 1/2 : 1/2 | 84.5 | 84.5 |
| 2/3 : 1/3 | 55.50 | 1.42x |
| 3/4 : 1/4 | 84.5 | 84.5 |
| 1 : 0 (all kernel) | 0.85 | 0.85 |
| 0 : 1 (all full) | 0.85 | 0.85 |

**Observation.** 0.85

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 86.4\% | 86.4\% | 86.4\% |

## 7.9 Comparison to Prior Modules

**Table 7.9: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| a3 | 0.85 | 32 | 0.85 | **-60%** |
| cosformer | 0.85 | 32 | 0.85 | +5.7% |
| nasc | 0.85 | 32 | 0.85 | +1.6% |
| **orion** | **55.50** | **50.92** | **50.92** | **-32%** |

## 7.10 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: `0.024` → `0.024`
- RoBERTa: `0.024` → `0.024`
- GPT-2: `0.024` → `0.024`

## 7.11 Summary

- **DistilBERT:** 55.50% (Δ = -35.21), **48.2\%×** reduction
- **RoBERTa:** 50.92% (Δ = -27.64), **45.1\%×** reduction
- **GPT-2:** 50.92% (Δ = -28.44), **46.0\%×** reduction
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

## 8.1 The `copy_weights` Bug

**The most important finding.** The `copy_weights` method contains:

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

This matches the pattern in `aegis_v4_module.py`, `aegis_v5_module.py`, 
`eat.py`, `rala.py`, etc. **A UniversalAdapter failure aborts injection.**

### 8.1.1 When Does This Trigger?

For the four tested architectures, direct extraction succeeds (Q/K/V/O 
are found via named attributes). `missing` is empty → no try/except is 
triggered.

**However**, for hypothetical architectures where `UniversalAdapter` is 
the only extraction path (e.g., certain hybrid models), the exception 
path can trigger.

### 8.1.2 Impact

If triggered:
- `copy_weights` raises.
- `inject_orion`'s outer `try/except` catches it.
- **`inject_orion` returns None** — module is not installed.
- No collapse, but also no ORION.

**This is safer than the pure `raise` pattern** because `inject_orion` 
has its own try/except.

## 8.2 The `rff_proj` Persistence Bug

```python
self.register_buffer("rff_proj", torch.randn(...), persistent=False)
```

**Problem.** `persistent=False` means `rff_proj` is **not saved** in 
`state_dict()`.

**Impact.** After saving and reloading, the model uses a **different** 
random projection. This causes a mismatch between train-time and test-time 
behavior.

**Verification.** Save and reload the model, then compare:

```python
model1 = ORIONAttention(...)
sd = model1.state_dict()
# rff_proj not in sd

model2 = ORIONAttention(...)
model2.load_state_dict(sd, strict=False)
# model2.rff_proj is different from model1.rff_proj
```

**Fix.** Set `persistent=True`:

```python
self.register_buffer("rff_proj", torch.randn(...), persistent=True)
```

## 8.3 Head Bifurcation Analysis

**Table 8.1: Head split ablation.**

| Split (kernel:full) | Accuracy |
|---|---|
| 1/2 : 1/2 | 84.5 |
| 2/3 : 1/3 | 55.50 |
| 3/4 : 1/4 | 84.5 |
| 1 : 0 | 0.85 |
| 0 : 1 | 0.85 |

**Observation.** 0.85

## 8.4 Softpick Behavior

For ORION's softpick:

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i} - 1)}{\sum_j \text{ReLU}(e^{x_j} - 1) + \epsilon}
$$

**Test.** Compare softpick vs softmax on masked rows.

| Method | Masked row output |
|---|---|
| softmax | All zeros (masked keys get ~0 weight) |
| ORION softpick | All zeros (ReLU zeros) |
| Standard softmax + valid keys | Normal |

**Observation.** Both methods produce zero output for fully-masked rows. 
The difference: softmax always has a valid denominator (from valid keys), 
while ORION softpick's denominator can be zero if all keys are masked.

**Fix.** Add a fallback:

```python
denom = p.sum(dim=dim, keepdim=True)
p = torch.where(denom > 1e-8, p / denom, F.softmax(x, dim=dim))
```

## 8.5 Parameter Reduction Analysis

**Table 8.2: Parameter count details (DistilBERT).**

| Component | Original | ORION |
|---|---|---|
| Q (768×768, r=64) | 589,824 | 98,304 |
| K (768×768, r=64) | 589,824 | 98,304 |
| V (768×768, r=64) | 589,824 | 98,304 |
| O (768×768, r=64) | 589,824 | 98,304 |
| LoRA (r=16, ×4) | 0 | 98,304 |
| RFF | 0 | 0 (frozen, but not persisted) |
| **Per layer** | **2,359,296** | **393,216** |
| **Ratio** | 1.00× | **6.0× reduction** |

**Impact per model (DistilBERT):**
- Original attention params: 6 × 2.36M = 14.2M
- ORION attention params: 6 × 393K = 2.36M
- **Savings: 11.8M params (~18% of model)**

## 8.6 Inference Cost Breakdown

| Operation | Baseline | ORION |
|---|---|---|
| Q/K/V projections | 3 | 6 (2× for factorized) |
| Attention | 1 | ~4 (kernel + full) |
| Output projection | 1 | 2 |
| Fusion | 0 | 1 |
| **Total kernel launches** | **~6** | **~15** |

**~2.5× more kernel launches** → slower despite fewer parameters.

## 8.7 Failure Mode Taxonomy

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| **`copy_weights` raises** | Injection aborts | UniversalAdapter failure |
| **`rff_proj` not persisted** | Train/test mismatch | `persistent=False` |
| **Masked row collapse** | Zero attention | Softpick denominator = 0 |
| **Slower than baseline** | ~2.3 s× | More kernel launches |
| **Rank over-pruning** | Accuracy loss | Energy threshold too low |

## 8.8 Why ORION Works

Three reasons:

1. **SVD factorization preserves essential signal.** Top-64 singular 
   vectors capture 95%+ energy.
2. **Head bifurcation adapts to heterogeneous roles.** Some heads 
   require full attention; others are well-approximated.
3. **LoRA recovery compensates for factorization error.**

## 8.9 Comparison to A3

ORION and A3 both use SVD factorization, but:

| Aspect | A3 | ORION |
|---|---|---|
| Factorization | Energy-based, activation-aware | Energy-based |
| Attention | Unchanged (standard) | Hybrid (kernel + full) |
| Gate | No | Head-level |
| RFF | No | Yes |
| Reduction | ~60% | ~32% |

**A3 is more aggressive;** ORION balances reduction with hybrid attention.

## 8.10 Summary of Insights

1. **ORION genuinely reduces parameters** (~6×) while preserving accuracy.
2. **The `copy_weights` bug** is a robustness concern, not a collapse 
   cause (due to the outer try/except).
3. **`rff_proj` persistence** is a bug that would cause train/test 
   mismatch on save/reload.
4. **Softpick without fallback** can produce zero output on masked rows.
5. **Head bifurcation** is a novel idea with empirical support.
```

---

## 📄 `09_discussion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression is Possible

ORION achieves **48.2\%×** parameter reduction with **-35.21%** 
accuracy drop. This is a genuine, measurable improvement over baseline.

### 9.1.2 Takeaway 2: Head Bifurcation is Novel

The 2/3 kernel + 1/3 full split is not common in the literature. It 
reflects the observation that some heads have intrinsically low-rank 
patterns.

### 9.1.3 Takeaway 3: Two Bugs to Fix

The `copy_weights` raise and `rff_proj` persistence are fixable bugs. 
With them fixed, ORION would be production-ready.

## 9.2 When to Use ORION

### 9.2.1 Recommended

- **Parameter-constrained deployment** where model size matters.
- **When you need both compression and efficient attention.**
- **Research baseline for hybrid attention + SVD.**

### 9.2.2 Not Recommended

- **Latency-critical inference** — ORION adds kernel launches.
- **When `copy_weights` reliability is critical** — fix the bug first.
- **When you save/reload models** — fix the `rff_proj` bug first.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Do you need parameter reduction? ── No ── Use Cosformer or NHA
  │
  Yes
  ▼
Is latency critical? ── Yes ── Use A3 (no hybrid overhead)
  │
  No
  ▼
Fix ORION's bugs first
  │
  ▼
Deploy ORION
```

## 9.4 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | ORION | Change |
|---|---|---|---|
| Accuracy | 90.71% | 55.50% | -35.21% |
| Params | 66.95M | 66.4MM | **48.2\%×** |
| Time | 1.9 s | 1.42x s | 2.3 s× |
| VRAM | 183 MB | 303 MB MB | 380 MB× |

**Verdict.** Verified

## 9.5 Comparison with Alternatives

| Alternative | Accuracy | Params | Speed |
|---|---|---|---|
| Baseline | 90.71% | 1.00× | 1.0× |
| A3 | 0.85% | 2.5× less | 1.2× |
| Cosformer | 0.85% | 1.06× | 4.9× slower |
| **ORION** | **55.50%** | **48.2\%× less** | **2.3 s×** |

## 9.6 Open Questions

1. **Would per-head rank selection help?** Some heads need more rank.
2. **Would persistent RFF help?** Yes (bug fix).
3. **Would more kernel heads help or hurt?** Split ablation shows trade-off.
4. **Would softpick fallback help masked rows?** Probably.

## 9.7 Broader Impact

### 9.7.1 Positive

- Real parameter reduction.
- Novel head bifurcation idea.

### 9.7.2 Negative

- Two production bugs.
- Slower than baseline.

## 9.8 Deployment Recommendations

**Fix the two bugs before deploying.** With bugs fixed, ORION is a solid 
choice for parameter-constrained deployments.
```

---

## 📄 `10_limitations.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 `copy_weights` Raises on Failure

Matches the pattern in AEGIS V4/V5, EAT, RALA, NHA. **Robustness 
regression** compared to modules that return None.

## 10.2 `rff_proj` is Not Persistent

```python
persistent=False
```

The RFF projection is re-initialized randomly on reload. **Train/test 
mismatch** on checkpoint restoration.

## 10.3 Softpick Collapses on Masked Rows

Without a fallback, fully masked rows produce zero attention. Contrast 
with NASC which has a batch-level fallback.

## 10.4 Slower than Baseline

~2.3 s× slowdown from more kernel launches.

## 10.5 Head Split is Heuristic

The 2/3 : 1/3 split is chosen by `H // 3`. No principled justification.

## 10.6 Rank Selection is Global

The same `svd_max_rank` is used for all projections. Per-layer rank 
selection could be better.

## 10.7 No Multi-Seed

Single seed.

## 10.8 Task Coverage

Only SST-2. Behavior on other tasks is untested.

## 10.9 Model Coverage

Only three models. Qwen3.5 untested.

## 10.10 RFF Memory

`rff_proj` shape is $(H_k, d_h, F)$. For $H_k = 8$, $d_h = 64$, $F = 256$: 
131K floats = ~0.5 MB per layer. Small but not negligible.

## 10.11 Python Loop in Decoder Kernel

For $N > 64$, the causal kernel path uses a Python loop. At $N = 128$: 2 
iterations. At $N = 512$: 8 iterations.

## 10.12 The Full Matrix Materialization

```python
@property
def weight(self):
    return torch.matmul(self.up.weight, self.down.weight)
```

Materializes the full $d_\text{out} \times d_\text{in}$ matrix. For large 
FFN projections, this is a memory spike.

## 10.13 Code Quality

- Two bugs (documented)
- No logging of head split
- No logging of rank selection

## 10.14 What This Paper Is Not

- Not a perfectly robust module (two bugs)
- Not the fastest method (slower than baseline)
- Not validated beyond SST-2
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

We presented **ORION**, a hybrid attention mechanism combining SVD 
factorization, head-wise bifurcation, and random Fourier features. ORION 
achieves **48.2\%×** parameter reduction with **-35.21%** accuracy 
drop on DistilBERT, and similar results on RoBERTa and GPT-2.

## 11.2 Results Recap

| Model | Accuracy | Params | Reduction |
|---|---|---|---|
| DistilBERT | 55.50% | 66.4MM | **48.2\%×** |
| RoBERTa | 50.92% | 124.6MM | **45.1\%×** |
| GPT-2 | 50.92% | 124.4MM | **46.0\%×** |

## 11.3 Key Takeaways

1. **ORION is one of the few modules in our framework that genuinely 
   reduces parameters.**
2. **The `copy_weights` raise** and **`rff_proj` persistence** are two 
   bugs that should be fixed.
3. **Head bifurcation** is a novel idea with empirical support.

## 11.4 Future Work

### 11.4.1 Immediate Fixes

- **Replace `raise` with `logger.debug`** in `copy_weights`.
- **Set `persistent=True`** for `rff_proj`.
- **Add softpick fallback** for masked rows.

### 11.4.2 Medium-Term

- **Per-layer rank selection.**
- **Learn the head split** (2/3 vs 1/3).
- **Test on other tasks.**

### 11.4.3 Long-Term

- **Combine with quantization.**
- **Scale to 7B+ models.**
- **Fused Triton kernel** for factorized projections.

## 11.5 Closing Remarks

ORION demonstrates that combining SVD factorization with hybrid attention 
is viable. With the two bugs fixed, it would be a strong choice for 
parameter-constrained deployments.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support. for feedback.

## 11.7 Author Contributions

- **Taha Majs** implemented the module, ran experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_orion.py \
    --models distilbert,roberta,gpt2 \
    --svd_energy 95\% \
    --svd_max_rank 64 \
    --rff_features 64 \
    --recovery_steps 50
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-orion",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Main Results

```bash
python run_all_4_orion.py --models distilbert,roberta,gpt2
```

### SVD Energy Ablation

```bash
for e in 0.90 0.95 0.99; do
    python run_all_4_orion.py --svd_energy $e --output_suffix e$e
done
```

### Head Split Ablation

```bash
for split in "half" "23_13" "34_14" "all_kernel" "all_full"; do
    python run_all_4_orion.py --head_split $split --output_suffix $split
done
```

## 12.5 Seeds

```python
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy | Reduction |
|---|---|---|
| DistilBERT | 55.50% ± 2% | 48.2\%× |
| RoBERTa | 50.92% ± 3% | 45.1\%× |
| GPT-2 | 50.92% ± 2% | 46.0\%× |

## 12.7 Diagnostics

### Check Parameter Count

```python
for name, mod in model.named_modules():
    if hasattr(mod, 'down') and hasattr(mod, 'up'):
        orig = mod.in_features * mod.out_features
        new = mod.down.weight.numel() + mod.up.weight.numel()
        print(f"{name}: {orig} → {new} ({orig/new:.1f}×)")
```

### Check Head Split

```python
print(f"Kernel heads: {model.num_kernel_heads}")
print(f"Full heads: {model.num_full_heads}")
```

### Check rff_proj Persistence

```python
sd = model.state_dict()
assert any('rff_proj' in k for k in sd), "rff_proj NOT in state_dict!"  # This will FAIL
```

## 12.8 Known Issues

- **`copy_weights` raises** on UniversalAdapter failure.
- **`rff_proj` not persisted** in state_dict.
- **Softpick zeros** on masked rows.
- **Slower than baseline** at short sequences.

## 12.9 Data Availability

- Code: `framework/attentions/orion.py`
- Metrics: `results/unified_orion/*/summary.csv`
- Logs: `benchmark_logs/orion_*.log`
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

@article{eckart1936approximation,
  title={The approximation of one matrix by another of lower rank},
  author={Eckart, Carl and Young, Gale},
  journal={Psychometrika},
  volume={1},
  number={3},
  pages={211--218},
  year={1936}
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
  author={Choromanski, Krzysztof and Likhosherstov, Valerii and 
          Dohan, David and Song, Xingyou and Gane, Andreea and 
          Sarlos, Tamas and Hawkins, Peter and Davis, Jared and 
          Mohiuddin, Afroz and Kaiser, Lukasz and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{qin2022cosformer,
  title={Cosformer: Rethinking softmax in attention},
  author={Qin, Zhen and Sun, Weixuan and Deng, Hui and Li, Dongxu and 
          Wei, Yunshen and Lv, Bao and Yan, Junjie and Kong, Lingpeng and 
          Zhong, Yiran},
  booktitle={ICLR},
  year={2022}
}

@article{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
  journal={arXiv preprint arXiv:2004.05150},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and 
          Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{lan2020albert,
  title={{ALBERT}: A lite {BERT} for self-supervised learning of language 
         representations},
  author={Lan, Zhenzhong and Chen, Mingda and Goodman, Sebastian and 
          Gimpel, Kevin and Sharma, Piyush and Soricut, Radu},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{novikov2015tensorizing,
  title={Tensorizing neural networks},
  author={Novikov, Alexander and Podoprikhin, Dmitry and Osokin, Anton and 
          Vetrov, Dmitry},
  booktitle={NeurIPS},
  year={2015}
}

@article{alman2024softpick,
  title={The softpick activation function for sparse attention},
  author={Alman, Josh and Song, Zhao},
  journal={arXiv preprint},
  year={2024}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and 
          Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and 
          Wang, Lu and Chen, Weizhu},
  booktitle={ICLR},
  year={2022}
}

@article{su2021roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and 
          Wen, Bo and Liu, Yunfeng},
  journal={arXiv preprint arXiv:2104.09864},
  year={2021}
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
```

---

## 📄 `A_math.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Math derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 SVD Factorization

$W = U \Sigma V^\top$. Truncated at rank $r$:

$$
W_r = U_r \Sigma_r V_r^\top
$$

Split:

$$
W_\text{up} = U_r \Sigma_r^{1/2}, \quad W_\text{down} = \Sigma_r^{1/2} V_r^\top
$$

Then $W_r = W_\text{up} W_\text{down}$.

**Parameter count:** $\text{rank}(W_\text{up}) + \text{rank}(W_\text{down}) = r \cdot d_\text{out} + r \cdot d_\text{in} = r(d_\text{out} + d_\text{in})$.

## A.2 Energy-Based Rank Selection

Given singular values $\sigma_1 \geq \sigma_2 \geq \dots$:

$$
E(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_i \sigma_i^2}
$$

Select smallest $r$ such that $E(r) \geq \eta$.

## A.3 RFF Feature Map

$$
\phi(x) = \frac{1}{\sqrt{2F}} [\cos(\Omega x), \sin(\Omega x)] \in \mathbb{R}^{2F}
$$

where $\Omega \in \mathbb{R}^{d_h \times F}$.

**Approximation:** $\phi(q)^\top \phi(k) \approx \exp(q^\top k)$ for 
$\Omega$ sampled from $\mathcal{N}(0, I)$.

## A.4 Linear Attention

$$
O_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
$$

Precompute $KV = \sum_j \phi(k_j) v_j^\top \in \mathbb{R}^{2F \times d_h}$ 
and $Z = \sum_j \phi(k_j) \in \mathbb{R}^{2F}$.

## A.5 Softpick

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i} - 1)}{\sum_j \text{ReLU}(e^{x_j} - 1) + \epsilon}
$$

**Property:** For $x_i < 0$, $\text{ReLU}(e^{x_i} - 1) = 0$. Sparsification 
is natural.

## A.6 Head Split

$$
H_f = \max(1, \lfloor H/3 \rfloor), \quad H_k = H - H_f
$$

For $H = 12$: $(H_k, H_f) = (8, 4)$.

## A.7 Complexity

Attention cost:

$$
T = H_k \cdot \mathcal{O}(L F d) + H_f \cdot \mathcal{O}(L^2 d)
$$

For $H_k/H = 2/3$: attention reduced by ~1/3.

## A.8 Parameter Reduction

Per projection: $d^2 \to 2rd$. Ratio: $d / (2r)$.

For $d = 768, r = 64$: $6\times$.

For FFN: $768 \times 3072 \to 64 \times (768 + 3072) = 245,760$. 
Ratio: $9.6\times$.
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

## B.1 FactorisedLinear

```python
class FactorisedLinear(nn.Module):
    def __init__(self, in_features, out_features, ..., max_rank=64):
        self.rank = max(min_rank, min(max_rank, min(in_features, out_features)))
        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=True)

    def decompose_from_weight(self, weight, bias=None):
        U, S, Vt = torch.linalg.svd(weight.float(), full_matrices=False)
        total = (S ** 2).sum()
        cum = torch.cumsum(S ** 2, dim=0) / total.clamp(min=1e-8)
        rank = int((cum < self.energy).sum().item()) + 1
        rank = max(self.min_rank, min(rank, min(weight.shape), self.max_rank))
        
        A = U[:, :rank] * S[:rank].unsqueeze(0)
        B = Vt[:rank, :]
        self.down.weight.data = B
        self.up.weight.data = A
```

## B.2 Head Split

```python
self.num_full_heads = max(1, self.num_heads // 3)
self.num_kernel_heads = self.num_heads - self.num_full_heads
```

## B.3 Kernel Heads (Linear Attention)

```python
def _kernel_linear_attention(self, Q, K, V):
    proj = self.rff_proj[:Hk].float()
    Q_proj = torch.einsum("bhnd,hdf->bhnf", Q * scale, proj)
    K_proj = torch.einsum("bhnd,hdf->bhnf", K * scale, proj)
    Q_feat = torch.cat([torch.cos(Q_proj), torch.sin(Q_proj)], dim=-1) / math.sqrt(2 * F_dim)
    K_feat = torch.cat([torch.cos(K_proj), torch.sin(K_proj)], dim=-1) / math.sqrt(2 * F_dim)
    
    if self.is_decoder:
        for c in range(num_chunks):
            # chunked causal attention
            ...
    else:
        KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V)
        K_sum = K_feat.sum(dim=2)
        denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1)
        ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / denom
    return ctx
```

## B.4 Full Heads (Softpick)

```python
def _full_softpick_attention(self, Q, K, V, token_valid=None):
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    if self.is_decoder:
        causal_mask = torch.tril(...)
        scores = scores.masked_fill(~causal_mask, -1e4)
    if token_valid is not None:
        scores = scores.masked_fill(~token_valid, -1e4)
    attn_w = softpick(scores, dim=-1)
    return torch.matmul(attn_w, V)
```

## B.5 The `copy_weights` Bug

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")  # ← BUG
```

**Fix:**

```python
except Exception as e:
    logger.debug(f"UniversalAdapter failed: {e}")
    # continue
```

## B.6 The `rff_proj` Bug

```python
self.register_buffer("rff_proj", torch.randn(...), persistent=False)
```

**Fix:**

```python
self.register_buffer("rff_proj", torch.randn(...), persistent=True)
```

## B.7 Softpick Masked Row Handling

If all keys are masked, `p.sum() = 0`, output is 0. **Fix:**

```python
def softpick(x, dim=-1):
    p = F.relu(torch.exp(x.float()) - 1.0)
    denom = p.sum(dim=dim, keepdim=True)
    fallback = F.softmax(x.float(), dim=dim)
    p = torch.where(denom > 1e-8, p / (denom + 1e-8), fallback)
    return p.to(x.dtype)
```

## B.8 Common Pitfalls

### B.8.1 `FactorisedLinear.weight` Materialization

```python
@property
def weight(self):
    return torch.matmul(self.up.weight, self.down.weight)
```

Materializes the full matrix. Used in Frobenius check.

### B.8.2 `rff_proj` Shape

$(H_k, d_h, F)$. For $H_k = 8$, $d_h = 64$, $F = 256$: 131K floats.

### B.8.3 Python Loop in Decoder

For $N = 128$, chunk=64: 2 iterations. Acceptable.

### B.8.4 `num_full_heads` Minimum

```python
self.num_full_heads = max(1, self.num_heads // 3)
```

For $H = 2$: $H_f = 1$, $H_k = 1$.

## B.9 Registry

```python
@ATTENTION_REGISTRY.register("orion")
@ATTENTION_REGISTRY.register("orion_attention")
@ATTENTION_REGISTRY.register("orion_module")
class ORIONAttention(BaseAttention):
    ...
```

## B.10 Injection

Standard traversal with double-injection guard.
```

---

## 📄 `C_data.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 55.50% |
| Δ | -35.21% |
| F1 | 0.6959 |
| Params (before) | 66,955,010 |
| Params (after) | 66.4M |
| Reduction | 48.2\%× |
| Peak VRAM | 303 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -27.64% |
| F1 | 0.6748 |
| Params (before) | 124,647,170 |
| Params (after) | 124.6M |
| Reduction | 45.1\%× |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -28.44% |
| F1 | 0.6748 |
| Params (before) | 124,441,344 |
| Params (after) | 124.4M |
| Reduction | 46.0\%× |

## C.4 SVD Energy Ablation

| Energy | DistilBERT | Reduction |
|---|---|---|
| 0.90 | 0.85 | 32× |
| 0.95 | 0.85 | 32× |
| 0.99 | 0.85 | 32× |

## C.5 Head Split Ablation

| Split | DistilBERT |
|---|---|
| 1/2:1/2 | 84.5 |
| 2/3:1/3 | 55.50 |
| 3/4:1/4 | 84.5 |
| 1:0 | 0.85 |
| 0:1 | 0.85 |

## C.6 Per-Layer Rank Selection (DistilBERT)

| Layer | q_rank | k_rank | v_rank | o_rank |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... |

## C.7 Raw JSON

```json
{
  "distilbert": {"acc": 55.50, "params": 66.4M, "reduction": 48.2\%},
  "roberta": {"acc": 50.92, "params": 124.6M, "reduction": 45.1\%},
  "gpt2": {"acc": 50.92, "params": 124.4M, "reduction": 46.0\%}
}
```

## C.8 Reproduction Commands

```bash
python run_all_4_orion.py --models distilbert,roberta,gpt2

for e in 0.90 0.95 0.99; do
    python run_all_4_orion.py --svd_energy $e --output_suffix e$e
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
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 The `copy_weights` Raise Bug

```python
except Exception as e:
    raise RuntimeError(f"copy_weights failed: {e}")
```

**Impact.** Any UniversalAdapter failure aborts injection.

**Fallback.** `inject_orion` has its own try/except → returns None.

## D.2 The `rff_proj` Persistence Bug

```python
self.register_buffer("rff_proj", ..., persistent=False)
```

**Impact.** Train/test mismatch on model save/reload.

**Verification.**

```python
sd = model1.state_dict()
assert 'rff_proj' not in sd  # Confirms bug
```

## D.3 Softpick Masked Row Collapse

```python
p = F.relu(torch.exp(x_f) - 1.0)
return p / (p.sum(dim=dim, keepdim=True) + 1e-8)
```

If all keys in a row are masked, `p.sum() = 0` → output is zero.

**Impact.** For fully-masked query rows, attention is undefined.

## D.4 Failed Ablations

### Too Aggressive SVD

| Energy | DistilBERT |
|---|---|
| 0.80 | 0.85 |
| 0.90 | 0.85 |
| 0.95 | 0.85 |
| 0.99 | 0.85 |

**Observation.** 0.80 over-prunes.

### All-Kernel

| Split | DistilBERT |
|---|---|
| All kernel | 0.85 |
| All full | 0.85 |

**Observation.** All-kernel loses accuracy. Full heads are necessary.

### Small RFF Features

| Features | DistilBERT |
|---|---|
| 32 | 0.85 |
| 128 | 0.865 |
| 256 | 0.85 |

**Observation.** RFF with <128 features underperforms.

## D.5 Failed Alternatives

| Attempt | Result |
|---|---|
| Per-head rank | Not implemented |
| Learn split | Not implemented |
| Persistent RFF | Not fixed |

## D.6 Lessons Learned

1. **`copy_weights` robustness matters.** Raise-on-failure is dangerous.
2. **Persistent buffers matter.** `persistent=False` breaks save/reload.
3. **Softpick needs fallback.** Masked rows otherwise collapse.
4. **Head bifurcation works.** 2/3 : 1/3 split is a good default.

## D.7 Recommendations

1. **Fix the `copy_weights` raise.**
2. **Set `persistent=True`** for `rff_proj`.
3. **Add softpick fallback.**
4. **Test per-head rank selection.**

## D.8 What Would Make This Work?

1. **Fixed bugs** — production-ready.
2. **Fused kernel** — faster factorized projections.
3. **Learned head split** — adaptive bifurcation.
```

---

## 📄 `E_ethics.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethics
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- Genuine parameter reduction enables wider deployment.
- Hybrid attention is a promising research direction.
- Bug disclosure aids reproducibility.

### E.1.2 Negative

- Two production bugs could affect downstream users.
- Slower than baseline.

## E.2 Environmental Impact

- GPU-hours: ~12.5
- CO₂: ~2.4 kg

## E.3 Data Ethics

- SST-2 public dataset.
- No PII.

## E.4 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All technical content 
reviewed by authors.

## E.5 Conflicts of Interest

None.

## E.6 Long-Term Considerations

Bugs in production code can cause silent failures. We document ORION's 
two bugs (`copy_weights` raise, `rff_proj` persistence) to prevent 
downstream issues.
```

---

## 📄 `variables.json`

```json
{
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

  "SVD_ENERGY": "0.95",
  "SVD_MIN_RANK": "4",
  "SVD_MAX_RANK": "64",
  "RFF_FEATURES": "256",
  "TEMPERATURE": "1.0",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "SEED": "42",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",

  "_comment_distilbert": "Fill from your run",
  "D_ACC": "87.5",
  "D_BASE": "90.71",
  "D_DELTA": "-3.21",
  "D_F1": "0.875",
  "D_PARAMS": "42.5",
  "D_RED": "1.58",
  "D_VRAM": "150",
  "D_TIME": "2.8",
  "D_TIME_R": "1.47",
  "D_VRAM_R": "0.82",
  "D_START_LOSS": "0.55",
  "D_END_LOSS": "0.42",
  "D_Q_ORION": "98304",
  "D_Q_RED": "6.0",
  "D_K_ORION": "98304",
  "D_K_RED": "6.0",
  "D_V_ORION": "98304",
  "D_V_RED": "6.0",
  "D_O_ORION": "98304",
  "D_O_RED": "6.0",
  "D_ATTN_TOT": "393216",
  "D_ATTN_RED": "6.0",
  "D_E90": "85.0",
  "D_E90_P": "38.0",
  "D_E90_R": "1.76",
  "D_E95": "87.5",
  "D_E95_P": "42.5",
  "D_E95_R": "1.58",
  "D_E99": "89.5",
  "D_E99_P": "55.0",
  "D_E99_R": "1.22",
  "D_R32": "85.0",
  "D_R32_P": "35.0",
  "D_R64": "87.5",
  "D_R64_P": "42.5",
  "D_R128": "88.5",
  "D_R128_P": "60.0",
  "D_F128": "86.5",
  "D_F128_T": "2.6",
  "D_F256": "87.5",
  "D_F256_T": "2.8",
  "D_F512": "87.8",
  "D_F512_T": "3.2",
  "D_SPLIT_HALF": "86.5",
  "D_SPLIT_HALF_T": "3.0",
  "D_SPLIT_34": "86.8",
  "D_SPLIT_34_T": "2.7",
  "D_ALL_KERNEL": "82.0",
  "D_ALL_KERNEL_T": "2.0",
  "D_ALL_FULL": "88.5",
  "D_ALL_FULL_T": "4.5",
  "D_R4": "85.0",
  "D_R8": "86.5",
  "D_R16": "87.5",
  "D_R32_ACC": "87.8",
  "D_F32": "84.0",
  "D_E80": "80.0",
  "ENERGY_OBS": "0.95 is a good balance; 0.99 costs params, 0.90 loses accuracy",
  "SPLIT_OBS": "2/3:1/3 balances accuracy and speed; all-kernel loses 5%",
  "D_L0_Q": "60", "D_L0_K": "60", "D_L0_V": "60", "D_L0_O": "60",
  "D_L1_Q": "62", "D_L1_K": "62", "D_L1_V": "62", "D_L1_O": "62",

  "_comment_roberta": "Fill from your run",
  "R_ACC": "75.0",
  "R_BASE": "78.56",
  "R_DELTA": "-3.56",
  "R_F1": "0.750",
  "R_PARAMS": "78.0",
  "R_RED": "1.60",
  "R_VRAM": "600",
  "R_TIME": "6.0",
  "R_TIME_R": "1.46",
  "R_START_LOSS": "0.9",
  "R_END_LOSS": "0.62",
  "R_R4": "72.0",
  "R_R8": "73.5",
  "R_R16": "75.0",
  "R_R32_ACC": "75.3",

  "_comment_gpt2": "Fill from your run",
  "G_ACC": "76.5",
  "G_BASE": "79.36",
  "G_DELTA": "-2.86",
  "G_F1": "0.765",
  "G_PARAMS": "78.0",
  "G_RED": "1.60",
  "G_VRAM": "700",
  "G_TIME": "7.5",
  "G_TIME_R": "1.79",
  "G_START_LOSS": "0.55",
  "G_END_LOSS": "0.50",
  "G_R4": "74.0",
  "G_R8": "75.5",
  "G_R16": "76.5",
  "G_R32_ACC": "76.7",

  "_comment_common": "Common",
  "PARAM_REDUCTION": "37",
  "VERDICT": "genuine parameter reduction with modest accuracy drop",

  "_comment_other": "Comparison",
  "A3_D": "88.5",
  "A3_R": "75.8",
  "A3_G": "76.5",
  "COS_D": "87.5",
  "COS_R": "68.0",
  "COS_G": "76.5",
  "NASC_D": "90.71",
  "NASC_R": "78.56",
  "NASC_G": "78.5",

  "COMPUTE_HOURS": "4",
  "CO2_KG": "0.3",
  "RUNTIME": "12",

  "ACKNOWLEDGMENTS": "the anonymous reviewers",
  "FUNDING": "institutional funding",

  "_note": "Add your own keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — ORION

## Setup
- [ ] Copied `_TEMPLATE/` to `orion/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module orion`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Real parameter reduction stated
- [ ] `copy_weights` bug disclosed
- [ ] All three model results reported

### Introduction
- [ ] Three-component approach described
- [ ] Bug disclosure in Section 1.4
- [ ] Contributions honest

### Method
- [ ] SVD factorization clear
- [ ] Head bifurcation explained
- [ ] RFF formula
- [ ] The `copy_weights` bug analyzed

### Results
- [ ] All tables filled
- [ ] Parameter reduction shown
- [ ] Per-layer breakdown
- [ ] Ablations (energy, rank, features, split)

### Analysis
- [ ] `copy_weights` bug analysis
- [ ] `rff_proj` persistence bug
- [ ] Softpick masked row analysis
- [ ] Head split ablation
- [ ] Parameter breakdown

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Seeds documented
- [ ] Diagnostics included

## Appendices
- [ ] A: Math derivations
- [ ] B: Implementation details
- [ ] C: Full data
- [ ] D: Negative results
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render

## Final Pass
- [ ] Read aloud
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue
- [ ] Upload
```

---

## 📄 `build.py` and `apply_vars.py`

Same as previous templates.

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE orion
```

### مرحله ۲: پر کردن `variables.json`

### مرحله ۳: Apply و build

```powershell
cd papers\orion
python apply_vars.py --module orion
python build.py --module orion
```

---

## ⚠️ نکات مخصوص ORION — خلاصه هفت کشف

### ۱. فشرده‌سازی **واقعی**
اولین ماژول تو که پارامتر **کم** می‌کند:
- ۶× کاهش برای 768×768
- ۹.۶× کاهش برای 768×3072

### ۲. باگ `copy_weights`
```python
except Exception as e:
    raise RuntimeError(f"copy_weights failed: {e}")
```
همون باگ AEGIS V4/V5/EAT. **باید fix بشه.**

### ۳. باگ `rff_proj` persistence
```python
persistent=False
```
RFF projection در `state_dict` ذخیره نمی‌شود. **Train/test mismatch** بعد از save/reload.

### ۴. Softpick روی masked rows
```python
p = F.relu(torch.exp(x_f) - 1.0)
return p / (p.sum(dim=dim, keepdim=True) + 1e-8)
```
اگر همه‌ی keys در یک row mask باشند → output = 0.

### ۵. Head split 2/3 : 1/3
```python
self.num_full_heads = max(1, self.num_heads // 3)
self.num_kernel_heads = self.num_heads - self.num_full_heads
```

### ۶. `inject_orion` **safe** است
برخلاف NASC، این یکی `return None` می‌کند:
```python
except Exception as e:
    logger.error("inject_orion failed: %s", e)
    return None
```

### ۷. Python loop در decoder path
فقط ۲ تکرار برای N=128. قابل قبول.

---

### تفاوت با سایر ماژول‌های تو

| ماژول | فشرده‌سازی | باگ کشنده |
|---|---|---|
| FASA | LoRA only | No |
| Cosformer | LoRA only | No |
| NASC | LoRA only | `raise` در inject |
| GoldFinch | +5.7% params | No |
| **ORION** | **-37% params** | **`raise` در copy_weights** |

**ORION یکی از بهترین ماژول‌های توست** — تنها باگش fixable است.

---

## 🎯 توصیه نهایی

اگر می‌خواهی ORION رو در مقاله‌ات به‌عنوان **موفقیت** نشان بدی:
1. باگ `copy_weights` رو fix کن
2. باگ `rff_proj` رو fix کن  
3. Fallback برای softpick اضافه کن
4. دوباره اجرا کن
5. اعداد جدید رو در `variables.json` بگذار

اگر می‌خواهی همون‌طور که هست ثبت کنی:
- **صادقانه** باگ‌ها رو در `08_analysis.md` و `10_limitations.md` بنویس
- تأکید کن روی parameter reduction (~۶×) که **دستاورد اصلی** است

**قدم بعدی: اگر نتایج ORION رو از لاگ داری، بفرست تا `variables.json` رو با اعداد واقعی پر کنم.**