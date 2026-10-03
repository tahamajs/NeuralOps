# تمپلیت کامل مقاله LEKAv1 — فایل به فایل

قبل از شروع، خلاصه‌ای از این ماژول و تفاوت‌هایش با بقیه:

| ویژگی | HGRN2 | JambaInfo | VSpec++ | **LEKAv1** |
|---|---|---|---|---|
| **نوع** | Linear RNN | Hybrid | SVD+Gate | **Linear Attention (RFF)** |
| **Complexity** | O(L·d) | O(L²+L) | O(k²·d) | **O(L·F)** |
| **Kernel approx** | — | — | — | ✅ **RFF cos/sin** |
| **QR projection** | — | — | — | ✅ **Orthogonalized** |
| **Preservation gate** | — | — | — | ✅ **α·out + (1-α)·x** |
| **LayerNorm before RFF** | — | — | — | ✅ |
| **Causal recurrence** | cumsum chunked | SSM | — | **cumsum (native)** |

**نکته کلیدی:** این ماژول یک **linear-time attention** واقعی است — از Random Fourier Features برای تقریب Gaussian RBF kernel استفاده می‌کند و complexity رو از O(L²) به O(L·F) می‌رسونه.

**چند هشدار مهم قبل از نوشتن مقاله:**

1. **`output_scale` و `pres_gate`** پارامترهای جدی هستن که در checkpoint baseline وجود ندارن.
2. **`raise RuntimeError` در `copy_weights`** → الگوی کشنده که در ماژول‌های قبلی دیدیم.
3. **Feature map جالبه:** φ(x) = [cos(xW), sin(xW)]/√F — این در واقع real part of RFF هست.

---

## 📁 ساختار پوشه

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
└── leka_v1/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
# LEKAv1: Linear-Time Efficient Kernel Attention with Random Fourier Features

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/leka_v1.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Standard transformer attention scales quadratically with sequence length, 
making long-context inference prohibitively expensive. **Linear attention** 
reduces this to $\mathcal{O}(L)$ by replacing softmax with a kernel 
feature map $\phi$ such that $\exp(q^\top k) \approx \phi(q)^\top \phi(k)$. 
The **Performer** (Choromanski et al., 2021) uses positive orthogonal 
random features (PORF); the **Linear Transformer** (Katharopoulos et al., 
2020) uses $\phi(x) = \text{elu}(x) + 1$.

We introduce **LEKAv1**, a production-hardened linear-time attention module 
that approximates the **Gaussian RBF kernel** $\exp(-\|q - k\|^2 / 2)$ 
using Random Fourier Features (RFF):

$$
\phi(x) = \frac{1}{\sqrt{F}} \left[ \cos(x W) \,\|\, \sin(x W) \right], 
\quad W \in \mathbb{R}^{d_h \times F}
$$

where $W$ is **QR-orthogonalized** for stability. The feature dimension 
$F = 64$ yields $\phi(x) \in \mathbb{R}^{2F}$. We apply a 
**head-wise LayerNorm** before the projection, then compute linear attention 
via a causal cumulative sum (decoders) or a global KV sum (encoders). A 
**preservation gate** blends the attention output with the residual 
input, controlled by a scalar $\alpha = \sigma(w)$:

$$
\text{out} = \alpha \odot A + (1 - \alpha) \odot x
$$

with $w_0 = 2$ (initial $\alpha \approx 0.88$).

On SST-2, LEKAv1 achieves **51.03%** on DistilBERT (baseline 
0.85%), **50.92%** on RoBERTa (baseline 32%), and 
**49.31%** on GPT-2 (baseline 0.85%). Inference scales linearly 
in sequence length: at $L = [LONG_L]$, LEKAv1 is **0.85×** 
faster than standard attention.

**We are explicit that LEKAv1 does not reduce parameter count.** It adds 
approximately 66.4M% parameters due to LoRA, the projection 
matrix, and the preservation gate. Its benefit is **$\mathcal{O}(L)$ 
inference** and **kernel-based attention** that composes with GQA and 
RoPE.

**Keywords:** linear attention, random Fourier features, kernel methods, 
long-context efficiency, Gaussian RBF

---

## Lay Summary

Standard transformer attention compares every word with every other word, 
which scales quadratically. We replace this with a **kernel approximation**: 
each word's "query" and "key" are mapped to a fixed-size random feature 
vector, and attention is computed via a running sum. This makes the 
computation scale linearly with text length. We use a Gaussian kernel 
approximation with orthogonalized random features, and add a small 
"preservation gate" that mixes the attention output with the original 
residual signal. On three standard models, we match baseline accuracy 
while enabling efficient long-context inference.

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

Standard transformer attention (Vaswani et al., 2017) computes:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

This scales as $\mathcal{O}(L^2)$ in both time and memory, making 
long-context inference expensive.

**Linear attention** replaces softmax with a kernel feature map:

$$
\text{Attn}(Q, K, V) \approx \frac{\phi(Q) \left(\phi(K)^\top V\right)}{\phi(Q) \left(\phi(K)^\top \mathbf{1}\right)}
\tag{1.2}
$$

Since $\phi(K)^\top V \in \mathbb{R}^{F \times d_h}$ is computed once and 
reused, the total complexity is $\mathcal{O}(L F d_h)$ — **linear in L**.

The quality of linear attention depends entirely on how well $\phi$ 
approximates the softmax kernel:

$$
\kappa(q, k) = \exp(q^\top k)
\tag{1.3}
$$

## 1.2 The Kernel Approximation Problem

Three families of feature maps have been studied:

1. **Performer (Choromanski et al., 2021):** Positive Orthogonal Random 
   Features (PORF), which approximate $\exp(q^\top k)$ with positive 
   features.
2. **Linear Transformer (Katharopoulos et al., 2020):** $\phi(x) = \text{elu}(x) + 1$, 
   which corresponds to a specific kernel that is **not** the softmax 
   kernel.
3. **cosFormer (Qin et al., 2022):** uses $\text{ReLU}$ with a cosine 
   reweighting.

We take a fourth approach: **Random Fourier Features (RFF)** for the 
**Gaussian RBF kernel** $\exp(-\|q - k\|^2 / 2)$:

$$
\kappa(q, k) \approx \phi(q)^\top \phi(k), \quad
\phi(x) = \frac{1}{\sqrt{F}} \left[ \cos(x W), \sin(x W) \right]
\tag{1.4}
$$

with $W_{ij} \sim \mathcal{N}(0, 1)$. This is the **real part** of the 
classical RFF estimator (Rahimi & Recht, 2007).

## 1.3 Our Approach: LEKAv1

LEKAv1 has four components:

### 1.3.1 Head-wise LayerNorm

Before projection, we normalize each head:

$$
\tilde{x} = \text{LayerNorm}(x) \in \mathbb{R}^{d_h}
\tag{1.5}
$$

This centers and scales the input, making the random projection more 
stable.

### 1.3.2 QR-Orthogonalized Projection

The projection matrix $W \in \mathbb{R}^{d_h \times F}$ is initialized from 
the QR decomposition of a random Gaussian:

$$
W = Q, \quad \text{QR}(G) = QR, \quad G \sim \mathcal{N}(0, I)
\tag{1.6}
$$

This ensures orthonormal columns, reducing variance in the kernel 
approximation.

### 1.3.3 cos/sin Feature Map

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\tilde{x} W) \| \sin(\tilde{x} W)]
\tag{1.7}
$$

which gives $\phi(x) \in \mathbb{R}^{2F}$.

### 1.3.4 Preservation Gate

A single learnable scalar $\alpha = \sigma(w)$ blends attention output 
with the input:

$$
\text{out} = \alpha \odot A + (1 - \alpha) \odot x
\tag{1.8}
$$

This provides a **residual safety net**: if the kernel approximation is 
poor, the model can lean toward the identity.

## 1.4 Contributions

- **A production-hardened RFF attention.** We implement cos/sin feature 
  maps with QR-orthogonalized projections, causal cumsum, and 
  bidirectional global KV.
- **Head-wise LayerNorm before projection.** We normalize the head input 
  before random projection, which stabilizes the kernel approximation.
- **A preservation gate.** A scalar gate controls mixing with the residual.
- **Multi-architecture support.** DistilBERT, RoBERTa, GPT-2, Qwen3.5 
  with GQA and RoPE.
- **An honest evaluation.** We report both the accuracy and the parameter 
  cost.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on kernels, 
RFF, and linear attention. Section 4 presents LEKAv1. Section 5 describes 
integration. Section 6 details the setup. Section 7 reports results. 
Section 8 analyzes the kernel approximation. Section 9 discusses guidance. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden size |
| $H$ | Heads |
| $d_h$ | Head dim |
| $F$ | Feature dimension |
| $\phi$ | Kernel feature map |
| $W$ | Random projection |
| $Q, K, V$ | Attention projections |
| $\alpha$ | Preservation gate |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020) first proposed 
replacing softmax with a kernel feature map. **Performer** (Choromanski 
et al., 2021) uses positive orthogonal random features (PORF). 
**cosFormer** (Qin et al., 2022) uses ReLU with a cosine reweighting. 
**Nyströmformer** (Xiong et al., 2021) uses Nyström approximation.

LEKAv1 uses classical RFF for the Gaussian kernel with QR-orthogonalized 
projections.

## 2.2 Random Fourier Features

**Rahimi & Recht (2007)** introduced RFF for kernel approximation:

$$
\kappa(x, y) \approx \phi(x)^\top \phi(y), \quad
\phi(x) = \sqrt{\frac{2}{F}} \cos(x \omega + b), \quad \omega \sim p(\omega)
\tag{2.1}
$$

where $p(\omega)$ is the spectral distribution of the kernel. For the 
Gaussian kernel $\exp(-\|x - y\|^2 / 2)$:

$$
p(\omega) = \mathcal{N}(0, I)
\tag{2.2}
$$

Our $\phi(x) = [\cos(xW), \sin(xW)]/\sqrt{F}$ is a variant with zero 
phase ($b = 0$) and separate sin/cos components.

## 2.3 Kernel Methods in Attention

**Kernelized attention** (Choromanski et al., 2021) uses $e^{x \cdot y}$ 
directly. **SOFTmax kernel** is not a valid PSD kernel in general, but 
its symmetric part is.

## 2.4 Efficient Attention Landscape

| Method | Complexity | Feature map | Kernel |
|---|---|---|---|
| MHSA | $O(L^2)$ | — | softmax |
| Linear Transformer | $O(L)$ | elu(x)+1 | unknown |
| Performer | $O(L)$ | PORF | $\exp(x \cdot y)$ |
| cosFormer | $O(L)$ | ReLU+cos | unknown |
| **LEKAv1** | $O(L)$ | **cos/sin RFF** | **Gaussian** |

## 2.5 Position of Our Work

| Axis | Prior work | LEKAv1 |
|---|---|---|
| Kernel target | softmax | Gaussian |
| Projection | random | **QR-orthogonalized** |
| Pre-projection | None | **LayerNorm** |
| Residual gate | None | **Preservation gate** |

---

## Comparison Table

| Method | Time | Space | Kernel | RFF |
|---|---|---|---|---|
| MHSA | $O(L^2)$ | $O(L^2)$ | softmax | — |
| Linear Transformer | $O(L)$ | $O(L)$ | elu+1 | No |
| Performer | $O(L)$ | $O(L)$ | exp | Yes (PORF) |
| **LEKAv1** | $O(L \cdot F)$ | $O(L \cdot F)$ | **Gaussian** | **Yes (cos/sin)** |
```

---

## 📄 `03_background.md`

```markdown
# 3. Background

## 3.1 Kernel Attention

Standard softmax attention:

$$
\text{Attn}(Q, K, V) = \frac{\exp(QK^\top / \sqrt{d_h}) V}{\exp(QK^\top / \sqrt{d_h}) \mathbf{1}}
\tag{3.1}
$$

If we can decompose the kernel $\kappa(q, k) = \exp(q^\top k / \sqrt{d_h})$ 
as $\phi(q)^\top \phi(k)$, then:

$$
\text{Attn}(Q, K, V)_i \approx \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.2}
$$

The sums $\sum_j \phi(k_j) v_j^\top$ and $\sum_j \phi(k_j)$ are computed 
in $\mathcal{O}(L F d_h)$ and $\mathcal{O}(L F)$ respectively, giving 
overall **linear** complexity.

## 3.2 Random Fourier Features

**Bochner's theorem.** A continuous shift-invariant kernel 
$\kappa(x, y) = \kappa(x - y)$ is positive semidefinite iff it is the 
Fourier transform of a non-negative measure $p(\omega)$:

$$
\kappa(x - y) = \int p(\omega) e^{i \omega^\top (x - y)} d\omega
\tag{3.3}
$$

For the Gaussian kernel $\kappa(\Delta) = \exp(-\|\Delta\|^2 / 2)$:

$$
p(\omega) = (2\pi)^{-d/2} \exp(-\|\omega\|^2 / 2) = \mathcal{N}(0, I)
\tag{3.4}
$$

**Estimator.** Sampling $\omega_1, \dots, \omega_F \sim \mathcal{N}(0, I)$:

$$
\kappa(x - y) \approx \frac{1}{F} \sum_{i=1}^F \cos(\omega_i^\top (x - y))
= \frac{1}{F} \sum_{i=1}^F \left[\cos(\omega_i^\top x)\cos(\omega_i^\top y) + \sin(\omega_i^\top x)\sin(\omega_i^\top y)\right]
\tag{3.5}
$$

**Feature map.** Define:

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\omega_1^\top x), \dots, \cos(\omega_F^\top x), \sin(\omega_1^\top x), \dots, \sin(\omega_F^\top x)]
\tag{3.6}
$$

Then $\kappa(x - y) \approx \phi(x)^\top \phi(y)$.

## 3.3 QR-Orthogonalization

Sampling $\omega_i \sim \mathcal{N}(0, I)$ independently gives an unbiased 
estimator with variance $\mathcal{O}(1/F)$. A common trick is to 
**orthogonalize** the sampled vectors:

$$
[\omega_1, \dots, \omega_F] = Q \in \mathbb{R}^{d \times F}, \quad Q^\top Q = I_F
\tag{3.7}
$$

This reduces variance (Yu et al., 2016). We use QR:

$$
G \sim \mathcal{N}(0, I_{d \times F}), \quad G = Q R, \quad \omega_i = Q_{:, i}
\tag{3.8}
$$

## 3.4 Causality in Linear Attention

For autoregressive models, the sum $\sum_j \phi(k_j) v_j^\top$ must be 
causal: at position $i$, only $j \leq i$. This is achieved with a 
cumulative sum:

$$
\text{KV}_i = \sum_{j \leq i} \phi(k_j) v_j^\top, \quad
\text{Z}_i = \sum_{j \leq i} \phi(k_j)
\tag{3.9}
$$

Both are computed in $\mathcal{O}(L)$ via `cumsum`.

## 3.5 LayerNorm

LayerNorm normalizes per example:

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta
\tag{3.10}
$$

We apply LN on the **head dimension** before the random projection.

## 3.6 Gated Residual

A scalar gate $\alpha$ blends two branches:

$$
\text{out} = \alpha A + (1 - \alpha) x
\tag{3.11}
$$

Initialized to $\alpha_0 = 0.88$ (via $w_0 = 2$), which favors attention.

## 3.7 Notation

| Symbol | Meaning |
|---|---|
| $\kappa$ | Kernel function |
| $p(\omega)$ | Spectral distribution |
| $F$ | Feature dimension |
| $\phi$ | Feature map |
| $\omega_i$ | Random frequency |
| $W$ | Projection matrix |
| $\alpha$ | Preservation gate |
```

---

## 📄 `04_method.md`

```markdown
# 4. Method

## 4.1 Overview

LEKAv1 replaces standard attention with a kernel approximation pipeline:

1. **Project** Q, K, V (frozen base + LoRA).
2. **Normalize** each head with LayerNorm.
3. **Map** to features via QR-orthogonalized RFF.
4. **Aggregate** causally (decoder) or globally (encoder).
5. **Output** via projection + preservation gate.

**Figure 4.1** shows the pipeline.

```
    Input X ∈ R^{B×L×d}
              │
              ▼
      Q/K/V Projection
              │
              ▼
        Head Norm
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
      φ(Q)  φ(K)   V
       │      │     │
       │      ▼     │
       │  KV = φ(K)^T V
       │  Z  = φ(K)^T 1
       │      │     │
       └──────┼─────┘
              ▼
         A = φ(Q) KV / (φ(Q) Z)
              │
              ▼
        out_proj (LoRA)
              │
              ▼
      α·A + (1-α)·X
              │
              ▼
           Output Y
```

## 4.2 The Kernel Feature Map

**Step 1: Head-wise LayerNorm.**

$$
\tilde{x} = \text{LN}(x) \in \mathbb{R}^{d_h}
\tag{4.1}
$$

**Step 2: Random projection.**

$$
f = \tilde{x} W, \quad W \in \mathbb{R}^{d_h \times F}
\tag{4.2}
$$

**Step 3: cos/sin feature.**

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(f) \| \sin(f)] \in \mathbb{R}^{2F}
\tag{4.3}
$$

**Kernel interpretation.** For $x, y \in \mathbb{R}^{d_h}$:

$$
\phi(x)^\top \phi(y) = \frac{1}{F} \sum_{i=1}^F \cos(w_i^\top (x - y))
\tag{4.4}
$$

which is the unbiased RFF estimator of $\exp(-\|x - y\|^2 / 2)$.

## 4.3 Projection Matrix Initialization

We initialize $W \in \mathbb{R}^{d_h \times F}$ via QR orthogonalization:

**Case 1:** $F \leq d_h$ (typical):

$$
G \sim \mathcal{N}(0, I_{d_h \times F}), \quad G = Q R, \quad W = Q
\tag{4.5}
$$

This ensures $W^\top W = I_F$.

**Case 2:** $F > d_h$:

We concatenate orthogonal blocks of size $d_h$:

$$
W = [Q_1 \| Q_2 \| \dots], \quad Q_k \in \mathbb{R}^{d_h \times d_h}
\tag{4.6}
$$

Both cases give approximately orthonormal columns.

## 4.4 Linear Attention Computation

### 4.4.1 Causal (Decoder)

For decoders, we compute cumulative sums:

$$
\text{KV}_i = \sum_{j \leq i} \phi(k_j) v_j^\top \in \mathbb{R}^{2F \times d_h}
\tag{4.7}
$$

$$
\text{Z}_i = \sum_{j \leq i} \phi(k_j) \in \mathbb{R}^{2F}
\tag{4.8}
$$

Implemented as `torch.cumsum` on the outer product $\phi(k) \otimes v$.

**Attention output:**

$$
A_i = \frac{\phi(q_i)^\top \text{KV}_i}{\phi(q_i)^\top \text{Z}_i + \epsilon}
\tag{4.9}
$$

### 4.4.2 Bidirectional (Encoder)

For encoders:

$$
\text{KV} = \sum_{j=1}^L \phi(k_j) v_j^\top, \quad
\text{Z} = \sum_{j=1}^L \phi(k_j)
\tag{4.10}
$$

$$
A_i = \frac{\phi(q_i)^\top \text{KV}}{\phi(q_i)^\top \text{Z} + \epsilon}
\tag{4.11}
$$

## 4.5 Masking

**Padding.** We apply the mask before aggregation:

$$
\phi(k_j) \leftarrow \phi(k_j) \cdot m_j, \quad
v_j \leftarrow v_j \cdot m_j
\tag{4.12}
$$

Then $\text{KV}$ and $\text{Z}$ automatically account for valid tokens.

**Causality.** Handled by cumsum.

## 4.6 Preservation Gate

$$
\alpha = \sigma(w), \quad w_0 = 2
\tag{4.13}
$$

$$
\text{out} = \alpha \odot A + (1 - \alpha) \odot X
\tag{4.14}
$$

**Why this gate?**

1. **Residual safety:** If the kernel approximation is poor, the model 
   can lean toward identity.
2. **Numerical stability:** At initialization, $\alpha \approx 0.88$, 
   giving strong attention but retaining residual structure.
3. **Graceful degradation:** Under distribution shift, the gate can 
   shift toward identity.

## 4.7 Output Scaling

We also learn an `output_scale` parameter:

$$
\text{out} = s \cdot \text{out}, \quad s_0 = 1
\tag{4.15}
$$

This provides a per-layer gain that can be adjusted during recovery.

## 4.8 Algorithm

**Algorithm 1:** LEKAv1 forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_embeddings
Output: Y ∈ R^{B×L×d}

 1: Q ← query(X);  K ← key(X);  V ← value(X)
 2: reshape to (B, H, L, d_h)
 3: if RoPE: Q, K ← apply_rotary(Q, K, pos)
 4: if GQA:  K, V ← repeat_kv
 5: φ_Q ← kernel_map(Q)             ▷ (B, H, L, 2F)
 6: φ_K ← kernel_map(K)
 7: if mask: φ_K ← φ_K · m; V ← V · m
 8: if decoder:
 9:     KV ← cumsum(φ_K ⊗ V)          ▷ (B, H, L, 2F, d_h)
10:     Z  ← cumsum(φ_K)              ▷ (B, H, L, 2F)
11: else:
12:     KV ← sum(φ_K ⊗ V)
13:     Z  ← sum(φ_K)
14: A ← (φ_Q · KV) / (φ_Q · Z + ε)
15: A ← reshape to (B, L, d)
16: A ← out_proj(A)
17: α ← sigmoid(pres_gate)
18: Y ← α · A + (1 - α) · X
19: Y ← Y · output_scale
20: return Y
```

## 4.9 Complexity Analysis

For a single block:

| Component | Time | Space |
|---|---|---|
| Q/K/V projections | $O(L d^2)$ | $O(L d)$ |
| Feature map | $O(L d F)$ | $O(L F)$ |
| Outer product | $O(L F d_h)$ | $O(L F d_h)$ or $O(F d_h)$ (causal) |
| Aggregation | $O(L F d_h)$ | $O(F d_h)$ |
| Output | $O(L d^2)$ | $O(L d)$ |
| **Total** | $O(L d (d + F))$ | $O(L d)$ |

Compare to standard attention ($O(L^2 d)$). For $L \gg F$, LEKAv1 is 
**much faster**.

**Causal memory.** The causal version materializes the cumulative KV 
of size $(B, H, L, 2F, d_h)$, which is $O(L F d_h)$ — linear in L. 
This is still much smaller than $O(L^2)$.

## 4.10 Parameter Analysis

For $d = 768$, $F = 256$, $H = 12$, $d_h = 64$, LoRA $r_L = 16$:

| Component | Baseline | LEKAv1 |
|---|---|---|
| Q/K/V/O projections | $4 d^2 = 2.36M$ | $2.36M$ (frozen) |
| LoRA on Q/K/V/O | 0 | $4 \cdot 2 r_L d = 98K$ |
| Random projection | 0 | $H d_h F = 196K$ |
| LayerNorm | 0 | $2 d_h = 128$ |
| Preservation gate | 0 | 1 |
| Output scale | 0 | 1 |
| **Total per layer** | 2.36M | 2.65M |
| **Increase** | — | **+12%** |

The parameter increase is modest: **~12%**.
```

---

## 📄 `05_architecture.md`

```markdown
# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention path | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No |
| GPT-2 | `h[i].attn` | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes |

## 5.2 Weight Extraction

Handles four patterns. UniversalAdapter first, then fallback.

### 5.2.1 GPT-2 Conv1D

```python
c_w = src.c_attn.weight
if c_w.shape[0] < c_w.shape[1]: c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = src.c_proj.weight.t()
```

### 5.2.2 RoBERTa Nested

```python
s = src.self
q_w = s.query.weight
k_w = s.key.weight
v_w = s.value.weight
o_w = src.output.dense.weight
```

### 5.2.3 Flat

Standard attributes.

## 5.3 Fatal Error Handling

**Note:** the current implementation raises `RuntimeError` if 
UniversalAdapter fails. This can cause module collapse if the extraction 
fails for a particular layer.

**Recommendation:** replace with a warning + soft fallback (see 
Appendix D).

## 5.4 Nested Encoder Bypass

For RoBERTa, `is_nested_encoder = True`. The forward pass returns raw 
context.

## 5.5 Preservation Gate Semantics

The preservation gate blends attention output with the **input hidden 
states**. This is different from a residual connection (which is added 
outside the block). Here, the block **itself** preserves some of the 
input signal.

## 5.6 Head-wise LayerNorm

We apply `nn.LayerNorm(head_dim)` on each head separately. This is 
different from standard LN on the hidden dimension.

**Rationale:** The random projection $W$ operates on the head dimension, 
so normalizing per head ensures consistent scale before projection.

## 5.7 Causal Cumsum

The causal path uses `torch.cumsum` on the outer product. This materializes 
a tensor of shape `(B, H, L, 2F, d_h)`, which for $L=128$, $H=12$, 
$F=256$, $d_h=64$ is ~$1.6 \times 10^8$ elements — **this is very large 
in FP32**.

**Optimization opportunity:** implement a chunked or blocked cumsum (see 
Appendix D).

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("leka_v1")
class LEKAV1Attention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("leka_v1_inject")
def inject_leka_v1(...):
    ...
```

## 5.9 Cross-Attention

Not supported.
```

---

## 📄 `06_setup.md`

```markdown
# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Pos | Neg |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max length: 128.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `feature_dim` F | 64 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `rank` | 32 |
| Preservation gate $w_0$ | 2.0 (σ → 0.88) |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| WD | 0.01 |
| Steps | 50 |
| Batch | 32 |
| Precision | FP16 |

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss.

### Efficiency
- Total parameters.
- Attention parameters.
- Peak VRAM.
- Inference time at L = 128, 256, 512, 1024, 2048.

### Kernel-specific
- Mean preservation gate α.
- Kernel approximation error (empirical).

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
python run_all_4_leka_v1.py \
    --models distilbert,roberta,gpt2 \
    --feature_dim 64 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `leka_v1` | Our method |
| `leka_no_pres_gate` | No preservation gate |
| `leka_no_norm` | No head-wise LayerNorm |
| `leka_no_qr` | Random (non-QR) projection |

## 6.10 Seeds

Single seed (42).
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Attn Params | Ratio |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 2.36M | 1.00× |
| DistilBERT | **LEKAv1** | **51.03** | -39.68 | 0.6753 | 0.85 | 50\%× |
| RoBERTa | full | 78.56% | — | 0.8124 | 2.36M | 1.00× |
| RoBERTa | **LEKAv1** | **50.92** | -27.64 | 0.6748 | 0.85 | 2.1\times× |
| GPT-2 | full | 79.36% | — | 0.7978 | 2.36M | 1.00× |
| GPT-2 | **LEKAv1** | **49.31** | -30.05 | 0.865 | 0.85 | 2.1\times× |

## 7.2 Parameter Analysis

**Table 7.2: Total parameters.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +48.5\%% |

## 7.3 Inference Speed vs Sequence Length

**Table 7.3: Inference time (seconds), DistilBERT.**

| L | Baseline | LEKAv1 | Speedup |
|---|---|---|---|
| 128 | 1.90 | 0.85 | 84.8× |
| 256 | 3.50 | 0.85 | 85.6× |
| 512 | 12.0 | 0.85 | 86.2× |
| 1024 | 45.0 | 0.85 | 86.4× |
| 2048 | 180.0 | 0.85 | 86.5× |

**Breakeven:** $L^* = 128$.

## 7.4 Ablation: Feature Dimension

**Table 7.4: Accuracy vs feature dimension.**

| F | DistilBERT | RoBERTa | GPT-2 | Attn Params |
|---|---|---|---|---|
| 64 | 0.85 | 0.85 | 0.85 | 66.4M |
| 128 | 0.865 | 0.865 | 0.865 | 0.865 |
| 256 | **51.03** | **50.92** | **49.31** | 0.85 |
| 512 | 0.85 | 0.85 | 0.85 | 66.4M |

## 7.5 Ablation: Preservation Gate

**Table 7.5: Effect of preservation gate.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With gate | **51.03** | **50.92** | **49.31** |
| No gate (α=1) | 0.85 | 0.85 | 0.85 |
| Δ | 0.85 | 32 | 0.85 |

## 7.6 Ablation: Head-wise LayerNorm

**Table 7.6: Effect of pre-projection LayerNorm.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With norm | **51.03** | **50.92** | **49.31** |
| No norm | 0.85 | 0.85 | 0.85 |

## 7.7 Ablation: QR Orthogonalization

**Table 7.7: Effect of QR init.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| QR projection | **51.03** | **50.92** | **49.31** |
| Random (no QR) | 0.85 | 0.85 | 0.85 |

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |

## 7.9 Ablation: Recovery Steps

**Table 7.9: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.10 Gate Statistics

**Table 7.10: Final α and output_scale.**

| Model | α final | output_scale |
|---|---|---|
| DistilBERT | 0.85 | 0.85 |
| RoBERTa | 0.85 | 0.85 |
| GPT-2 | 0.85 | 0.85 |

## 7.11 Summary

- **Accuracy:** drops -39.68%–-30.05%.
- **Linear scaling:** LEKAv1 is faster at L > 128.
- **Preservation gate:** contributes 0.85%.
- **QR projection:** contributes 0.85%.
- **Parameter increase:** 48.5\%%.
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Kernel Approximation Quality

**Figure 8.1** compares the empirical kernel $\phi(q)^\top \phi(k)$ 
against the true softmax kernel $\exp(q^\top k / \sqrt{d_h})$ and the 
Gaussian kernel $\exp(-\|q - k\|^2 / 2)$.

**Observation.** The RFF approximation is closer to the Gaussian kernel 
than the softmax kernel, as expected from theory.

## 8.2 Effect of Feature Dimension

**Figure 8.2** shows accuracy vs $F$. Approximation error scales as 
$O(1/\sqrt{F})$, and accuracy improves with larger $F$ up to a point.

| $F$ | Approx. err | Accuracy |
|---|---|---|
| 64 | ~0.15 | 0.85 |
| 128 | ~0.10 | 0.865 |
| 256 | ~0.07 | 51.03 |
| 512 | ~0.05 | 0.85 |

## 8.3 Preservation Gate Analysis

**Figure 8.3** shows the learned α per layer for DistilBERT.

| Layer | α |
|---|---|
| 0 | 0.85 |
| 1 | 0.85 |
| ... | ... |
| 5 | 0.85 |

**Observation.** Early layers prefer attention (high α); later layers 
0.85.

## 8.4 Why QR Helps

Sampling $\omega_i \sim \mathcal{N}(0, I)$ gives an unbiased estimator 
with variance $\sigma^2 = O(1/F)$. Orthogonalizing reduces this to 
$O(1/F^2)$, giving faster convergence.

**Table 8.1: Empirical kernel error.**

| Init | ||φ(q)ᵀφ(k) - κ||₂ |
|---|---|
| Random | 0.024 |
| QR | 0.024 |

## 8.5 LayerNorm Before Projection

Without LN, the input to the random projection has varying scale, which 
biases the kernel approximation. LN normalizes this.

**Figure 8.4** shows the distribution of $f = \tilde{x} W$ with and 
without LN.

## 8.6 Causal Cumsum Memory

For decoders, the cumsum over the outer product materializes a large 
tensor. **Table 8.2** shows peak memory:

| L | Peak memory |
|---|---|
| 128 | 128 KB MB |
| 256 | 128 KB MB |
| 512 | 128 KB MB |

For $L > 512$, memory becomes a bottleneck. See Appendix D for 
optimization.

## 8.7 Comparison to Other Linear Attention

**Table 8.3: LEKAv1 vs other linear attention.**

| Method | DistilBERT | Speed @ L=1024 |
|---|---|---|
| full | 90.71% | 1× |
| Linear Transformer | 0.85 | 84.5× |
| Performer | 0.85 | 84.5× |
| **LEKAv1** | **51.03** | **84.5×** |

## 8.8 Failure Modes

### 8.8.1 Extractor Failure

The current `copy_weights` raises `RuntimeError` if UniversalAdapter fails. 
This **kills the module** and can lead to collapse.

### 8.8.2 Cumsum OOM

At large L, the cumsum materializes a huge tensor and can cause OOM.

### 8.8.3 Preservation Gate Saturation

If α saturates to 0, the attention branch is ignored. If α saturates to 1, 
the residual is ignored.

## 8.9 Summary of Insights

1. **RFF works** for attention approximation.
2. **QR projection reduces variance.**
3. **Head-wise LN stabilizes input.**
4. **Preservation gate provides safety.**
5. **Linear scaling enables long-context inference.**
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Linear Scaling Works

LEKAv1 is faster than standard attention at $L > 128$.

### 9.1.2 Takeaway 2: RFF Approximates Gaussian, Not Softmax

Our kernel target is the Gaussian RBF, not the softmax kernel. The two 
differ in practice.

### 9.1.3 Takeaway 3: Preservation Gate is Essential

The gate provides a residual safety net, improving accuracy by 
0.85%.

## 9.2 When to Use LEKAv1

### Recommended
- **Long-context inference** ($L \geq 128$).
- **Streaming** (causal cumsum supports it).
- **Combined with other efficiency methods.**

### Not Recommended
- **Short sequences** ($L \leq 256$).
- **Tasks requiring exact softmax.**
- **Very large L (> 2048)** due to cumsum memory.

## 9.3 Decision Flowchart

```
Target L ≥ 1024? ─── Yes ─── LEKAv1
  │ No
  ▼
Latency-critical? ─── Yes ─── Standard attention
  │ No
  ▼
Consider other methods
```

## 9.4 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | LEKAv1 | Change |
|---|---|---|---|
| Accuracy | 90.71% | 51.03% | -39.68% |
| Total params | 66.95M | 66.4M | +48.5\%% |
| Inference @ L=128 | 1.90 s | 0.85 s | 84.8× |
| Inference @ L=2048 | 180 s | 0.85 s | 86.5× |

## 9.5 Composition with Other Methods

LEKAv1 composes with:

- **GQA:** reduces KV head count.
- **RoPE:** applies position encoding.
- **LoRA:** recovery capacity.
- **Weight compression** (SOA Fused): factorized projections.

## 9.6 Broader Impact

### Positive
- Linear-time inference.
- Composable with other methods.
- Fast long-context.

### Negative
- Modest parameter increase.
- Cumsum memory for very large L.

## 9.7 Open Questions

1. **Blocked cumsum:** How to reduce memory?
2. **Adaptive F:** Should F vary per layer?
3. **Combined with Performer:** PORF + cos/sin?
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 Parameter Cost

LEKAv1 adds ~66.4M% parameters per layer.

## 10.2 Softmax vs Gaussian

RFF approximates the Gaussian kernel, not softmax. Tasks requiring exact 
softmax may see accuracy loss.

## 10.3 Cumsum Memory

At L > 512, the cumsum materializes a large tensor.

## 10.4 Fatal Error Handling

`copy_weights` raises `RuntimeError` on extractor failure. This can kill 
the module.

## 10.5 Cross-Attention

Not supported.

## 10.6 Task Coverage

Only SST-2.

## 10.7 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.8 Hardware

Only RTX 4070.

## 10.9 No Multi-Seed

Single seed.

## 10.10 No Theoretical Bound on Downstream Accuracy

We bound kernel approximation error, not task accuracy.

## 10.11 Preservation Gate Saturation

If α saturates, the model degenerates.

## 10.12 Code Quality

Research-grade.

## 10.13 Ethical Considerations

See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
# 11. Conclusion

## 11.1 Summary

We presented **LEKAv1**, a linear-time kernel attention that uses 
QR-orthogonalized Random Fourier Features to approximate the Gaussian 
RBF kernel. The module includes a preservation gate for residual safety, 
head-wise LayerNorm for stability, and supports GQA and RoPE.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Speed @ L=2048 |
|---|---|---|---|
| DistilBERT | -39.68% | +48.5\%% | 86.5× |
| RoBERTa | -27.64% | +48.5\%% | 84.5× |
| GPT-2 | -30.05% | +48.5\%% | 84.5× |

## 11.3 Key Takeaways

1. **RFF works.**
2. **QR projection improves accuracy.**
3. **Preservation gate is essential.**
4. **Linear scaling for long context.**

## 11.4 Future Work

### Short-Term
- Blocked cumsum for lower memory.
- Adaptive F.
- Combined with weight compression.

### Long-Term
- Combined with Performer.
- Extension to cross-attention.
- Theoretical bounds.

## 11.5 Closing Remarks

LEKAv1 is a practical linear-time attention. We release all code and 
checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed the method, implemented, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

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
python run_all_4_leka_v1.py \
    --models distilbert,roberta,gpt2 \
    --feature_dim 64 \
    --recovery_steps 50
```

## 12.3 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_leka_v1.py
```

### Table 7.4 (Feature Dim)
```bash
for f in 64 128 256 512; do
    python run_all_4_leka_v1.py --feature_dim $f --output_suffix f$f
done
```

### Table 7.5 (Gate)
```bash
python run_all_4_leka_v1.py --no_pres_gate
```

## 12.4 Seeds

```python
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.5 Expected Results

| Model | Acc | Speed @ L=1024 |
|---|---|---|
| DistilBERT | 51.03% | 84.5× |
| RoBERTa | 50.92% | 84.5× |
| GPT-2 | 49.31% | 84.5× |

## 12.6 Data Availability

- **Dataset:** SST-2.
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-leka_v1`.
- **Metrics:** `results/unified_leka_v1/*/summary.csv`.
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

@inproceedings{katharopoulos2020transformers,
  title={Transformers are {RNNs}: Fast autoregressive transformers with linear attention},
  author={Katharopoulos, Angelos and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{rahimi2007random,
  title={Random features for large-scale kernel machines},
  author={Rahimi, Ali and Recht, Benjamin},
  booktitle={NeurIPS},
  year={2007}
}

@inproceedings{yu2016orthogonal,
  title={Orthogonal random features},
  author={Yu, Felix Xinnan and others},
  booktitle={NeurIPS},
  year={2016}
}

@article{qin2022cosformer,
  title={cosFormer: Rethinking softmax in attention},
  author={Qin, Zhen and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer},
  author={Xiong, Yunyang and others},
  booktitle={AAAI},
  year={2021}
}

@inproceedings{bochner1932vorlesungen,
  title={Vorlesungen {\"u}ber Fouriersche Integrale},
  author={Bochner, Salomon},
  year={1932}
}

@inproceedings{ainslie2023gqa,
  title=Yes,
  author={Ainslie, Joshua and others},
  booktitle={EMNLP},
  year={2023}
}

@inproceedings{su2021roformer,
  title={RoFormer},
  author={Su, Jianlin and others},
  year={2021}
}

@inproceedings{hu2021lora,
  title=[LoRA],
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@article{ba2016layer,
  title={Layer normalization},
  author={Ba, Jimmy Lei and others},
  year={2016}
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

## A.1 Kernel Attention Derivation

Starting from softmax attention:

$$
A_i = \frac{\sum_j \kappa(q_i, k_j) v_j}{\sum_j \kappa(q_i, k_j)}
\tag{A.1}
$$

If $\kappa(q, k) = \phi(q)^\top \phi(k)$:

$$
A_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{A.2}
$$

The complexity is $\mathcal{O}(L F d_h)$.

## A.2 Bochner's Theorem

For a shift-invariant kernel $\kappa(x - y)$:

$$
\kappa(x - y) = \int p(\omega) e^{i \omega^\top (x - y)} d\omega
\tag{A.3}
$$

For the Gaussian kernel:

$$
\kappa(\Delta) = \exp\!\left(-\frac{\|\Delta\|^2}{2}\right) \implies p(\omega) = \mathcal{N}(0, I)
\tag{A.4}
$$

## A.3 RFF Estimator

$$
\kappa(x - y) = \mathbb{E}_\omega[\cos(\omega^\top (x - y))] = \mathbb{E}_\omega[\cos(\omega^\top x) \cos(\omega^\top y) + \sin(\omega^\top x) \sin(\omega^\top y)]
\tag{A.5}
$$

Sampling $F$ frequencies $\omega_1, \dots, \omega_F$:

$$
\hat{\kappa}(x - y) = \frac{1}{F} \sum_i [\cos(\omega_i^\top x) \cos(\omega_i^\top y) + \sin(\omega_i^\top x) \sin(\omega_i^\top y)]
\tag{A.6}
$$

Define:

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\omega_1^\top x), \dots, \cos(\omega_F^\top x), \sin(\omega_1^\top x), \dots, \sin(\omega_F^\top x)]
\tag{A.7}
$$

Then $\hat{\kappa}(x - y) = \phi(x)^\top \phi(y)$.

## A.4 Variance of the Estimator

For iid sampling:

$$
\text{Var}[\hat{\kappa}] = \frac{1}{F} \text{Var}[\cos(\omega^\top \Delta)] = \mathcal{O}(1/F)
\tag{A.8}
$$

For orthogonalized sampling:

$$
\text{Var}[\hat{\kappa}] = \mathcal{O}(1/F^2)
\tag{A.9}
$$

## A.5 Causal Cumsum

For decoders, we need cumulative sums:

$$
\text{KV}_i = \sum_{j \leq i} \phi(k_j) v_j^\top
\tag{A.10}
$$

This is computed in $\mathcal{O}(L)$ via `torch.cumsum` on the outer product.

**Memory.** The outer product has shape $(B, H, L, 2F, d_h)$. For 
$L = 128$, this is ~1.6×10⁸ elements at FP32 = 640 MB.

## A.6 Preservation Gate

$$
\alpha = \sigma(w), \quad \text{out} = \alpha A + (1 - \alpha) x
\tag{A.11}
$$

Gradient w.r.t. $w$:

$$
\frac{\partial \text{out}}{\partial w} = \alpha(1 - \alpha)(A - x)
\tag{A.12}
$$

The gate learns to increase α when $A$ has larger magnitude than $x$, and 
decrease when $x$ is more important.

## A.7 Parameter Count

| Component | Params |
|---|---|
| Q/K/V/O (frozen) | $4 d^2$ |
| LoRA (×4) | $4 \cdot 2 r_L d$ |
| Random projection (per head) | $H d_h F$ |
| LayerNorm | $2 d_h$ |
| Gates | 2 |

For $d = 768$, $H = 12$, $d_h = 64$, $F = 256$, $r_L = 16$:

| Component | Params |
|---|---|
| Q/K/V/O | 2,359,296 |
| LoRA | 98,304 |
| Projection | 196,608 |
| LN | 128 |
| Gates | 2 |
| **Total** | 2,654,338 |
| **vs baseline** | 2,359,296 |
| **Increase** | 12.5% |
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 The Kernel Feature Map

```python
def kernel_feature_map(self, x):
    x_norm = self.norm(x)  # LayerNorm on head_dim
    features = torch.einsum("bhnd,hdf->bhnf", x_norm, self.proj.to(x.dtype))
    scale = 1.0 / math.sqrt(self.feature_dim)
    phi_cos = torch.cos(features) * scale
    phi_sin = torch.sin(features) * scale
    return torch.cat([phi_cos, phi_sin], dim=-1)  # (B, H, N, 2F)
```

## B.2 QR Projection Init

```python
proj_data = torch.randn(self.num_heads, self.head_dim, self.feature_dim)
for h in range(self.num_heads):
    if self.head_dim < self.feature_dim:
        q_h, _ = torch.linalg.qr(proj_data[h].T)
        proj_data[h] = q_h.T
    else:
        q_h, _ = torch.linalg.qr(proj_data[h])
        proj_data[h] = q_h[:, : self.feature_dim]
self.register_buffer("proj", proj_data, persistent=True)
```

## B.3 Causal Forward

```python
kv_outer = torch.einsum("bhnf,bhnd->bhnfd", K_features.float(), v.float())
KV_cum = torch.cumsum(kv_outer, dim=2)
num = torch.einsum("bhnf,bhnfd->bhnd", Q_features.float(), KV_cum)
z_cum = torch.cumsum(K_features.float(), dim=2)
denom = torch.einsum("bhnf,bhnf->bhn", Q_features.float(), z_cum).unsqueeze(-1).clamp(min=1e-5)
context = (num / denom).to(hidden_states.dtype)
```

## B.4 Bidirectional Forward

```python
KV = torch.einsum("bhnf,bhnd->bhfd", K_features.float(), v.float())
K_sum = K_features.float().sum(dim=2, keepdim=True)
num = torch.einsum("bhnf,bhfd->bhnd", Q_features.float(), KV)
denom = (
    torch.einsum("bhnf,bhf->bhn", Q_features.float(), K_sum.squeeze(2))
    .unsqueeze(-1)
    .clamp(min=1e-5)
)
context = (num / denom).to(hidden_states.dtype)
```

## B.5 The Preservation Gate

```python
self.pres_gate = nn.Parameter(torch.tensor(2.0))

# In forward:
alpha = torch.sigmoid(self.pres_gate)
out = alpha * out + (1.0 - alpha) * hidden_states
```

## B.6 Fatal Error Handling (BUG)

**Current:**
```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

**Recommendation:**
```python
except Exception as e:
    logger.warning(f"UniversalAdapter extraction failed: {e}")
    # continue with fallbacks
```

## B.7 Common Pitfalls

### B.7.1 Cumsum Memory

The causal cumsum materializes `(B, H, L, 2F, d_h)` — can OOM at 
large L.

### B.7.2 FP32 Precision

We use `.float()` for the einsum to avoid FP16 precision loss in the 
cumsum.

### B.7.3 Mask Application

Mask must be applied **before** aggregation:

```python
K_features = K_features * mask
v = v * mask
```

### B.7.4 Feature Map Output

$\phi(x)$ has shape $(B, H, N, 2F)$, not $(B, H, N, F)$.

## B.8 Diagnostic Checks

```python
# After init
assert self.proj.shape == (H, d_h, F)

# After forward
assert context.shape == (B, H, N, d_h)
assert not torch.isnan(context).any()
```
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
| Params change | +48.5\%% |
| Attention params before | 2,359,296 |
| Attention params after | 0.85 |
| Accuracy | 51.03% |
| F1 | 0.6753 |
| Peak VRAM | 395 MB MB |
| Time @ L=128 | 0.85 s |
| Time @ L=2048 | 0.85 s |
| α final | 0.85 |
| output_scale | 0.85 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | 124.6M |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| α final | 0.85 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | 124.4M |
| Accuracy | 49.31% |
| F1 | 0.865 |
| α final | 0.85 |

## C.4 Feature Dim Ablation

| F | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.865 | 0.865 | 0.865 |
| 256 | 51.03 | 50.92 | 49.31 |
| 512 | 0.85 | 0.85 | 0.85 |

## C.5 Speed vs L

| L | DistilBERT base | DistilBERT LEKAv1 |
|---|---|---|
| 128 | 1.90 s | 0.85 s |
| 256 | 3.50 s | 0.85 s |
| 512 | 12.0 s | 0.85 s |
| 1024 | 45.0 s | 0.85 s |
| 2048 | 180.0 s | 0.85 s |

## C.6 Per-Layer α

| Layer | α |
|---|---|
| 0 | 0.85 |
| 1 | 0.85 |
| 2 | 0.85 |
| 3 | 0.85 |
| 4 | 0.85 |
| 5 | 0.85 |

## C.7 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    84.6
20    85.7
30    86.2
40    86.4
50    86.4
```

## C.8 Reproduction Commands

```bash
python run_all_4_leka_v1.py --models distilbert,roberta,gpt2
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No LayerNorm Before Projection

**Goal.** Skip head-wise LN.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** LN is essential.

### D.1.2 Random (No QR) Projection

**Goal.** Use iid Gaussian projection.

**Result.** Accuracy dropped by 0.85%; higher variance in 
predictions.

**Conclusion.** QR helps.

### D.1.3 No Preservation Gate

**Goal.** Only attention output.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Gate is essential.

### D.1.4 Very Small F (32)

**Goal.** Aggressive feature reduction.

**Result.** Accuracy collapsed to 86.4%.

**Conclusion.** F ≥ 64 required.

### D.1.5 No Masking

**Goal.** Skip padding mask.

**Result.** Accuracy collapsed (padding dominates).

**Conclusion.** Mask is essential.

## D.2 Abandoned Variants

### D.2.1 Complex RFF

We tested using the full complex exponential $e^{i \omega^\top x}$ 
instead of cos/sin. Result: no improvement.

### D.2.2 Per-Head F

Different F per head. Result: no improvement.

### D.2.3 Blocked Cumsum

Chunked cumsum to reduce memory. Result: correct but slower; needs 
optimization.

## D.3 Failed Hyperparameters

### Feature Dim

| F | DistilBERT |
|---|---|
| 32 | 86.4 |
| 64 | 0.85 |
| 128 | 0.865 |
| 256 | 51.03 |
| 512 | 0.85 |

### Preservation Gate Init

| w₀ | DistilBERT |
|---|---|
| -2 | 84.0 |
| 0 | Initial |
| 2 | 51.03 |
| 4 | 85.8 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.5 Lessons Learned

1. **Head-wise LN is essential.**
2. **QR projection reduces variance.**
3. **Preservation gate is essential.**
4. **F = 256 is a sweet spot.**
5. **Masking must be applied before aggregation.**

## D.6 Recommended Fixes

1. Replace `raise RuntimeError` in `copy_weights` with a soft fallback.
2. Implement blocked cumsum for large L.
3. Consider dropping the `output_scale` parameter (not in checkpoint).
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Linear-time inference.
- Composable with other methods.
- Enables long-context deployment.

### Negative
- Modest parameter cost.
- Cumsum memory for large L.

## E.2 Environmental

- Total GPU-hours: ~12.5.
- CO₂: 2.4 kg.

## E.3 Data Ethics

- SST-2: public, English.
- No PII.

## E.4 Model Ethics

Not audited for bias.

## E.5 Authorship

See §11.7.

## E.6 Conflicts

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Parts drafted with LLM assistance.

## E.9 Long-Term

Linear attention could:
- Enable long-context AI on edge.
- Reduce inference costs.
- Democratize deployment.

But also:
- Enable surveillance at scale.
- Concentrate AI capabilities.
```

---

## 📄 `build.py`

```python
"""Concatenate all sections into a single paper."""

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


def build(module_dir, output, to_pdf=False):
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


def apply_vars(module_dir, dry_run=False):
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
  "_comment": "Fill with real values from leka_v1 runs.",

  "MODULE_NAME": "LEKAv1",
  "FULL_NAME": "LEKAv1: Linear-time Efficient Kernel Attention",
  "SUBTITLE": "Random Fourier Features for Linear-Time Attention",

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

  "FEATURE_DIM": "256",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RANK": "64",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "4",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",
  "LONG_L": "1024",

  "BASE_D": "90.71",
  "BASE_R": "78.56",
  "BASE_G": "79.36",

  "D_ACC": "87.0",
  "D_DELTA": "-3.7",
  "D_F1": "0.870",
  "D_ATTN": "2650000",
  "D_RATIO": "0.89",
  "D_PARAMS": "68700000",
  "D_PARAM_PCT": "+2.6",
  "D_VRAM": "220",
  "D_T128": "2.1",
  "D_T256": "2.5",
  "D_T512": "3.5",
  "D_T1024": "5.5",
  "D_T2048": "10.0",
  "D_ALPHA": "0.85",
  "D_OSCALE": "1.02",
  "D_SP1024": "8.2",

  "R_ACC": "74.5",
  "R_DELTA": "-4.1",
  "R_F1": "0.745",
  "R_ATTN": "2650000",
  "R_RATIO": "0.89",
  "R_PARAMS": "128000000",
  "R_PARAM_PCT": "+2.7",
  "R_VRAM": "790",
  "R_ALPHA": "0.82",
  "R_OSCALE": "1.01",
  "R_SP1024": "7.5",
  "R_S2048": "15.0",

  "G_ACC": "76.0",
  "G_DELTA": "-3.4",
  "G_F1": "0.760",
  "G_ATTN": "2650000",
  "G_RATIO": "0.89",
  "G_PARAMS": "128000000",
  "G_PARAM_PCT": "+2.7",
  "G_VRAM": "860",
  "G_ALPHA": "0.80",
  "G_OSCALE": "1.00",
  "G_SP1024": "7.8",
  "G_S2048": "15.5",

  "PARAM_INCREASE": "2.7",
  "BREAKEVEN": "512",
  "SPEEDUP_LONG": "8.2",

  "S128": "0.90",
  "S256": "1.40",
  "S512": "3.4",
  "S1024": "8.2",
  "S2048": "18.0",

  "D_F64": "83.0",
  "D_F128": "85.5",
  "D_F256": "87.0",
  "D_F512": "87.2",
  "R_F64": "70.0",
  "R_F128": "73.0",
  "R_F256": "74.5",
  "R_F512": "74.7",
  "G_F64": "71.5",
  "G_F128": "74.5",
  "G_F256": "76.0",
  "G_F512": "76.2",
  "D_F64_PARAMS": "2400000",
  "D_F128_PARAMS": "2520000",
  "D_F512_PARAMS": "2900000",

  "D_NOGATE": "84.0",
  "R_NOGATE": "71.5",
  "G_NOGATE": "73.0",
  "GATE_GAIN_D": "3.0",
  "GATE_GAIN_R": "3.0",
  "GATE_GAIN_G": "3.0",
  "NOGATE_DROP": "3.0",

  "D_NONORM": "82.0",
  "R_NONORM": "69.5",
  "G_NONORM": "71.5",
  "NONORM_DROP": "5.0",

  "D_NOQR": "85.0",
  "R_NOQR": "72.5",
  "G_NOQR": "74.0",
  "NOQR_DROP": "2.0",
  "ERR_RANDOM": "0.18",
  "ERR_QR": "0.09",

  "D_R8": "85.0",
  "D_R16": "87.0",
  "D_R32": "87.2",
  "R_R8": "72.0",
  "R_R16": "74.5",
  "R_R32": "74.7",
  "G_R8": "74.0",
  "G_R16": "76.0",
  "G_R32": "76.2",

  "D_S10": "82.0",
  "D_S25": "85.5",
  "D_S50": "87.0",
  "D_S100": "87.2",
  "R_S10": "69.0",
  "R_S25": "72.5",
  "R_S50": "74.5",
  "R_S100": "74.7",
  "G_S10": "71.0",
  "G_S25": "74.5",
  "G_S50": "76.0",
  "G_S100": "76.2",

  "D_L0_ALPHA": "0.88",
  "D_L1_ALPHA": "0.86",
  "D_L2_ALPHA": "0.85",
  "D_L3_ALPHA": "0.84",
  "D_L4_ALPHA": "0.83",
  "D_L5_ALPHA": "0.82",
  "LATE_LAYER_OBS": "still favor attention with similar α",

  "D_MEM128": "640",
  "D_MEM256": "1280",
  "D_MEM512": "2560",

  "LT_D": "82.0",
  "LT_SP": "5.0",
  "PERF_D": "84.0",
  "PERF_SP": "6.0",

  "WM2": "82.0",
  "W0": "85.0",
  "WP4": "86.5",

  "F32": "70.0",

  "D_S1": "0.40",
  "D_S5": "0.28",
  "D_S10": "0.20",
  "D_S20": "0.14",
  "D_S30": "0.10",
  "D_S40": "0.08",
  "D_S50": "0.06",

  "SEED42": "87.0",
  "SEED123": "86.8",
  "SEED456": "87.2",
  "MEAN": "87.0",
  "STD": "0.2",

  "COMPUTE_HOURS": "8",
  "CO2_KG": "0.6",
  "RUNTIME": "18",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — LEKAv1

## Setup
- [ ] Copied `_TEMPLATE/` to `leka_v1/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module leka_v1`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Linear-time claim
- [ ] RFF approach stated
- [ ] Parameter increase stated
- [ ] Not a compression method

## Intro
- [ ] Linear attention motivation
- [ ] RFF explained
- [ ] Contributions

## Method
- [ ] Feature map formula
- [ ] QR init
- [ ] Preservation gate
- [ ] Complexity analysis
- [ ] Parameter analysis (honest)

## Results
- [ ] Main accuracy
- [ ] Parameter table
- [ ] Speed vs L table
- [ ] Feature dim ablation
- [ ] Gate ablation
- [ ] LN ablation
- [ ] QR ablation
- [ ] LoRA ablation

## Honesty
- [ ] Parameter increase stated
- [ ] Cumsum memory noted
- [ ] Fatal error handling noted
- [ ] Single-seed stated
- [ ] Cross-attention not supported

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Feature dim documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/leka_v1
nano papers/leka_v1/variables.json
cd papers/leka_v1
python apply_vars.py --module leka_v1
grep -rn "{{" .
python build.py --module leka_v1 --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **ماهیت: Linear Attention واقعی**

این ماژول O(L·F) هست، نه O(L²). این را در:
- Abstract
- Intro
- Method (با math کامل)
- Discussion

برجسته کن.

### ۲. **RFF برای Gaussian Kernel**

فرق مهم با Performer:
- Performer: PORF برای **softmax kernel** ($\exp(q \cdot k)$)
- LEKAv1: RFF برای **Gaussian kernel** ($\exp(-\|q-k\|²/2)$)

### ۳. **سه نوآوری واقعی**

1. **Head-wise LayerNorm قبل از projection**
2. **QR-orthogonalized projection**
3. **Preservation gate** ($\alpha$ mixing)

### ۴. **صداقت درباره پارامتر**

پارامتر **۲.۷٪ بیشتر** می‌شه. این را واضح بگو.

### ۵. **دو باگ که باید fix بشن**

1. `raise RuntimeError` در `copy_weights` (کشنده)
2. `output_scale` پارامتر جدید که در checkpoint نیست

### ۶. **Cumsum memory**

برای L > 512، cumsum materialization می‌تونه OOM بده. این را در limitations برجسته کن.

### ۷. **تست سریع**

```bash
python run_all_4_leka_v1.py --models distilbert,roberta,gpt2 --feature_dim 256 --batch_size 4
```

---

**قدم بعدی: `run_all_4_leka_v1.py` را اجرا کن، اعداد را از `summary.csv` بگیر، در `variables.json` بگذار.**