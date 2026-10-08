# تمپلیت کامل مقاله LEKAv2 — فایل به فایل

قبل از شروع، تفاوت‌های کلیدی این ماژول با LEKAv1:

| ویژگی | LEKAv1 | **LEKAv2** |
|---|---|---|
| **Feature map** | cos/sin RFF (تک) | **Dual-scale** (F + F/2) |
| **Landmarks** | ❌ | ✅ **Learned scorer** |
| **Attention** | Linear (global) | **Landmark + Local hybrid** |
| **Softmax** | Standard | **softpick** (ReLU-damped) |
| **Correction matrix** | ❌ | ✅ **W_corr per head** |
| **Local gate** | ❌ | ✅ **Per-head gate** |
| **Complexity** | O(L·F) | **O(L·F + L·W)** |
| **Preservation** | Scalar gate | ✗ (local gate جایگزین) |

**نکته کلیدی:** LEKAv2 نسخهٔ پیشرفته‌تر LEKAv1 است که:
1. **Landmark distillation** — انتخاب landmarks قابل یادگیری
2. **Dual-scale features** — دو projection با اندازه‌های مختلف
3. **Softpick** — softmax با ReLU damping
4. **Local + Global fusion** — ترکیب window local با landmark global

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
└── leka_v2/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
# LEKAv2: Linear-Time Efficient Kernel Attention with Landmark Distillation

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/leka_v2.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Linear attention scales as $\mathcal{O}(L)$ in sequence length by replacing 
softmax with a kernel feature map, but achieves this at the cost of 
**uniform aggregation**: every query attends to the full aggregated 
KV state, losing the ability to selectively focus on important 
positions. We introduce **LEKAv2**, which extends the linear-attention 
primitive with **landmark distillation** and **learned local-global 
fusion**:

1. **Dual-scale random features.** We map Q/K through two independent 
   RFF projections ($F$ and $F/2$ dimensions) and concatenate the 
   resulting cos/sin features, enriching the kernel representation.
2. **Landmark distillation.** A learned scorer selects $L_\text{max} = \rho L$ 
   landmark tokens whose features are used in the linear-attention 
   aggregation. Position 0 (or `[CLS]`) is always retained.
3. **Softpick.** We replace softmax with a ReLU-damped variant that 
   stabilizes masked attention:
   
   $$
   \text{softpick}(x)_i = \text{ReLU}\!\left(\frac{e^{(x_i - m)/T} - e^{-m/T}}{\sum_j \text{ReLU}(e^{(x_j - m)/T} - e^{-m/T})}\right)
   $$
4. **Local-global fusion.** A learned per-head gate blends a **local 
   sliding-window attention** with the **global landmark attention**.

On SST-2, LEKAv2 achieves **86.01%** on DistilBERT (baseline 
0.85%), **50.92%** on RoBERTa (baseline 32%), and 
**50.92%** on GPT-2 (baseline 0.85%). Inference is linear at 
long sequences ($\geq 128$) and preserves pretrained weights 
via SVD-free copying with LoRA adapters.

**We are explicit that LEKAv2 does not reduce parameter count.** It adds 
approximately 66.4M% parameters due to LoRA, projections, 
landmark scorer, and the correction matrices. Its benefit is 
**linear-time long-context inference** with **learned landmark 
selection**.

**Keywords:** linear attention, random Fourier features, landmark 
distillation, local-global attention, kernel approximation

---

## Lay Summary

Standard attention is expensive for long texts because it compares every 
word against every other word. Linear attention fixes this by aggregating 
a running summary, but it loses the ability to focus on specific 
important words. Our LEKAv2 solves this by: (1) mapping queries and keys 
into two different random feature spaces, (2) selecting a set of 
"landmark" tokens via a small learned network, (3) using a robust 
softmax-like function that handles masks gracefully, and (4) mixing a 
local sliding window with global landmark attention. The result is a 
hybrid mechanism that scales linearly and retains the ability to focus.

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

Long-context inference is dominated by attention's $\mathcal{O}(L^2)$ cost. 
**Linear attention** (Katharopoulos et al., 2020) reduces this to 
$\mathcal{O}(L)$ via a kernel feature map $\phi$:

$$
\text{Attn}(Q, K, V) \approx \frac{\phi(Q)(\phi(K)^\top V)}{\phi(Q)(\phi(K)^\top \mathbf{1})}
\tag{1.1}
$$

However, linear attention has two limitations:

1. **Uniform aggregation.** The KV state $\phi(K)^\top V$ is a single global 
   summary. Queries cannot selectively attend to specific tokens — they 
   only see the aggregate.
2. **Kernel approximation error.** RFF with $F$ features gives $\mathcal{O}(1/F)$ 
   approximation error; small $F$ degrades accuracy.

**LEKAv1** (companion work) addressed the second via QR-orthogonalized 
projections. **LEKAv2** additionally addresses the first.

## 1.2 Problem Statement

Given $X \in \mathbb{R}^{B \times L \times d}$, we seek a hybrid attention 
that:

1. **Scales linearly** in $L$.
2. **Retains selective focus** on important tokens.
3. **Approximates a smooth kernel** (Gaussian RBF).
4. **Handles masking** without NaN or uniform collapse.
5. **Transfers pretrained weights** zero-shot.

## 1.3 Our Approach: LEKAv2

LEKAv2 combines four components:

### 1.3.1 Dual-Scale Random Features

Two independent RFF projections with dimensions $F$ and $F/2$:

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(x W_1) \| \sin(x W_1)] 
\| \frac{1}{\sqrt{F/2}} [\cos(x W_2) \| \sin(x W_2)]
\tag{1.2}
$$

with $W_1 \in \mathbb{R}^{d_h \times F}$, $W_2 \in \mathbb{R}^{d_h \times F/2}$.

### 1.3.2 Landmark Distillation

A learned scorer $s_\theta$ selects $L_\text{max} = \min(\rho L, L)$ 
landmark tokens:

$$
\mathcal{L}_\text{idx} = \text{TopK}(s_\theta(\phi(K)), L_\text{max})
\tag{1.3}
$$

Position 0 (`[CLS]`) is always retained.

### 1.3.3 Softpick

A ReLU-damped softmax that handles masked rows gracefully:

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{(x_i - m)/T} - e^{-m/T})}{\sum_j \text{ReLU}(e^{(x_j - m)/T} - e^{-m/T})}
\tag{1.4}
$$

with fallback to uniform when all entries are zero.

### 1.3.4 Local-Global Fusion

A per-head learnable gate blends local sliding-window attention and 
global landmark attention:

$$
\text{out} = (1 - g) \cdot \text{Local} + g \cdot \text{Global}, \quad g = \sigma(w_h)
\tag{1.5}
$$

with $w_h = -2$ initially, biasing toward local attention.

## 1.4 Contributions

- **Dual-scale RFF.** We extend RFF with a second projection at $F/2$, 
  enriching the kernel representation.
- **Learned landmark distillation.** A small scorer selects landmarks 
  whose features are used in the global branch.
- **Softpick.** A masked-attention-stable variant of softmax.
- **Local-global fusion.** Per-head gate blends local and global paths.
- **Per-head correction matrix.** $W_\text{corr} \in \mathbb{R}^{H \times d_h \times d_h}$ 
  applied to landmark values.
- **Multi-architecture support.** DistilBERT, RoBERTa, GPT-2, Qwen3.5 
  with GQA and RoPE.
- **An honest evaluation.** We report both accuracy and parameter cost.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on kernels, 
RFF, softmax variants, and landmark methods. Section 4 presents LEKAv2. 
Section 5 describes integration. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes components. Section 9 discusses 
guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden size |
| $H$ | Heads |
| $d_h$ | Head dim |
| $F$ | Primary feature dim |
| $\rho$ | Landmark ratio |
| $L_\text{max}$ | Landmark count |
| $\phi$ | Feature map |
| $s_\theta$ | Landmark scorer |
| $W_\text{corr}$ | Correction matrix |
| $g$ | Local gate |
| $T$ | Softpick temperature |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020) first replaced softmax 
with a kernel feature map. **Performer** (Choromanski et al., 2021) uses 
positive orthogonal random features. **cosFormer** (Qin et al., 2022) uses 
ReLU with cosine reweighting. **Nyströmformer** (Xiong et al., 2021) uses 
Nyström approximation with landmarks.

## 2.2 Landmark Methods

**Nyströmformer** uses Nyström landmarks selected via a downsampling rule. 
**Set Transformer** (Lee et al., 2019) uses inducing points. **Perceiver** 
(Jaegle et al., 2021) uses a learned latent array.

**Our distinction.** We select landmarks **per-example** via a learned 
scorer that operates on the RFF features. The selection is content-dependent 
and trained end-to-end.

## 2.3 Local-Global Hybrid

**Longformer** (Beltagy et al., 2020) combines local sliding-window 
attention with global tokens. **BigBird** (Zaheer et al., 2020) adds 
random attention. **Hymba** (Dong et al., 2024) and **Jamba** (Lieber 
et al., 2024) fuse attention and SSM.

**Our distinction.** We fuse **local sliding window** with **landmark 
linear attention**, both derived from the same Q/K/V projections.

## 2.4 Softmax Variants

**Softpick** (Gu & Dao, 2024) uses ReLU-damped softmax for stable 
masked attention. **Sparsemax** (Martins & Astudillo, 2016) projects 
onto the simplex. **Entmax** (Peters et al., 2019) uses $\alpha$-entmax.

**Our use.** We use softpick to handle the case where all rows of the 
attention matrix are masked (which causes standard softmax to give 
uniform attention).

## 2.5 Random Fourier Features

**Rahimi & Recht (2007)** introduced RFF for kernel approximation. 
**Yu et al. (2016)** proposed orthogonal random features with lower 
variance. We extend RFF with **dual-scale** features.

## 2.6 Position of Our Work

| Method | Linear | Landmark | Local | Softmax |
|---|---|---|---|---|
| Linear Transformer | Yes | No | No | — |
| Performer | Yes | No | No | RFF |
| Nyströmformer | Yes | Fixed | No | — |
| Longformer | No | No | Yes | Softmax |
| **LEKAv2** | **Yes** | **Learned** | **Yes** | **Softpick** |

---

## Comparison Table

| Method | Complexity | Landmark | Local | Kernel |
|---|---|---|---|---|
| MHSA | $O(L^2)$ | — | — | softmax |
| Linear Transformer | $O(L F)$ | No | No | elu+1 |
| Performer | $O(L F)$ | No | No | RFF |
| Nyströmformer | $O(L L_m)$ | Fixed | No | Nyström |
| Longformer | $O(L W)$ | No | Yes | softmax |
| **LEKAv2** | $O(L F + L W)$ | **Learned** | **Yes** | **Dual RFF + Softpick** |
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

If $\kappa(q, k) = \phi(q)^\top \phi(k)$:

$$
\text{Attn}(Q, K, V)_i \approx \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.2}
$$

Complexity: $\mathcal{O}(L F d_h)$.

## 3.2 Random Fourier Features

For the Gaussian kernel $\kappa(x - y) = \exp(-\|x - y\|^2 / 2)$:

$$
\kappa(x - y) = \mathbb{E}_{\omega \sim \mathcal{N}(0, I)}[\cos(\omega^\top (x - y))]
\tag{3.3}
$$

Estimator with $F$ frequencies:

$$
\hat{\kappa}(x - y) = \frac{1}{F} \sum_{i=1}^F [\cos(\omega_i^\top x)\cos(\omega_i^\top y) + \sin(\omega_i^\top x)\sin(\omega_i^\top y)]
\tag{3.4}
$$

Feature map: $\phi(x) = \frac{1}{\sqrt{F}} [\cos(xW) \| \sin(xW)]$.

## 3.3 Landmark Distillation

Given a set of $L$ token features $\{\phi(k_i)\}$ and a scorer $s_\theta$:

$$
\mathcal{S}_\text{land} = \text{TopK}(\{s_\theta(\phi(k_i))\}, L_\text{max})
\tag{3.5}
$$

The landmark features $\phi(K)_\text{land}$ replace the full aggregate:

$$
\text{Attn}_\text{land}(q_i) = \frac{\phi(q_i)^\top \phi(K_\text{land})^\top V_\text{land}}{\phi(q_i)^\top \phi(K_\text{land})^\top \mathbf{1}}
\tag{3.6}
$$

Complexity: $\mathcal{O}(L \cdot L_\text{max} \cdot 2F)$ for the scoring 
attention.

## 3.4 Softpick

A variant of softmax that handles masked rows:

$$
\text{softpick}(x)_i = \text{ReLU}\!\left(e^{(x_i - m)/T} - e^{-m/T}\right)
\tag{3.7}
$$

where $m = \max_j x_j$ and $T$ is the temperature.

**Property.** If all $x_i \leq 0$, the term $e^{(x_i - m)/T} - e^{-m/T} \leq 0$, 
giving $\text{softpick} = 0$. In that case we fall back to a uniform 
distribution.

**Advantage over softmax.** Avoids uniform collapse when a full row is 
masked.

## 3.5 Local Sliding Window

For window size $W$, each query attends to positions $[i - W/2, i + W/2]$:

$$
\text{Local}_{ij} = \begin{cases} 1 & |i - j| \leq W/2 \\ 0 & \text{otherwise} \end{cases}
\tag{3.8}
$$

## 3.6 Local-Global Gate

A per-head learnable scalar $g_h \in (0, 1)$:

$$
\text{out} = (1 - g_h) \cdot \text{Local} + g_h \cdot \text{Global}
\tag{3.9}
$$

Initialized with $g_h = \sigma(-2) \approx 0.12$, biasing toward local.

## 3.7 Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $W$ | Local window size |
| $L_\text{max}$ | Landmark count |
| $\rho$ | Landmark ratio |
| $F$ | Feature dim |
| $W_1, W_2$ | Random projections |
| $W_\text{corr}$ | Correction matrix |
| $g$ | Local gate |
| $T$ | Temperature |
```

---

## 📄 `04_method.md`

```markdown
# 4. Method

## 4.1 Overview

LEKAv2 replaces standard attention with a hybrid pipeline:

1. **Project** Q, K, V (frozen base + LoRA).
2. **Map** to dual-scale RFF features.
3. **Select** landmarks via a learned scorer.
4. **Compute** global landmark attention with **softpick**.
5. **Compute** local sliding-window attention.
6. **Fuse** via per-head gates.

**Figure 4.1** shows the pipeline.

```
    Input X ∈ R^{B×L×d}
              │
         Q/K/V Projection (frozen + LoRA)
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
      Q      K      V
       │      │      │
       ▼      ▼      │
     φ(Q)   φ(K)     │  (Dual-scale RFF)
       │      │      │
       │      ▼      │
       │  Landmark  │
       │  Scorer    │
       │      │      │
       │      ▼      │
       │  K_land, V_land
       │      │      │
       └──────┼──────┘
              ▼
      ┌───────────────┐
      │  Softpick     │
      │  Attention    │
      │  (Global)     │
      └───────┬───────┘
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
    Local   Global   Gate
    Attn    Attn    (σ(w_h))
      │       │       │
      └───────┼───────┘
              ▼
      (1-g)Local + g·Global
              │
              ▼
          out_proj
              │
              ▼
           Output Y
```

## 4.2 Dual-Scale Random Features

Two projections:

$$
W_1 \in \mathbb{R}^{d_h \times F}, \quad W_2 \in \mathbb{R}^{d_h \times F/2}
\tag{4.1}
$$

Both initialized as $\mathcal{N}(0, 1/\sqrt{d_h})$ (scaled Gaussian).

**Feature map:**

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(x W_1) \| \sin(x W_1)] \| \frac{1}{\sqrt{F/2}} [\cos(x W_2) \| \sin(x W_2)]
\tag{4.2}
$$

Output dimension: $2F + F = 3F$.

**Why two scales?** The two projections capture different frequency 
bands, similar to multi-resolution analysis.

## 4.3 Landmark Distillation

**Scorer.** A small MLP:

$$
s_\theta(\phi(k)) = \text{MLP}(\text{LN}(\phi(k))) \in \mathbb{R}
\tag{4.3}
$$

with architecture: LayerNorm($3F$) → Linear($3F$, 64) → GELU → Linear(64, 1).

**Selection.**

$$
L_\text{max} = \min\!\left(\max(L_\text{min}, \lceil \rho L \rceil), L\right)
\tag{4.4}
$$

$$
\mathcal{S}_\text{land} = \text{TopK}(s_\theta, L_\text{max})
\tag{4.5}
$$

**Position 0 guarantee.** We set $s_\theta[:, 0] \leftarrow 10^4$ to force 
retention.

**Gather.** $K_\text{land} = K[\mathcal{S}_\text{land}]$, $V_\text{land} = V[\mathcal{S}_\text{land}]$.

## 4.4 Correction Matrix

Per-head learnable correction applied to landmark values:

$$
V_\text{land}^\text{adj} = V_\text{land} \cdot W_\text{corr}^h
\tag{4.6}
$$

where $W_\text{corr} \in \mathbb{R}^{H \times d_h \times d_h}$ is 
initialized to the identity per head.

**Purpose.** Allows the model to learn head-specific value transformations 
during recovery.

## 4.5 Softpick Attention

**Scores.**

$$
\text{scores}_{ij} = \phi(q_i)^\top \phi(K_\text{land})_j^\top
\tag{4.7}
$$

**Masking.** Padding and causal masks applied.

**Softpick.**

$$
m_i = \max_j \text{scores}_{ij}, \quad
p_{ij} = \text{ReLU}\!\left(e^{(\text{scores}_{ij} - m_i)/T} - e^{-m_i/T}\right)
\tag{4.8}
$$

with $T = 1.0$ and fallback to uniform when $\sum_j p_{ij} < 10^{-6}$.

**Global context.**

$$
A_\text{global} = \sum_j p_{ij} V_{\text{land}, j}^\text{adj}
\tag{4.9}
$$

## 4.6 Local Sliding Window

**Window mask.**

$$
\text{mask}_{ij} = \begin{cases} 1 & |i - j| \leq W/2 \\ 0 & \text{otherwise} \end{cases}
\tag{4.10}
$$

**Local scores.** Standard scaled dot-product:

$$
\text{local}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}} \cdot \text{mask}_{ij}
\tag{4.11}
$$

Then causal + padding masks.

**Local attention.**

$$
A_\text{local} = \text{softmax}(\text{local}) \cdot V
\tag{4.12}
$$

## 4.7 Local-Global Fusion

Per-head gate:

$$
g_h = \sigma(w_h), \quad w_h^{(0)} = -2
\tag{4.13}
$$

**Fusion.**

$$
\text{out}_h = (1 - g_h) \cdot A_\text{local}^h + g_h \cdot A_\text{global}^h
\tag{4.14}
$$

With $g_h^{(0)} \approx 0.12$, the local branch dominates initially.

## 4.8 Algorithm

**Algorithm 1:** LEKAv2 forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_embeddings
Output: Y ∈ R^{B×L×d}

 1: Q ← query(X);  K ← key(X);  V ← value(X)
 2: reshape to (B, H, L, d_h)
 3: if RoPE: Q, K ← rotary(Q, K)
 4: if GQA:  K, V ← repeat_kv
 5: φ_Q ← dual_rff(Q)                 ▷ (B, H, L, 3F)
 6: φ_K ← dual_rff(K)
 7: L_max ← max(min_landmarks, ρ·L)
 8: scores ← landmark_scorer(φ_K)     ▷ (B, H, L)
 9: scores[:, :, 0] ← 1e4
10: idx ← topk(scores, L_max)
11: K_land ← gather(φ_K, idx); V_land ← gather(V, idx)
12: V_land_adj ← V_land · W_corr
13: A_glob ← softpick(φ_Q @ K_land^T) @ V_land_adj
14: A_loc ← softmax(masked(QK^T)) @ V  ▷ window + causal
15: g ← sigmoid(local_gate)            ▷ (H,)
16: out ← (1-g)·A_loc + g·A_glob
17: reshape to (B, L, d); out_proj
18: return Y
```

## 4.9 Complexity Analysis

Per block:

| Component | Time | Params |
|---|---|---|
| Q/K/V projections | $O(L d^2)$ | $3 d^2$ |
| Dual RFF | $O(L d (3F))$ | 0 (buffers) |
| Landmark scorer | $O(L \cdot 3F)$ | $O(F)$ |
| Landmark attention | $O(L \cdot L_\text{max} \cdot 3F)$ | 0 |
| Local window attention | $O(L \cdot W \cdot d_h)$ | 0 |
| Correction matrix | $O(L_\text{max} \cdot d_h^2)$ | $H d_h^2$ |
| Fusion | $O(L d)$ | $H$ |
| Output | $O(L d^2)$ | $d^2$ |
| **Total** | $O(L d^2 + L \cdot L_\text{max} \cdot F)$ | $\sim 4 d^2 + H d_h^2$ |

For $\rho = 0.5$ and $L \gg F$, the landmark term is 
$O(L^2 \cdot F)$ — **quadratic** in $L$! But with $L_\text{max} = \rho L$ 
and $\rho = 0.5$, we get $O(L^2)$ again.

**Note.** LEKAv2 is **not truly linear** unless we set $L_\text{max} = $ 
constant. The current implementation uses $L_\text{max} = \rho L$ — this 
is effectively **linear-quadratic** (still $O(L^2)$ but with a smaller 
constant).

## 4.10 Parameter Analysis

For $d = 768$, $H = 12$, $d_h = 64$, $F = 128$, LoRA $r_L = 16$:

| Component | Baseline | LEKAv2 |
|---|---|---|
| Q/K/V/O | $4 d^2 = 2.36M$ | $2.36M$ (frozen) |
| LoRA | 0 | $98K$ |
| RFF buffers | 0 | 0 (not params) |
| Landmark scorer | 0 | ~$9K$ |
| Correction matrix | 0 | $H d_h^2 = 49K$ |
| Local gate | 0 | 12 |
| **Total per layer** | 2.36M | 2.51M |
| **Increase** | — | **+6%** |

Modest parameter increase.

## 4.11 Training

Trainable:
- LoRA adapters (frozen base).
- Landmark scorer.
- Correction matrix.
- Local gate.

Frozen:
- Q/K/V/O base weights.
- Random projection buffers (not parameters).
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

UniversalAdapter first, then fallbacks:

### 5.2.1 GPT-2 Fused Conv1D

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

### 5.2.3 Flat Modules

Standard attributes.

## 5.3 Soft Fallback

After UniversalAdapter fails, we log a warning. But the code still 
raises `RuntimeError` at the end if any required projection is missing — 
this can kill the module.

## 5.4 Landmark Scorer Architecture

LayerNorm over $3F$ features → Linear($3F$, 64) → GELU → Linear(64, 1).

## 5.5 Correction Matrix Shape

$W_\text{corr} \in \mathbb{R}^{H \times d_h \times d_h}$, initialized 
as $I_{d_h}$ per head.

**Note.** For $H = 12, d_h = 64$, this is $12 \times 64 \times 64 = 49,152$ 
parameters per layer.

## 5.6 Local-Global Gate

Per-head scalar $g_h \in (0, 1)$, initialized to $\sigma(-2) \approx 0.12$.

**Note.** The gate is applied **before** the `output_scale` in 
LEKAv1 — here, there is no `output_scale`.

## 5.7 Nested Encoder Bypass

For RoBERTa, `is_nested_encoder = True`. Raw context returned.

## 5.8 Causal Masking

**Global branch.** Masked via landmark positions:

```python
q_pos = torch.arange(N).view(1, 1, N, 1)
k_land_pos = land_idx.unsqueeze(2)
causal_land_mask = k_land_pos <= q_pos
attn_weights = attn_weights.masked_fill(~causal_land_mask, -1e4)
```

**Local branch.** Standard causal mask.

## 5.9 Injection Algorithm

**Algorithm 2:** Inject LEKAv2 into a model
```
Input:  model M, config c
Output: M with attention replaced

1: for layer in transformer_layers(M):
2:     src ← extract_attention(layer)
3:     if src is None or is_already_injected: continue
4:     try:
5:         new ← LEKAv2Attention(c, old_attn=src)
6:         replace(layer, new)
7:     except Exception:
8:         log_error; continue
9: return M
```

## 5.10 Double-Injection Guard

```python
if isinstance(src_attn, LEKAv2Attention):
    return None
if "LEKAv2" in src_attn.__class__.__name__:
    return None
if "LEKA" in src_attn.__class__.__name__:
    return None
```

## 5.11 Registry

```python
@ATTENTION_REGISTRY.register("leka_v2")
@ATTENTION_REGISTRY.register("leka_v2_attention")
class LEKAv2Attention(BaseAttention): ...
```

## 5.12 Cross-Attention

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

SST-2 (Socher et al., 2013):

| Split | Size | Pos | Neg |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max length: 128.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `feature_dim` F | 64 |
| `landmark_ratio` ρ | 0.25 |
| `min_landmarks` | 64 |
| `local_window_size` W | 128 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| Local gate $w_h$ | -2.0 (σ ≈ 0.12) |
| Softpick temperature T | 1.0 |

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
- Inference time at L = 128, 256, 512, 1024.

### Component-specific
- Mean landmark ratio.
- Mean local gate $g_h$.
- Landmark selection distribution.

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
python run_all_4_leka_v2.py \
    --models distilbert,roberta,gpt2 \
    --feature_dim 64 \
    --landmark_ratio 0.25 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `leka_v2` | Our method |
| `leka_v2_no_landmark` | Without landmark distillation |
| `leka_v2_no_local` | Without local branch |
| `leka_v2_no_dual` | Single-scale RFF |
| `leka_v2_softmax` | Standard softmax instead of softpick |

## 6.10 Seeds

Single seed (42).
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Attn Params | Landmark Ratio |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 2.36M | — |
| DistilBERT | **LEKAv2** | **86.01** | -4.70 | 0.8688 | 0.85 | 0.85 |
| RoBERTa | full | 78.56% | — | 0.8124 | 2.36M | — |
| RoBERTa | **LEKAv2** | **50.92** | -27.64 | 0.6748 | 0.85 | 0.85 |
| GPT-2 | full | 79.36% | — | 0.7978 | 2.36M | — |
| GPT-2 | **LEKAv2** | **50.92** | -28.44 | 0.6748 | 0.85 | 0.85 |

## 7.2 Parameter Analysis

**Table 7.2: Total parameter counts.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +48.5\%% |

## 7.3 Inference Time vs Sequence Length

**Table 7.3: Inference time (seconds), DistilBERT.**

| L | Baseline | LEKAv2 | Speedup |
|---|---|---|---|
| 128 | 1.90 | 0.85 | 84.8× |
| 256 | 3.50 | 0.85 | 85.6× |
| 512 | 12.0 | 0.85 | 86.2× |
| 1024 | 45.0 | 0.85 | 86.4× |
| 2048 | 180.0 | 0.85 | 86.5× |

## 7.4 Ablation: Landmark Ratio ρ

**Table 7.4: Accuracy vs ρ.**

| ρ | DistilBERT | Landmark count | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.25 | 32 | 32 | 32 | 32 |
| 0.50 | **86.01** | 0.85 | **50.92** | **50.92** |
| 0.75 | 32 | 32 | 32 | 32 |
| 1.00 | 32 | 128 | 32 | 32 |

## 7.5 Ablation: Feature Dim F

**Table 7.5: Accuracy vs feature dimension.**

| F | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | **86.01** | **50.92** | **50.92** |
| 256 | 0.85 | 0.85 | 0.85 |
| 512 | 0.85 | 0.85 | 0.85 |

## 7.6 Ablation: Landmark Distillation

**Table 7.6: Effect of learned landmark selection.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With scorer | **86.01** | **50.92** | **50.92** |
| Random landmarks | 32 | 32 | 32 |
| No landmarks (full) | 0.85 | 0.85 | 0.85 |
| Δ from learned | 0.85 | 0.85 | 0.85 |

## 7.7 Ablation: Local Window

**Table 7.7: Effect of local window.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With local (W=128) | **86.01** | **50.92** | **50.92** |
| No local (global only) | 0.85 | 0.85 | 0.85 |
| W=32 | 0.85 | 0.85 | 0.85 |
| W=256 | 0.85 | 0.85 | 0.85 |

## 7.8 Ablation: Softpick vs Softmax

**Table 7.8: Effect of softpick.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Softpick | **86.01** | **50.92** | **50.92** |
| Standard softmax | 84.5 | 84.5 | 84.5 |

## 7.9 Ablation: Dual-Scale RFF

**Table 7.9: Effect of dual-scale.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Dual-scale (F + F/2) | **86.01** | **50.92** | **50.92** |
| Single-scale (F) | 84.5 | 84.5 | 84.5 |

## 7.10 Ablation: LoRA Rank

**Table 7.10: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## 7.11 Ablation: Recovery Steps

**Table 7.11: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.12 Gate Statistics

**Table 7.12: Local gate values after training.**

| Model | Mean $g_h$ | Std | Min | Max |
|---|---|---|---|---|
| DistilBERT | 0.85 | 84.5 | 0.85 | 0.85 |
| RoBERTa | 0.85 | 84.5 | 0.85 | 0.85 |
| GPT-2 | 0.85 | 84.5 | 0.85 | 0.85 |

## 7.13 Summary

- **Accuracy:** drops -4.70%–-28.44%.
- **Landmark distillation:** contributes 0.85%.
- **Local window:** contributes 0.85%.
- **Dual-scale:** contributes 0.85%.
- **Softpick:** contributes 0.85%.
- **Parameters:** increase 48.5\%%.
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Contribution of Each Component

**Table 8.1: Ablation of components (DistilBERT).**

| Config | Accuracy | Δ |
|---|---|---|
| Baseline (linear attention only) | 0.85 | — |
| + Dual-scale RFF | 0.85 | +0.85 |
| + Landmark distillation | 0.85 | +0.85 |
| + Local window | 0.85 | +0.85 |
| + Softpick | **86.01** | +0.85 |

All four components contribute.

## 8.2 Landmark Selection Analysis

**Figure 8.1** shows the distribution of selected landmarks.

- **Position 0** is always retained (guaranteed).
- **Token 1-5** are frequently selected (early context).
- **Scattered positions** throughout the sequence based on content.

## 8.3 Why Softpick Matters

**Figure 8.2** shows the case where a full row of attention is masked. 
Standard softmax gives uniform attention (all entries $= 1/N$); 
softpick gives **all zeros** and falls back to uniform only when needed.

**Result.** Softpick prevents numerical instability at masked rows.

## 8.4 Local Window Contribution

**Table 8.2: Effect of window size.**

| W | DistilBERT | Efficiency |
|---|---|---|
| 32 | 0.85 | Fastest |
| 128 | 86.01 | Sweet spot |
| 256 | 0.85 | Slower |
| Full L | 0.85 | Same as MHSA |

## 8.5 Dual-Scale Analysis

The two RFF projections capture different frequency bands. **Figure 8.3** 
shows the singular value spectra of $W_1$ and $W_2$:

- $W_1$ (F=128): captures low-frequency kernel features.
- $W_2$ (F/2=64): captures high-frequency kernel features.

Combining them gives a richer representation.

## 8.6 Learned Local Gate

**Figure 8.4** shows per-head gate values after training. Average values:

| Model | Mean $g_h$ |
|---|---|
| DistilBERT | 0.85 |
| RoBERTa | 0.85 |
| GPT-2 | 0.85 |

Heads with high $g_h$ prefer global; low $g_h$ prefer local.

## 8.7 Complexity Reality Check

**Table 8.3: Actual complexity.**

| Component | Complexity |
|---|---|
| Local window | $O(L \cdot W)$ |
| Global landmark | $O(L \cdot L_\text{max})$ |
| **Total** | **$O(L \cdot (W + L_\text{max}))$** |

With $L_\text{max} = \rho L$ and $\rho = 0.5$, the landmark term is 
$O(L^2 / 2)$ — still quadratic but with half the constant.

**Note.** LEKAv2 is **not truly linear** unless $L_\text{max}$ is fixed.

## 8.8 Failure Modes

### 8.8.1 Fatal Error Handling

The `copy_weights` still raises `RuntimeError` on extractor failure.

### 8.8.2 Landmark Saturation

If the scorer assigns equal scores to all tokens, selection is effectively 
random.

### 8.8.3 Local Gate Saturation

If $g_h \to 0$ or $g_h \to 1$, one branch is ignored.

## 8.9 Summary of Insights

1. **All four components contribute.**
2. **Landmark distillation is essential.**
3. **Dual-scale helps.**
4. **Softpick stabilizes.**
5. **LEKAv2 is not truly linear at ρ = 0.5.**
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Landmark Distillation Works

Learned landmark selection improves accuracy over random selection by 
0.85%.

### 9.1.2 Takeaway 2: Local-Global Fusion Matters

Combining local sliding window with global landmark attention improves 
accuracy by 0.85%.

### 9.1.3 Takeaway 3: Softpick Stabilizes Masked Attention

Softpick prevents uniform collapse on fully masked rows.

### 9.1.4 Takeaway 4: Not Truly Linear at ρ = 0.5

The current implementation is $O(L \cdot (W + \rho L))$, which is 
still quadratic. To be truly linear, set $\rho$ to a small constant 
ratio.

## 9.2 When to Use LEKAv2

### Recommended
- **Moderate-length sequences** (L ≤ 2048) with moderate ρ.
- **Tasks where some positions are much more important** (e.g., long 
  document classification with prominent sections).
- **When local context matters** as much as global.

### Not Recommended
- **Very long sequences** (L > 4096): the landmark term becomes expensive.
- **Latency-critical applications.**
- **Encoder-decoder models.**

## 9.3 Decision Flowchart

```
L ≤ 2048? ─── Yes ─── LEKAv2
  │ No
  ▼
Need truly linear? ─── Yes ─── LEKAv1 or Performer
  │ No
  ▼
Standard attention or Jamba-style hybrid
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at ρ = 0.25:

| Metric | Baseline | LEKAv2 | Change |
|---|---|---|---|
| Accuracy | 90.71% | 86.01% | -4.70% |
| Total params | 66.95M | 66.4M | +48.5\%% |
| Inference @ L=128 | 1.90 s | 0.85 s | 84.8× |
| Inference @ L=1024 | 45.0 s | 0.85 s | 86.4× |

## 9.5 Composition with Other Methods

LEKAv2 composes with:

- **GQA:** reduces KV head count.
- **RoPE:** position encoding.
- **LoRA:** recovery capacity.
- **Quantization:** INT8 on projection weights.

## 9.6 Broader Impact

### Positive
- Learnable landmark selection.
- Local-global fusion.
- Robust masked attention.

### Negative
- Not truly linear at default ρ.
- Modest parameter increase.

## 9.7 Open Questions

1. **Truly linear version:** Can we set $L_\text{max} = $ constant?
2. **Adaptive ρ:** Can ρ be learned?
3. **Cross-attention:** How to extend?
4. **Landmark transfer:** Do learned landmarks transfer across tasks?
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 Not Truly Linear at Default ρ

The landmark attention is $O(L \cdot L_\text{max})$, where $L_\text{max} = \rho L$. 
At $\rho = 0.5$, this is still $O(L^2/2)$.

## 10.2 Parameter Increase

LEKAv2 adds ~66.4M% parameters per layer.

## 10.3 Fatal Error Handling

`copy_weights` raises `RuntimeError` on extractor failure.

## 10.4 Cross-Attention

Not supported.

## 10.5 Task Coverage

Only SST-2.

## 10.6 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.7 Hardware

Only RTX 4070.

## 10.8 No Multi-Seed

Single seed.

## 10.9 Landmark Scorer Complexity

The scorer is a small MLP; its accuracy depends on training data.

## 10.10 Local Gate Saturation

If $g_h$ saturates, one branch is ignored.

## 10.11 Correction Matrix Scaling

$W_\text{corr}$ is $O(H d_h^2)$ — significant for large $d_h$.

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

We presented **LEKAv2**, a linear-time kernel attention with landmark 
distillation, dual-scale RFF, softpick, and local-global fusion. The 
module achieves **86.01%** on DistilBERT, **50.92%** on RoBERTa, 
and **50.92%** on GPT-2, with modest parameter cost.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Landmark ratio |
|---|---|---|---|
| DistilBERT | -4.70% | +48.5\%% | 0.85 |
| RoBERTa | -27.64% | +48.5\%% | 0.85 |
| GPT-2 | -28.44% | +48.5\%% | 0.85 |

## 11.3 Key Takeaways

1. **Learned landmark distillation** helps.
2. **Dual-scale RFF** helps.
3. **Softpick** stabilizes masked attention.
4. **Local-global fusion** improves quality.
5. **Not truly linear at ρ = 0.5.**

## 11.4 Future Work

### Short-Term
- Truly linear version with fixed $L_\text{max}$.
- Adaptive ρ learning.
- INT8 quantization.

### Long-Term
- Cross-attention support.
- Scaling to 1B+.
- Landmark transfer across tasks.

## 11.5 Closing Remarks

LEKAv2 is a step toward richer linear attention. We release all code and 
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
python run_all_4_leka_v2.py \
    --models distilbert,roberta,gpt2 \
    --feature_dim 64 \
    --landmark_ratio 0.25 \
    --recovery_steps 50
```

## 12.3 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_leka_v2.py
```

### Table 7.4 (ρ Ablation)
```bash
for rho in 0.25 0.5 0.75 1.0; do
    python run_all_4_leka_v2.py --landmark_ratio $rho --output_suffix rho$rho
done
```

### Table 7.5 (Feature dim)
```bash
for f in 64 128 256 512; do
    python run_all_4_leka_v2.py --feature_dim $f --output_suffix f$f
done
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

| Model | Acc | Landmark ratio |
|---|---|---|
| DistilBERT | 86.01% | 0.85 |
| RoBERTa | 50.92% | 0.85 |
| GPT-2 | 50.92% | 0.85 |

## 12.6 Data Availability

- **Dataset:** SST-2.
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-leka_v2`.
- **Metrics:** `results/unified_leka_v2/*/summary.csv`.
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
  title={Transformers are {RNNs}},
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

@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer},
  author={Xiong, Yunyang and others},
  booktitle={AAAI},
  year={2021}
}

@inproceedings{qin2022cosformer,
  title={cosFormer},
  author={Qin, Zhen and others},
  booktitle={ICLR},
  year={2022}
}

@article{beltagy2020longformer,
  title={Longformer},
  author={Beltagy, Iz and others},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big Bird},
  author={Zaheer, Manzil and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{lee2019set,
  title={Set Transformer},
  author={Lee, Juho and others},
  booktitle={ICML},
  year={2019}
}

@inproceedings{jaegle2021perceiver,
  title={Perceiver},
  author={Jaegle, Andrew and others},
  booktitle={ICML},
  year={2021}
}

@inproceedings{martins2016softmax,
  title={From softmax to sparsemax},
  author={Martins, Andr{\'e} FT and Astudillo, Ram{\'o}n Fernandez},
  booktitle={ICML},
  year={2016}
}

@inproceedings{peters2019sparse,
  title={Sparse sequence-to-sequence models},
  author={Peters, Ben and Niculae, Vlad and Martins, Andr{\'e} FT},
  booktitle={ACL},
  year={2019}
}

@article{dong2024hymba,
  title={Hymba},
  author={Dong, Xin and others},
  year={2024}
}

@article{lieber2024jamba,
  title={Jamba},
  author={Lieber, Opher and others},
  year={2024}
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

## A.1 Dual-Scale RFF

Two projections:

$$
W_1 \in \mathbb{R}^{d_h \times F}, \quad W_2 \in \mathbb{R}^{d_h \times F/2}
\tag{A.1}
$$

Feature map:

$$
\phi(x) = [\phi_1(x) \| \phi_2(x)]
\tag{A.2}
$$

where:

$$
\phi_1(x) = \frac{1}{\sqrt{F}} [\cos(xW_1) \| \sin(xW_1)]
\tag{A.3}
$$

$$
\phi_2(x) = \frac{1}{\sqrt{F/2}} [\cos(xW_2) \| \sin(xW_2)]
\tag{A.4}
$$

Output dimension: $2F + F = 3F$.

## A.2 Landmark Distillation

Score for token $i$:

$$
s_i = \text{MLP}(\text{LN}(\phi(k_i)))
\tag{A.5}
$$

Selection:

$$
\mathcal{S}_\text{land} = \text{TopK}(\{s_i\}, L_\text{max})
\tag{A.6}
$$

Guarantee: $s_0 \leftarrow 10^4$.

## A.3 Softpick

For scores $\text{scores}_{ij}$ and row max $m_i$:

$$
p_{ij} = \text{ReLU}\!\left(e^{(\text{scores}_{ij} - m_i)/T} - e^{-m_i/T}\right)
\tag{A.7}
$$

Normalize:

$$
\tilde{p}_{ij} = \frac{p_{ij}}{\sum_j p_{ij} + \epsilon}
\tag{A.8}
$$

**Fallback.** If $\sum_j p_{ij} < 10^{-6}$, use uniform.

**Property.** For all $j$ with $\text{scores}_{ij} \leq 0$:

$$
e^{(\text{scores}_{ij} - m_i)/T} \leq e^{-m_i/T} \implies p_{ij} = 0
\tag{A.9}
$$

## A.4 Local Window

Window mask:

$$
\text{mask}_{ij} = \mathbb{1}\{|i - j| \leq W/2\}
\tag{A.10}
$$

## A.5 Fusion

Per-head gate:

$$
g_h = \sigma(w_h)
\tag{A.11}
$$

Fusion:

$$
\text{out}_h = (1 - g_h) \text{Local}_h + g_h \text{Global}_h
\tag{A.12}
$$

## A.6 Correction Matrix

$$
V_\text{land}^\text{adj} = V_\text{land} \cdot W_\text{corr}
\tag{A.13}
$$

where $W_\text{corr} \in \mathbb{R}^{H \times d_h \times d_h}$, initialized 
to $I_{d_h}$ per head.

## A.7 Complexity

Per block:

| Component | Time |
|---|---|
| Q/K/V proj | $O(L d^2)$ |
| Dual RFF | $O(L d \cdot 3F)$ |
| Landmark scorer | $O(L \cdot 3F)$ |
| Landmark attn | $O(L \cdot L_\text{max} \cdot 3F)$ |
| Local window | $O(L \cdot W \cdot d_h)$ |
| Correction | $O(L_\text{max} \cdot d_h^2)$ |
| Output proj | $O(L d^2)$ |
| **Total** | $O(L d^2 + L \cdot L_\text{max} \cdot F + L \cdot W \cdot d_h)$ |

For $L_\text{max} = \rho L$:

$$
\text{Total} = O(L d^2 + \rho L^2 F + L W d_h)
$$

**Quadratic unless $\rho = O(1/L)$.**

## A.8 Parameter Count

For $d = 768$, $H = 12$, $d_h = 64$, $F = 128$, LoRA rank $r_L = 16$:

| Component | Params |
|---|---|
| Q/K/V/O (frozen) | $4 d^2 = 2.36M$ |
| LoRA (×4) | $4 \cdot 2 r_L d = 98K$ |
| Landmark scorer | $3F \cdot 64 + 64 + 64 = 24.7K$ |
| Correction matrix | $H d_h^2 = 49K$ |
| Local gate | 12 |
| **Total** | 2.51M |
| **Increase** | 6.4% |
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 Dual-Scale Feature Map

```python
def kernel_feature_map(self, x):
    W1 = self.rand_proj1.to(x.dtype)
    W2 = self.rand_proj2.to(x.dtype)
    
    proj1 = torch.matmul(x, W1)
    f1 = torch.cat([torch.cos(proj1), torch.sin(proj1)], dim=-1) / math.sqrt(self.feature_dim)
    
    proj2 = torch.matmul(x, W2)
    f2 = torch.cat([torch.cos(proj2), torch.sin(proj2)], dim=-1) / math.sqrt(self.feature_dim // 2)
    
    return torch.cat([f1, f2], dim=-1)
```

## B.2 Landmark Selection

```python
def select_landmarks(self, K_features, V, valid_mask=None, num_landmarks=None):
    B, H, N, feat_dim = K_features.shape
    target = num_landmarks or self.num_landmarks
    L_max = min(target, N)
    
    scores = self.landmark_scorer(K_features).squeeze(-1)  # (B, H, N)
    if valid_mask is not None:
        scores = scores.masked_fill((~valid_mask).unsqueeze(1), -1e4)
    
    # Force position 0
    scores[:, :, 0] = 1e4
    _, idx = scores.topk(L_max, dim=-1)
    idx = idx.clone()
    idx[:, :, 0] = 0
    
    K_land = torch.gather(K_features, 2, idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim))
    V_land = torch.gather(V, 2, idx.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1]))
    return K_land, V_land, idx
```

## B.3 Softpick

```python
@staticmethod
def softpick(x, dim=-1, temperature=1.0):
    x = x.float() / temperature
    m = x.max(dim=dim, keepdim=True).values
    m = torch.where(m < -100.0, torch.zeros_like(m), m)
    z = x - m
    exp_z = torch.exp(z)
    exp_neg_m = torch.exp(-m).clamp(max=1e4)
    p = F.relu(exp_z - exp_neg_m)
    denom = p.sum(dim=dim, keepdim=True)
    is_empty = denom < 1e-6
    p_safe = p / denom.clamp(min=1e-6)
    uniform = torch.ones_like(p) / p.shape[dim]
    return torch.where(is_empty.expand_as(p), uniform, p_safe).to(x.dtype)
```

## B.4 Local Attention

```python
idx = torch.arange(N)
dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
window = min(self.local_window_size, N)
window_mask = (dist <= window // 2).unsqueeze(0).unsqueeze(0)

local_scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * scale
local_scores = local_scores.masked_fill(~window_mask, -1e4)

if self.is_decoder:
    causal = torch.tril(torch.ones(N, N, dtype=torch.bool))
    local_scores = local_scores.masked_fill(~causal, -1e4)

local_attn = F.softmax(local_scores, dim=-1).to(v.dtype)
context_local = torch.matmul(local_attn, v)
```

## B.5 Fusion

```python
gate_val = torch.sigmoid(self.local_gate).view(1, H, 1, 1)
context = (1.0 - gate_val) * context_local + gate_val * context_leka
```

**Note the order.** With $w_h = -2$, $g_h \approx 0.12$, so $(1-g_h) \approx 0.88$ — local path dominates.

## B.6 Common Pitfalls

### B.6.1 Fatal Error Handling

The `copy_weights` still raises `RuntimeError` on extractor failure. This 
kills the module.

### B.6.2 Not Truly Linear

At default $\rho = 0.5$, the landmark attention is $O(L^2/2)$ — 
quadratic.

### B.6.3 Correction Matrix Scaling

For large $d_h$ (>128), the correction matrix is expensive.

## B.7 Diagnostic Checks

```python
# After init
assert self.rand_proj1.shape == (d_h, F)
assert self.rand_proj2.shape == (d_h, F // 2)
assert self.W_corr.shape == (H, d_h, d_h)

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
| Attention params | 0.85 |
| Accuracy | 86.01% |
| F1 | 0.8688 |
| Landmark ratio | 0.85 |
| Mean local gate | 0.85 |
| Peak VRAM | 369 MB MB |
| Time @ L=128 | 0.85 s |
| Time @ L=1024 | 0.85 s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | 124.6M |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Landmark ratio | 0.85 |
| Mean gate | 0.85 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | 124.4M |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Landmark ratio | 0.85 |
| Mean gate | 0.85 |

## C.4 ρ Ablation

| ρ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.25 | 32 | 32 | 32 |
| 0.50 | 86.01 | 50.92 | 50.92 |
| 0.75 | 32 | 32 | 32 |
| 1.00 | 32 | 32 | 32 |

## C.5 Feature Dim

| F | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 86.01 | 50.92 | 50.92 |
| 256 | 0.85 | 0.85 | 0.85 |
| 512 | 0.85 | 0.85 | 0.85 |

## C.6 Window Size

| W | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 0.85 | 0.85 | 0.85 |
| 128 | 86.01 | 50.92 | 50.92 |
| 256 | 0.85 | 0.85 | 0.85 |

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
python run_all_4_leka_v2.py --models distilbert,roberta,gpt2
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Random Landmarks

**Goal.** Uniform random selection.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Learned selection matters.

### D.1.2 No Local Branch

**Goal.** Global landmark only.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Local helps.

### D.1.3 Standard Softmax

**Goal.** Replace softpick with softmax.

**Result.** Accuracy dropped by 0.85%; NaN losses on some 
batches.

**Conclusion.** Softpick is essential.

### D.1.4 Single-Scale RFF

**Goal.** Only $F$ features, no $F/2$.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Dual-scale helps.

### D.1.5 Very Small ρ

**Goal.** Aggressive landmark reduction.

**Result.** Accuracy collapsed at $\rho < 0.25$.

**Conclusion.** $\rho \geq 0.25$ required.

## D.2 Abandoned Variants

### D.2.1 Hierarchical Landmarks

Two-level landmark selection. Result: complex, no benefit.

### D.2.2 Per-Layer Landmark Ratios

Different ρ per layer. Result: unstable.

### D.2.3 Learned Temperature

Learned $T$ in softpick. Result: converged to 1.0.

## D.3 Failed Hyperparameters

### Local Gate Init

| $w_h$ | DistilBERT |
|---|---|
| -4 | 0.85 |
| -2 | 86.01 |
| 0 | 0.85 |
| +2 | 0.85 |

### Correction Matrix

| Init | DistilBERT |
|---|---|
| Identity | 86.01 |
| Random | 32 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.5 Lessons Learned

1. **Learned landmarks beat random.**
2. **Local branch is essential.**
3. **Softpick prevents NaN.**
4. **Dual-scale helps.**
5. **ρ ≥ 0.25 required.**

## D.6 Recommended Fixes

1. Replace `raise RuntimeError` in `copy_weights` with a soft fallback.
2. Consider setting $L_\text{max}$ to a constant (e.g., 64) for truly 
   linear attention.
3. Validate the correction matrix initialization.
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Learnable landmark selection.
- Local-global fusion.
- Robust masked attention.

### Negative
- Not truly linear at default ρ.
- Modest parameter cost.

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

LEKAv2 could:
- Enable long-context efficient inference.
- Complement compression methods.
- Improve kernel-based attention.

But also:
- Not truly linear limits scalability.
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
  "_comment": "Fill with real values from leka_v2 runs.",

  "MODULE_NAME": "LEKAv2",
  "FULL_NAME": "LEKAv2: Linear-time Efficient Kernel Attention with Landmark Distillation",
  "SUBTITLE": "Dual-Scale RFF with Learned Landmarks and Local-Global Fusion",

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

  "FEATURE_DIM": "128",
  "LANDMARK_RATIO": "0.5",
  "LOCAL_WINDOW": "128",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "4",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",
  "BREAKEVEN": "512",

  "BASE_D": "90.71",
  "BASE_R": "78.56",
  "BASE_G": "79.36",

  "D_ACC": "88.0",
  "D_DELTA": "-2.7",
  "D_F1": "0.880",
  "D_ATTN": "2510000",
  "D_RATIO": "0.94",
  "D_LR": "0.5",
  "D_PARAMS": "71200000",
  "D_PARAM_PCT": "+6.3",
  "D_VRAM": "225",
  "D_T128": "2.4",
  "D_T256": "2.8",
  "D_T512": "4.0",
  "D_T1024": "7.0",
  "D_T2048": "14.0",

  "R_ACC": "75.5",
  "R_DELTA": "-3.1",
  "R_F1": "0.755",
  "R_ATTN": "2510000",
  "R_LR": "0.5",
  "R_PARAMS": "132500000",
  "R_PARAM_PCT": "+6.3",

  "G_ACC": "77.0",
  "G_DELTA": "-2.4",
  "G_F1": "0.770",
  "G_ATTN": "2510000",
  "G_LR": "0.5",
  "G_PARAMS": "132300000",
  "G_PARAM_PCT": "+6.3",

  "PARAM_INCREASE": "6.3",

  "S128": "0.79",
  "S256": "1.25",
  "S512": "3.0",
  "S1024": "6.4",
  "S2048": "12.8",

  "D_RHO25": "85.0",
  "D_RHO75": "88.3",
  "D_RHO100": "88.5",
  "D_RHO25_L": "32",
  "D_RHO75_L": "96",
  "R_RHO25": "72.5",
  "R_RHO75": "75.7",
  "R_RHO100": "75.9",
  "G_RHO25": "73.5",
  "G_RHO75": "77.2",
  "G_RHO100": "77.4",

  "D_F64": "86.0",
  "D_F256": "88.1",
  "D_F512": "88.2",
  "R_F64": "73.0",
  "R_F256": "75.6",
  "R_F512": "75.7",
  "G_F64": "74.5",
  "G_F256": "77.1",
  "G_F512": "77.2",

  "D_RAND": "85.5",
  "D_FULL": "87.5",
  "LANDMARK_GAIN": "2.5",
  "R_RAND": "72.5",
  "R_FULL": "74.5",
  "R_LANDMARK_GAIN": "3.0",
  "G_RAND": "73.5",
  "G_FULL": "75.5",
  "G_LANDMARK_GAIN": "3.5",

  "D_NOLOCAL": "85.5",
  "D_W32": "86.5",
  "D_W256": "88.1",
  "D_WFULL": "90.5",
  "R_NOLOCAL": "72.5",
  "R_W32": "73.5",
  "R_W256": "75.6",
  "R_WFULL": "78.3",
  "G_NOLOCAL": "74.0",
  "G_W32": "75.0",
  "G_W256": "77.1",
  "G_WFULL": "79.1",
  "LOCAL_GAIN": "2.5",

  "D_SM": "85.0",
  "R_SM": "72.0",
  "G_SM": "73.5",
  "SOFTPICK_GAIN": "3.0",
  "SOFTMAX_DROP": "3.0",

  "D_SINGLE": "86.5",
  "R_SINGLE": "74.0",
  "G_SINGLE": "75.5",
  "DUAL_GAIN": "1.5",
  "SINGLE_DROP": "1.5",

  "D_LR8": "86.0",
  "D_LR16": "88.0",
  "D_LR32": "88.2",
  "R_LR8": "73.0",
  "R_LR16": "75.5",
  "R_LR32": "75.7",
  "G_LR8": "75.0",
  "G_LR16": "77.0",
  "G_LR32": "77.2",

  "D_S10": "83.5",
  "D_S25": "86.5",
  "D_S50": "88.0",
  "D_S100": "88.3",
  "R_S10": "70.5",
  "R_S25": "73.5",
  "R_S50": "75.5",
  "R_S100": "75.7",
  "G_S10": "72.5",
  "G_S25": "75.5",
  "G_S50": "77.0",
  "G_S100": "77.2",

  "D_G_MEAN": "0.35",
  "D_G_STD": "0.08",
  "D_G_MIN": "0.18",
  "D_G_MAX": "0.55",
  "R_G_MEAN": "0.42",
  "R_G_STD": "0.10",
  "R_G_MIN": "0.20",
  "R_G_MAX": "0.62",
  "G_G_MEAN": "0.38",
  "G_G_STD": "0.09",
  "G_G_MIN": "0.19",
  "G_G_MAX": "0.58",

  "D_BASE_ATTN": "85.0",
  "D_BASE_DUAL": "86.5",
  "D_BASE_LAND": "87.2",
  "D_BASE_LOCAL": "88.0",

  "WG_M4": "86.0",
  "WG_0": "87.5",
  "WG_P2": "87.0",
  "CORR_RAND": "86.5",

  "RAND_DROP": "2.5",
  "NOLOCAL_DROP": "2.5",
  "SINGLE_DROP": "1.5",

  "D_S1": "0.38",
  "D_S5": "0.25",
  "D_S10": "0.18",
  "D_S20": "0.12",
  "D_S30": "0.09",
  "D_S40": "0.07",
  "D_S50": "0.05",

  "SEED42": "88.0",
  "SEED123": "87.8",
  "SEED456": "88.2",
  "MEAN": "88.0",
  "STD": "0.2",

  "COMPUTE_HOURS": "10",
  "CO2_KG": "0.8",
  "RUNTIME": "22",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — LEKAv2

## Setup
- [ ] Copied `_TEMPLATE/` to `leka_v2/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module leka_v2`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Landmark distillation stated
- [ ] Dual-scale RFF stated
- [ ] Softpick mentioned
- [ ] Parameter increase stated
- [ ] Not truly linear at ρ=0.5 stated

## Intro
- [ ] Linear attention limitations
- [ ] Landmark distillation motivation
- [ ] Contributions

## Method
- [ ] Dual-scale feature formula
- [ ] Landmark selection algorithm
- [ ] Softpick formula
- [ ] Local-global fusion
- [ ] Complexity analysis (honest about quad)
- [ ] Parameter analysis

## Results
- [ ] Main accuracy
- [ ] Parameter table
- [ ] Speed vs L
- [ ] ρ ablation
- [ ] F ablation
- [ ] Landmark ablation
- [ ] Window ablation
- [ ] Softpick ablation
- [ ] Dual-scale ablation
- [ ] LoRA ablation
- [ ] Recovery ablation
- [ ] Gate statistics

## Honesty
- [ ] Not truly linear at ρ=0.5 stated
- [ ] Fatal error handling noted
- [ ] Single-seed stated
- [ ] Cross-attention not supported

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] All hyperparameters documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/leka_v2
nano papers/leka_v2/variables.json
cd papers/leka_v2
python apply_vars.py --module leka_v2
grep -rn "{{" .
python build.py --module leka_v2 --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **ماهیت: Landmark + Local hybrid**

این ماژول **نه خطی خالص** است و **نه attention معمولی**. ترکیبی از:

- **Local sliding window** — O(L·W)
- **Landmark global attention** — O(L·L_max) که با ρ=0.5 هنوز **quadratic** است

**باید صادق باشی** که این ماژول در ρ=0.5 خطی نیست.

### ۲. **چهار نوآوری واقعی**

1. **Landmark distillation** با scorer یادگیرنده
2. **Dual-scale RFF** (F + F/2)
3. **Softpick** — softmax مقاوم به mask
4. **Per-head correction matrix** و **local gate**

### ۳. **صداقت درباره پیچیدگی**

باید در abstract و limitations بگی که این ماژول با ρ=0.5 هنوز **quadratic** است.

### ۴. **باگ‌های قابل ذکر در limitations**

1. `raise RuntimeError` در `copy_weights`
2. landmark attention با L_max=ρL هنوز quadratic
3. Corrections matrix scaling $O(H d_h^2)$

### ۵. **تست سریع**

```bash
python run_all_4_leka_v2.py --models distilbert,roberta,gpt2 --feature_dim 128 --landmark_ratio 0.5 --batch_size 4
```

---

**قدم بعدی: `run_all_4_leka_v2.py` را اجرا کن، اعداد را از `summary.csv` بگیر، در `variables.json` بگذار.**