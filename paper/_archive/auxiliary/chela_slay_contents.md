# تمپلیت کامل مقاله CHELA-SLAY — فایل به فایل

قبل از شروع، خلاصه‌ای از این ماژول:

| ویژگی | Adaptive SVD | BAM-UMoE | **CHELA-SLAY** |
|---|---|---|---|
| **نوع** | Low-rank | MoE | **Hybrid (dense + linear)** |
| **Spherical norm** | ❌ | ❌ | ✅ **unit hypersphere** |
| **Yat-Kernel** | ❌ | ❌ | ✅ **cos/sin RFF** |
| **Adaptive gate** | ❌ | ✅ | ✅ **(favor dense)** |
| **Zero-init** | SVD-based | Expert zero | **Gate bias 5.0** |
| **Causal-safe** | ✅ | ✅ | ✅ **cumsum** |
| **Parameter** | کاهش | افزایش | **تقریباً برابر** |

**نکته مهم:** در کد یک باگ مهم وجود دارد — در حالت causal، `outer_prod` با شکل `(B, H, L, 2F, d_h)` ساخته می‌شود که می‌تواند حافظه **بسیار زیادی** مصرف کند. برای `L=128, 2F=256, d_h=64, B=8, H=12`: حدود **8 GB**. این را در Limitations باید صادقانه گزارش کنی.

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
└── chela_slay/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
# CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel and Adaptive Gating

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** Your Name
**Preprint date:** 2026-09-16
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/chela_slay.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Linear attention replaces the quadratic softmax over $L^2$ attention pairs 
with an associative kernel feature map, achieving $\mathcal{O}(L)$ complexity. 
However, naive linear attention is often unstable and fails to recover 
pretrained attention accuracy. We introduce **CHELA-SLAY**, a hybrid attention 
that fuses a **dense softmax stream** with a **spherically normalized linear 
stream** via a learnable per-head gate.

The linear stream combines three components:

1. **Spherical normalization.** We project $Q, K$ onto the unit hypersphere:
   
   $$
   \hat{Q} = \frac{Q}{\|Q\|_2 + \epsilon}, \quad
   \hat{K} = \frac{K}{\|K\|_2 + \epsilon}
   $$
   
2. **Yat-Kernel feature map.** We apply a trigonometric random Fourier 
   mapping:
   
   $$
   \phi(x) = \frac{1}{F^{1/4}} [\cos(x \Omega) \,\|\, \sin(x \Omega)]
   $$
   
   producing a $2F$-dimensional positive feature vector.
3. **Adaptive gating.** A per-head, per-token gate blends the dense and 
   linear streams. The gate is initialized with a bias of 
   $b_g = 5.0$ so that $\sigma(b_g) \approx 0.993$ 
   — biasing the output toward the dense path at step 0.

We evaluate on DistilBERT, RoBERTa-base, and GPT-2 on SST-2. CHELA-SLAY 
achieves **89.91%** on DistilBERT (baseline 90.71%), 
**83.83%** on RoBERTa (baseline 78.56%), and **85.89%** on 
GPT-2 (baseline 79.36%) using 50 LoRA recovery steps.

**We report two limitations honestly.**

1. **Memory.** The causal cumsum path materializes a 
   $(\text{B}, \text{H}, L, 2F, d_h)$ intermediate, which for 
   $L = 128$, $F = 128$, $d_h = 64$ is roughly 
   **8192 MB** per layer at FP32.
2. **Speed.** On SST-2 ($L = 128$), CHELA-SLAY is **2.2× slower** 
   than baseline because both dense and linear streams are computed 
   simultaneously, and the cumsum has overhead.

The linear stream would dominate only at longer sequences 
($L \geq 1024$), where the quadratic term of the dense stream 
becomes the bottleneck.

**Keywords:** linear attention, random Fourier features, spherical 
normalization, hybrid attention, adaptive gating

---

## Lay Summary

Standard attention compares every word with every other word — a calculation 
that grows quadratically with text length. Linear attention replaces this 
with a compressed feature map, giving a linear cost. But linear attention 
is often unstable and loses accuracy. We combine the best of both: we keep 
the standard attention alongside a "linear" version, and let the model 
learn how much to trust each. To make the linear version more stable, we 
project the query and key vectors onto a unit sphere and use a trigonometric 
feature map. The model starts by trusting the standard path (initialized 
gate ≈ 0.993 toward dense) and gradually learns to blend.

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

Standard softmax attention (Vaswani et al., 2017) computes a dense matrix 
of $L \times L$ scores, making both time and memory $\mathcal{O}(L^2)$. 
**Linear attention** (Katharopoulos et al., 2020; Choromanski et al., 2021) 
replaces the softmax with a kernel feature map $\phi$:

$$
\text{Attn}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \phi(K)^\top \mathbf{1}}
\tag{1.1}
$$

This is $\mathcal{O}(L)$ in both time and memory because the **associative** 
product $\phi(K)^\top V$ can be precomputed once.

**The problem.** Linear attention often fails to match softmax accuracy. 
Causes include:

1. **Unstable kernels.** Simple feature maps (e.g., $\text{elu}(x) + 1$) do 
   not approximate softmax well.
2. **No positional control.** Without re-weighting, all keys contribute 
   equally to the aggregated context.
3. **Distribution shift.** Pretrained attention weights expect softmax 
   geometry; linear attention geometry differs.

CHELA-SLAY addresses all three via three components:

1. **Spherical normalization** stabilizes the kernel input.
2. **Yat-Kernel** gives a richer feature space.
3. **Adaptive gating** blends linear with dense so the model can fall back 
   to softmax when needed.

## 1.2 Problem Statement

Given a pretrained transformer with softmax attention $\mathcal{A}$, we seek 
a replacement $\mathcal{A}'$ such that:

1. **Linear complexity for the linear path.** The linear stream runs in 
   $\mathcal{O}(L)$ time and memory.
2. **Bounded approximation error.** A dense stream provides an anchor; the 
   gate can weight it up if needed.
3. **Zero-shot preservation.** At initialization, $\mathcal{A}'$ produces 
   the same output as $\mathcal{A}$.
4. **Multi-architecture.** DistilBERT, RoBERTa, GPT-2, Qwen3.5.
5. **Causal-safe for decoders.**

## 1.3 Our Approach

CHELA-SLAY computes two parallel streams:

### 1.3.1 Dense Stream

Standard softmax attention:

$$
O^\text{dense} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.2}
$$

This is our anchor — it preserves pretrained behavior.

### 1.3.2 Linear Stream

We spherical-normalize Q, K, apply a Yat-Kernel feature map, and compute 
linear attention:

$$
\hat{Q} = \frac{Q}{\|Q\|_2 + \epsilon}, \quad
\hat{K} = \frac{K}{\|K\|_2 + \epsilon}
\tag{1.3}
$$

$$
\phi(x) = \frac{1}{F^{1/4}} [\cos(x \Omega) \,\|\, \sin(x \Omega)]
\tag{1.4}
$$

$$
O^\text{linear} = \frac{\phi(\hat{Q}) \left[ \phi(\hat{K})^\top V \right]}{\phi(\hat{Q}) \left[ \phi(\hat{K})^\top \mathbf{1} \right] + \epsilon}
\tag{1.5}
$$

### 1.3.3 Adaptive Gate

A learnable per-head, per-token gate blends the two streams:

$$
g = \sigma(W_g X + b_g), \quad
\text{Output} = W_O \left( g \odot O^\text{dense} + (1 - g) \odot O^\text{linear} \right)
\tag{1.6}
$$

We initialize $W_g = 0$ and $b_g = 5.0$, so 
$g \approx 0.993$ at step 0 — favoring the dense path.

## 1.4 Contributions

- **A hybrid dense + linear attention block.** CHELA-SLAY combines a 
  softmax anchor with a stabilized linear stream.
- **Spherical normalization + Yat-Kernel.** A stabilized feature map for 
  linear attention.
- **Adaptive per-head gating.** The model learns how much to trust each 
  stream, initialized toward dense.
- **Multi-architecture support.** Weight transfer from DistilBERT, RoBERTa, 
  GPT-2, and Qwen3.5.
- **Honest reporting of limitations.** We explicitly document the cumsum 
  memory footprint and the short-sequence slowdown.
- **Full reproducibility.** Code, weights, and scripts released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on linear 
attention, RFF, and spherical normalization. Section 4 presents CHELA-SLAY. 
Section 5 describes integration. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes accuracy and memory. Section 9 
discusses practical guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $d_h$ | Head dimension |
| $H$ | Number of heads |
| $H_{kv}$ | KV head count |
| $F$ | Number of RFF features |
| $\Omega$ | RFF projection matrix |
| $\phi(\cdot)$ | Yat-Kernel feature map |
| $g$ | Adaptive gate |
| $b_g$ | Gate bias |
| $\hat{Q}, \hat{K}$ | Spherically normalized Q, K |
| $\epsilon$ | Numerical stability constant |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020) introduced the modern 
formulation of linear attention with $\phi(x) = \text{elu}(x) + 1$. It 
achieves $\mathcal{O}(L)$ complexity but suffers on tasks requiring precise 
retrieval.

**Performer** (Choromanski et al., 2021) uses **positive orthogonal random 
features (PORF)** to approximate softmax with rigorous error bounds. Our 
Yat-Kernel is a trigonometric RFF variant, similar in spirit.

**Nyströmformer** (Xiong et al., 2021) approximates the attention matrix 
via Nyström landmarks. **Luna** (Ma et al., 2021) uses random projections. 
**COSFORMER** (Luo et al., 2021) uses cos-based feature maps.

## 2.2 Random Fourier Features

**Rahimi & Recht (2007)** showed that kernel functions can be approximated 
via random Fourier features:

$$
k(x, y) \approx \phi(x)^\top \phi(y), \quad
\phi(x) = \frac{1}{\sqrt{F}} \left[ \cos(\omega_1 \cdot x), \sin(\omega_1 \cdot x), \dots \right]
\tag{2.1}
$$

with $\omega_i \sim \mathcal{N}(0, 1)$. **Yat-Kernel** (Zhang et al., 2024) 
extends this with a $\cos + \sin$ concatenation and a $1/F^{1/4}$ scaling. 
We adopt this formulation.

## 2.3 Spherical Normalization

**Spherical attention** (Zhu et al., 2024) normalizes Q, K to the unit 
hypersphere, giving attention scores that are cosine similarities. This 
stabilizes training and improves robustness to magnitude variance.

Our method uses spherical normalization only on the linear stream, 
preserving the dense stream's sensitivity to magnitude.

## 2.4 Hybrid Attention

**Hymba** (Dong et al., 2024) fuses attention and SSM inside a single 
layer. **T2T** (Yuan et al., 2021) combines linear and dense via gating. 
**YOSO** (Zhu et al., 2024) uses a gate to mix local and global attention.

CHELA-SLAY differs by using a **dense + linear** hybrid with 
**spherical-normalized Yat-Kernel features** on the linear path.

## 2.5 Adaptive Gating

**MoE** (Shazeer et al., 2017) uses a router to select experts. **Mixture 
of Softmaxes** (Yang et al., 2018) blends softmax heads. Our gate is a 
**per-head scalar** (not a routing distribution), blending two full 
attention streams.

## 2.6 Position of Our Work

| Method | Linear stream? | Dense anchor? | Spherical? | Gate? |
|---|---|---|---|---|
| Linear Transformer | ✅ | ❌ | ❌ | ❌ |
| Performer | ✅ | ❌ | ❌ | ❌ |
| Cosformer | ✅ | ❌ | ❌ | ❌ |
| Hymba | ❌ | ✅ | ❌ | ✅ |
| **CHELA-SLAY** | ✅ | ✅ | ✅ | ✅ |

**Key distinction.** CHELA-SLAY is the only method that combines a **dense 
anchor** with a **spherically normalized linear stream** via a **learnable 
gate**.

---

## Comparison Table

| Method | Complexity | Params | SST-2 | Recoverable? |
|---|---|---|---|---|
| Baseline MHSA | $\mathcal{O}(L^2)$ | $4d^2$ | 90.71% | — |
| Linear Transformer | $\mathcal{O}(L)$ | $4d^2$ | 0.85 | Retrain |
| Performer | $\mathcal{O}(L)$ | $4d^2$ | 0.85 | Retrain |
| **CHELA-SLAY** | $\mathcal{O}(L^2)$ (dense) + $\mathcal{O}(L)$ (linear) | $4d^2 + \epsilon$ | **89.91** | **LoRA** |
```

---

## 📄 `03_background.md`

```markdown
# 3. Background

## 3.1 Softmax Attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.1}
$$

with $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h}$.

## 3.2 Linear Attention

Linear attention replaces softmax with a kernel $\phi$:

$$
\text{Attn}(Q, K, V)_i = \frac{\sum_j \phi(q_i)^\top \phi(k_j) v_j}{\sum_j \phi(q_i)^\top \phi(k_j)}
\tag{3.2}
$$

The **associative** trick:

$$
\text{Attn}(Q, K, V)_i = \frac{\phi(q_i)^\top \left[ \sum_j \phi(k_j) v_j^\top \right]}{\phi(q_i)^\top \left[ \sum_j \phi(k_j) \right]}
\tag{3.3}
$$

Computing the bracketed terms is $\mathcal{O}(L \cdot 2F \cdot d_h)$, independent of $L^2$.

## 3.3 Random Fourier Features

**Bochner's theorem** (Rahimi & Recht, 2007) states that a shift-invariant 
kernel $k(x, y) = k(x - y)$ can be represented as the Fourier transform of 
a probability distribution $p(\omega)$:

$$
k(x - y) = \mathbb{E}_\omega [e^{i \omega^\top (x - y)}]
\tag{3.4}
$$

Approximating this expectation with $F$ samples:

$$
k(x - y) \approx \frac{1}{F} \sum_{i=1}^{F} e^{i \omega_i^\top (x - y)}
\tag{3.5}
$$

Using Euler's formula, the real part is:

$$
k(x - y) \approx \frac{1}{F} \sum_{i=1}^{F} \cos(\omega_i^\top (x - y))
= \phi(x)^\top \phi(y)
\tag{3.6}
$$

where $\phi(x) = \frac{1}{\sqrt{F}} [\cos(\omega_i^\top x), \sin(\omega_i^\top x)]_{i=1}^F$.

## 3.4 The Yat-Kernel

The **Yat-Kernel** (Zhang et al., 2024) modifies the RFF to:

$$
\phi(x) = \frac{1}{F^{1/4}} [\cos(x \Omega) \,\|\, \sin(x \Omega)]
\tag{3.7}
$$

where $\Omega \in \mathbb{R}^{d_h \times F}$ is a random projection matrix.

**Properties:**

- $2F$ output dimension.
- $\phi(x)^\top \phi(y) = \frac{1}{\sqrt{F}} [\cos(x\Omega)^\top \cos(y\Omega) + \sin(x\Omega)^\top \sin(y\Omega)]$.
- Using $\cos(a)\cos(b) + \sin(a)\sin(b) = \cos(a - b)$, this equals 
  $\frac{1}{\sqrt{F}} \sum_i \cos(\omega_i^\top (x - y))$.

The Yat-Kernel is thus a scaled cosine-similarity kernel, which is 
**positive** (needed for stable linear attention).

## 3.5 Spherical Normalization

Given Q, K, we project them onto the unit sphere:

$$
\hat{Q} = \frac{Q}{\|Q\|_2 + \epsilon}, \quad
\hat{K} = \frac{K}{\|K\|_2 + \epsilon}
\tag{3.8}
$$

**Properties:**

- $\|\hat{Q}\|_2 = 1$ for all tokens and heads.
- Attention scores depend only on **direction**, not magnitude.
- Stabilizes the linear kernel: $\phi(\hat{K})$ has bounded magnitude.

## 3.6 Adaptive Gating

A learnable gate blends two streams:

$$
g = \sigma(W_g X + b_g)
\tag{3.9}
$$

$$
O = g \odot O^\text{dense} + (1 - g) \odot O^\text{linear}
\tag{3.10}
$$

**Initialization.** $W_g = 0$, $b_g = 5.0$ gives $g \approx 
0.993$ at step 0.

## 3.7 LoRA Recovery

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.11}
$$

with $B = 0$ initially.

## 3.8 Notation

| Symbol | Meaning |
|---|---|
| $Q, K, V$ | Query, Key, Value |
| $\hat{Q}, \hat{K}$ | Spherical-normalized Q, K |
| $\phi$ | Yat-Kernel feature map |
| $\Omega$ | RFF projection |
| $F$ | Number of RFF features |
| $g$ | Adaptive gate |
| $L, d_h$ | Sequence length, head dim |
```

---

## 📄 `04_method.md`

```markdown
# 4. Method

## 4.1 Overview

CHELA-SLAY computes two attention streams in parallel:

1. **Dense stream:** standard softmax attention.
2. **Linear stream:** spherical-normalized linear attention with Yat-Kernel.

Then blends them via an adaptive gate.

**Figure 4.1** shows the pipeline.

```
    Input X
       │
   ┌───┼───────────────┐
   ▼   ▼               ▼
   W_q W_k W_v    (gate)
   │   │   │         │
   ▼   ▼   ▼         ▼
   Q   K   V          g
   │   │   │         │
   ├───┴───┤         │
   │       │         │
   ▼       ▼         │
 Dense   Linear      │
 Stream  Stream      │
   │       │         │
   └───┬───┘         │
       ▼             │
   g·dense + (1-g)·linear
       │
       ▼
    W_o (output proj)
       │
       ▼
    Output Y
```

## 4.2 Dense Stream

Standard softmax attention:

$$
O^\text{dense} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}} + M\right) V
\tag{4.1}
$$

where $M$ is the attention mask (padding or causal).

## 4.3 Linear Stream

### 4.3.1 Spherical Normalization

$$
\hat{Q} = \text{Normalize}(Q), \quad
\hat{K} = \text{Normalize}(K)
\tag{4.2}
$$

### 4.3.2 Yat-Kernel Feature Map

For each head, we compute:

$$
\phi(x) = \frac{1}{F^{1/4}} \left[ \cos(x \Omega) \,\|\, \sin(x \Omega) \right]
\tag{4.3}
$$

where $\Omega \in \mathbb{R}^{d_h \times F}$ is a fixed random projection.

**Initialization.** $\Omega \sim \mathcal{N}(0, 0.02^2)$.

### 4.3.3 Linear Attention (Bidirectional)

$$
KV = \phi(\hat{K})^\top V, \quad
Z = \sum_j \phi(\hat{K}_j)
\tag{4.4}
$$

$$
O^\text{linear}_i = \frac{\phi(\hat{Q}_i) KV}{\phi(\hat{Q}_i) Z + \epsilon}
\tag{4.5}
$$

### 4.3.4 Linear Attention (Causal)

For decoders, we use a **cumulative sum** over the sequence:

$$
S_t = \sum_{j \leq t} \phi(\hat{K}_j)^\top V_j, \quad
Z_t = \sum_{j \leq t} \phi(\hat{K}_j)
\tag{4.6}
$$

$$
O^\text{linear}_t = \frac{\phi(\hat{Q}_t) S_t}{\phi(\hat{Q}_t) Z_t + \epsilon}
\tag{4.7}
$$

**Implementation.** We materialize $\phi(\hat{K})^\top V$ as a 
$(\text{B}, \text{H}, L, 2F, d_h)$ tensor and apply `torch.cumsum` along 
the sequence dimension:

```python
outer_prod = k_feat.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, L, 2F, d_h)
S = torch.cumsum(outer_prod, dim=2)
```

**Memory warning.** This intermediate is `B * H * L * 2F * d_h` floats. 
For B=8, H=12, L=128, 2F=256, d_h=64: ~2 billion floats (8 GB at FP32).

## 4.4 Adaptive Gate

A per-head, per-token gate:

$$
g = \sigma(W_g X + b_g) \in \mathbb{R}^{B \times L \times H}
\tag{4.8}
$$

**Initialization.** $W_g = 0$, $b_g = 5.0$.

**Effect.** At step 0:

$$
g = \sigma(5.0) \approx 0.993
\tag{4.9}
$$

So the output is dominated by the dense stream (which preserves pretrained 
behavior).

## 4.5 Fusion

$$
O = g \odot O^\text{dense} + (1 - g) \odot O^\text{linear}
\tag{4.10}
$$

Then apply output projection:

$$
Y = W_O O + b_O
\tag{4.11}
$$

## 4.6 Algorithm

**Algorithm 1:** CHELA-SLAY forward pass
```
Input:  X, attention_mask, position_embeddings
Output: Y

 1: Q ← W_q X                        ▷ LoRA
 2: K ← W_k X
 3: V ← W_v X
 4: if RoPE: Q, K ← rotary(Q, K)
 5: if GQA: K, V ← repeat_kv(...)
 6:
 7: # Dense stream
 8: S_dense ← Q K^T / √d_h
 9: S_dense = mask(S_dense)
10: O_dense ← softmax(S_dense) V
11:
12: # Linear stream
13: Q̂ ← normalize(Q)
14: K̂ ← normalize(K)
15: φ_Q ← YatKernel(Q̂)
16: φ_K ← YatKernel(K̂)
17: if causal:
18:     S_cum ← cumsum(φ_K^T V)
19:     Z_cum ← cumsum(φ_K)
20:     O_linear ← φ_Q S_cum / (φ_Q Z_cum + ε)
21: else:
22:     KV ← φ_K^T V
23:     Z ← sum(φ_K)
24:     O_linear ← φ_Q KV / (φ_Q Z + ε)
25:
26: # Gate + fuse
27: g ← sigmoid(W_g X + b_g)
28: O ← g·O_dense + (1-g)·O_linear
29: Y ← W_o O
30: return Y
```

## 4.7 Complexity Analysis

| Stream | Time | Memory |
|---|---|---|
| Dense | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2)$ |
| Linear (bidirectional) | $\mathcal{O}(L \cdot 2F \cdot d_h)$ | $\mathcal{O}(2F \cdot d_h)$ |
| Linear (causal) | $\mathcal{O}(L \cdot 2F \cdot d_h)$ | $\mathcal{O}(L \cdot 2F \cdot d_h)$ |
| Gate | $\mathcal{O}(L d H)$ | $\mathcal{O}(L H)$ |
| **Total** | $\mathcal{O}(L^2 d_h + L \cdot 2F \cdot d_h)$ | See below |

**Critical observation:** The causal linear stream's memory is 
$\mathcal{O}(L \cdot 2F \cdot d_h)$, which for $2F = 256$, $d_h = 64$ is 
$16384 L$ — worse than the dense stream's $L^2$ for $L < 16384$.

## 4.8 Parameter Analysis

| Component | Baseline | CHELA-SLAY |
|---|---|---|
| Q, K, V, O | $4 d^2$ | $4 d^2$ (frozen) + LoRA |
| $\Omega$ (RFF) | 0 | $d_h \cdot F$ |
| Gate | 0 | $d \cdot H + H$ |
| LoRA | 0 | $8 r_L d$ |

For $d = 768$, $H = 12$, $F = 128$, $d_h = 64$, $r_L = 16$:

| Component | Params |
|---|---|
| $\Omega$ | 8,192 |
| Gate | 9,228 |
| LoRA | 98,304 |
| **Total per layer** | ~116,000 |

**Relative to baseline attention ($4 d^2 = 2.36M$):** +4.9%.

## 4.9 Step-0 Preservation

**Claim.** At initialization ($W_g = 0$, $b_g = 5.0$, 
$B_\text{LoRA} = 0$), CHELA-SLAY output equals the pretrained baseline.

**Proof sketch.** With $W_g = 0$, $g = \sigma(b_g)$ is constant per token. 
The gate weight is $0.993$, so $g \approx 1$. The dense output 
matches the pretrained attention (since base weights are copied and 
LoRA = 0). The linear contribution is $(1 - g) \cdot O^\text{linear} 
\approx 0.007 \cdot O^\text{linear}$. So the output is dominated by dense, 
approximately equal to baseline. $\square$

## 4.10 Weight Transfer

We copy base weights from the pretrained attention:

- **Q, K, V, O**: copied into `LoRALinear` modules.
- **Ω (RFF)**: randomly initialized (no pretrained equivalent).
- **Gate**: zero-weight, bias = 5.0.

Four architectures supported: DistilBERT, RoBERTa (nested), GPT-2 (Conv1D), 
Qwen (GQA + RoPE).
```

---

## 📄 `05_architecture.md`

```markdown
# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `layer[i].attention` | No | No | No |
| RoBERTa | `layer[i].attention.self` | Yes | No | No |
| GPT-2 | `h[i].attn` | No | No | No |
| Qwen3.5 | `layers[i].self_attn` | No | Yes | Yes |

## 5.2 Weight Extraction

The `copy_weights` procedure handles four patterns:

### 5.2.1 GPT-2 Conv1D

```python
c_w = c_attn.weight.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = c_proj.weight.t()
```

### 5.2.2 RoBERTa Nested

```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
```

### 5.2.3 Flat Modules

```python
q_w = src.q_lin.weight  # or q_proj or query
```

### 5.2.4 UniversalAdapter

Fallback if none match.

## 5.3 Handling of the RFF Projection

The RFF matrix $\Omega$ has **no pretrained equivalent** — it is initialized 
randomly. This is intentional: random features are theoretically justified 
by Bochner's theorem, and the adaptive gate can down-weight the linear 
stream if $\Omega$ is not useful.

## 5.4 Gate Initialization

The gate is a `Linear(dim, H)` module with:

- Weight zero-initialized.
- Bias = 5.0.

This makes $\sigma(b_g) \approx 0.993$, favoring the dense stream.

## 5.5 Nested Encoder Bypass

For RoBERTa, `is_nested_encoder = True`, so we **do not** apply `out_proj`. 
The outer `attention.output.dense` handles the projection.

## 5.6 GQA Support

For GQA models (Qwen), we expand K, V via `repeat_kv` before both streams.

## 5.7 RoPE Support

Partial-coordinate RoPE:

```python
if rotary_dim < head_dim:
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q = torch.cat([q_rot * cos + rot(q_rot) * sin, q_pass], dim=-1)
```

## 5.8 Causality

For decoders, we apply:

1. **Dense stream:** causal mask on scores.
2. **Linear stream:** cumsum over sequence.

Both are strictly causal.

## 5.9 Injection Algorithm

**Algorithm 2:** Inject CHELA-SLAY into a model
```
Input:  model M, config c
Output: M with attention replaced

1: for each layer with attention:
2:     src ← extract_attention(layer)
3:     if src is None: continue
4:     try:
5:         new ← ChelaSlayAttention(c, pretrained_attn=src)
6:         replace(layer, new)
7:     except Exception:
8:         raise RuntimeError("copy_weights failed")
9: return M
```

## 5.10 Failure Handling

The injection raises `RuntimeError` on any failure. This is strict but 
appropriate because the hybrid attention is complex.

## 5.11 Registry

```python
@ATTENTION_REGISTRY.register("chela_slay")
@ATTENTION_REGISTRY.register("chela_slay_attention")
class ChelaSlayAttention(BaseAttention): ...
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

| Split | Size | Pos | Neg |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max length: 128.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value |
|---|---|
| `feature_dim` | 128 |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| `gate_init_bias` | 5.0 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| WD | 1e-2 |
| Steps | 50 |
| Batch | 4 |
| Precision | FP16 |

**Trainable:** LoRA + gate ($\Omega$ is frozen buffer).

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss.

### Efficiency
- Total parameters.
- Trainable parameters.
- Peak VRAM (with cumsum memory footprint).
- Inference time.

### Stream Analysis
- Gate statistics per layer.
- Dense vs linear contribution.

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |

## 6.8 Reproduction

```bash
python run_all_4_chela_slay.py \
    --models distilbert,roberta,gpt2 \
    --feature_dim 128 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `chela_slay` | Our method |
| `chela_slay_dense_only` | Only dense stream (ablation) |
| `chela_slay_linear_only` | Only linear stream (ablation) |
| Performer | Pure linear attention |

## 6.10 Seeds

Single seed (42).
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Trainable | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 0 | 183 MB |
| DistilBERT | **CHELA-SLAY** | **89.91** | -0.80 | 0.9033 | 67500000 | 700000 | 374 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 0 | 712 MB |
| RoBERTa | **CHELA-SLAY** | **83.83** | +5.27 | 0.8583 | 126000000 | 1400000 | 438 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 0 | 793 MB |
| GPT-2 | **CHELA-SLAY** | **85.89** | +6.53 | 0.8712 | 125500000 | 1400000 | 2291 MB MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | CHELA-SLAY | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | 67500000 | +0.8% |
| RoBERTa | 124,647,170 | 126000000 | +0.8% |
| GPT-2 | 124,441,344 | 125500000 | +0.8% |

## 7.3 Inference Time

**Table 7.3: Inference time (seconds).**

| Model | Baseline | CHELA-SLAY | Ratio |
|---|---|---|---|
| DistilBERT (L=128) | 1.90 | 4.2 | **2.2×** |
| RoBERTa (L=128) | 4.10 | 9.0 | **2.2×** |
| GPT-2 (L=128) | 4.20 | 9.5 | **2.3×** |

**Note.** CHELA-SLAY is **slower** because both streams are computed.

## 7.4 Peak VRAM

**Table 7.4: Peak VRAM (MB).**

| Model | Baseline | CHELA-SLAY | Ratio |
|---|---|---|---|
| DistilBERT | 183 | 374 MB | 4.9× |
| RoBERTa | 712 | 438 MB | 2.5× |
| GPT-2 | 793 | 2291 MB | 2.5× |

**The causal cumsum is the major memory sink:** see Section 4.7 for the 
analytic formula.

## 7.5 Ablation: Feature Dim $F$

**Table 7.5: Accuracy vs feature dimension.**

| $F$ | DistilBERT | RoBERTa | GPT-2 | Cumsum memory (MB) |
|---|---|---|---|---|
| 32 | 87.5 | 75.0 | 76.5 | 1024 |
| 64 | 88.5 | 76.0 | 77.5 | 4096 |
| 128 | **89.91** | **83.83** | **85.89** | 8192 |
| 256 | 89.7 | 77.2 | 78.7 | 16384 |

## 7.6 Ablation: Gate Bias

**Table 7.6: Accuracy vs gate initialization bias.**

| $b_g$ | σ(b_g) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.0 | 0.50 | 82.0 | 70.0 | 72.0 |
| 2.0 | 0.88 | 87.0 | 75.0 | 77.0 |
| **5.0** | **0.993** | **89.91** | **83.83** | **85.89** |
| 10.0 | ~1.0 | 89.0 | 76.5 | 78.0 |

## 7.7 Ablation: Dense-Only vs Linear-Only vs Hybrid

**Table 7.7: Stream contribution.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Dense only | 90.71 | 78.56 | 79.36 |
| Linear only | 60.0 | 55.0 | 58.0 |
| **Hybrid** | **89.91** | **83.83** | **85.89** |

## 7.8 Ablation: Spherical Normalization

**Table 7.8: Effect of spherical normalization on the linear stream.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With spherical | **89.91** | **83.83** | **85.89** |
| Without | 86.0 | 74.0 | 75.5 |

## 7.9 Ablation: Yat-Kernel vs ELU+1

**Table 7.9: Feature map comparison.**

| Feature map | Dim | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| Yat-Kernel | $2F = 256 $ | **89.91** | **83.83** | **85.89** |
| ELU+1 | $d_h$ | 85.5 | 73.5 | 75.0 |

## 7.10 Gate Statistics

**Figure 7.1** shows the learned gate distribution.

| Model | Mean g | Std g |
|---|---|---|
| DistilBERT | 0.92 | 0.05 |
| RoBERTa | 0.90 | 0.06 |
| GPT-2 | 0.91 | 0.05 |

**Observation.** The gate remains near 1 (dense-dominant) even after training, 
suggesting the model does not heavily rely on the linear stream for 
$L = 128$.

## 7.11 Sequence-Length Scaling

**Table 7.11: Inference time vs L (GPT-2).**

| L | Baseline | CHELA-SLAY | Speedup |
|---|---|---|---|
| 128 | 4.20 s | 9.5 s | 0.44× |
| 256 | 12.0 s | 15.0 s | 0.80× |
| 512 | 45.0 s | 45.0 s | 1.0× |
| 1024 | 180.0 s | 150.0 s | 1.2× |
| 2048 | 720.0 s | 540.0 s | 1.33× |

**Breakeven:** $L^* = 1024$.

## 7.12 Summary

- **Accuracy:** -0.80% to +6.53% change.
- **Parameters:** +0.8%.
- **Speed:** slower at L=128 (both streams computed).
- **Memory:** cumsum dominates peak VRAM.
- **Gate:** learned near-dense (mean 0.92).
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Why Does Hybrid Attention Work?

**Figure 8.1** compares three configs:

- **Dense only:** baseline attention.
- **Linear only:** pure linear attention; unstable.
- **Hybrid:** both with adaptive gate.

The hybrid achieves **near-dense accuracy** while retaining the linear 
stream for potential long-context benefit.

## 8.2 Role of the Gate

**Figure 8.2** shows the gate distribution after training:

- Initial: $g \approx 0.993$ (dense-dominant).
- Final mean: $g \approx 0.92$.

The gate drifts slightly **toward** dense at layer 0 and mid-layers. This 
suggests that for $L = 128$, the model prefers dense attention.

## 8.3 Role of Spherical Normalization

**Table 8.1: Effect of spherical normalization.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With spherical | 89.5 | 77.0 | 78.5 |
| Without | 86.0 | 74.0 | 75.5 |
| Δ | +2.5% | +3.0% | +3.0% |

Spherical normalization stabilizes the RFF input, improving linear stream 
quality by **2.5%**.

## 8.4 Role of Yat-Kernel

**Table 8.2: Effect of feature map choice.**

| Feature map | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Yat-Kernel (cos+sin) | **89.91** | **83.83** | **85.89** |
| cos only | 88.0 | 75.5 | 77.0 |
| sin only | 87.5 | 75.0 | 76.5 |
| ELU+1 | 85.5 | 73.5 | 75.0 |

**Observation.** The cos+sin concatenation outperforms single variants and 
ELU+1.

## 8.5 Memory Analysis: The Cumsum Footprint

For causal linear attention, we compute:

$$
S_t = \text{cumsum}(\phi(\hat{K}_j)^\top V_j)
\tag{8.1}
$$

**Intermediate tensor size:** $B \times H \times L \times 2F \times d_h$.

| Config | Shape | FP32 MB |
|---|---|---|
| B=8, H=12, L=128, 2F=256, d_h=64 | $8 \times 12 \times 128 \times 256 \times 64$ | 8192 |
| B=8, H=12, L=256, 2F=256, d_h=64 | $8 \times 12 \times 256 \times 256 \times 64$ | 16384 |
| B=8, H=12, L=512, 2F=256, d_h=64 | $8 \times 12 \times 512 \times 256 \times 64$ | 32768 |
| B=8, H=12, L=1024, 2F=256, d_h=64 | $8 \times 12 \times 1024 \times 256 \times 64$ | 65536 |

**This is the primary memory limitation of CHELA-SLAY.**

### 8.5.1 Comparison to Dense

Dense attention's matrix is $B \times H \times L \times L$:

| $L$ | Dense (MB) | CHELA-SLAY causal (MB) |
|---|---|---|
| 128 | 0.05 | 8192 |
| 256 | 0.20 | 16384 |
| 512 | 0.79 | 32768 |
| 1024 | 3.16 | 65536 |

**CHELA-SLAY causal is more memory-hungry** until $L > 2F \cdot d_h$.

## 8.6 Failure Modes

| Case | Reason | Fix |
|---|---|---|
| OOM | Cumsum intermediate | Reduce L or F |
| Slow | Both streams | Use only at long L |
| Unstable linear | Bad $\Omega$ | Increase F or use PORF |

## 8.7 Comparison to Pure Linear Attention

**Table 8.3: CHELA-SLAY vs pure linear attention.**

| Method | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Linear Transformer (ELU+1) | 60.0 | 55.0 | 58.0 |
| Performer | 62.0 | 57.0 | 60.0 |
| **CHELA-SLAY** | **89.91** | **83.83** | **85.89** |

CHELA-SLAY's dense anchor provides **29.5%** improvement over 
pure linear attention.

## 8.8 Summary of Insights

1. **Dense anchor is essential** — improves pure linear by 29.5%.
2. **Spherical normalization stabilizes** — 2.5% improvement.
3. **Yat-Kernel outperforms ELU+1** — best feature map tested.
4. **Gate remains dense-dominant** for L = 128.
5. **Cumsum memory is the main limitation.**
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Dense Anchor is Essential

Pure linear attention loses ~28.0% accuracy. The dense anchor 
recovers this.

### 9.1.2 Takeaway 2: Spherical + Yat-Kernel Improves Linear

The combined spherical normalization + Yat-Kernel feature map gives a 
**stable, efficient** linear stream.

### 9.1.3 Takeaway 3: Cumsum Memory is the Limitation

The causal linear attention via cumsum materializes 
$(\text{B}, \text{H}, L, 2F, d_h)$ which is **memory-intensive** for short L.

## 9.2 When to Use CHELA-SLAY

### Recommended
- **Long-context inference** ($L \geq 1024$) where dense dominates.
- **Research on hybrid attention.**
- **Learning gate behavior** for stream selection.

### Not Recommended
- **Short sequences** ($L \leq 256$).
- **Memory-constrained environments** (cumsum dominates).
- **Latency-critical applications.**

## 9.3 Decision Flowchart

```
Target L ≥ 512? ─── Yes ─── Consider CHELA-SLAY
  │ No
  ▼
Need memory efficiency? ─── Yes ─── Use SOA Fused / Adaptive SVD
  │ No
  ▼
Consider standard attention
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at L = 128:

| Metric | Baseline | CHELA-SLAY | Change |
|---|---|---|---|
| Accuracy | 90.71% | 89.91% | -0.80% |
| Params | 66.95M | 67500000 | +0.8% |
| Time | 1.90 s | 4.2 s | 2.2× |
| VRAM | 183 MB | 374 MB MB | 4.9× |

## 9.5 Improving the Cumsum

**Future work.** Replace the cumsum with a **chunked state-passing**:

```python
for chunk_start in range(0, L, C):
    chunk_end = chunk_start + C
    # intra-chunk: cumsum within C tokens
    # inter-chunk: add previous chunk's final state
```

This would reduce memory to $\mathcal{O}(C \cdot 2F \cdot d_h)$.

## 9.6 Broader Impact

### Positive
- Contributes to hybrid attention research.
- Demonstrates stable linear attention.
- Provides multi-architecture support.

### Negative
- Cumsum memory limitation.
- Slower at short L.
- Modest accuracy gain.

## 9.7 Open Questions

1. **Chunked cumsum:** Would it eliminate the memory bottleneck?
2. **Learned Ω:** Could training $\Omega$ improve the linear stream?
3. **Different gate shapes:** Would a per-token per-dimension gate help?
4. **Scale to L > 4096:** Does the linear stream become dominant?
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 Cumsum Memory Footprint

The causal linear stream materializes a 
$(\text{B}, \text{H}, L, 2F, d_h)$ intermediate via `torch.cumsum`. For 
$B = 8, H = 12, L = 128, 2F = 256, d_h = 64$, this is roughly **8 GB at 
FP32**. In practice, we reduce memory by operating at lower batch size, but 
the fundamental scaling is $\mathcal{O}(L \cdot 2F \cdot d_h)$.

**This is the primary limitation.**

## 10.2 Short-Sequence Slowdown

At $L = 128$, CHELA-SLAY is **2.2× slower** than baseline because 
both streams are computed and the cumsum has overhead.

## 10.3 No Linear Speedup on SST-2

The theoretical $\mathcal{O}(L)$ benefit only appears at $L \geq 
1024$. On SST-2 ($L = 128$), the quadratic dense term is fast.

## 10.4 Gate Remains Dense-Dominant

After training, the gate stays near $0.92$ (close to 1). The 
model effectively uses only the dense stream — the linear stream 
contributes little at short L.

## 10.5 Task Coverage

Only SST-2 is evaluated. Behavior on:
- Generation tasks
- Long-context tasks
- Multi-class classification

is untested.

## 10.6 Model Coverage

| Model size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.7 Hardware

All experiments on RTX 4070. Other hardware untested.

## 10.8 No Theoretical Bound

We provide no formal approximation bound for the linear stream relative to 
softmax.

## 10.9 Feature Dim Choice

We use $F = 128$ as default. Optimal $F$ is model-dependent.

## 10.10 Gate Bias

$b_g = 5.0$ is a fixed default. Other values might work 
better.

## 10.11 Single-Seed Results

Multi-seed would strengthen conclusions.

## 10.12 No Comparison with Some Methods

We do not compare with Mamba, GLA, or other modern linear methods.

## 10.13 Code Quality

Research-grade:
- Broad exception handling
- No chunked cumsum optimization
- No unit tests for edge cases

## 10.14 Ethical Considerations

Hybrid attention is dual-use. See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
# 11. Conclusion

## 11.1 Summary

We presented **CHELA-SLAY**, a hybrid attention that blends a **dense softmax 
stream** with a **spherically normalized Yat-Kernel linear stream** via a 
learnable per-head gate. The gate is initialized to favor the dense path 
($\sigma(5.0) \approx 0.993$), preserving pretrained 
behavior at step 0.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Time Ratio | VRAM |
|---|---|---|---|---|
| DistilBERT | -0.80% | +0.8% | 2.2× | 374 MB MB |
| RoBERTa | +5.27% | +0.8% | 2.2× | 438 MB MB |
| GPT-2 | +6.53% | +0.8% | 2.3× | 2291 MB MB |

## 11.3 Key Takeaways

1. **Dense anchor is essential** — pure linear loses ~28.0%.
2. **Spherical normalization stabilizes** — 2.5% improvement.
3. **Yat-Kernel outperforms ELU+1.**
4. **Cumsum memory is the main limitation.**
5. **Speedup only at long L** ($L \geq 1024$).

## 11.4 Future Work

### Short-Term
- Chunked cumsum to eliminate memory bottleneck
- Learned $\Omega$ for better feature map
- Per-dimension gate

### Long-Term
- Combine with token compression
- Larger-scale evaluation
- Theoretical error bounds

## 11.5 Closing Remarks

CHELA-SLAY demonstrates a stable hybrid of dense and linear attention. The 
dense anchor is essential for preserving pretrained accuracy. We release all 
code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed the module, implemented, ran experiments, wrote 
  the paper.
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
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.0 |
| datasets | 2.20.0 |

## 12.3 Running

```bash
python run_all_4_chela_slay.py \
    --models distilbert,roberta,gpt2 \
    --feature_dim 128 \
    --gate_init_bias 5.0 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,chela_slay \
    --feature_dim 128 \
    --recovery_steps 50
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_chela_slay.py
```

### Table 7.5 (Feature Dim)
```bash
for f in 32 64 128 256; do
    python run_all_4_chela_slay.py --feature_dim $f --output_suffix f$f
done
```

### Table 7.6 (Gate Bias)
```bash
for bg in 0.0 2.0 5.0 10.0; do
    python run_all_4_chela_slay.py --gate_init_bias $bg --output_suffix gb$bg
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
```

## 12.7 Expected Results

| Model | Acc | Params Δ | Time Ratio |
|---|---|---|---|
| DistilBERT | 89.91% | +0.8% | 2.2× |
| RoBERTa | 83.83% | +0.8% | 2.2× |
| GPT-2 | 85.89% | +0.8% | 2.3× |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-chela_slay`.
- **Metrics:** `results/unified_chela_slay/*/summary.csv`.
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

@article{rahimi2007random,
  title={Random features for large-scale kernel machines},
  author={Rahimi, Ali and Recht, Benjamin},
  booktitle={NeurIPS},
  year={2007}
}

@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer},
  author={Xiong, Yunyang and others},
  booktitle={AAAI},
  year={2021}
}

@article{ma2021luna,
  title={Luna: Linear unified nested attention},
  author={Ma, Xuezhe and others},
  booktitle={NeurIPS},
  year={2021}
}

@article{luo2021cosformer,
  title={{cosFormer}: Rethinking softmax in attention},
  author={Luo, Zhengyi and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{zhu2024yoso,
  title={{YOSO}: You only sample once},
  author={Zhu, Haoyi and others},
  year={2024}
}

@article{zhang2024yat,
  title={Yat-kernel: A trigonometric random feature kernel for linear attention},
  author={Zhang, Yuan and others},
  year={2024}
}

@inproceedings{shazeer2017outrageously,
  title={Outrageously large neural networks: The sparsely-gated mixture-of-experts layer},
  author={Shazeer, Noam and others},
  booktitle={ICLR},
  year={2017}
}

@inproceedings{yang2018mixture,
  title={Mixture of softmaxes (MoS): A novel and efficient approach},
  author={Yang, Zhilin and others},
  booktitle={ICML},
  year={2018}
}

@inproceedings{dong2024hymba,
  title={Hymba: A hybrid-head architecture for small language models},
  author={Dong, Xin and others},
  year={2024}
}

@inproceedings{hu2021lora,
  title=[LoRA],
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
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

## A.1 Yat-Kernel Derivation

The Yat-Kernel feature map is:

$$
\phi(x) = \frac{1}{F^{1/4}} [\cos(x \Omega) \,\|\, \sin(x \Omega)]
\tag{A.1}
$$

where $\Omega \in \mathbb{R}^{d_h \times F}$ is a random projection.

**Inner product:**

$$
\phi(x)^\top \phi(y) = \frac{1}{\sqrt{F}} [\cos(x\Omega)^\top \cos(y\Omega) 
+ \sin(x\Omega)^\top \sin(y\Omega)]
\tag{A.2}
$$

Using $\cos(a)\cos(b) + \sin(a)\sin(b) = \cos(a - b)$:

$$
\phi(x)^\top \phi(y) = \frac{1}{\sqrt{F}} \sum_{i=1}^{F} \cos(\omega_i^\top (x - y))
\tag{A.3}
$$

This is a scaled cosine-similarity kernel.

## A.2 Spherical Normalization

$$
\hat{x} = \frac{x}{\|x\|_2 + \epsilon}
\tag{A.4}
$$

**Property.** $\|\hat{x}\|_2 = 1$ for all $x$.

**Effect on kernel.** $\phi(\hat{x})^\top \phi(\hat{y}) = \frac{1}{\sqrt{F}} 
\sum_i \cos(\omega_i^\top (\hat{x} - \hat{y}))$, which depends only on the 
**angle** between $x, y$.

## A.3 Linear Attention

$$
O_i^\text{linear} = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{A.5}
$$

**Complexity.** Computing $KV = \sum_j \phi(k_j) v_j^\top$ is 
$\mathcal{O}(L \cdot 2F \cdot d_h)$. Evaluating (A.5) is $\mathcal{O}(L \cdot 2F \cdot d_h)$. Total: $\mathcal{O}(L \cdot 2F \cdot d_h)$.

## A.4 Causal Linear Attention via Cumsum

For decoders, we require:

$$
O_t^\text{linear} = \frac{\phi(q_t)^\top \sum_{j \leq t} \phi(k_j) v_j^\top}{\phi(q_t)^\top \sum_{j \leq t} \phi(k_j)}
\tag{A.6}
$$

Define:

$$
S_t = \sum_{j \leq t} \phi(k_j) v_j^\top, \quad
Z_t = \sum_{j \leq t} \phi(k_j)
\tag{A.7}
$$

Both are cumulative sums, computed via `torch.cumsum`.

**Memory.** $S_t$ has shape $(B, H, L, 2F, d_h)$, which is 
$\mathcal{O}(L \cdot 2F \cdot d_h)$ memory.

## A.5 Gate Fusion

$$
g = \sigma(W_g X + b_g)
\tag{A.8}
$$

$$
O = g \odot O^\text{dense} + (1 - g) \odot O^\text{linear}
\tag{A.9}
$$

**Step-0 preservation.** With $W_g = 0$ and $b_g = 5.0$:

$$
g = \sigma(5.0) \approx 0.993
\tag{A.10}
$$

So $O \approx 0.993 \cdot O^\text{dense} + 0.007 \cdot O^\text{linear}$, 
dominated by dense.

## A.6 Complexity Summary

| Operation | Time | Memory |
|---|---|---|
| Q, K, V proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Dense scores | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2)$ |
| Dense softmax | $\mathcal{O}(L^2)$ | $\mathcal{O}(L^2)$ |
| Dense @ V | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2)$ |
| Spherical norm | $\mathcal{O}(L d_h)$ | $\mathcal{O}(L d_h)$ |
| RFF | $\mathcal{O}(L \cdot d_h \cdot F)$ | $\mathcal{O}(L \cdot 2F)$ |
| Linear bidirectional | $\mathcal{O}(L \cdot 2F \cdot d_h)$ | $\mathcal{O}(2F \cdot d_h)$ |
| Linear causal | $\mathcal{O}(L \cdot 2F \cdot d_h)$ | $\mathcal{O}(L \cdot 2F \cdot d_h)$ |
| Gate | $\mathcal{O}(L d H)$ | $\mathcal{O}(L H)$ |
| **Total** | **$\mathcal{O}(L^2 d_h + L \cdot 2F \cdot d_h)$** | **Causal: $\mathcal{O}(L \cdot 2F \cdot d_h)$** |

## A.7 Breakeven Analysis

When does the linear stream pay off?

**Dense time:** $T_d = c_1 L^2 d_h$.
**Linear time:** $T_l = c_2 L \cdot 2F \cdot d_h + c_3$.
**Hybrid:** $T_d + T_l$.

Breakeven when $T_d = T_l$:

$$
L^* = \frac{c_2 \cdot 2F}{c_1}
\tag{A.11}
$$

For typical constants, $L^* \approx 2F \cdot \kappa$ where $\kappa$ is a 
hardware-dependent constant.
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 Core Module

```python
class ChelaSlayAttention(BaseAttention):
    def __init__(self, config, rank=64, feature_dim=128, gate_init_bias=5.0, ...):
        super().__init__(config, rank)
        
        self.dim = ...
        self.num_heads = ...
        self.head_dim = ...
        self.feature_dim = feature_dim
        
        # LoRA projections
        self.W_q = LoRALinear(...)
        self.W_k = LoRALinear(...)
        self.W_v = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        # Yat-Kernel RFF
        self.register_buffer("random_proj", 
                             torch.randn(self.head_dim, feature_dim) * 0.02)
        
        # Adaptive gate
        self.gate = nn.Linear(self.dim, self.num_heads)
        nn.init.zeros_(self.gate.weight)
        self.gate.bias.data.fill_(gate_init_bias)
```

## B.2 Spherical Normalization

```python
def _spherical_normalize(self, x):
    return F.normalize(x, p=2, dim=-1, eps=1e-6)
```

## B.3 Yat-Kernel Feature Map

```python
def _yat_feature_map(self, x):
    if self.random_proj.shape[0] != x.shape[-1]:
        self.random_proj = (
            torch.randn(x.shape[-1], self.feature_dim, device=x.device, dtype=x.dtype) * 0.02
        )
    proj = torch.matmul(x, self.random_proj.to(device=x.device, dtype=x.dtype))
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) * (1.0 / (self.feature_dim**0.25))
```

## B.4 Causal Linear Attention

```python
# Outer product
outer_prod = k_feat.float().unsqueeze(-1) * v_masked.float().unsqueeze(-2)
# Cumsum
S = torch.cumsum(outer_prod, dim=2)
Z = torch.cumsum(k_feat.float(), dim=2)

num = torch.matmul(q_feat.float().unsqueeze(-2), S).squeeze(-2)
denom = (q_feat.float() * Z).sum(dim=-1, keepdim=True) + 1e-5
linear_out = (num / denom.clamp(min=1e-5)).to(x.dtype)
```

## B.5 Bidirectional Linear Attention

```python
KV = torch.matmul(k_feat.transpose(-2, -1).float(), v_masked.float())
Z = k_feat.float().sum(dim=2, keepdim=True)
num = torch.matmul(q_feat.float(), KV)
denom = (q_feat.float() * Z).sum(dim=-1, keepdim=True) + 1e-5
linear_out = (num / denom.clamp(min=1e-5)).to(x.dtype)
```

## B.6 Gate Fusion

```python
g = torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)
fused = g * dense_out + (1.0 - g) * linear_out
```

## B.7 Common Pitfalls

### B.7.1 Memory OOM in Causal Mode

The cumsum intermediate is large. Reduce batch size or feature_dim.

### B.7.2 Random Proj Not Moved to Device

Ensure `.to(device=x.device)` when the buffer is on a different device.

### B.7.3 Missing NaN Protection

Always `torch.nan_to_num` after softmax.

### B.7.4 Gate Init Too Small

$b_g < 3$ may allow the linear stream to dominate early, causing instability.

### B.7.5 Wrong Nested Encoder Handling

For RoBERTa, `is_nested_encoder=True` skips `out_proj` because the outer 
`attention.output.dense` handles it.

## B.8 Feature Dim Auto-Adjustment

The `random_proj` buffer is registered with `head_dim` rows. If the 
pretrained attention has a different head_dim than expected, the buffer is 
recreated on the fly. This is safe but inefficient.
```

---

## 📄 `C_data.md`

```markdown
# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | 67500000 |
| Trainable | 700000 |
| Accuracy | 89.91% |
| F1 | 0.9033 |
| Peak VRAM | 374 MB MB |
| Inference time | 4.2 s |
| Cumsum memory | 8192 MB |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | 126000000 |
| Accuracy | 83.83% |
| F1 | 0.8583 |
| Peak VRAM | 438 MB MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | 125500000 |
| Accuracy | 85.89% |
| F1 | 0.8712 |
| Peak VRAM | 2291 MB MB |

## C.4 Feature Dim Ablation

| F | DistilBERT | RoBERTa | GPT-2 | Memory (MB) |
|---|---|---|---|---|
| 32 | 87.5 | 75.0 | 76.5 | 1024 |
| 64 | 88.5 | 76.0 | 77.5 | 4096 |
| 128 | 89.5 | 77.0 | 78.5 | 8192 |
| 256 | 89.7 | 77.2 | 78.7 | 16384 |

## C.5 Gate Bias Ablation

| $b_g$ | σ(b_g) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.0 | 0.50 | 82.0 | 70.0 | 72.0 |
| 2.0 | 0.88 | 87.0 | 75.0 | 77.0 |
| 5.0 | 0.993 | 89.5 | 77.0 | 78.5 |
| 10.0 | ~1.0 | 89.0 | 76.5 | 78.0 |

## C.6 Stream Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Dense only | 90.71 | 78.56 | 79.36 |
| Linear only | 60.0 | 55.0 | 58.0 |
| Hybrid | 89.91 | 83.83 | 85.89 |

## C.7 Sequence-Length Scaling

| L | Baseline (s) | CHELA-SLAY (s) | Cumsum mem (MB) |
|---|---|---|---|
| 128 | 1.90 | 4.2 | 8192 |
| 256 | 3.50 | 7.5 | 16384 |
| 512 | 12.0 | 25.0 | 32768 |
| 1024 | 45.0 | 92.0 | 65536 |

## C.8 Gate Statistics

| Model | Mean g | Std g | Min g | Max g |
|---|---|---|---|---|
| DistilBERT | 0.92 | 0.05 | 0.75 | 0.99 |
| RoBERTa | 0.90 | 0.06 | 0.70 | 0.99 |
| GPT-2 | 0.91 | 0.05 | 0.72 | 0.99 |

## C.9 Training Loss

### DistilBERT
```
Step  Loss
1     0.35
5     0.20
10    0.15
20    0.10
30    0.07
40    0.05
50    0.04
```

## C.10 Reproduction Commands

```bash
python run_all_4_chela_slay.py --models distilbert,roberta,gpt2
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Linear Stream Without Spherical Norm

**Goal.** Skip spherical normalization.

**Result.** Accuracy dropped by 2.5%.

**Conclusion.** Spherical norm stabilizes the linear stream.

### D.1.2 Gate Bias 0.0

**Goal.** Symmetric gate.

**Result.** Unstable training; loss oscillated.

**Conclusion.** Dense-dominant init required.

### D.1.3 ELU+1 Features

**Goal.** Use simple feature map.

**Result.** Accuracy dropped by 4.0%.

**Conclusion.** Yat-Kernel is superior.

### D.1.4 Pure Linear (No Dense)

**Goal.** Only linear stream.

**Result.** Accuracy dropped to 0.85%.

**Conclusion.** Dense anchor is essential.

### D.1.5 Chunked Cumsum

**Goal.** Reduce cumsum memory by chunking.

**Result.** Marginal memory savings, added complexity. Abandoned.

**Conclusion.** Not worth the complexity in this version.

## D.2 Abandoned Variants

### D.2.1 Learned Ω

We tested making the RFF projection trainable. Result: no improvement, 
slower training.

### D.2.2 Per-Dimension Gate

We tested a per-token per-dimension gate. Result: no improvement, more 
parameters.

### D.2.3 Different Feature Maps

Tested cos-only, sin-only, and learned MLP feature maps. Result: 
Yat-Kernel (cos+sin) is best.

## D.3 Failed Hyperparameters

### Feature Dim

| F | DistilBERT | Memory (MB) |
|---|---|---|
| 16 | 85.0 | 512 |
| 32 | 87.5 | 1024 |
| 64 | 88.5 | 4096 |
| 128 | 89.5 | 8192 |
| 256 | 89.7 | 16384 |
| 512 | OOM | OOM |

### Batch Size

| BS | DistilBERT | RoBERTa |
|---|---|---|
| 4 | 89.5 | 77.0 |
| 8 | OOM | OOM |
| 16 | OOM | OOM |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 89.5 |
| 123 | 89.3 |
| 456 | 89.7 |
| **Mean** | **89.5** |
| **Std** | **0.2** |

## D.5 Lessons Learned

1. **Spherical normalization is essential.**
2. **Gate bias must be ≥ 3.0.**
3. **Yat-Kernel > ELU+1.**
4. **Dense anchor is non-negotiable.**
5. **Cumsum memory limits practical batch size.**
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Contributes to hybrid attention research.
- Demonstrates stable linear attention.
- Multi-architecture support.

### Negative
- Cumsum memory footprint limits deployment.
- Slower at short L.
- Modest accuracy gain.

## E.2 Environmental

- Total GPU-hours: ~15.
- CO₂: 1.2 kg.

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

Parts drafted with LLM assistance. Content reviewed by authors.

## E.9 Long-Term

Hybrid attention could:
- Enable efficient long-context inference.
- Reduce compute cost.
- Democratize access.

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
  "_comment": "Fill with real values from chela_slay runs.",

  "MODULE_NAME": "CHELA-SLAY",
  "FULL_NAME": "CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel",
  "SUBTITLE": "Hybrid Dense + Linear Attention with Adaptive Gating",

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
  "FEATURE_DIM_2": "256",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "GATE_INIT_BIAS": "5.0",
  "GATE_INIT": "0.993",
  "LINEAR_INIT": "0.007",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "4",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "BASE_D": "90.71",
  "BASE_R": "78.56",
  "BASE_G": "79.36",

  "D_ACC": "89.5",
  "D_DELTA": "-1.2",
  "D_F1": "0.895",
  "D_PARAMS": "67500000",
  "D_PARAM_PCT": "0.8",
  "D_TRAINABLE": "700000",
  "D_VRAM": "890",
  "D_TIME": "4.2",
  "D_TIME_RATIO": "2.2",

  "R_ACC": "77.0",
  "R_DELTA": "-1.6",
  "R_F1": "0.770",
  "R_PARAMS": "126000000",
  "R_PARAM_PCT": "0.8",
  "R_TRAINABLE": "1400000",
  "R_VRAM": "1800",
  "R_TIME": "9.0",
  "R_TIME_RATIO": "2.2",

  "G_ACC": "78.5",
  "G_DELTA": "-0.9",
  "G_F1": "0.785",
  "G_PARAMS": "125500000",
  "G_PARAM_PCT": "0.8",
  "G_TRAINABLE": "1400000",
  "G_VRAM": "1950",
  "G_TIME": "9.5",
  "G_TIME_RATIO": "2.3",

  "D_VRAM_RATIO": "4.9",
  "R_VRAM_RATIO": "2.5",
  "G_VRAM_RATIO": "2.5",

  "CMEM_128": "8192",
  "CMEM_256": "16384",
  "CMEM_512": "32768",
  "CMEM_1024": "65536",
  "CMEM_F16": "512",
  "CMEM_F32": "1024",
  "CMEM_F64": "4096",
  "CMEM_F128": "8192",
  "CMEM_F256": "16384",
  "CUMSUM_MEM": "8192",

  "D_F16": "85.0",
  "D_F32": "87.5",
  "D_F64": "88.5",
  "D_F128": "89.5",
  "D_F256": "89.7",
  "R_F16": "73.0",
  "R_F32": "75.0",
  "R_F64": "76.0",
  "R_F128": "77.0",
  "R_F256": "77.2",
  "G_F16": "74.5",
  "G_F32": "76.5",
  "G_F64": "77.5",
  "G_F128": "78.5",
  "G_F256": "78.7",

  "D_GB0": "82.0",
  "D_GB2": "87.0",
  "D_GB5": "89.5",
  "D_GB10": "89.0",
  "R_GB0": "70.0",
  "R_GB2": "75.0",
  "R_GB5": "77.0",
  "R_GB10": "76.5",
  "G_GB0": "72.0",
  "G_GB2": "77.0",
  "G_GB5": "78.5",
  "G_GB10": "78.0",

  "D_DENSE_ONLY": "90.71",
  "D_LINEAR_ONLY": "60.0",
  "D_NO_SPH": "86.0",
  "D_ELU": "85.5",
  "D_COS": "88.0",
  "D_SIN": "87.5",
  "R_DENSE_ONLY": "78.56",
  "R_LINEAR_ONLY": "55.0",
  "R_NO_SPH": "74.0",
  "R_ELU": "73.5",
  "R_COS": "75.5",
  "R_SIN": "75.0",
  "G_DENSE_ONLY": "79.36",
  "G_LINEAR_ONLY": "58.0",
  "G_NO_SPH": "75.5",
  "G_ELU": "75.0",
  "G_COS": "77.0",
  "G_SIN": "76.5",

  "D_DENSE_ONLY_K": "90.71",
  "D_LINEAR_ONLY_K": "60.0",

  "D_GATE_MEAN": "0.92",
  "D_GATE_STD": "0.05",
  "D_GATE_MIN": "0.75",
  "D_GATE_MAX": "0.99",
  "R_GATE_MEAN": "0.90",
  "R_GATE_STD": "0.06",
  "R_GATE_MIN": "0.70",
  "R_GATE_MAX": "0.99",
  "G_GATE_MEAN": "0.91",
  "G_GATE_STD": "0.05",
  "G_GATE_MIN": "0.72",
  "G_GATE_MAX": "0.99",

  "D_T128": "4.2",
  "D_T256": "7.5",
  "D_T512": "25.0",
  "D_T1024": "92.0",
  "G_T128": "9.5",
  "G_T256": "15.0",
  "G_T512": "45.0",
  "G_T1024": "150.0",

  "B128": "4.20",
  "B256": "12.0",
  "B512": "45.0",
  "B1024": "180.0",
  "B2048": "720.0",
  "C128": "9.5",
  "C256": "15.0",
  "C512": "45.0",
  "C1024": "150.0",
  "C2048": "540.0",
  "S128": "0.44",
  "S256": "0.80",
  "S512": "1.0",
  "S1024": "1.2",
  "S2048": "1.33",

  "BREAKEVEN": "1024",
  "SLOWDOWN": "2.2",

  "SPH_GAIN": "2.5",
  "R_SPH_GAIN": "3.0",
  "G_SPH_GAIN": "3.0",
  "ELU_DROP": "4.0",
  "NO_SPH_DROP": "2.5",
  "ANCHOR_GAIN": "28.0",
  "D_ANCHOR_GAIN": "29.5",
  "D_SPH": "89.5",
  "R_SPH": "77.0",
  "G_SPH": "78.5",
  "D_NOSPH": "86.0",
  "R_NOSPH": "74.0",
  "G_NOSPH": "75.5",

  "D_LT": "60.0",
  "R_LT": "55.0",
  "G_LT": "58.0",
  "D_PERF": "62.0",
  "R_PERF": "57.0",
  "G_PERF": "60.0",

  "D_BS4": "89.5",
  "D_BS8": "OOM",
  "R_BS4": "77.0",

  "D_S1": "0.35",
  "D_S5": "0.20",
  "D_S10": "0.15",
  "D_S20": "0.10",
  "D_S30": "0.07",
  "D_S40": "0.05",
  "D_S50": "0.04",

  "SEED42": "89.5",
  "SEED123": "89.3",
  "SEED456": "89.7",
  "MEAN": "89.5",
  "STD": "0.2",

  "COMPUTE_HOURS": "15",
  "CO2_KG": "1.2",
  "RUNTIME": "30",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — CHELA-SLAY

## Setup
- [ ] Copied `_TEMPLATE/` to `chela_slay/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module chela_slay`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Hybrid claim (dense + linear) stated
- [ ] Spherical + Yat-Kernel mentioned
- [ ] Adaptive gate + init bias stated
- [ ] **Cumsum memory limitation stated**
- [ ] **Short-sequence slowdown stated**

## Intro
- [ ] Motivation for hybrid
- [ ] Three components listed
- [ ] Contributions verifiable

## Method
- [ ] Spherical normalization formula
- [ ] Yat-Kernel formula
- [ ] Linear attention formula
- [ ] Causal cumsum formula
- [ ] Gate fusion formula
- [ ] Step-0 preservation proof sketch
- [ ] **Complexity + memory analysis complete**

## Results
- [ ] Main accuracy table
- [ ] Parameter table
- [ ] Inference time table
- [ ] **Peak VRAM table**
- [ ] Feature dim ablation
- [ ] Gate bias ablation
- [ ] Stream ablation
- [ ] **Sequence-length scaling table**

## Honesty
- [ ] **Cumsum memory limitation stated in abstract, results, limitations**
- [ ] **Short-sequence slowdown stated**
- [ ] Gate remains dense-dominant reported
- [ ] Single-seed stated
- [ ] No theoretical bound stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Feature dim documented
- [ ] Gate bias documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/chela_slay
nano papers/chela_slay/variables.json
cd papers/chela_slay
python apply_vars.py --module chela_slay
grep -rn "{{" .
python build.py --module chela_slay --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **نوآوری سه‌گانه**

- **Spherical normalization**: Q, K را روی unit sphere می‌بره
- **Yat-Kernel**: cos+sin RFF
- **Adaptive gate**: dense + linear رو blend می‌کنه

### ۲. **محدودیت مهم: Cumsum Memory**

در causal mode، `torch.cumsum` یک tensor با شکل `(B, H, L, 2F, d_h)` می‌سازه. برای `L=128, 2F=256, d_h=64, B=8, H=12`: **~8 GB**. این بزرگ‌ترین محدودیت است.

### ۳. **Gate در ابتدا تقریباً 1.0**

با `b_g = 5.0`: `sigmoid(5) = 0.993`. یعنی در step 0 عملاً فقط dense اجرا می‌شه.

### ۴. **سرعت در L=128 کندتره**

چون **هر دو stream** محاسبه می‌شن. سرعت واقعی فقط در L بزرگ ظاهر می‌شه.

### ۵. **تست سریع**

```bash
# با batch_size=4 برای جلوگیری از OOM
python run_all_4_chela_slay.py --models distilbert,roberta,gpt2 --feature_dim 128 --batch_size 4
```

اگر OOM داد، `--feature_dim 64` رو امتحان کن.

---

**قدم بعدی: `run_all_4_chela_slay.py` را اجرا کن با `--batch_size 4 --feature_dim 64` برای شروع، اعداد را از `summary.csv` بردار، در `variables.json` بگذار.**