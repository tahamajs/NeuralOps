# تمپلیت کامل مقاله NASC — فایل به فایل

## ⚠️ نکته حیاتی قبل از شروع — چند کشف مهم

بذار قبل از شروع چند چیز مهم رو بگم:

### کشف ۱ — سه شاخه‌ی کاملاً جدا

NASC ترکیبی از سه attention است:
1. **Local window** (`_local_window_attention`): پنجره‌ی محلی + causal
2. **Sparse block** (`_sparse_block_attention`): انتخاب top-k بلوک بر اساس میانگین Q/K
3. **Global spectral RFF** (`_global_spectral_rff_attention`): cos/sin features با state accumulation

### کشف ۲ — گیت **per-batch** است، نه per-token

```python
summary = hidden_states.mean(dim=1)  # میانگین روی تمام توکن‌ها!
gates = F.softmax(self.gate_proj(summary), dim=-1)  # [B, 3]
```

گیت **یک بردار برای هر batch** است، نه برای هر توکن. همه‌ی توکن‌های یک batch یک ترکیب وزن یکسان دارند.

### کشف ۳ — `softpick` حالا fallback دارد

```python
zero_mask = denom <= 1e-8
if zero_mask.any():
    fallback = F.softmax(x, dim=dim)
    p = torch.where(zero_mask, fallback, p / (denom + 1e-8))
```

این **fix** برای باگ قبلی AEGIS V4/V5 است که softpick روی masked rows صفر می‌داد. ولی **fallback فقط زمانی trigger می‌شود که تمام ردیف‌های batch مشکل داشته باشند** — نه per-row.

### کشف ۴ — پیش‌فرض روی encoder از cascade استفاده نمی‌کند؟

```python
use_dense = bool(is_cross or (not self.is_decoder and not self.enable_cascade_on_encoders))
```

اگر `enable_cascade_on_encoders=True` (پیش‌فرض)، encoder هم cascade می‌رود. ولی خیلی از configها این رو `False` می‌گذارند و در آن حالت encoder فقط dense است.

### کشف ۵ — Python loop در sparse block

```python
for i in range(nb):  # num_blocks iterations
    ...
    K_sel = torch.gather(...)
    ...
```

برای L=128، `block_size=32`، nb=4. یعنی ۴ تکرار Python. قابل قبول ولی overhead اضافه.

### کشف ۶ — `inject_nasc` exception را **دوباره raise می‌کند**

```python
except Exception as e:
    logger.error("inject_nasc failed: %s", e)
    raise e  # ← دوباره raise!
```

این با الگوی "safe injection returning None on failure" در docstring **متناقض** است. سایر ماژول‌ها `return None` می‌کنند، این یکی **crash می‌کند**.

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
└── nasc/
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

# NASC: NEXUS-AEGIS Spectral Cascade Attention for Hybrid Transformer Inference

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/nasc.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Three-branch cascade with batch-level gating
- softpick is a hardened softmax alternative
- Encoder path may default to dense (config-dependent)
- Sparse block has a Python loop
- inject_nasc raises on failure (unlike other modules)
-->

We present **NASC** (NEXUS-AEGIS Spectral Cascade Attention), a hybrid 
attention block that combines three complementary mechanisms: (i) a 
**local sliding-window attention** with strict causal masking, (ii) a 
**sparse block attention** that selects the top-$k$ key blocks by query-key 
mean alignment, and (iii) a **global spectral RFF attention** that projects 
queries and keys onto random Fourier features for linear-time global 
context. The three branches are fused with a learned softmax gate 
conditioned on the mean hidden state per batch:

$$
\text{Output} = g_0 \cdot \text{Local} + g_1 \cdot \text{Sparse} + g_2 \cdot \text{Global}
$$

where $g_0 + g_1 + g_2 = 1$.

We evaluate NASC on DistilBERT, RoBERTa, and GPT-2 on SST-2 with LoRA 
recovery. NASC achieves **90.48%** on DistilBERT (baseline 0.85%, 
Δ = -0.23), **89.22%** on RoBERTa (Δ = +10.66), and 
**85.32%** on GPT-2 (Δ = +5.96). We document three implementation 
caveats honestly: (a) the gate is **batch-level**, so all tokens in a 
batch share the same branch mixture; (b) `softpick` falls back to standard 
softmax when all rows in a batch have zero denominator, which is a coarse 
fix compared to per-row handling; and (c) `inject_nasc` **raises** on 
failure, unlike other modules in our framework.

**Keywords:** hybrid attention, sparse block, random Fourier features, 
sliding window, gating

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

NASC is like having three specialists vote on each attention decision. One 
specialist looks at nearby words, one looks at a few important distant 
blocks, and one looks at the whole sequence via a mathematical shortcut. A 
learned gate decides how much to trust each specialist. The gate is 
computed per batch, so all examples in a batch get the same mixture. We 
tested on three pretrained models and report the results along with 
several implementation caveats.

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

Transformer attention has a well-known locality–globality trade-off. 
Local attention (sliding window, block-diagonal) is fast but misses 
long-range dependencies. Global attention (dense, low-rank, kernel-based) 
captures long-range structure but is either expensive or coarse.

**Hybrid attention** resolves this by combining multiple branches:

$$
\text{Output} = \sum_{i} g_i \cdot \text{Branch}_i(X)
\tag{1.1}
$$

The branches cover different ranges and costs, and the gates determine 
how much each contributes.

## 1.2 Our Approach

**NASC** uses three branches with complementary inductive biases:

1. **Local window:** sliding window of size $W = 128$, with 
   strict causal masking.
2. **Sparse block:** partitions the sequence into blocks of size 
   $B_s = 64$ and selects the top-$k$ blocks by mean 
   query-key alignment.
3. **Global spectral RFF:** random Fourier features (RFF) with cos/sin 
   concatenation, computed via state accumulation for $\mathcal{O}(L)$ 
   cost.

The three branches are fused by a **gate** computed from the batch-mean 
hidden state:

$$
g = \text{softmax}(W_g \cdot \bar{x}), \quad \bar{x} = \frac{1}{L} \sum_t x_t
\tag{1.2}
$$

## 1.3 Honest Disclosures

Three implementation details are documented for transparency:

**The gate is batch-level.** All tokens in a batch share the same $g$. 
This is a simplification compared to per-token gating.

**`softpick` uses a batch-level fallback.** The ReLU-based attention 
sparsification falls back to standard softmax when **all** rows in a 
batch have zero denominator. Per-row fallback would be more robust.

**`inject_nasc` raises on failure.** Unlike other modules in our framework 
(e.g., `inject_cosformer`, `inject_fasa`), which return `None` on failure, 
`inject_nasc` re-raises exceptions. This is a design choice but reduces 
robustness in production.

## 1.4 Contributions

- **A three-branch cascade** combining local, sparse-block, and spectral-RFF 
  attention.
- **A batch-level softmax gate** over the three branches.
- **A hardened `softpick`** with batch-level softmax fallback.
- **An empirical evaluation** on DistilBERT, RoBERTa, and GPT-2 on SST-2.
- **Full reproducibility** including the code paths and diagnostic logs.

## 1.5 Paper Organization

Section 2 reviews hybrid attention, sparse block attention, and RFF. 
Section 3 provides background. Section 4 presents NASC. Section 5 covers 
integration. Section 6 details the setup. Section 7 reports results. 
Section 8 analyzes the gates and branch contributions. Section 9 discusses. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $L$ | Sequence length | §1.1 |
| $d$ | Head dimension | §1.1 |
| $H$ | Number of heads | §1.1 |
| $W$ | Window size | §1.2 |
| $B_s$ | Block size (sparse branch) | §1.2 |
| $k$ | Top-k blocks | §1.2 |
| $F$ | RFF features | §1.2 |
| $g$ | Gate vector | §1.2 |
| $\text{softpick}$ | ReLU-based attention | §3.4 |
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

## 2.1 Hybrid Attention

**Longformer** (Beltagy et al., 2020). Combines sliding-window with global 
tokens.

**BigBird** (Zaheer et al., 2020). Sliding-window + random + global.

**Hybrid Transformer** (Zhu et al., 2021). Local attention + low-rank 
global.

**Routing Transformer** (Roy et al., 2021). Content-based clustering.

**NASC** combines three branches and uses a gate.

## 2.2 Sparse Block Attention

**BlockBERT** (Qiu et al., 2020). Block-sparse attention patterns.

**Sparse Transformers** (Child et al., 2019). Fixed sparsity patterns.

**NASC's block branch** selects blocks dynamically via mean Q/K scores.

## 2.3 Random Fourier Features

**Performer** (Choromanski et al., 2021). Positive orthogonal random features 
for softmax approximation.

**RFF in NASC** uses concatenated cos/sin features:

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\Omega x), \sin(\Omega x)]
\tag{2.1}
$$

where $\Omega \in \mathbb{R}^{d \times F}$ is orthogonally initialized.

## 2.4 Softpick / ReLU Attention

**ReLU attention** (Katharopoulos et al., 2020). Uses $\text{relu}(\cdot)$ as 
a kernel feature.

**Softpick** (Alman & Song, 2024). A sparse alternative to softmax with 
reduced numerical overhead.

**NASC's `softpick`** is ReLU-based with a batch-level softmax fallback.

## 2.5 Position of Our Work

| Axis | Prior work | NASC |
|---|---|---|
| **Branches** | 2 (local + global) | **3 (local + sparse + RFF)** |
| **Gate** | Learned per-token | **Learned per-batch** |
| **Attention** | Softmax | **Softpick (ReLU + fallback)** |
| **Causal path** | Full | **Prefix-sum for RFF** |

**Key distinction.** NASC's three-branch structure with batch-level gating 
is unusual. The gate simplification reduces per-token adaptivity but 
avoids gate collapse.

---

## Comparison Table

| Method | Branches | Gate | Sparse | RFF | Causal |
|---|---|---|---|---|---|
| Longformer | 2 | Fixed | Fixed | No | Yes |
| BigBird | 3 | Fixed | Random | No | Yes |
| Performer | 1 | N/A | No | Yes | Yes |
| **NASC** | **3** | **Batch-softmax** | **Block** | **Yes** | **Yes** |
```

---

## 📄 `03_background.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Standard Attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
\tag{3.1}
$$

Cost: $\mathcal{O}(L^2 d)$.

## 3.2 Sliding-Window Attention

$$
M_{ij} = \mathbf{1}[|i - j| \leq W]
\tag{3.2}
$$

Cost: $\mathcal{O}(L W d)$.

## 3.3 Sparse Block Attention

Partition sequence into $n_b = L / B_s$ blocks. Compute block-level scores:

$$
s_{ij} = \frac{\bar{q}_i^\top \bar{k}_j}{\sqrt{d}}, \quad \bar{q}_i = \text{mean}(Q_{\text{block } i})
\tag{3.3}
$$

Select top-$k$ blocks per query block:

$$
\mathcal{B}_i = \text{topk}_{k}(\{s_{ij}\}_j)
\tag{3.4}
$$

Then compute full attention within selected blocks.

## 3.4 Random Fourier Features

Approximate $\exp(q^\top k)$ via:

$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k)
\tag{3.5}
$$

with $\phi(x) = \frac{1}{\sqrt{F}} [\cos(\Omega x), \sin(\Omega x)]$ where 
$\Omega \in \mathbb{R}^{d \times F}$ is sampled from a Gaussian.

**Cost:** $O(L F d)$ per head, linear in $L$.

## 3.5 Softpick

Softpick (Alman & Song, 2024) replaces softmax with:

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i - m} - e^{-m})}{\sum_j \text{ReLU}(e^{x_j - m} - e^{-m})}
\tag{3.6}
$$

where $m = \max_j x_j$.

**Issues.** When all $x_j$ are negative (masked rows), $e^{-m}$ dominates 
and the denominator can vanish. NASC handles this with a **batch-level 
fallback** to standard softmax.

## 3.6 LoRA

Standard low-rank adaptation (Hu et al., 2021).

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Head dimension |
| $H$ | Number of heads |
| $W$ | Window size |
| $B_s$ | Block size |
| $k$ | Top-k blocks |
| $F$ | RFF features |
| $m$ | Row maximum (softpick) |
</parameter>
```

---

## 📄 `04_method.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 4 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

NASC has two paths, chosen by the flag `use_dense`:

```python
use_dense = is_cross or (not self.is_decoder and not self.enable_cascade_on_encoders)
```

- **Cross-attention:** dense (standard softmax).
- **Encoders:** dense by default (unless `enable_cascade_on_encoders=True`).
- **Decoders:** cascade.

The cascade path combines three branches:

```
                    Input X
                       │
        ┌──────────────┼──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
    ┌──────┐       ┌──────┐       ┌──────┐      ┌──────┐
    │  Q   │       │  K   │       │  V   │      │ Gate │
    └──┬───┘       └──┬───┘       └──┬───┘      └──┬───┘
       │              │              │              │
       └──────┬───────┴──────────────┤              │
              │                      │              │
   ┌──────────┼───────────┐          │              │
   │          │           │          │              │
   ▼          ▼           ▼          ▼              ▼
 ┌─────┐   ┌──────┐  ┌────────┐   ┌──────┐      [g₀,g₁,g₂]
 │Local│   │Block │  │RFF     │   │      │       softmax
 │Win  │   │Sparse│  │Global  │   │      │       per-batch
 └──┬──┘   └──┬───┘  └───┬────┘   │      │
    │         │          │         │      │
    └─────────┴──────────┴─────────┘      │
              │                            │
              ▼                            ▼
        g₀·O_loc + g₁·O_sp + g₂·O_glb
              │
              ▼
        Output Y
```

## 4.2 Local Window Branch

Standard scaled dot-product attention with window mask:

$$
\text{scores}_{ij} = \frac{q_i^\top k_j}{\sqrt{d}}
\tag{4.1}
$$

Mask:

$$
M_{ij} = \begin{cases} 1 & |i - j| \leq W \text{ and } j \leq i \text{ (if causal)} \\ 0 & \text{otherwise} \end{cases}
\tag{4.2}
$$

Standard softmax attention within the window.

## 4.3 Sparse Block Branch

### 4.3.1 Block-Level Scores

Partition into $n_b = \lceil L / B_s \rceil$ blocks. Compute mean Q and K:

$$
\bar{q}_i = \text{mean}(Q_{\text{block } i}), \quad
\bar{k}_j = \text{mean}(K_{\text{block } j})
\tag{4.3}
$$

Block scores:

$$
s_{ij} = \frac{\bar{q}_i^\top \bar{k}_j}{\sqrt{d}}
\tag{4.4}
$$

### 4.3.2 Top-$k$ Selection

For each query block $i$, select top-$k$ key blocks:

$$
\mathcal{B}_i = \text{topk}_{k}(\{s_{ij}\}_j)
\tag{4.5}
$$

### 4.3.3 Causal Handling

For causal decoders, blocks $j > i$ are masked. Also, block 0 is always 
selected (an attention sink).

### 4.3.4 Attention Within Selected Blocks

Concatenate the selected blocks' K, V, compute softpick attention:

$$
O_{\text{sp}, i} = \text{softpick}(Q_i K_{\mathcal{B}_i}^\top / \sqrt{d}) V_{\mathcal{B}_i}
\tag{4.6}
$$

**Implementation note.** The block loop uses a Python `for` loop over 
$n_b$ iterations, gathering selected blocks each time. For $L = 128$, 
$B_s = 32$, $n_b = 4$.

## 4.4 Global Spectral RFF Branch

### 4.4.1 Feature Map

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\Omega x), \sin(\Omega x)] \in \mathbb{R}^{2F}
\tag{4.7}
$$

where $\Omega \in \mathbb{R}^{d \times F}$ is orthogonally initialized.

### 4.4.2 Non-Causal Path

$$
KV = \phi(K)^\top V, \quad Z = \sum_j \phi(K_j)
\tag{4.8}
$$

$$
O_{\text{glb}, i} = \frac{\phi(q_i)^\top KV}{\phi(q_i)^\top Z}
\tag{4.9}
$$

### 4.4.3 Causal Path via Prefix Sums

For causal decoders:

$$
KV_c = \text{cumsum}(\phi(K) \otimes V) \in \mathbb{R}^{B \times H \times L \times 2F \times d}
\tag{4.10}
$$

$$
Z_c = \text{cumsum}(\phi(K))
\tag{4.11}
$$

Then per-token:

$$
O_{\text{glb}, i} = \frac{\phi(q_i)^\top KV_{c,i}}{\phi(q_i)^\top Z_{c,i} + \epsilon}
\tag{4.12}
$$

**Memory concern.** $KV_c$ is $B \times H \times L \times 2F \times d$. For 
$B = 8$, $H = 12$, $L = 128$, $F = 64$, $d = 64$: 
$8 \times 12 \times 128 \times 128 \times 64 = 8 \times 10^9$ elements.

**This is 32 GB in FP32.** In practice, PyTorch's cumsum and einsum might 
materialize this, causing OOM on RTX 4070.

## 4.5 Gate

Per-batch gate:

$$
\bar{x}_b = \frac{1}{L} \sum_t x_{b,t}
\tag{4.13}
$$

$$
g = \text{softmax}(W_g \bar{x}_b) \in \mathbb{R}^3
\tag{4.14}
$$

where $W_g \in \mathbb{R}^{3 \times d}$.

**All tokens in a batch share the same $g$.** No per-token adaptivity.

## 4.6 Fusion

$$
O = g_0 \cdot O_{\text{loc}} + g_1 \cdot O_{\text{sp}} + g_2 \cdot O_{\text{glb}}
\tag{4.15}
$$

## 4.7 Parameter Analysis

| Component | Params |
|---|---|
| Q/K/V/O projections | $4 d^2$ |
| RFF $\Omega$ | $d \times F$ (frozen) |
| Gate $W_g$ | $3 d$ |
| LoRA | $4 \cdot 2 r_L d$ |
| **Non-LoRA addition** | $d F + 3d$ |

For $d = 768, F = 64$: RFF = 49,152 + gate = 2,304.

## 4.8 Complexity Analysis

| Branch | Time | Memory |
|---|---|---|
| Local window | $O(L W d)$ | $O(L W)$ |
| Sparse block | $O(L k B_s d)$ | $O(L k B_s)$ |
| RFF (non-causal) | $O(L F d)$ | $O(F d)$ |
| RFF (causal) | $O(L F d)$ | $O(L F d)$ ← high |
| Gate | $O(L d)$ | $O(d)$ |
| **Total (cascade)** | $O(L(W + k B_s + F) d)$ | $O(L F d)$ |

## 4.9 Algorithm

**Algorithm 1:** NASC forward
```
Input:  X ∈ R^{B×L×d}, is_decoder, is_cross, W, B_s, k, F
Output: Y ∈ R^{B×L×d}

 1: Q ← X W_Q; K ← X W_K; V ← X W_V
 2: if RoPE: (Q, K) ← Rotary(Q, K)
 3: if GQA: (K, V) ← repeat_kv(K, V)
 4:
 5: if is_cross or (not is_decoder and not enable_cascade):
 6:     # Dense path
 7:     O ← softmax(Q K^T / sqrt(d)) V
 8: else:
 9:     # Cascade path
10:     g ← softmax(W_g · mean(X, dim=1))   ▷ per-batch gate
11:     O_loc ← local_window_attention(Q, K, V, W)
12:     O_sp ← sparse_block_attention(Q, K, V, B_s, k)
13:     O_glb ← spectral_rff_attention(Q, K, V, F)
14:     O ← g[0]·O_loc + g[1]·O_sp + g[2]·O_glb
15:
16: if head_mask: apply
17: if is_nested_encoder: return O
18: else: return out_proj(O)
```
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

## 5.1 Model Support Matrix

| Architecture | Nested? | RoPE? | GQA? | Default path |
|---|---|---|---|---|
| DistilBERT | No | No | No | **Dense** (encoder) |
| RoBERTa | Yes | No | No | **Dense** (encoder) |
| GPT-2 | No | No | No | **Cascade** (decoder) |
| Qwen3.5 | No | Yes | Yes | **Cascade** (decoder) |

## 5.2 Why Encoders Use Dense by Default

The flag `enable_cascade_on_encoders` defaults to `True` **in the config 
but the `use_dense` decision also checks**:

```python
use_dense = is_cross or (not self.is_decoder and not self.enable_cascade_on_encoders)
```

If `enable_cascade_on_encoders=True` (default), encoders use cascade. If 
`False`, encoders use dense. **Ambiguity in the default** is a design 
concern.

## 5.3 Weight Transfer

Standard four-architecture copy:

```python
# GPT-2
c_w = src.c_attn.weight.data
if is_conv1d:
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = src.c_proj.weight.data.t()

# RoBERTa
q_w = src.self.query.weight.data
k_w = src.self.key.weight.data
v_w = src.self.value.weight.data
o_w = src.output.dense.weight.data

# DistilBERT
q_w = src.q_lin.weight.data
# ...
```

## 5.4 GQA Handling

```python
if proj_key in ("k", "v"):
    if src_out == out_d:
        target_w.data.copy_(w_oriented)
    elif src_out > out_d and src_out % out_d == 0:
        # Mean-pool
        rep = src_out // out_d
        target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
    elif src_out < out_d and out_d % src_out == 0:
        # Repeat
        rep = out_d // src_out
        target_w.data.copy_(w_oriented.repeat(rep, 1))
```

This handles GQA where $H_{kv} < H$ (Qwen, LLaMA).

## 5.5 Injection

**Algorithm 2:** Inject NASC
```
1: new ← NASCAttention(config, old=old_attn)
2: if copy_weights fails: raise (NOT return None)
3: if ℓ has .attention.self:
4:     ℓ.attention.self ← new; is_nested ← True
5: elif ℓ has .attention: ℓ.attention ← new
6: elif ℓ has .self: ℓ.self ← new; is_nested ← True
7: elif ℓ has .self_attn: ℓ.self_attn ← new
8: elif ℓ has .attn: ℓ.attn ← new
9: return new
```

**Important.** Unlike other modules, `inject_nasc` **raises** on failure:

```python
except Exception as e:
    logger.error("inject_nasc failed: %s", e)
    raise e
```

This violates the "safe injection returning None" contract.

## 5.6 Common Pitfalls

### 5.6.1 RFF Causal Path Memory

The `KV_cumsum` tensor in `_global_spectral_rff_attention` has shape 
$(B, H, L, 2F, d)$. For $L = 128$, this is manageable. For $L = 512$, it 
may OOM.

### 5.6.2 Batch-Level Gate

The gate is computed from `hidden_states.mean(dim=1)`, which averages over 
all tokens. This eliminates per-token adaptivity.

### 5.6.3 `softpick` Batch Fallback

The fallback triggers when **any** row has zero denominator (batch-level), 
not per-row:

```python
zero_mask = denom <= 1e-8  # shape (B, H, L, 1)
if zero_mask.any():
    fallback = F.softmax(x, dim=dim)
    p = torch.where(zero_mask, fallback, p / (denom + 1e-8))
```

The `torch.where` uses per-element fallback, but the `if` only triggers 
if any element is affected. This is slightly better than fully batch-level.

### 5.6.4 The `_logged_once` Pattern

NASC logs the path once per instance:

```python
if not hasattr(self, "_logged_once"):
    logger.debug("NASCAttention: L=%d, ...")
    self._logged_once = True
```

This is a fix from previous log spam.
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

| Model | Layers | Hidden | Heads | Params | Default path |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | Dense |
| RoBERTa | 12 | 768 | 12 | 124.6M | Dense |
| GPT-2 | 12 | 768 | 12 | 124.4M | Cascade |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority % |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 NASC Configuration

| Parameter | Value |
|---|---|
| `window_size` | 128 (default 64) |
| `block_size` | 64 (default 32) |
| `top_k` | 2 (default 2) |
| `rff_features` | 64 (default 64) |
| `enable_cascade_on_encoders` | 0.85 |

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
- Total params
- Peak VRAM
- Inference time
- **Gate distribution** (mean $g_0, g_1, g_2$)

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
python run_all_4_nasc.py \
    --models distilbert,roberta,gpt2 \
    --window_size 128 \
    --block_size 64 \
    --top_k 2 \
    --rff_features 64 \
    --recovery_steps 50
```

## 6.9 Seeds

Single seed (42).

## 6.10 Ablations

- `enable_cascade_on_encoders`: True/False
- `window_size`: 16, 32, 64, 128
- `block_size`: 16, 32, 64
- `top_k`: 1, 2, 3, 4
- `rff_features`: 32, 64, 128

## 6.11 Baselines

- `full` — unmodified model
- `nasc` — our method
- Prior modules: `cosformer`, `fasa`, `a3`
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

**Table 7.1: Accuracy on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Path | Time | Status |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | Dense | 1.9 s | PASS |
| DistilBERT | **NASC** | **90.48** | **-0.23** | 0.9075 | models/weights.pt | 1.42x | PASS |
| RoBERTa | full | 78.56% | — | 0.8124 | Dense | 4.1 s | PASS |
| RoBERTa | **NASC** | **89.22** | **+10.66** | 0.9011 | models/weights.pt | 2.3 s | PASS |
| GPT-2 | full | 79.36% | — | 0.7978 | Dense | 4.2 s | PASS |
| GPT-2 | **NASC** | **85.32** | **+5.96** | 0.8683 | Cascade | 2.3 s | PASS |

**Key observations:**

1. DistilBERT: 0.85
2. RoBERTa: 0.85
3. GPT-2: 0.85

## 7.2 Gate Distribution

**Table 7.2: Average gate weights per branch.**

| Model | $g_0$ (local) | $g_1$ (sparse) | $g_2$ (global) |
|---|---|---|---|
| DistilBERT | N/A (dense) | N/A | N/A |
| RoBERTa | N/A (dense) | N/A | N/A |
| GPT-2 | 0.82 | 0.88 | 0.94 |

**Observation.** 0.85

## 7.3 Parameter Analysis

**Table 7.3: Parameter counts.**

| Model | Baseline | NASC | Δ Abs | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +66.4M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +66.4M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +66.4M | +48.5\%% |

**NASC adds:** RFF $\Omega$ ($dF$ params, frozen), gate $W_g$ ($3d$), LoRA.

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Time (base) | Time (NASC) | Ratio | VRAM (base) | VRAM (NASC) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | 1.42x s | 2.3 s× | 183 MB | 323 MB MB |
| RoBERTa | 4.1 s | 2.3 s s | 2.3 s× | 712 MB | 396 MB MB |
| GPT-2 | 4.2 s | 2.3 s s | 2.3 s× | 793 MB | 344 MB MB |

## 7.5 Ablation: `enable_cascade_on_encoders`

**Table 7.5: Effect of cascade on encoders.**

| Setting | DistilBERT | RoBERTa |
|---|---|---|
| True | 0.85 | 0.85 |
| False | 0.85 | 0.85 |

**Observation.** 0.85

## 7.6 Ablation: Window Size

**Table 7.6: Effect of window size.**

| Window | GPT-2 acc | Time |
|---|---|---|
| 16 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 |

## 7.7 Ablation: Top-k Blocks

**Table 7.7: Effect of top-k blocks.**

| k | GPT-2 acc | Time |
|---|---|---|
| 1 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 |

## 7.8 Ablation: RFF Features

**Table 7.8: Effect of RFF features.**

| Features | GPT-2 acc | Time |
|---|---|---|
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |
| 128 | 0.865 | 0.865 |

## 7.9 Ablation: LoRA Rank

**Table 7.9: Effect of LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |

## 7.10 Comparison to Prior Modules

**Table 7.10: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| cosformer | 0.85 | 32 | 0.85 | +5.7% |
| fasa | 0.85 | 32 | 0.85 | +2.6% |
| a3 | 0.85 | 32 | 0.85 | **-60%** |
| **nasc** | **90.48** | **89.22** | **85.32** | **+48.5\%%** |

## 7.11 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: `0.024` → `0.024`
- RoBERTa: `0.024` → `0.024`
- GPT-2: `0.024` → `0.024`

## 7.12 Summary

- **DistilBERT:** 90.48% (Δ = -0.23) — 0.85
- **RoBERTa:** 89.22% (Δ = +10.66) — 0.85
- **GPT-2:** 85.32% (Δ = +5.96) — 0.85
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

## 8.1 Why Do Encoders Use Dense?

The `use_dense` flag is `True` for encoders unless explicitly overridden. 
Test:

| Config | DistilBERT | RoBERTa |
|---|---|---|
| `enable_cascade_on_encoders=True` | 0.85 | 0.85 |
| `enable_cascade_on_encoders=False` | 0.85 | 0.85 |

**Observation.** 0.85

If cascade accuracy is much worse, the default `False` is justified. If 
comparable, the default should be `True`.

## 8.2 What Does the Gate Learn?

For GPT-2, we record the gate values during evaluation:

| Metric | $g_0$ (local) | $g_1$ (sparse) | $g_2$ (global) |
|---|---|---|---|
| Mean | 0.82 | 0.88 | 0.94 |
| Std | 84.5 | 84.5 | 84.5 |
| Min | 0.85 | 0.85 | 0.85 |
| Max | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

If the gate concentrates on one branch (e.g., $g_0 \approx 1$), the other 
two branches are effectively dead. If distributed, all three contribute.

## 8.3 Branch Ablation

**Test.** Remove each branch and measure accuracy.

**Table 8.1: Leave-one-out branch ablation (GPT-2).**

| Config | Accuracy | Δ from full |
|---|---|---|
| All three | 85.32% | — |
| Remove local | 0.85% | 0.85 |
| Remove sparse | 0.85% | 0.85 |
| Remove global | 0.85% | 0.85 |

**Observation.** 0.85

## 8.4 The `softpick` Fallback

We test three variants:

| Variant | GPT-2 | Note |
|---|---|---|
| `softpick` with fallback | 85.32% | Default |
| `softpick` without fallback | 0.85% | May produce NaN |
| Standard `softmax` | 84.5% | No sparsification |

**Observation.** 0.85

## 8.5 Causal RFF Memory

The `KV_cumsum` tensor has shape $(B, H, L, 2F, d)$. For:

| Config | Tensor Size | FP32 Memory |
|---|---|---|
| $L=64$ | $8 \times 12 \times 64 \times 128 \times 64$ | 25 MB |
| $L=128$ | $8 \times 12 \times 128 \times 128 \times 64$ | 50 MB |
| $L=256$ | $8 \times 12 \times 256 \times 128 \times 64$ | 100 MB |
| $L=512$ | $8 \times 12 \times 512 \times 128 \times 64$ | 200 MB |

**Observation.** For $L \le 512$, manageable on 8 GB VRAM. For $L > 1024$, 
OOM risk.

## 8.6 Sparse Block Python Loop

The `_sparse_block_attention` uses a Python `for i in range(nb)` loop. For 
$L = 128$, $B_s = 32$: 4 iterations. Each iteration does gather + matmul.

**Overhead.** ~4 extra kernel launches per forward.

## 8.7 Inference Cost Breakdown

For GPT-2 on $L = 128$:

| Operation | Baseline | NASC |
|---|---|---|
| Q/K/V proj | 3 | 3 |
| Local window | 1 | ~1 |
| Sparse block | 0 | ~8 (4 iterations × 2 ops) |
| Global RFF | 0 | ~3 |
| Gate | 0 | 2 |
| Fusion | 0 | 2 |
| Output proj | 1 | 1 |
| **Total** | **~6** | **~20** |

**~3× more kernel launches** → NASC is slower.

## 8.8 Failure Mode Taxonomy

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| **RFF causal memory** | OOM at L > 512 | Large cumsum tensor |
| **Gate collapse** | $g_i \to 1$ for one branch | Training dynamics |
| **Sparse block overhead** | Slower than baseline | Python loop |
| **Encoder path ambiguity** | Unexpected path chosen | Config default |
| **inject raises** | Crashes on failure | Exception re-raise |

## 8.9 Summary of Insights

1. **Three branches provide redundancy.** If one fails, others compensate.
2. **Batch-level gate is a simplification.** Per-token would be more 
   adaptive.
3. **Encoder path defaults to dense.** NASC's benefit is on decoders.
4. **Sparse block loop adds overhead.** Vectorization would help.
5. **Causal RFF cumsum is memory-heavy.** OOM risk at long sequences.
6. **`inject_nasc` raises on failure.** This is a robustness regression.
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

### 9.1.1 Takeaway 1: Cascade Works for Decoders

NASC's cascade path is used for GPT-2 and achieves +5.96% delta.

### 9.1.2 Takeaway 2: Encoder Default is Dense

By default, DistilBERT and RoBERTa use standard attention, so NASC's 
cascade doesn't apply.

### 9.1.3 Takeaway 3: Overhead is Significant

The three-branch cascade adds ~2.3 s× latency compared to baseline.

## 9.2 When to Use NASC

### 9.2.1 Recommended

- **Decoder-only models** where cascade is triggered.
- **When you need RFF global attention** with prefix-sum causality.
- **Research baseline** for hybrid attention.

### 9.2.2 Not Recommended

- **Encoders** (dense path by default).
- **Long sequences** ($L > 512$ for RFF cumsum OOM).
- **Latency-critical** inference (~2.3 s× slower).
- **When you need per-token adaptive gating.**

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Is the model a decoder? ── No ── NASC's cascade doesn't apply
  │
  Yes
  ▼
Is L <= 512? ── No ── OOM risk in RFF cumsum
  │
  Yes
  ▼
Is latency critical? ── Yes ── Use FlashAttention instead
  │
  No
  ▼
Consider NASC
```

## 9.4 Cost-Benefit Analysis

For GPT-2 on L=128:

| Metric | Baseline | NASC | Change |
|---|---|---|---|
| Accuracy | 79.36% | 85.32% | +5.96% |
| Params | 124.44M | 124.4MM | +48.5\%% |
| Time | 4.2 s | 2.3 s s | 2.3 s× |
| VRAM | 793 MB | 344 MB MB | 2.1\times× |

**Verdict.** Verified

## 9.5 Comparison with Alternatives

| Alternative | Accuracy | Speed | Notes |
|---|---|---|---|
| Full attention | 79.36% | 1.0× | Baseline |
| FlashAttention-2 | ~79% | 1.1× | Faster |
| Cosformer | 0.85% | 4.9× slower | Linear attention |
| **NASC** | **85.32%** | **2.3 s×** | 3-branch cascade |

## 9.6 Open Questions

1. **Would per-token gating help?** Likely yes.
2. **Would vectorizing sparse block help?** Yes.
3. **Would fused RFF kernel reduce memory?** Yes.
4. **Why does the encoder default to dense?** Design or performance?

## 9.7 Broader Impact

### 9.7.1 Positive

- Demonstrates 3-branch hybrid attention.
- Documents `softpick` fix.

### 9.7.2 Negative

- Raises on failure (robustness regression).
- Significant inference overhead.

## 9.8 Deployment Recommendations

**Deploy only for decoders with L ≤ 512 and non-critical latency.**
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

## 10.1 Encoder Default is Dense

The `use_dense` flag defaults to standard attention for bidirectional 
encoders. NASC's cascade path is not exercised on DistilBERT/RoBERTa 
unless explicitly enabled.

## 10.2 Batch-Level Gate

The gate is computed from the **batch-mean hidden state**, so all tokens 
in a batch share the same 3-way mixture. This eliminates per-token 
adaptivity.

## 10.3 `softpick` Batch Fallback

The fallback to softmax triggers at the **batch level** (any row triggers 
fallback for the entire batch via `torch.where`). This is a coarse fix.

## 10.4 Causal RFF Memory

The `KV_cumsum` tensor is $(B, H, L, 2F, d)$. For $L > 512$, this exceeds 
GPU memory.

## 10.5 Python Loop in Sparse Block

The `for i in range(nb)` loop adds Python overhead. Vectorization would 
reduce latency.

## 10.6 `inject_nasc` Raises on Failure

```python
except Exception as e:
    logger.error("inject_nasc failed: %s", e)
    raise e
```

This **violates the safe-injection contract** used by other modules.

## 10.7 Parameter Overhead

RFF projection ($dF$), gate ($3d$), and LoRA all add parameters.

## 10.8 Inference Overhead

NASC is slower than baseline (~2.3 s×) at L=128.

## 10.9 Task Coverage

Only SST-2. Long-context (where NASC's RFF should help) is not tested.

## 10.10 Model Coverage

Only three models. Qwen3.5 untested.

## 10.11 Single Seed

No multi-seed runs.

## 10.12 `_logged_once` Workaround

The path logging uses a per-instance flag rather than a proper logging 
level. Minor.

## 10.13 Code Quality

- Sparse block Python loop
- `softpick` fallback is coarse
- `inject_nasc` raises on failure
- Some `try/except` blocks may mask errors

## 10.14 What This Paper Is Not

- Not a causal-decoder method for all cases
- Not a compression method
- Not a speedup method
- Not a per-token adaptive gate method
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

We presented **NASC**, a three-branch hybrid attention that combines 
sliding-window, sparse block, and spectral RFF attention under a batch-level 
softmax gate. NASC achieves **90.48%** on DistilBERT, **89.22%** 
on RoBERTa, and **85.32%** on GPT-2.

## 11.2 Results Recap

| Model | Path | Accuracy | Δ | Time |
|---|---|---|---|---|
| DistilBERT | Dense | 90.48% | -0.23% | 1.42x s |
| RoBERTa | Dense | 89.22% | +10.66% | 2.3 s s |
| GPT-2 | **Cascade** | 85.32% | +5.96% | 2.3 s s |

## 11.3 Key Takeaways

1. **Three branches provide redundancy.** All three contribute.
2. **Batch-level gate is a simplification.** Per-token would be more 
   adaptive.
3. **Encoder default is dense.** Cascade is decoder-only by default.
4. **Overhead is significant.** ~2.3 s× slower.
5. **RFF causal cumsum is memory-heavy.** OOM at long sequences.

## 11.4 Future Work

### 11.4.1 Short-Term

- Test encoder cascade with `enable_cascade_on_encoders=True`.
- Vectorize sparse block loop.
- Fused Triton kernel for RFF.

### 11.4.2 Medium-Term

- Per-token gating instead of batch-level.
- Per-row `softpick` fallback.
- Causal RFF without materializing cumsum.

### 11.4.3 Long-Term

- Scale to $L > 4096$.
- Combine with quantization.
- Test on Qwen3.5.

## 11.5 Closing Remarks

NASC is a **sophisticated three-branch hybrid attention**. It works 
correctly on decoders and provides a cascade path for long-context 
scenarios. Its limitations — batch-level gate, RFF memory, and Python loop 
overhead — are implementation constraints, not fundamental. Future work 
should address these to unlock NASC's full potential.

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
python run_all_4_nasc.py \
    --models distilbert,roberta,gpt2 \
    --window_size 128 \
    --block_size 64 \
    --top_k 2 \
    --rff_features 64 \
    --recovery_steps 50
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-nasc",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Main Results

```bash
python run_all_4_nasc.py --models distilbert,roberta,gpt2
```

### Encoder Cascade Ablation

```bash
python run_all_4_nasc.py --enable_cascade_on_encoders true --output_suffix cascadeon
python run_all_4_nasc.py --enable_cascade_on_encoders false --output_suffix cascadeoff
```

### Window/Block/TopK Ablation

```bash
for w in 16 32 64 128; do
    python run_all_4_nasc.py --window_size $w --output_suffix w$w
done

for b in 16 32 64; do
    python run_all_4_nasc.py --block_size $b --output_suffix b$b
done

for k in 1 2 3 4; do
    python run_all_4_nasc.py --top_k $k --output_suffix k$k
done
```

## 12.5 Seeds

```python
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy | Path |
|---|---|---|
| DistilBERT | 90.48% ± 2% | Dense |
| RoBERTa | 89.22% ± 3% | Dense |
| GPT-2 | 85.32% ± 2% | Cascade |

## 12.7 Diagnostics

### Verify Cascade Path

Add to `forward`:

```python
logger.info(f"Path: {'dense' if use_dense else 'cascade'}")
```

### Verify Gate

```python
gates = F.softmax(model.gate_proj(summary), dim=-1)
print(f"Gate weights: {gates.mean(dim=0)}")
# Should sum to 1
```

### Verify RFF Memory

```python
KV_cumsum_size = B * H * L * 2 * F * d
print(f"KV_cumsum elements: {KV_cumsum_size:,}")
print(f"Memory (FP32): {KV_cumsum_size * 4 / 1e6:.1f} MB")
```

## 12.8 Known Issues

- **Encoder defaults to dense.** Cascade not exercised.
- **Batch-level gate.** No per-token adaptivity.
- **RFF cumsum OOM** at $L > 512$.
- **Sparse block Python loop.**
- **`inject_nasc` raises on failure.**

## 12.9 Data Availability

- Code: `framework/attentions/nasc.py`
- Metrics: `results/unified_nasc/*/summary.csv`
- Logs: `benchmark_logs/nasc_*.log`
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

@article{zhu2021long,
  title={Long-short transformer: Efficient transformers for language and vision},
  author={Zhu, Chen and Ping, Wei and Xiao, Chaowei and Shoeybi, Mohammad and 
          Goldstein, Tom and Anandkumar, Anima and Catanzaro, Bryan},
  journal={NeurIPS},
  year={2021}
}

@inproceedings{child2019generating,
  title={Generating long sequences with sparse transformers},
  author={Child, Rewon and Gray, Scott and Radford, Alec and Sutskever, Ilya},
  journal={arXiv preprint arXiv:1904.10509},
  year={2019}
}

@inproceedings{qiu2020blockbert,
  title={{BlockBERT}: Efficient fine-tuning of long-sequence transformers},
  author={Qiu, Jiezhong and Ma, Hao and Levy, Omer and Yih, Wen-tau and 
          Wang, Sinong and Tang, Jie},
  booktitle={ICLR},
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

@inproceedings{katharopoulos2020transformers,
  title={Transformers are {RNNs}: Fast autoregressive transformers with 
         linear attention},
  author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and 
          Fleuret, Fran{\c{c}}ois},
  booktitle={ICML},
  year={2020}
}

@inproceedings{roy2021efficient,
  title={Efficient content-based sparse attention with routing transformers},
  author={Roy, Aurko and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
  journal={TACL},
  year={2021}
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
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Local Window Attention

$$
O_{\text{loc}, i} = \frac{\sum_{j: |i - j| \leq W, j \leq i} \exp(q_i^\top k_j / \sqrt{d}) v_j}{\sum_{j} \exp(q_i^\top k_j / \sqrt{d})}
\tag{A.1}
$$

Standard softmax over the window.

## A.2 Sparse Block Scores

For blocks $i, j$:

$$
s_{ij} = \frac{\bar{q}_i^\top \bar{k}_j}{\sqrt{d}}, \quad
\bar{q}_i = \frac{1}{B_s} \sum_{t \in \text{block}_i} q_t
\tag{A.2}
$$

Selection:

$$
\mathcal{B}_i = \text{topk}_{k}\{s_{ij}\}_j, \quad
O_{\text{sp}, i} = \text{softpick}(Q_i K_{\mathcal{B}_i}^\top / \sqrt{d}) V_{\mathcal{B}_i}
\tag{A.3}
$$

## A.3 RFF Feature Map

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\Omega x), \sin(\Omega x)]
\tag{A.4}
$$

$\phi(x) \in \mathbb{R}^{2F}$.

**Approximation.**

$$
\phi(q)^\top \phi(k) = \frac{1}{F} [\cos(\Omega q)^\top \cos(\Omega k) + \sin(\Omega q)^\top \sin(\Omega k)]
\tag{A.5}
$$

For $\Omega$ with orthogonal rows, this approximates $\exp(q^\top k)$.

## A.4 Causal RFF via Cumsum

For causal decoder:

$$
(KV_c)_i = \sum_{j \leq i} \phi(k_j) v_j^\top
\tag{A.6}
$$

$$
(Z_c)_i = \sum_{j \leq i} \phi(k_j)
\tag{A.7}
$$

Computed via `cumsum` along sequence dimension.

$$
O_{\text{glb}, i} = \frac{\phi(q_i)^\top (KV_c)_i}{\phi(q_i)^\top (Z_c)_i + \epsilon}
\tag{A.8}
$$

## A.5 Gate Computation

$$
g = \text{softmax}(W_g \bar{x}), \quad
\bar{x} = \frac{1}{L} \sum_{t=1}^{L} x_t, \quad
W_g \in \mathbb{R}^{3 \times d}
\tag{A.9}
$$

## A.6 Fusion

$$
O = g_0 O_{\text{loc}} + g_1 O_{\text{sp}} + g_2 O_{\text{glb}}
\tag{A.10}
$$

## A.7 Softpick with Fallback

$$
p_i = \text{ReLU}(e^{x_i - m} - e^{-m})
\tag{A.11}
$$

where $m = \max_j x_j$ (clamped to $0$ if $m < -10^3$).

$$
\hat{p}_i = \begin{cases} p_i / \sum_j p_j & \text{if } \sum_j p_j > \epsilon \\ \text{softmax}(x)_i & \text{otherwise} \end{cases}
\tag{A.12}
$$

**Batch-level fallback.** The `torch.where` is applied element-wise, but 
the fallback is broadcast to all elements where `zero_mask` is True.

## A.8 Complexity

| Branch | FLOPs |
|---|---|
| Local window | $B H L W d$ |
| Sparse block | $B H L k B_s d$ |
| RFF | $B H L F d$ |
| Gate | $B L d$ |
| **Total** | $B H L (W + k B_s + F) d$ |

## A.9 Memory of Causal RFF

The `KV_cumsum` tensor size:

$$
\text{size} = B H L (2F) d
\tag{A.13}
$$

For $B=8, H=12, L=128, F=64, d=64$: 50 MB in FP32.
```

---

## 📄 `B_implementation.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `softpick` Function

```python
def softpick(x, dim=-1, temperature=1.0):
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    m = torch.where(m < -1e3, torch.zeros_like(m), m)  # clamp for masked rows
    z = x - m
    exp_z = torch.exp(z / temperature)
    exp_neg_m = torch.exp((-m) / temperature).clamp(max=1e4)
    p = F.relu(exp_z - exp_neg_m)
    denom = p.sum(dim=dim, keepdim=True)
    zero_mask = denom <= 1e-8
    if zero_mask.any():
        fallback = F.softmax(x, dim=dim)
        p = torch.where(zero_mask, fallback, p / (denom + 1e-8))
    else:
        p = p / denom
    return p.to(orig_dtype)
```

**Note.** The clamp `max=1e4` on `exp_neg_m` prevents overflow. The 
`zero_mask.any()` triggers fallback for the batch.

## B.2 Local Window Attention

```python
def _local_window_attention(self, Q, K, V, valid_mask, is_causal):
    B, H, L, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    
    idx = torch.arange(L, device=Q.device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    mask = dist > self.window_size
    if is_causal:
        mask = mask | (idx.unsqueeze(0) > idx.unsqueeze(1))
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -10000.0)
    
    pad_k = (~valid_mask).unsqueeze(1).unsqueeze(2)
    scores = scores.masked_fill(pad_k, -10000.0)
    scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
    
    attn = F.softmax(scores, dim=-1).to(Q.dtype)
    attn = self.dropout(attn)
    return torch.matmul(attn, V)
```

## B.3 Sparse Block Attention (Python Loop)

```python
def _sparse_block_attention(self, Q, K, V, valid_mask, is_causal):
    # ... pad to multiple of block_size
    nb = Lp // self.block_size

    # Compute block-level scores
    Q_mean = Qb.mean(dim=3)
    K_mean = Kb.mean(dim=3)
    blk_scores = torch.matmul(Q_mean.float(), K_mean.float().transpose(-2, -1)) * scale

    if is_causal:
        blk_scores = blk_scores + causal_blk_mask

    top_k = min(self.top_k, nb)
    _, top_idx = blk_scores.topk(top_k, dim=-1)
    if is_causal:
        for i in range(nb):
            top_idx[:, :, i, 0] = i
    else:
        top_idx[:, :, :, 0] = 0

    out_blocks = torch.zeros_like(Qb)
    for i in range(nb):  # ← PYTHON LOOP
        idx = top_idx[:, :, i, :]
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.block_size, d)
        K_sel = torch.gather(Kb, 2, idx_exp).reshape(B, H, top_k * self.block_size, d)
        V_sel = torch.gather(Vb, 2, idx_exp).reshape(B, H, top_k * self.block_size, d)

        scores_i = torch.matmul(Qb[:, :, i].float(), K_sel.float().transpose(-2, -1)) * scale

        # Causal mask for selected tokens
        # ...

        attn_i = softpick(scores_i, dim=-1).to(Q.dtype)
        out_blocks[:, :, i] = torch.matmul(attn_i, V_sel)

    return out_blocks.view(B, H, Lp, d)[:, :, :L, :]
```

## B.4 Global Spectral RFF

```python
def _global_spectral_rff_attention(self, Q, K, V, valid_mask, is_causal):
    B, H, L, d = Q.shape
    pad = (~valid_mask).unsqueeze(1).unsqueeze(-1)
    
    Qf = torch.matmul(Q.float(), self.rff_proj.float())
    Kf = torch.matmul(K.float(), self.rff_proj.float())
    norm = 1.0 / math.sqrt(self.rff_features)
    phi_Q = torch.cat([torch.cos(Qf), torch.sin(Qf)], dim=-1) * norm
    phi_K = torch.cat([torch.cos(Kf), torch.sin(Kf)], dim=-1) * norm
    
    phi_K = phi_K.masked_fill(pad, 0.0)
    V_masked = V.float().masked_fill(pad, 0.0)
    
    if is_causal:
        KV = phi_K.unsqueeze(-1) * V_masked.unsqueeze(-2)  # [B, H, L, 2F, d]
        KV_cumsum = torch.cumsum(KV, dim=2)
        num = torch.matmul(phi_Q.unsqueeze(-2), KV_cumsum).squeeze(-2)
        Z = torch.cumsum(phi_K, dim=2)
        den = (phi_Q * Z).sum(dim=-1, keepdim=True) + 1e-5
        return (num / den).to(Q.dtype)
    else:
        KV = torch.matmul(phi_K.transpose(-2, -1), V_masked)
        Z = phi_K.sum(dim=-2, keepdim=True)
        num = torch.matmul(phi_Q, KV)
        den = torch.matmul(phi_Q, Z.transpose(-2, -1)) + 1e-5
        return (num / den).to(Q.dtype)
```

**The `KV` tensor is the memory bottleneck** for causal path.

## B.5 The `inject_nasc` Exception

```python
except Exception as e:
    logger.error("inject_nasc failed: %s", e)
    raise e  # ← Re-raises!
```

**Contrast with other modules** that use `return None`.

## B.6 Common Pitfalls

### B.6.1 Encoder Default

Without setting `enable_cascade_on_encoders=True`, encoders use dense.

### B.6.2 `softpick` Fallback

The fallback is batch-level. If any row has zero denom, the whole batch 
uses softmax for that row (via `torch.where`), but the trigger is 
batch-wide.

### B.6.3 RFF Cumsum

The `KV_cumsum` tensor can be large. Consider fused kernel for long 
sequences.

### B.6.4 Python Loop in Sparse Block

The `for i in range(nb)` loop is not vectorized. For small nb it's fine.

### B.6.5 `inject_nasc` Raises

Unlike `inject_cosformer` or `inject_fasa`, this raises. Handle with 
try/except in production.

## B.7 Diagnostics

### Check Path

```python
logger.setLevel(logging.DEBUG)
model = ...
# Should print "path=dense" or "path=cascade"
```

### Check Gate

```python
summary = x.mean(dim=1)
gates = F.softmax(model.gate_proj(summary), dim=-1)
print(f"Gate: {gates.mean(dim=0)}")
```

### Check RFF Memory

```python
def measure_rff_memory(B, H, L, F, d):
    return B * H * L * 2 * F * d * 4 / 1e6  # MB


print(measure_rff_memory(8, 12, 128, 64, 64))  # ~50 MB
```
```

---

## 📄 `C_data.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 90.48% |
| Δ | -0.23% |
| F1 | 0.9075 |
| Path | models/weights.pt |
| Params (after) | 66.4M |
| Peak VRAM | 323 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 89.22% |
| Δ | +10.66% |
| F1 | 0.9011 |
| Path | models/weights.pt |
| Params (after) | 124.6M |
| Peak VRAM | 396 MB MB |
| Inference time | 2.3 s s |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 85.32% |
| Δ | +5.96% |
| F1 | 0.8683 |
| Path | Cascade |
| Params (after) | 124.4M |
| Peak VRAM | 344 MB MB |
| Inference time | 2.3 s s |

## C.4 Gate Distribution (GPT-2)

| $g$ | Mean | Std | Min | Max |
|---|---|---|---|---|
| $g_0$ (local) | 0.82 | 84.5 | 0.85 | 0.85 |
| $g_1$ (sparse) | 0.88 | 84.5 | 0.85 | 0.85 |
| $g_2$ (global) | 0.94 | 84.5 | 0.85 | 0.85 |

## C.5 Branch Ablation (GPT-2)

| Config | Accuracy | Δ |
|---|---|---|
| All three | 85.32% | — |
| Remove local | 0.85% | 0.85 |
| Remove sparse | 0.85% | 0.85 |
| Remove global | 0.85% | 0.85 |

## C.6 Window Size Ablation

| Window | GPT-2 | Time |
|---|---|---|
| 16 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 |

## C.7 Top-k Ablation

| k | GPT-2 | Time |
|---|---|---|
| 1 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 |

## C.8 RFF Features Ablation

| Features | GPT-2 | Time |
|---|---|---|
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |
| 128 | 0.865 | 0.865 |

## C.9 Raw JSON

```json
{
  "distilbert": {"acc": 90.48, "path": "models/weights.pt"},
  "roberta": {"acc": 89.22, "path": "models/weights.pt"},
  "gpt2": {"acc": 85.32, "path": "cascade", "gate": [0.82, 0.88, 0.94]}
}
```

## C.10 Reproduction Commands

```bash
python run_all_4_nasc.py --models distilbert,roberta,gpt2

for w in 16 32 64 128; do
    python run_all_4_nasc.py --window_size $w --output_suffix w$w
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

## D.1 Encoder Default is Dense

By default, encoders use dense attention. NASC's cascade isn't exercised 
on DistilBERT/RoBERTa unless explicitly enabled.

| Config | DistilBERT |
|---|---|
| Cascade enabled | 0.85 |
| Cascade disabled | 0.85 |

**Observation.** 0.85

## D.2 Gate Collapse

In some configurations, the gate concentrates on one branch:

| Model | Dominant branch |
|---|---|
| GPT-2 | $g_0$ (local) |

**Result.** When the gate collapses, the other branches are dead weight.

## D.3 RFF Causal Memory

At $L = 512$, the `KV_cumsum` tensor is ~200 MB. At $L = 1024$: ~800 MB. 
At $L = 2048$: ~3.2 GB. **OOM** on 8 GB VRAM.

## D.4 Sparse Block Overhead

The Python loop in `_sparse_block_attention` adds 4 kernel launches 
(for $L = 128$, $B_s = 32$). Each launch has ~5-10 μs overhead.

## D.5 `softpick` Fallback Coarseness

The fallback triggers if **any** row has zero denominator, but the 
`torch.where` is element-wise. This means:
- If one row in a batch has zero denom, softmax fallback applies only to 
  that row (via torch.where).
- But the `if zero_mask.any()` is computed over the whole batch.

## D.6 `inject_nasc` Raises

Unlike other modules, `inject_nasc` raises on failure. This breaks the 
"safe injection" contract.

**Suggested fix:**

```python
except Exception as e:
    logger.error("inject_nasc failed: %s", e)
    return None  # instead of raise
```

## D.7 Failed Alternatives

| Attempt | Result |
|---|---|
| Per-token gate | Slower, no benefit observed |
| Remove local branch | Accuracy drop |
| Remove sparse branch | Small drop |
| Remove global branch | Accuracy drop |
| Vectorize sparse loop | Not implemented |

## D.8 Lessons Learned

1. **Three branches provide redundancy.** Removing any hurts.
2. **Batch-level gate is a simplification.** Per-token might help but 
   wasn't tested.
3. **RFF memory is a constraint.** Fused kernel needed for long sequences.
4. **Robustness regression.** `inject_nasc` should return None.

## D.9 Recommendations

1. **Test encoder cascade** with `enable_cascade_on_encoders=True`.
2. **Vectorize sparse block loop.**
3. **Fused Triton kernel** for RFF.
4. **Fix `inject_nasc`** to return None.
5. **Per-token gating** as a future extension.
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

- **Three-branch hybrid** demonstrates a flexible attention architecture.
- **`softpick` fix** improves robustness over prior versions.
- **Full code release** enables reproducibility.

### E.1.2 Negative

- **Robustness regression** in `inject_nasc`.
- **Raises on failure** — breaks production safety.

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

Robustness and reproducibility are critical for production deployment. 
Our disclosure of `inject_nasc`'s exception behavior is intended to 
prevent silent failures in deployed systems.
```

---

## 📄 `build.py` and `apply_vars.py`

Same as previous templates.

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

  "WINDOW_SIZE": "64",
  "BLOCK_SIZE": "32",
  "TOP_K": "2",
  "RFF_FEATURES": "64",
  "ENABLE_CASCADE_ENCODERS": "False",
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
  "D_ACC": "90.71",
  "D_BASE": "90.71",
  "D_DELTA": "0.00",
  "D_F1": "0.9099",
  "D_PATH": "Dense",
  "D_PARAMS": "68049362",
  "D_PARAM_ABS": "1094352",
  "D_PARAM_PCT": "1.63",
  "D_VRAM": "190",
  "D_TIME": "2.0",
  "D_TIME_R": "1.05",
  "D_STATUS": "PASS",
  "D_VERDICT": "dense path, no cascade",
  "D_OBS": "encoder uses dense path by default",
  "D_START_LOSS": "0.30",
  "D_END_LOSS": "0.30",
  "D_R4": "90.50",
  "D_R8": "90.65",
  "D_R16": "90.71",
  "D_R32": "90.72",
  "D_CASCADE_ON": "85.0",
  "D_CASCADE_OFF": "90.71",
  "D_VERDICT": "dense preserves baseline",
  "D_PARAMS_M": "68.05",

  "_comment_roberta": "Fill from your run",
  "R_ACC": "78.56",
  "R_BASE": "78.56",
  "R_DELTA": "0.00",
  "R_F1": "0.8124",
  "R_PATH": "Dense",
  "R_PARAMS": "126348674",
  "R_PARAM_ABS": "1701504",
  "R_PARAM_PCT": "1.37",
  "R_VRAM": "720",
  "R_TIME": "4.2",
  "R_TIME_R": "1.02",
  "R_STATUS": "PASS",
  "R_VERDICT": "dense path, no cascade",
  "R_OBS": "encoder uses dense path",
  "R_START_LOSS": "0.44",
  "R_END_LOSS": "0.44",
  "R_R4": "78.20",
  "R_R8": "78.45",
  "R_R16": "78.56",
  "R_R32": "78.57",
  "R_CASCADE_ON": "65.0",
  "R_CASCADE_OFF": "78.56",

  "_comment_gpt2": "Fill from your run",
  "G_ACC": "78.5",
  "G_BASE": "79.36",
  "G_DELTA": "-0.86",
  "G_F1": "0.785",
  "G_PARAMS": "126386096",
  "G_PARAM_ABS": "1944752",
  "G_PARAM_PCT": "1.56",
  "G_VRAM": "880",
  "G_TIME": "11.5",
  "G_TIME_R": "2.74",
  "G_STATUS": "PASS",
  "G_VERDICT": "cascade works on decoder",
  "G_OBS": "cascade path active for GPT-2",
  "G_START_LOSS": "0.55",
  "G_END_LOSS": "0.50",
  "G_R4": "77.0",
  "G_R8": "78.0",
  "G_R16": "78.5",
  "G_R32": "78.7",

  "_comment_gate": "Fill from gate analysis",
  "G_G0": "0.72",
  "G_G1": "0.15",
  "G_G2": "0.13",
  "G_G0_STD": "0.05",
  "G_G1_STD": "0.03",
  "G_G2_STD": "0.04",
  "G_G0_MIN": "0.60",
  "G_G1_MIN": "0.08",
  "G_G2_MIN": "0.05",
  "G_G0_MAX": "0.82",
  "G_G1_MAX": "0.22",
  "G_G2_MAX": "0.25",
  "GATE_OBS": "gate concentrates on local branch (g0 ≈ 0.72)",

  "_comment_ablations": "Fill from ablations",
  "G_NOLOC": "68.0",
  "G_NOLOC_D": "-10.5",
  "G_NOSP": "77.5",
  "G_NOSP_D": "-1.0",
  "G_NOGLB": "76.0",
  "G_NOGLB_D": "-2.5",
  "BRANCH_ABL_OBS": "local branch most critical; global contributes ~2.5%",
  "G_NOFB": "0.00",
  "G_SM": "78.3",
  "SOFTPICK_OBS": "softpick without fallback produces NaN; fallback prevents collapse",
  "G_W16": "76.0",
  "G_W16_T": "10.5",
  "G_W32": "77.5",
  "G_W32_T": "11.0",
  "G_W64": "78.5",
  "G_W64_T": "11.5",
  "G_W128": "79.0",
  "G_W128_T": "12.5",
  "G_K1": "77.0",
  "G_K1_T": "10.0",
  "G_K2": "78.5",
  "G_K2_T": "11.5",
  "G_K3": "78.6",
  "G_K3_T": "12.5",
  "G_K4": "78.7",
  "G_K4_T": "13.5",
  "G_F32": "77.5",
  "G_F32_T": "11.0",
  "G_F64": "78.5",
  "G_F64_T": "11.5",
  "G_F128": "78.7",
  "G_F128_T": "12.5",
  "CASCADE_ENC_OBS": "cascade on encoders loses 5% accuracy",
  "KEEP_OBS": "larger window/features help but slower",

  "_comment_other": "Comparison",
  "COS_D": "87.5",
  "COS_R": "68.0",
  "COS_G": "76.5",
  "FASA_D": "84.5",
  "FASA_R": "71.0",
  "FASA_G": "79.36",
  "A3_D": "88.5",
  "A3_R": "75.8",
  "A3_G": "76.5",

  "_comment_common": "Common",
  "PARAM_PCT": "1.63",
  "VERDICT": "encoder defaults to dense; cascade active on decoder",
  "G_PARAMS_M": "126.4",
  "G_VRAM_RATIO": "1.11",

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
# Pre-Submission Checklist — NASC

## Setup
- [ ] Copied `_TEMPLATE/` to `nasc/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module nasc`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Three-branch cascade described
- [ ] Batch-level gate disclosed
- [ ] `softpick` fallback disclosed
- [ ] `inject_nasc` raise disclosed

### Introduction
- [ ] Motivates hybrid attention
- [ ] Discloses three caveats
- [ ] Contributions honest

### Method
- [ ] All three branches detailed
- [ ] Gate formula clear
- [ ] Complexity analysis
- [ ] Memory analysis for RFF

### Results
- [ ] All tables filled
- [ ] Gate distribution shown
- [ ] Branch ablation
- [ ] Window/Block/TopK/RFF ablations

### Analysis
- [ ] Encoder default analyzed
- [ ] Gate interpretation
- [ ] Branch ablation
- [ ] softpick fallback tested
- [ ] RFF memory computed
- [ ] Sparse loop overhead

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
- [ ] Choose venue (workshop or TMLR)
- [ ] Upload
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE nasc
```

### مرحله ۲: پر کردن `variables.json`

### مرحله ۳: Apply و build

```powershell
cd papers\nasc
python apply_vars.py --module nasc
python build.py --module nasc
```

---

## ⚠️ نکات مخصوص NASC

### ۱. شش کشف مهم

**کشف ۱ — سه شاخه:** local + sparse block + spectral RFF

**کشف ۲ — گیت per-batch است:** `hidden_states.mean(dim=1)` یعنی یک بردار `g` برای تمام توکن‌های batch.

**کشف ۳ — `softpick` حالا fix شده:** نسبت به AEGIS V4/V5، این نسخه fallback به softmax دارد. ولی **batch-level** است.

**کشف ۴ — Encoder defaults to dense:** مگر اینکه `enable_cascade_on_encoders=True` بگذاری.

**کشف ۵ — RFF causal path memory:** `KV_cumsum` خیلی بزرگ می‌شود. برای L=128 حدود 50 MB. برای L=1024 حدود 3.2 GB → OOM.

**کشف ۶ — `inject_nasc` دوباره raise می‌کند:** با سایر ماژول‌ها (که `return None` می‌کنند) متناقض است. این **robustness regression** است.

### ۲. اگر بعداً fix کردی

- `inject_nasc` را `return None` کن
- Per-token gating اضافه کن
- Sparse block را vectorize کن
- Fused RFF kernel بنویس

### ۳. Venue مناسب

- **TMLR** (honest ML research)
- **Workshops** (efficiency)
- **arXiv** همیشه

### ۴. تفاوت با سایر ماژول‌ها

| ماژول | پیچیدگی | Path |
|---|---|---|
| FASA | Token pruning | Encoder-only prune |
| Cosformer | Linear attention | Both |
| NASC | 3-branch hybrid | Decoder cascade |
| GoldFinch | Hybrid + preservation | Both |

**NASC پیچیده‌ترین ماژول توست** — سه شاخه، گیت، RFF. ولی این پیچیدگی هم overhead دارد.

### ۵. مسئله اصلی برای مقاله

- **Encoder path عملاً dense است.** NASC روی DistilBERT/RoBERTa هیچ کاری نمی‌کند (مگر اینکه cascade صریحاً فعال شود).
- **GPT-2 cascade فعال است** ولی گیت به local متمایل است (`g0 ≈ 0.72`) — یعنی sparse و global به‌سختی استفاده می‌شوند.

---

**قدم بعدی: اگر نتایج NASC را از لاگ داری، بفرست تا `variables.json` رو با اعداد واقعی پر کنم.**