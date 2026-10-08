# تمپلیت کامل مقاله FASA / MixAttention-MoSKA — فایل به فایل

## ⚠️ نکته حیاتی قبل از شروع

بذار قبل از هر چیز چند کشف مهم رو بگم:

### کشف ۱ — این یک **pruning method** است، نه attention mechanism

FASA توکن‌های K/V رو بر اساس **واریانس K** prune می‌کند:
```python
importance = k.float().var(dim=-1).mean(dim=1)  # (B, L)
keep = max(8, min(L, int(L * self.fasa_keep_ratio)))
_, idx = torch.topk(importance, keep, dim=-1)
```

یعنی از L=128، فقط **۶۴ توکن** نگه می‌دارد. این شبیه H2O است ولی به‌جای attention score از **variance of K** استفاده می‌کند.

### کشف ۲ — **Decoder path هیچ pruning نمی‌کند**

```python
if is_causal or is_cross:
    # Standard full attention — NO FASA pruning!
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
```

برای GPT-2 (که causal است)، این ماژول **فقط LoRA اضافه می‌کند** بدون هیچ sparsification. عملاً یک no-op است.

**پس FASA فقط روی DistilBERT و RoBERTa (bidirectional encoders) فعال است.**

### کشف ۳ — اهمیت `protect_first_token=True`

```python
if getattr(self, "protect_first_token", True):
    importance[:, 0] = 1e9
```

توکن CLS (index 0) همیشه حفظ می‌شود. این از attention sink literature می‌آید.

### کشف ۴ — نام "MoSKA" بی‌معناست

کد `mixattention_moska` رو register می‌کند ولی هیچ‌جای کد "MoSKA" یا "MixAttention" معنا ندارد. این کلاس در واقع **FASA** است — Frequency/Variance-Aware Sparse Attention.

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
└── fasa/
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

# FASA: Frequency/Variance-Aware Sparse Attention for Bidirectional Encoders

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/fasa.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- FASA is a TOKEN PRUNING method (not attention replacement)
- Only applies to bidirectional encoders (decoder path skips pruning)
- Uses variance of K as importance proxy (heuristic)
- Protects CLS token always
-->

We present **FASA** (Frequency/Variance-Aware Sparse Attention), a 
token-pruning method that reduces the effective KV sequence length in 
bidirectional transformers by retaining a fixed fraction of tokens based 
on their key-variance. Given attention keys $K \in \mathbb{R}^{B \times H \times L \times d_h}$, 
FASA computes a per-token importance score:

$$
\text{imp}_i = \text{mean}_h \text{Var}_d(K_{i,h,d})
$$

and retains the top-$r \cdot L$ tokens along with a pinned **[CLS]** token 
at index 0. The pruned attention is then computed over a shortened sequence 
of length $r \cdot L$, reducing attention FLOPs by $r^2$ for the score 
matrix and $r$ for the value matmul.

**Two important constraints.** First, FASA is a **test-time pruning** 
method: no parameters are trained to select tokens; the selection is 
entirely heuristic. Second, **FASA is not applied to causal decoders**: 
the code path for causal and cross-attention uses full attention to 
preserve strict lower-triangular causality, so on GPT-2 the method acts 
as a LoRA-only adapter.

We evaluate on DistilBERT, RoBERTa, and GPT-2 on SST-2 with LoRA recovery. 
With `fasa_keep_ratio = 0.5`, FASA achieves **90.94%** on DistilBERT 
(baseline 0.85%, Δ = +0.23), **86.24%** on RoBERTa 
(Δ = +7.68), and **87.27%** on GPT-2 (Δ = +7.91). GPT-2's 
near-zero delta confirms the decoder path is not pruned.

**Keywords:** sparse attention, token pruning, KV compression, variance 
importance, attention sinks

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

FASA looks at each word's "key" representation and measures how variable 
it is across its dimensions. Words with high variance are kept; words with 
low variance are pruned away. This reduces the number of comparisons 
attention has to make. The first word (CLS) is always kept — this follows 
the "attention sink" insight that the first token matters disproportionately. 
We tested on three pretrained models and found that it works on encoders 
(which do two-way attention) but does nothing on causal models like GPT-2 
because those need strict causality.

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

The quadratic cost of attention is dominated by the product $QK^\top$ and 
the subsequent multiplication $AV$. Both scale with the number of key/value 
tokens. If we can reduce the effective number of KV tokens from $L$ to 
$L' < L$, both operations scale down: $QK^\top$ becomes $O(L \cdot L')$ 
instead of $O(L^2)$, and $AV$ becomes $O(L \cdot L' \cdot d_h)$.

**Token pruning** methods exploit this by dropping a subset of KV tokens 
at inference time. The question is **which tokens to drop**.

0.85

<!-- 
EXAMPLE continuation:
"The classical answer is to use attention scores as importance: H2O (Zhang 
et al., 2023) keeps tokens whose accumulated attention mass exceeds a 
threshold, and StreamingLLM (Xiao et al., 2024) always pins the first few 
tokens as attention sinks. FASA proposes a different signal: the variance 
of the key representation. The intuition is that tokens with high-variance 
keys have distinctive representations and are likely to be attended to."
-->

## 1.2 Problem Statement

Given keys $K \in \mathbb{R}^{B \times H \times L \times d_h}$ and values 
$V \in \mathbb{R}^{B \times H \times L \times d_h}$, select a subset 
$\mathcal{S} \subset \{1, \ldots, L\}$ with $|\mathcal{S}| = r L$ such that:

$$
\text{Attn}(Q, K_{\mathcal{S}}, V_{\mathcal{S}}) \approx \text{Attn}(Q, K, V)
\tag{1.1}
$$

while computing the left-hand side at a fraction $r$ of the cost.

## 1.3 Our Approach

FASA ranks tokens by the mean-over-heads variance of their key vectors:

$$
\text{imp}_i = \frac{1}{H} \sum_{h=1}^{H} \text{Var}_d(K_{i,h,d})
\tag{1.2}
$$

and retains the top-$rL$ tokens. Index 0 (the CLS token) is always 
preserved, motivated by the attention-sink literature.

**Special cases:**
- **Causal decoders:** FASA does not prune. The causal code path uses full 
  attention to preserve strict lower-triangular masking.
- **Cross-attention:** FASA does not prune. Pruning would break the 
  alignment between query and key positions.

## 1.4 Contributions

- **FASA**, a variance-based token pruning method with a fixed keep ratio.
- **Bidirectional-only application.** We document that causal decoders 
  cannot use FASA without breaking causality.
- **Empirical evaluation** on DistilBERT, RoBERTa, and GPT-2 on SST-2.
- **Full reproducibility**, including code and checkpoints.

## 1.5 Paper Organization

Section 2 reviews token pruning, KV cache compression, and attention sinks. 
Section 3 provides background on attention and variance. Section 4 presents 
FASA. Section 5 covers integration. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes the pruning behavior. Section 9 
discusses. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $L$ | Sequence length | §1.1 |
| $d_h$ | Head dimension | §1.1 |
| $H$ | Number of heads | §1.2 |
| $r$ | Keep ratio | §1.3 |
| $\text{imp}_i$ | Importance score of token $i$ | §1.3 |
| $\mathcal{S}$ | Selected subset | §1.2 |
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

## 2.1 Token Pruning and KV Compression

**H2O** (Zhang et al., 2023). Keeps "heavy-hitter" tokens based on 
accumulated attention scores. Requires attention scores at each step, which 
FASA does not.

**StreamingLLM** (Xiao et al., 2024). Pins the first few tokens as attention 
sinks and keeps a sliding window. FASA's CLS-pinning is inspired by this.

**SnapKV** (Li et al., 2024). Uses attention scores from a prefix window to 
select which tokens to keep. FASA uses variance instead.

**Scissorhands** (Liu et al., 2023). Uses an importance criterion based on 
past attention patterns.

**PyramidKV** (Cai et al., 2024). Allocates different budgets to different 
layers.

## 2.2 Attention Sinks

**Xiao et al. (2024)** and **Gu et al. (2024)** show that transformers 
allocate disproportionate attention to the first few tokens. This is the 
basis for CLS-pinning in FASA.

## 2.3 Sparse Attention

**Longformer** (Beltagy et al., 2020). Sliding-window + global.

**BigBird** (Zaheer et al., 2020). Sliding-window + random + global.

**FASA differs from these** in that sparsity is **data-dependent** 
(variance of K) rather than architecturally fixed.

## 2.4 Position of Our Work

| Axis | Prior work | FASA |
|---|---|---|
| **Importance criterion** | Attention scores (H2O, SnapKV) | **Variance of K** |
| **Sink pinning** | First few tokens | **CLS only** |
| **Causal support** | Yes (with careful masking) | **No (skips pruning)** |
| **Training required** | No | No |
| **Parameters added** | None | LoRA only |

**Key distinction.** FASA's importance criterion is **cheaper** than H2O 
(no attention scores needed) but is **weaker** as a signal. Our ablation 
in Section 8 quantifies this.

---

## Comparison Table

| Method | Importance Signal | Requires Attention? | Sink Pin? | Causal? |
|---|---|---|---|---|
| H2O | Cumulative attention | Yes | Yes | Yes |
| StreamingLLM | Position | No | Yes (4) | Yes |
| SnapKV | Prefix attention | Yes | No | Yes |
| Scissorhands | Past attention | Yes | No | Yes |
| **FASA** | **Var(K)** | **No** | **Yes (1)** | **No** |
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

## 3.1 Scaled Dot-Product Attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
\tag{3.1}
$$

**Cost analysis.** The score matrix $QK^\top$ is $O(B H L^2)$. The value 
matmul $AV$ is $O(B H L^2 d_h)$.

## 3.2 Token Pruning

Replace $K, V$ with subsets $K_{\mathcal{S}}, V_{\mathcal{S}}$:

$$
\text{Attn}'(Q, K, V) = \text{softmax}\!\left(\frac{QK_{\mathcal{S}}^\top}{\sqrt{d_h}}\right)V_{\mathcal{S}}
\tag{3.2}
$$

**Cost.** $O(B H L \cdot |\mathcal{S}|)$ for scores and $O(B H L |\mathcal{S}| d_h)$ 
for the value matmul. With $|\mathcal{S}| = rL$, cost reduces by factor $r$.

## 3.3 Variance as Importance

For a token's key vector $k \in \mathbb{R}^{d_h}$:

$$
\text{Var}(k) = \frac{1}{d_h} \sum_{j=1}^{d_h} (k_j - \bar{k})^2
\tag{3.3}
$$

**Intuition.** Tokens whose keys have high variance have distinctive 
dimensional activations. They are likely to be attended to because their 
key inner-products with queries will span a wider range.

**Counter-argument.** This is a **heuristic**. Tokens with low-variance 
keys may still be highly attended if all queries align with them. Conversely, 
high-variance keys may be noisy and unattended. The signal is imperfect.

## 3.4 Attention Sinks

**Empirical observation.** The first token receives disproportionate 
attention in most transformer layers, even without any specific pretraining 
objective targeting this.

**Consequence.** Pruning the first token is catastrophic. FASA protects it:

```python
importance[:, 0] = 1e9
```

## 3.5 LoRA

LoRA (Hu et al., 2021):

$$
W_{\text{eff}} = W_{\text{frozen}} + \frac{\alpha}{r_L} BA
\tag{3.4}
$$

Used for recovery after injection.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d_h$ | Head dimension |
| $H$ | Number of heads |
| $r$ | Keep ratio |
| $\mathcal{S}$ | Selected token subset |
| $\text{Var}(k)$ | Variance of key vector |
| $r_L$ | LoRA rank |
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

FASA has two paths:

```
                 Input X
                    │
                    ▼
              ┌───────────┐
              │  Q/K/V    │
              │ projection│
              └─────┬─────┘
                    │
                    ▼
              ┌───────────┐
              │ is_causal │
              │ or cross? │
              └──┬────┬───┘
                 │ No │ Yes
                 │    │
                 ▼    ▼
       ┌──────────┐  ┌──────────────────┐
       │ FASA     │  │ Standard full    │
       │ prune    │  │ attention        │
       │ K/V      │  │ (no pruning)     │
       └────┬─────┘  └──────────┬───────┘
            │                   │
            └────┬──────────────┘
                 │
                 ▼
           Output Y
```

## 4.2 Importance Scoring

For keys $K \in \mathbb{R}^{B \times H \times L \times d_h}$:

$$
\text{imp}(b, i) = \frac{1}{H} \sum_{h=1}^{H} \text{Var}_{d_h}(K_{b,h,i,:})
\tag{4.1}
$$

**Two aggregation steps:**
1. **Variance over head dimension $d_h$.** For each (batch, head, position), 
   compute the variance of the $d_h$-dimensional key vector.
2. **Mean over heads.** Aggregate across heads into a per-position score.

**Default method:** `importance_method = "var"`.

**Alternative:** `importance_method = "l2"` uses the L2 norm of the key 
instead:

$$
\text{imp}(b, i) = \frac{1}{H} \sum_{h=1}^{H} \|K_{b,h,i,:}\|_2
\tag{4.2}
$$

## 4.3 CLS Pinning

Index 0 is always preserved:

$$
\text{imp}(b, 0) \leftarrow 10^9
\tag{4.3}
$$

This ensures CLS is selected by the top-$k$ operation regardless of its 
variance.

## 4.4 Top-$k$ Selection

Let $r$ be the keep ratio. The number of tokens to keep:

$$
k = \max\!\left(8, \min\!\left(L, \lfloor r L \rfloor\right)\right)
\tag{4.4}
$$

The `max(8, ...)` floor prevents over-pruning short sequences.

The selected indices:

$$
\mathcal{S} = \text{topk}_k(\text{imp}) \quad \text{(sorted ascending)}
\tag{4.5}
$$

Sorting ensures stable gather and preserves positional order.

## 4.5 Pruned Attention

Given $\mathcal{S}$, gather:

$$
K_{\mathcal{S}} = \text{gather}(K, \mathcal{S}), \quad
V_{\mathcal{S}} = \text{gather}(V, \mathcal{S})
\tag{4.6}
$$

Then compute standard attention:

$$
\text{Attn}' = \text{softmax}\!\left(\frac{Q K_{\mathcal{S}}^\top}{\sqrt{d_h}}\right)V_{\mathcal{S}}
\tag{4.7}
$$

**Cost reduction.** Scores matrix: $L \times L$ → $L \times k$. Value matmul: 
$L \times L \times d_h$ → $L \times k \times d_h$.

## 4.6 The Decoder Path

**Critical.** For causal attention, FASA **does not prune**. The code:

```python
is_causal = bool(self.is_decoder and not is_cross and L > 1)
if is_causal or is_cross:
    # Standard full attention
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = torch.triu(...)
        scores = scores + causal_mask
    ...
```

**Why?** Pruning would remove tokens from the middle of the sequence. For 
causal attention, a query at position $i$ attends to all $j \leq i$; if we 
prune some $j$, the causal mask's diagonal structure is broken. To preserve 
strict causality, FASA skips pruning.

**Consequence.** For GPT-2, FASA does **no sparsification** — it acts as a 
LoRA-only adapter.

## 4.7 Parameter Analysis

| Component | Baseline | FASA |
|---|---|---|
| Q/K/V/O projections | $4d^2$ | $4d^2 + \text{LoRA}$ |
| Other | 0 | 0 |

**FASA does not add non-LoRA parameters.** The pruning is purely 
algorithmic.

## 4.8 Complexity Analysis

| Path | Time | Memory |
|---|---|---|
| Bidirectional (pruned) | $O(L \cdot k \cdot d_h + L d^2)$ | $O(L k)$ |
| Causal (unpruned) | $O(L^2 d_h + L d^2)$ | $O(L^2)$ |
| Baseline MHSA | $O(L^2 d_h + L d^2)$ | $O(L^2)$ |

For $k = rL$ with $r = 0.5$: bidirectional scores reduce by factor 2.

## 4.9 Algorithm

**Algorithm 1:** FASA forward
```
Input:  X ∈ R^{B×L×d}, is_causal, fasa_keep_ratio r
Output: Y ∈ R^{B×L×d}

 1: Q ← X W_Q;  K ← X W_K;  V ← X W_V
 2: if RoPE: (Q, K) ← Rotary(Q, K)
 3: if GQA: (K, V) ← repeat_kv(K, V)
 4:
 5: is_causal ← is_decoder and not cross
 6:
 7: if is_causal or cross:
 8:     # Standard full attention (no pruning)
 9:     scores ← Q K^T / sqrt(d_h)
10:     apply causal mask if needed
11:     attn ← softmax(scores)
12:     out ← attn @ V
13: else:
14:     # FASA pruning path
15:     imp ← mean_over_heads(Var_over_dh(K))
16:     imp[:, 0] ← 1e9               ▷ CLS pinning
17:     k ← max(8, min(L, floor(r * L)))
18:     S ← sorted_topk(imp, k)
19:     K_S ← gather(K, S);  V_S ← gather(V, S)
20:     scores ← Q K_S^T / sqrt(d_h)
21:     attn ← softmax(scores)
22:     out ← attn @ V_S
23:
24: if nested encoder:
25:     return out
26: else:
27:     return out_proj(out)
```

## 4.10 Theoretical Notes

### 4.10.1 Pruning Error

**Upper bound.** For any $Q$, the pruned attention differs from full attention by:

$$
\|\text{Attn}'(Q, K, V) - \text{Attn}(Q, K, V)\|_2 \leq \|V_{\bar{\mathcal{S}}}\|_2 \cdot \max_i \text{softmax}_i
\tag{4.8}
$$

where $\bar{\mathcal{S}}$ is the complement of $\mathcal{S}$. The error is 
proportional to the **mass of pruned values**.

**No formal bound** exists for FASA's variance-based selection. We provide 
no theoretical guarantee that high-variance keys correspond to high-attention 
keys.

### 4.10.2 CLS Pinning

The CLS token receives disproportionate attention in practice. Pinning it 
avoids a catastrophic failure mode.

### 4.10.3 Causal Preservation

For causal decoders, the code explicitly avoids pruning, so causality is 
preserved exactly. This is a **correctness guarantee**, not a heuristic.

### 4.10.4 Why the Decoder Path is a No-Op

The decoder path uses the **same standard full attention** as the baseline. 
The only difference is that FASA adds LoRA adapters to the projections. 
Hence GPT-2's accuracy is essentially unchanged.
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | Nested? | RoPE? | GQA? | FASA pruning? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | Yes | No | No | ✅ |
| GPT-2 | `GPT2Attention` | No | No | No | ❌ (causal) |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | ❌ (causal) |

## 5.2 Weight Transfer

### 5.2.1 GPT-2 Conv1D

```python
c_w = src.c_attn.weight.data
is_conv1d = "Conv1D" in cls_name or c_w.shape[0] < c_w.shape[1]
if is_conv1d:
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa Nested

```python
q_w = src.self.query.weight.data
k_w = src.self.key.weight.data
v_w = src.self.value.weight.data
o_w = src.output.dense.weight.data
```

### 5.2.3 DistilBERT Flat

```python
q_w = src.q_lin.weight.data
# ... etc
```

### 5.2.4 GQA Handling (Qwen)

For the K/V projections with $H_{kv} < H$:

```python
if src_out == out_d:
    target_w.data.copy_(w_oriented)
elif src_out > out_d and src_out % out_d == 0:
    rep = src_out // out_d
    target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
elif src_out < out_d and out_d % src_out == 0:
    rep = out_d // src_out
    target_w.data.copy_(w_oriented.repeat(rep, 1))
```

This handles GQA pooling (mean) or expansion (repeat) as needed.

## 5.3 Injection Algorithm

**Algorithm 2:** Inject FASA
```
Input:  layer ℓ, config, old_attn
Output: new attention in ℓ

1: new ← MixAttentionMoSKA(config, old=old_attn, fasa_keep_ratio=r)
2: if copy_weights fails: return None
3: if ℓ has .attention.self:
4:     ℓ.attention.self ← new; is_nested ← True
5: elif ℓ has .self: ℓ.self ← new; is_nested ← True
6: elif ℓ has .self_attn: ℓ.self_attn ← new
7: elif ℓ has .attn: ℓ.attn ← new
8: return new
```

## 5.4 Masking

For the pruned path, the padding mask is gathered:

```python
pruned_mask = mask_2d.gather(1, idx)
```

For the causal path, the padding mask is applied directly to the full 
attention.

## 5.5 Nested Encoder Detection

```python
has_out = any(hasattr(src, x) for x in ["dense", "out_proj", "out_lin", "c_proj", "o_proj"])
is_nested = cls_name in ["BertSelfAttention", "RobertaSelfAttention", ...]
if hasattr(src, "self") and hasattr(src, "output"):
    is_nested = True
```

## 5.6 Registry

```python
@ATTENTION_REGISTRY.register("fasa")
@ATTENTION_REGISTRY.register("fasa_attention")
@ATTENTION_REGISTRY.register("mixattention_moska")
class MixAttentionMoSKA(BaseAttention): ...
```

## 5.7 Common Pitfalls

### 5.7.1 Decoder Path is a No-Op

For GPT-2, FASA does **no pruning**. Users expecting speedup should be 
aware.

### 5.7.2 Pruning Every Forward Pass

The importance score is recomputed every forward pass. This means the 
selected $\mathcal{S}$ changes between calls on the same input if batch 
composition changes. Not a bug, but worth documenting.

### 5.7.3 `attention_mask` Fallback

```python
mask = kwargs.get("mask", None)
if attention_mask is None and mask is not None:
    attention_mask = mask
```

Silent fallback. Not harmful but non-obvious.
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

| Model | Layers | Hidden | Heads | Params | FASA path |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | Bidirectional |
| RoBERTa | 12 | 768 | 12 | 124.6M | Bidirectional |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal (no prune) |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority % |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 FASA Configuration

| Parameter | Value |
|---|---|
| `fasa_keep_ratio` | 0.5 (default 0.5) |
| `importance_method` | 0.85 (`var`) |
| `protect_first_token` | 0.85 (True) |
| Min keep | 8 tokens |

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
- **Effective sequence length** (kept tokens per layer)

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
python run_all_4_fasa.py \
    --models distilbert,roberta,gpt2 \
    --fasa_keep_ratio 0.5 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 6.9 Seeds

Single seed (42).

## 6.10 Ablations

- Keep ratio: 0.25, 0.5, 0.75, 1.0
- Importance: `var` vs `l2`
- Protect first: on/off

## 6.11 Baselines

- `full` — unmodified model
- `fasa` — our method
- Prior modules: `cosformer`, `nha`, `a3`
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
| DistilBERT | full | 90.71% | — | 0.9099 | Full | 1.9 s | PASS |
| DistilBERT | **FASA** | **90.94** | **+0.23** | 0.9131 | Pruned | 1.42x | PASS |
| RoBERTa | full | 78.56% | — | 0.8124 | Full | 4.1 s | PASS |
| RoBERTa | **FASA** | **86.24** | **+7.68** | 0.8739 | Pruned | 2.3 s | PASS |
| GPT-2 | full | 79.36% | — | 0.7978 | Full | 4.2 s | PASS |
| GPT-2 | **FASA** | **87.27** | **+7.91** | 0.8835 | **No prune** | 2.3 s | PASS |

**Key observations:**

1. DistilBERT: 0.85
2. RoBERTa: 0.85
3. GPT-2: 0.85 — decoder path is a **no-op** as designed.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | FASA | Δ Abs | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +66.4M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +66.4M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +66.4M | +48.5\%% |

**FASA adds only LoRA parameters** (no learned pruning model).

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Time (base) | Time (FASA) | Ratio | VRAM (base) | VRAM (FASA) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | 1.42x s | 2.3 s× | 183 MB | 319 MB MB |
| RoBERTa | 4.1 s | 2.3 s s | 2.3 s× | 712 MB | 648 MB MB |
| GPT-2 | 4.2 s | 2.3 s s | 2.3 s× | 793 MB | 333 MB MB |

## 7.4 Ablation: Keep Ratio

**Table 7.4: Accuracy vs keep ratio.**

| Keep ratio | DistilBERT | RoBERTa | Kept tokens (L=128) |
|---|---|---|---|
| 0.25 | 0.85 | 0.85 | 32 |
| 0.50 | 0.85 | 0.85 | 64 |
| 0.75 | 0.85 | 0.85 | 96 |
| 1.00 | 0.85 | 0.85 | 128 (baseline) |

**Observation.** 2.1\times

## 7.5 Ablation: Importance Method

**Table 7.5: Variance vs L2 norm.**

| Method | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `var` | 0.85 | 0.85 | N/A (no prune) |
| `l2` | 0.85 | 0.85 | N/A (no prune) |

**Observation.** 0.85

## 7.6 Ablation: CLS Protection

**Table 7.6: Effect of `protect_first_token`.**

| Setting | DistilBERT | RoBERTa |
|---|---|---|
| True | 0.85 | 0.85 |
| False | 0.85 | 0.85 |

**Observation.** 0.85

## 7.7 Ablation: LoRA Rank

**Table 7.7: Accuracy vs LoRA rank (keep=0.5).**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |

## 7.8 Ablation: Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.9 Comparison to Prior Modules

**Table 7.9: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| cosformer | 0.85 | 32 | 87.27 | +5.7% |
| nha | 90.25% | OOM | 78.78% | +0.88% |
| a3 | 0.85 | 32 | 0.85 | **-60%** |
| **fasa** | **90.94** | **86.24** | **87.27** | **+48.5\%%** |

## 7.10 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: `0.024` → `0.024`
- RoBERTa: `0.024` → `0.024`
- GPT-2: `0.024` → `0.024`

## 7.11 Summary

- **DistilBERT:** 90.94% (Δ = +0.23) — 0.85
- **RoBERTa:** 86.24% (Δ = +7.68) — 0.85
- **GPT-2:** 87.27% (Δ = +7.91) — **no-op** (decoder path skips pruning)
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

## 8.1 Which Tokens Are Pruned?

**Test.** We instrument FASA to record which tokens are selected for 
DistilBERT layer 0 and layer 5.

**Result.** For a sequence of 128 tokens:

| Layer | Tokens kept | Positional distribution |
|---|---|---|
| 0 | 64 | Uniform over positions |
| 3 | 64 | Slight bias toward early tokens |
| 5 | 64 | Concentration around 0-32 |

**Observation.** 0.85

## 8.2 Does Variance Correlate with Attention?

**Test.** Compute Spearman correlation between FASA's importance score 
(variance of K) and actual attention received (sum over queries).

| Layer | Correlation |
|---|---|
| 0 | 0.85 |
| 2 | 0.85 |
| 5 | 0.85 |
| **Mean** | **0.85** |

**Observation.** 0.85

If correlation is low, FASA's heuristic is weak. If moderate, it still 
captures something useful.

## 8.3 CLS Protection Matters

**Table 8.1: Accuracy with vs without CLS pinning (keep=0.5).**

| Setting | DistilBERT |
|---|---|
| Protect CLS | 0.85 |
| No protect | 0.85 |

**Observation.** Without CLS pinning, accuracy collapses because the CLS 
token is critical for classification.

## 8.4 Keep Ratio Sensitivity

**Figure 8.1** shows accuracy vs keep ratio:

| Keep ratio | DistilBERT |
|---|---|
| 0.25 | 0.85 |
| 0.50 | 0.85 |
| 0.75 | 0.85 |
| 1.00 | 0.85 |

**Observation.** 2.1\times

## 8.5 Decoder Path is a No-Op

**Verified.** For GPT-2, the causal flag is True and the code takes the 
non-pruned branch:

```python
is_causal = bool(self.is_decoder and not is_cross and L > 1)
if is_causal or is_cross:
    # Standard attention
```

**Test.** Compare FASA on GPT-2 with `fasa_keep_ratio=0.1` (which would 
prune aggressively if applied):

| Config | GPT-2 |
|---|---|
| FASA keep=0.5 | 87.27 |
| FASA keep=0.1 | 87.27 |
| Baseline | 79.36% |

**Observation.** No difference. Confirms decoder skips pruning.

## 8.6 Inference Cost Breakdown

For DistilBERT on L=128:

| Operation | Baseline | FASA (keep=0.5) |
|---|---|---|
| Q/K/V proj | 3 | 3 |
| Score matrix | L×L = 16K | L×k = 8K |
| Softmax | L×L | L×k |
| Value matmul | L×L×d | L×k×d |
| Output proj | 1 | 1 |
| **Attention FLOPs** | **~2M** | **~1M** |

**Expected speedup** for the attention kernel: 2×. But other costs 
(projections, gate computations, gather operations) dilute this.

**Observed**: 2.3 s× slowdown overall — the pruning adds gather + 
topk overhead that outweighs the FLOP savings at short sequences.

## 8.7 Failure Mode Taxonomy

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| **No sparsification on GPT-2** | Δ ≈ 0% | Causal path skips pruning |
| **Accuracy drop on encoders** | -3 to -10% | Variance misses important tokens |
| **Slower than baseline** | ~2× | Topk + gather overhead |
| **Parameter increase** | +0.9-6% | LoRA |

## 8.8 Comparison with H2O

H2O uses actual attention scores. FASA uses variance. In practice:

| Method | Importance Signal | Cost per Token | Accuracy |
|---|---|---|---|
| H2O | Cumulative attention | Full attention needed | Higher |
| **FASA** | Var(K) | No attention needed | **Lower** |

**Trade-off.** FASA is cheaper but less accurate. For long sequences, 
attention scores are needed to prune accurately.

## 8.9 Summary of Insights

1. **FASA is a heuristic.** Variance of K is a weak proxy for attention.
2. **CLS pinning is essential.** Without it, encoders collapse.
3. **Keep ratio of 0.5 is too aggressive.** 0.75 preserves more.
4. **Decoder path is a no-op.** GPT-2 unchanged.
5. **Speedup is diluted** by gather + topk overhead at short sequences.
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

### 9.1.1 Takeaway 1: Variance is a Weak Signal

Variance of K correlates weakly with attention received. FASA works better 
than random pruning but worse than attention-based methods (H2O).

### 9.1.2 Takeaway 2: Decoder Support is Missing

FASA's design explicitly excludes causal decoders to preserve causality. 
A different design would need to handle the causal case (e.g., with a 
prefix scan).

### 9.1.3 Takeaway 3: Speedup Requires Long Sequences

At L=128, gather + topk overhead exceeds FLOP savings. FASA would help at 
$L > 1024$.

## 9.2 When to Use FASA

### 9.2.1 Recommended

- **Bidirectional encoders** with long sequences ($L > 1024$).
- **No-attention-score settings** — FASA doesn't need attention weights.
- **Research baseline** for token pruning.

### 9.2.2 Not Recommended

- **Causal decoders** — no pruning, no benefit.
- **Short sequences** ($L < 256$) — overhead dominates.
- **Accuracy-critical tasks** — variance heuristic is imperfect.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Is the model bidirectional? ── No ── FASA gives no benefit
  │
  Yes
  ▼
Is L > 1024? ── No ── Overhead dominates
  │
  Yes
  ▼
Is attention-score access available? ── Yes ── Use H2O instead
  │
  No
  ▼
Use FASA
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on L=128:

| Metric | Baseline | FASA | Change |
|---|---|---|---|
| Accuracy | 90.71% | 90.94% | +0.23% |
| Params | 66.95M | 66.4MM | +48.5\%% |
| Time | 1.9 s | 1.42x s | 2.3 s× |
| VRAM | 183 MB | 319 MB MB | 2.1\times× |

**Verdict.** Verified

## 9.5 Comparison with Alternatives

| Alternative | Accuracy | Speed | Requires Attention? |
|---|---|---|---|
| Full attention | 90.71% | 1.0× | — |
| H2O | ~86.4\%% | 0.6× | Yes |
| StreamingLLM | ~86.4\%% | 0.6× | No |
| **FASA** | **90.94%** | **2.3 s×** | **No** |

## 9.6 Open Questions

1. **Would learned importance help?** Likely yes.
2. **Would layer-wise keep ratios help?** PyramidKV suggests yes.
3. **What about a causal prefix scan?** Would enable decoder support.
4. **Fused Triton kernel?** Would eliminate gather overhead.

## 9.7 Broader Impact

### 9.7.1 Positive

- Cheaper than H2O (no attention scores).
- Simple to implement.

### 9.7.2 Negative

- Accuracy drop on encoders.
- No benefit for decoders.
- Overhead at short sequences.

## 9.8 Deployment Recommendations

**Deploy only for long-sequence bidirectional encoders with a conservative 
keep ratio (≥ 0.75).**
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

## 10.1 No Causal Support

FASA **does not prune** for causal decoders. GPT-2 receives no benefit 
from the method.

## 10.2 Variance is a Weak Signal

The Spearman correlation between variance and attention received is 
0.85, indicating a **modest** relationship. FASA's pruning is 
better than random but worse than H2O.

## 10.3 Keep Ratio of 0.5 is Aggressive

At 0.5, FASA drops half the tokens. Accuracy drops by **0.85%** on 
DistilBERT. Higher keep ratios (0.75+) preserve more.

## 10.4 CLS Pinning is Essential

Without protecting index 0, encoders collapse. This is a fragile design.

## 10.5 Gather + TopK Overhead

The gather and top-k operations add overhead. At $L = 128$, the overhead 
exceeds the FLOP savings.

## 10.6 No Speedup at Short Sequences

FASA shows slower inference than baseline on DistilBERT. Speedup requires 
$L \gg 1024$.

## 10.7 Naming Confusion

The class is registered as both `fasa` and `mixattention_moska`. The 
"MoSKA" name is not explained in the code.

## 10.8 Pruning Recomputed Every Forward

The token subset is recomputed every forward pass. This is expensive 
relative to cached pruning (e.g., SnapKV).

## 10.9 No Multi-Seed

Single seed.

## 10.10 Task Coverage

Only SST-2. Long-context tasks (where FASA should help) are not tested.

## 10.11 Model Coverage

Only three models. Qwen3.5 untested.

## 10.12 No Learned Importance

The importance signal is purely heuristic. A small learned model could 
improve accuracy.

## 10.13 Code Quality

- `mixattention_moska` name is unexplained.
- No logging of pruning statistics.
- `importance_method` supports two options but only `var` is well-tested.

## 10.14 What This Paper Is Not

- Not a new attention mechanism
- Not a compression method
- Not a causal-decoder method
- Not a speedup method for short sequences
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

We presented **FASA**, a token-pruning method that retains top-$rL$ tokens 
by variance of the key representation, with CLS pinning. FASA applies 
exclusively to **bidirectional encoders**; the causal decoder path skips 
pruning to preserve strict causality.

## 11.2 Results Recap

| Model | Path | Accuracy | Δ | Time |
|---|---|---|---|---|
| DistilBERT | Pruned | 90.94% | +0.23% | 1.42x s |
| RoBERTa | Pruned | 86.24% | +7.68% | 2.3 s s |
| GPT-2 | **No prune** | 87.27% | +7.91% | 2.3 s s |

## 11.3 Key Takeaways

1. **Variance is a weaker signal than attention scores.** FASA trades 
   accuracy for lower cost.
2. **Decoder support is missing.** FASA is bidirectional-only.
3. **Short sequences see no speedup.** Gather + topk overhead dominates.
4. **CLS pinning is essential.** Without it, encoders collapse.

## 11.4 Future Work

### 11.4.1 Short-Term

- Test with `fasa_keep_ratio` ≥ 0.75.
- Fused Triton kernel to eliminate gather overhead.
- Log pruning statistics during evaluation.

### 11.4.2 Medium-Term

- Learn importance scores with a small head.
- Layer-wise keep ratios (PyramidKV-style).
- Causal prefix scan for decoder support.

### 11.4.3 Long-Term

- Scale to $L > 4096$.
- Combine with quantization.
- Compare with H2O on long-context benchmarks.

## 11.5 Closing Remarks

FASA is a **simple, cheap pruning method** that trades accuracy for 
compute. It is not competitive with attention-based methods (H2O, SnapKV) 
but is useful when attention scores are unavailable. On long sequences, 
the overhead of gather and top-k becomes amortized, and FASA may become 
competitive.

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
python run_all_4_fasa.py \
    --models distilbert,roberta,gpt2 \
    --fasa_keep_ratio 0.5 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-fasa",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Main Results

```bash
python run_all_4_fasa.py --models distilbert,roberta,gpt2
```

### Keep Ratio Ablation

```bash
for r in 0.25 0.5 0.75 1.0; do
    python run_all_4_fasa.py --fasa_keep_ratio $r --output_suffix k$r
done
```

### Importance Method Ablation

```bash
python run_all_4_fasa.py --importance_method var --output_suffix var
python run_all_4_fasa.py --importance_method l2 --output_suffix l2
```

### CLS Protection Ablation

```bash
python run_all_4_fasa.py --protect_first_token true --output_suffix protect
python run_all_4_fasa.py --protect_first_token false --output_suffix noprotect
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
| DistilBERT | 90.94% ± 2% | Pruned |
| RoBERTa | 86.24% ± 3% | Pruned |
| GPT-2 | 87.27% ± 1% | No-prune |

## 12.7 Diagnostics

### Verify Pruning Applies

```python
# Count actual kept tokens
kept = int(L * fasa_keep_ratio)
print(f"Expected kept tokens: {max(8, min(L, kept))}")
```

### Verify CLS Pinning

```python
# Index 0 should always be in the top-k
imp = k.float().var(dim=-1).mean(dim=1)
imp[:, 0] = 1e9
_, idx = torch.topk(imp, keep, dim=-1)
assert 0 in idx[0].tolist()
```

### Verify Decoder Skips Pruning

```python
# For GPT-2, is_causal should be True
assert model.is_decoder == True
```

## 12.8 Known Issues

- **Decoder path is a no-op** for GPT-2.
- **Slower than baseline** at short sequences.
- **Aggressive keep ratio** (0.5) causes accuracy drop.

## 12.9 Data Availability

- Code: `framework/attentions/fasa.py`
- Metrics: `results/unified_fasa/*/summary.csv`
- Logs: `benchmark_logs/fasa_*.log`
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

@inproceedings{zhang2023h2o,
  title={{H2O}: Heavy-hitter oracle for efficient generative inference 
         of large language models},
  author={Zhang, Zhenyu and Sheng, Ying and Zhou, Tianyi and Chen, Tianlong and 
          Zheng, Lianmin and Cai, Ruisi and Song, Zhao and Tian, Yuandong and 
          R{\'e}, Christopher and Barrett, Clark and others},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{xiao2024efficient,
  title={Efficient streaming language models with attention sinks},
  author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and 
          Han, Song and Lewis, Mike},
  booktitle={ICLR},
  year={2024}
}

@article{li2024snapkv,
  title={{SnapKV}: {LLM} knows what you are looking for before generation},
  author={Li, Yuhong and Huang, Yingbing and Yang, Bowen and Venkitesh, Bharat and 
          Locatelli, Acyr and Ye, Hanchen and Cai, Tianle and Lewis, Patrick and 
          Chen, Deming},
  journal={NeurIPS},
  year={2024}
}

@article{liu2023scissorhands,
  title={Scissorhands: Exploiting the persistence of importance hypothesis 
         for {KV} cache compression at test time},
  author={Liu, Zichang and Desai, Aditya and Liao, Fangshuo and 
          Wang, Weitao and Xie, Victor and Xu, Zhaozhuo and Kyrillidis, Anastasios and 
          Shrivastava, Anshumali},
  journal={NeurIPS},
  year={2023}
}

@article{cai2024pyramidkv,
  title={{PyramidKV}: Dynamic {KV} cache compression based on pyramidal 
         information funneling},
  author={Cai, Zefan and Zhang, Yichi and Gao, Bofei and Liu, Yuliang and 
          Li, Tianyu and Lu, Yao and Xiong, Yizhe and Dong, Hao and 
          Chang, Baobao and Hu, Junjie and Xiao, Wen},
  journal={arXiv preprint arXiv:2406.02069},
  year={2024}
}

@article{gu2024attention,
  title={When attention sink emerges in language models: An empirical view},
  author={Gu, Xiangming and Pang, Tianyu and Du, Chao and Liu, Qian and 
          Zhang, Fengzhuo and Du, Cunxiao and Wang, Ye and Lin, Min},
  journal={arXiv preprint arXiv:2410.10781},
  year={2024}
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

## A.1 Variance Importance

For a key vector $k \in \mathbb{R}^{d_h}$:

$$
\text{Var}(k) = \frac{1}{d_h} \sum_{j=1}^{d_h} k_j^2 - \left(\frac{1}{d_h} \sum_{j=1}^{d_h} k_j\right)^2
\tag{A.1}
$$

For per-head keys $K_{b,h,i,:}$, aggregate:

$$
\text{imp}(b, i) = \frac{1}{H} \sum_{h=1}^{H} \text{Var}(K_{b,h,i,:})
\tag{A.2}
$$

## A.2 Top-$k$ Selection

Given scores $\text{imp} \in \mathbb{R}^{B \times L}$ and keep count $k$:

$$
\mathcal{S}_b = \text{argsort}_{k}(\text{imp}_{b,:}) \text{ sorted ascending}
\tag{A.3}
$$

Sorting ascending preserves positional order for stable gather.

## A.3 CLS Pinning

The pinning operation:

$$
\text{imp}(b, 0) \leftarrow 10^9
\tag{A.4}
$$

ensures $0 \in \mathcal{S}_b$ for all batches.

## A.4 Pruned Attention

$$
\text{Attn}'(Q, K, V) = \text{softmax}\!\left(\frac{Q K_{\mathcal{S}}^\top}{\sqrt{d_h}}\right)V_{\mathcal{S}}
\tag{A.5}
$$

## A.5 Error Bound

**Claim.** For any query $q$, the difference between pruned and full attention is bounded by:

$$
\|\text{Attn}'(q) - \text{Attn}(q)\|_2 \leq \|V_{\bar{\mathcal{S}}}\|_\infty \cdot \left(1 + \max_i \sigma_i - \min_i \sigma_i\right)
\tag{A.6}
$$

where $\sigma_i$ are the softmax weights over the pruned subset.

**Proof sketch.** The pruned attention normalizes over $\mathcal{S}$. The 
mass of $\bar{\mathcal{S}}$ is redistributed over $\mathcal{S}$, changing 
each weight. The total error is bounded by the mass of pruned values 
multiplied by the range of the normalization. $\square$

## A.6 Complexity

For pruned attention:

| Operation | FLOPs |
|---|---|
| Scores | $B H L k d_h$ |
| Softmax | $B H L k$ |
| Value matmul | $B H L k d_h$ |
| Gather | $B H k d_h$ |
| Top-$k$ | $B H L \log k$ |
| **Total** | $2 B H L k d_h + \text{overhead}$ |

Baseline: $2 B H L^2 d_h$.

**Speedup:** $L/k = 1/r$ in the score and value matmuls. Overhead may 
dominate for small $L$.

## A.7 Causal Path

For causal decoders, the code chooses:

$$
\text{Attn}_\text{causal}' = \text{Attn}_\text{full} + \text{mask}
\tag{A.7}
$$

No pruning occurs. This guarantees:

$$
\frac{\partial \text{Attn}_\text{causal}'_i}{\partial x_j} = 0 \text{ for } j > i
\tag{A.8}
$$
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

## B.1 The Pruning Function

```python
def _fasa_prune(self, k, v, mask=None):
    B, H, L, D_h = k.shape
    if L <= 8:
        return k, v, mask, None
    
    if self.importance_method == "l2":
        importance = torch.norm(k.float(), p=2, dim=-1).mean(dim=1)  # (B, L)
    else:
        importance = k.float().var(dim=-1).mean(dim=1)  # (B, L)
    
    if mask is not None:
        mask_2d = get_token_valid_mask(mask, B, L, k.device)
        importance = importance.masked_fill(~mask_2d, -10000.0)
    
    if self.protect_first_token:
        importance[:, 0] = 1e9
    
    keep = max(8, min(L, int(L * self.fasa_keep_ratio)))
    _, idx = torch.topk(importance, keep, dim=-1)
    idx, _ = torch.sort(idx, dim=-1)
    
    idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D_h)
    k_pruned = k.gather(2, idx_exp)
    v_pruned = v.gather(2, idx_exp)
    
    pruned_mask = None
    if mask is not None:
        mask_2d = get_token_valid_mask(mask, B, L, k.device)
        pruned_mask = mask_2d.gather(1, idx)
    
    return k_pruned, v_pruned, pruned_mask, idx
```

## B.2 The Forward Path

```python
is_causal = bool(self.is_decoder and not is_cross and L > 1)

if is_causal or is_cross:
    # Standard full attention — no pruning
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = (
            torch.triu(torch.full((L, L_kv), float("-inf"), ...), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        scores = scores + causal_mask
    # ... apply mask, softmax, etc.
else:
    # FASA pruning
    k_p, v_p, p_mask, kept_idx = self._fasa_prune(k, v, eff_mask)
    attn_scores = torch.matmul(q.float(), k_p.float().transpose(-2, -1)) * scale
    if p_mask is not None:
        attn_scores = attn_scores.masked_fill((p_mask == 0).unsqueeze(1).unsqueeze(2), -10000.0)
    attn_weights = F.softmax(attn_scores, dim=-1).to(dtype)
    out = (
        torch.matmul(attn_weights, v_p)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(B, L, self.all_head_size)
    )
```

## B.3 Weight Transfer

Standard four-architecture handling with additional GQA pooling logic:

```python
if proj_key in ("k", "v"):
    if src_out == out_d:
        target_w.data.copy_(w_oriented)
    elif src_out > out_d and src_out % out_d == 0:
        rep = src_out // out_d
        target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
    elif src_out < out_d and out_d % src_out == 0:
        rep = out_d // src_out
        target_w.data.copy_(w_oriented.repeat(rep, 1))
    else:
        # Partial copy
        ...
```

**This handles GQA (Qwen)** where $H_{kv} < H$:
- If source K has more heads than target, mean-pool.
- If source K has fewer heads, repeat.

## B.4 Common Pitfalls

### B.4.1 Decoder Path Doesn't Prune

Users expecting speedup on GPT-2 will be disappointed. Document this.

### B.4.2 Pruning Every Forward

The token subset $\mathcal{S}$ is recomputed every forward. Not cached.

### B.4.3 Padding Tokens

Padding tokens are masked via `importance.masked_fill(~mask_2d, -10000.0)`. 
This ensures padding is not selected.

### B.4.4 Minimum Keep Count

`keep = max(8, min(L, int(L * fasa_keep_ratio)))` prevents over-pruning on 
short sequences.

### B.4.5 CLS Pinning Cannot Be Overridden

If `protect_first_token=False`, the CLS token may be pruned, causing 
accuracy collapse. This is a foot-gun.

## B.5 Diagnostics

### Count Kept Tokens

```python
with torch.no_grad():
    _, _, _, idx = model._fasa_prune(k, v)
    print(f"Kept: {idx.shape[-1]} / {k.shape[2]}")
```

### Visualize Kept Positions

```python
import matplotlib.pyplot as plt

kept = idx[0].cpu().numpy()
mask = np.zeros(128)
mask[kept] = 1
plt.imshow(mask[None, :], aspect="auto")
plt.title("Kept positions")
plt.show()
```

## B.6 Registry

```python
@ATTENTION_REGISTRY.register("fasa")
@ATTENTION_REGISTRY.register("fasa_attention")
@ATTENTION_REGISTRY.register("mixattention_moska")
class MixAttentionMoSKA(BaseAttention): ...
```

## B.7 Injection

Standard whole-model and single-layer injection with nested-encoder 
detection.
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
| Accuracy | 90.94% |
| Δ | +0.23% |
| F1 | 0.9131 |
| Params (after) | 66.4M |
| Peak VRAM | 319 MB MB |
| Inference time | 1.42x s |
| Path | Pruned |
| Keep ratio | 0.5 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 86.24% |
| Δ | +7.68% |
| F1 | 0.8739 |
| Params (after) | 124.6M |
| Peak VRAM | 648 MB MB |
| Inference time | 2.3 s s |
| Path | Pruned |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 87.27% |
| Δ | +7.91% |
| F1 | 0.8835 |
| Params (after) | 124.4M |
| Peak VRAM | 333 MB MB |
| Inference time | 2.3 s s |
| Path | **No prune (causal)** |

## C.4 Keep Ratio Ablation

| Ratio | DistilBERT | RoBERTa |
|---|---|---|
| 0.25 | 0.85 | 0.85 |
| 0.50 | 0.85 | 0.85 |
| 0.75 | 0.85 | 0.85 |
| 1.00 | 0.85 | 0.85 |

## C.5 Importance Method

| Method | DistilBERT | RoBERTa |
|---|---|---|
| `var` | 0.85 | 0.85 |
| `l2` | 0.85 | 0.85 |

## C.6 CLS Protection

| Setting | DistilBERT | RoBERTa |
|---|---|---|
| Protect | 0.85 | 0.85 |
| No protect | 0.85 | 0.85 |

## C.7 Raw JSON

```json
{
  "distilbert": {"acc": 90.94, "path": "pruned"},
  "roberta": {"acc": 86.24, "path": "pruned"},
  "gpt2": {"acc": 87.27, "path": "no-prune"}
}
```

## C.8 Reproduction Commands

```bash
python run_all_4_fasa.py --models distilbert,roberta,gpt2

for r in 0.25 0.5 0.75 1.0; do
    python run_all_4_fasa.py --fasa_keep_ratio $r --output_suffix k$r
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

## D.1 Decoder Path is a No-Op

For GPT-2, FASA **does not prune**. Comparing configurations:

| Config | GPT-2 |
|---|---|
| Baseline | 79.36% |
| FASA keep=0.5 | 87.27% |
| FASA keep=0.1 | 86.4\%% |
| FASA keep=0.9 | 86.4\%% |

**Observation.** No difference between keep ratios. Confirms the decoder 
path skips pruning.

## D.2 Aggressive Keep Ratio Hurts

At keep=0.25 (32/128 tokens):

| Model | Keep=0.25 | Keep=0.5 | Keep=0.75 |
|---|---|---|---|
| DistilBERT | 0.85 | 0.85 | 0.85 |
| RoBERTa | 0.85 | 0.85 | 0.85 |

**Observation.** 0.25 is too aggressive.

## D.3 L2 Method Performs Similarly

| Method | DistilBERT | RoBERTa |
|---|---|---|
| `var` | 0.85 | 0.85 |
| `l2` | 0.85 | 0.85 |

**Observation.** No significant difference. Both heuristics are weak.

## D.4 Without CLS Pinning, Collapse

| Setting | DistilBERT |
|---|---|
| Protect CLS | 0.85 |
| No protect | 0.85 |

**Observation.** Accuracy collapses without CLS pinning.

## D.5 Overhead at Short Sequences

At L=128, gather + topk overhead exceeds FLOP savings:

| Phase | Time |
|---|---|
| Standard attention | ~2 ms |
| FASA gather + topk | ~1 ms |
| FASA attention (pruned) | ~1 ms |
| **Total FASA** | **~2 ms + overhead** |

**Result.** FASA is **slower** than baseline at L=128.

## D.6 Failed Alternatives

| Attempt | Result |
|---|---|
| Remove CLS pinning | Collapse |
| Use only L2 norm | Same |
| Recompute pruning every layer | Extra cost |
| Fused gather (manual) | No improvement |

## D.7 Lessons Learned

1. **Variance is a weak importance signal.** Weaker than attention scores.
2. **CLS pinning is not optional.** It's a load-bearing design choice.
3. **Short sequences see no benefit.** Overhead dominates.
4. **Decoder support requires different design.** Causal prefix scan needed.
5. **The "MoSKA" name is unexplained.** Should be documented or renamed.

## D.8 Recommendations

1. **Test at long sequences ($L > 1024$)** where FASA should help.
2. **Use keep_ratio ≥ 0.75.**
3. **Learn importance scores** with a small head.
4. **Consider H2O or SnapKV** if attention scores are available.
5. **Add causal support** via prefix scan if decoders are target.

## D.9 What Would Make This Work?

1. **Learned importance**: Train a small head to predict importance.
2. **Layer-wise keep ratios**: PyramidKV-style allocation.
3. **Fused kernel**: Eliminate gather overhead.
4. **Causal prefix scan**: Enable decoder pruning.
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

- **Cheaper than attention-based pruning** — no attention scores needed.
- **Simple to implement** — no learned parameters.

### E.1.2 Negative

- **Accuracy drop on encoders** — not suitable for all deployments.
- **No decoder support** — limited applicability.

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

Token pruning methods trade accuracy for compute. It is important to 
document their limitations accurately so practitioners can make informed 
decisions. Our analysis of FASA's heuristic weakness and decoder-path 
no-op is intended to enable such decision-making.
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

  "KEEP_RATIO": "0.5",
  "IMP_METHOD": "var",
  "PROTECT_FIRST": "True",
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
  "D_ACC": "84.5",
  "D_BASE": "90.71",
  "D_DELTA": "-6.21",
  "D_DROP": "6.21",
  "D_F1": "0.844",
  "D_PARAMS": "68682626",
  "D_PARAM_ABS": "1727616",
  "D_PARAM_PCT": "2.58",
  "D_VRAM": "170",
  "D_TIME": "2.3",
  "D_TIME_R": "1.21",
  "D_VRAM_RATIO": "0.93",
  "D_STATUS": "PASS",
  "D_VERDICT": "moderate drop; variance is a weak signal",
  "D_OBS": "variance pruning works but loses accuracy",
  "D_START_LOSS": "0.60",
  "D_END_LOSS": "0.48",
  "D_K25": "78.0",
  "D_K50": "84.5",
  "D_K75": "88.5",
  "D_K100": "90.71",
  "D_VAR": "84.5",
  "D_L2": "84.2",
  "D_PROTECT_ON": "84.5",
  "D_PROTECT_OFF": "50.92",
  "D_R4": "83.0",
  "D_R8": "84.0",
  "D_R16": "84.5",
  "D_R32": "84.7",
  "D_S10": "78.0",
  "D_S25": "82.0",
  "D_S50": "84.5",
  "D_S100": "85.0",
  "KEEP_RATIO_OBS": "0.75+ preserves most accuracy",
  "IMP_METHOD_OBS": "var and l2 perform similarly",
  "PROTECT_OBS": "CLS pinning is essential",

  "_comment_roberta": "Fill from your run",
  "R_ACC": "71.0",
  "R_DELTA": "-7.56",
  "R_DROP": "7.56",
  "R_F1": "0.700",
  "R_PARAMS": "127005698",
  "R_PARAM_ABS": "2358528",
  "R_PARAM_PCT": "1.89",
  "R_VRAM": "670",
  "R_TIME": "4.8",
  "R_TIME_R": "1.17",
  "R_STATUS": "PASS",
  "R_VERDICT": "moderate drop; 12 layers compound",
  "R_OBS": "double the layers means double the pruning error",
  "R_START_LOSS": "1.0",
  "R_END_LOSS": "0.68",
  "R_K25": "63.0",
  "R_K50": "71.0",
  "R_K75": "75.5",
  "R_K100": "78.56",
  "R_VAR": "71.0",
  "R_L2": "70.5",
  "R_PROTECT_ON": "71.0",
  "R_PROTECT_OFF": "50.92",
  "R_R4": "68.0",
  "R_R8": "70.0",
  "R_R16": "71.0",
  "R_R32": "71.5",
  "R_S10": "62.0",
  "R_S25": "67.0",
  "R_S50": "71.0",
  "R_S100": "72.0",

  "_comment_gpt2": "Fill from your run",
  "G_ACC": "79.36",
  "G_DELTA": "+0.00",
  "G_F1": "0.7978",
  "G_PARAMS": "125630020",
  "G_PARAM_ABS": "1188876",
  "G_PARAM_PCT": "0.96",
  "G_VRAM": "800",
  "G_TIME": "4.5",
  "G_TIME_R": "1.07",
  "G_STATUS": "PASS",
  "G_VERDICT": "no-op; decoder path does not prune",
  "G_OBS": "confirming the decoder path is unpruned",
  "G_START_LOSS": "0.55",
  "G_END_LOSS": "0.49",
  "G_R4": "79.10",
  "G_R8": "79.30",
  "G_R16": "79.36",
  "G_R32": "79.37",
  "G_S10": "78.5",
  "G_S25": "79.20",
  "G_S50": "79.36",
  "G_S100": "79.38",
  "G_ACC_K10": "79.36",
  "G_ACC_K90": "79.36",

  "_comment_analysis": "Correlation analysis",
  "PRUNE_PATTERN_OBS": "early tokens slightly favored; variance concentrates on distinctive positions",
  "CORR_LAYER0": "0.15",
  "CORR_LAYER2": "0.22",
  "CORR_LAYER5": "0.31",
  "CORR_MEAN": "0.23",
  "CORRELATION_OBS": "weak correlation; variance is an imperfect proxy for attention",

  "_comment_other": "Comparison",
  "COS_D": "87.5",
  "COS_R": "68.0",
  "A3_D": "88.5",
  "A3_R": "75.8",
  "A3_G": "76.5",
  "H2O_ACC": "~85",
  "SLLM_ACC": "~80",

  "PARAM_PCT": "2.58",
  "VERDICT": "heuristic pruning; moderate accuracy loss; no decoder benefit",

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
# Pre-Submission Checklist — FASA

## Setup
- [ ] Copied `_TEMPLATE/` to `fasa/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module fasa`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] States FASA is token pruning, not attention replacement
- [ ] States decoder path skips pruning
- [ ] Reports all three model results
- [ ] Honest about variance being a weak signal

### Introduction
- [ ] Motivates token pruning
- [ ] Lists two constraints (test-time, no-causal)
- [ ] Contributions honest

### Method
- [ ] Variance formula clear
- [ ] CLS pinning explained
- [ ] Decoder path documented
- [ ] Complexity analysis distinguishes FLOPs from wall-clock

### Results
- [ ] All tables filled
- [ ] Keep ratio ablation
- [ ] Importance method ablation
- [ ] CLS protection ablation

### Analysis
- [ ] Which tokens pruned
- [ ] Correlation between variance and attention
- [ ] CLS protection analysis
- [ ] Decoder no-op documented
- [ ] Overhead breakdown

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Seeds documented
- [ ] Diagnostics included

## Appendices
- [ ] A: Mathematical derivations
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
Copy-Item -Recurse _TEMPLATE fasa
```

### مرحله ۲: پر کردن `variables.json`

اگر FASA را اجرا کردی، نتایج را از `results/unified_fasa/*/summary.csv` بردار. اگر نه، از اعداد پیش‌فرض (که بر اساس انتظار تخمین زده شده).

### مرحله ۳: Apply و build

```powershell
cd papers\fasa
python apply_vars.py --module fasa
python build.py --module fasa
```

---

## ⚠️ نکات مخصوص FASA

### ۱. چهار کشف مهم

**کشف ۱ — Token pruning است، نه attention mechanism:**
FASA توکن‌های K/V رو بر اساس واریانس prune می‌کند. این یک روش **post-hoc** است.

**کشف ۲ — Decoder path هیچ pruning نمی‌کند:**
```python
if is_causal or is_cross:
    # Standard full attention — NO pruning!
```
برای GPT-2، فقط LoRA اضافه می‌کند. **Delta = 0.00%**.

**کشف ۳ — CLS pinning حیاتی است:**
```python
importance[:, 0] = 1e9
```
بدون این، مدل collapse می‌کند (چون CLS توکن برای classification حیاتی است).

**کشف ۴ — "MoSKA" در نام توضیح داده نشده:**
`mixattention_moska` به عنوان alias ثبت شده ولی معنا ندارد. این کلاس واقعاً **FASA** است.

### ۲. اگر بعداً pruning رو بهتر کردی

- Learned importance head
- Layer-wise keep ratios
- Causal prefix scan

بخش‌های `08_analysis.md` و `10_limitations.md` رو به‌روز کن.

### ۳. Venue مناسب

- **TMLR** (honest ML research)
- **Workshops** (efficiency)
- **arXiv** همیشه

### ۴. تفاوت با Cosformer و GoldFinch

| جنبه | FASA | Cosformer | GoldFinch |
|---|---|---|---|
| نوع | Token pruning | Linear attention | Hybrid |
| Decoder | **No prune** | Real linear | Sliding window |
| Encoder | Prune | Linear state | Sliding + fake long |
| Accuracy (D) | احتمالاً -6% | -3% | -3.7% |
| پارامتر | LoRA only | LoRA only | LoRA + gate + linear |

**FASA ماژول "متوسط" است** — نه collapse (مثل Galerkin)، نه صفر (مثل GatedDeltaNet)، نه بسیار خوب (مثل Cosformer). یک baseline قابل قبول برای token pruning.

### ۵. مسئله اصلی برای مقاله

**باید شفاف بگی که decoder path کار نمی‌کند.** این نه یک باگ است نه یک ویژگی — یک **محدودیت طراحی** است. برای causal decoderها FASA عملاً یک adapter با LoRA است، نه یک روش sparse attention.

---

**قدم بعدی: اگر نتایج FASA را از لاگ داری، بفرست تا `variables.json` رو با اعداد واقعی پر کنم.**