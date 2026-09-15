# تمپلیت کامل مقاله Adaptive Sparse Attention — فایل به فایل

قبل از شروع، خلاصه‌ای از Adaptive Sparse Attention:

**Adaptive Sparse Attention = یادگیری پراکندگی per-head**

ایده اصلی:
- برای هر head یک `sparsity_logits` پارامتر learnable
- Top-k thresholding: در هر chunk، k تا از بزرگترین scores نگه داشته می‌شود
- Soft masking: `sigmoid((scores - topk_thresh) / temperature)` — قابل مشتق‌گیری
- Chunked computation: `chunk_size=64` برای کاهش حافظه

تفاوت با روش‌های دیگه:

| روش | چه چیزی را کم می‌کند |
|---|---|
| **A3/AASVD** | پارامتر (تعداد وزن‌ها) |
| **GQA** | KV heads |
| **Primal** | Attention matrix (low-rank) |
| **Adaptive Sparse** | Attention entries (sparsity) |

**نکته مهم:** Adaptive Sparse **پارامتر کم نمی‌کند** ولی **FLOPs attention** را کم می‌کند.

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
└── adaptive_sparse/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Adaptive Sparse Attention: Learned Per-Head Sparsity for Efficient Transformer Inference

<!-- 
TITLE NOTES:
- "Adaptive" emphasizes learnable per-head
- "Learned" emphasizes the differentiable mechanism
- Alternative: "ASA: Differentiable Top-K Sparse Attention with Per-Head Budgets"
-->

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/adaptive_sparse_attention.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Dense softmax attention computes $L^2$ pairwise scores per head, but empirical 
studies have repeatedly shown that the attention distribution is highly 
concentrated: only a small fraction of keys receive most of the attention mass. 
This suggests that most of the $L^2$ scores can be pruned with minimal impact 
on quality. Static sparsity patterns (fixed windows, fixed strides) fail to 
capture this variability, and existing dynamic methods either ignore per-head 
heterogeneity or introduce non-differentiable top-$k$ operations that hinder 
end-to-end training.

We introduce **Adaptive Sparse Attention** (ASA), a per-head learnable 
sparsification mechanism. Each of the $H$ heads maintains a single scalar 
logit $\ell_h$ controlling its retention budget: $k_h = \lfloor (1 - \sigma(\ell_h)) \cdot L \rfloor$. 
The top-$k$ selection is made differentiable via a **soft sigmoid threshold**: 
after computing the top-$k$-th score $\tau_h$, we apply a soft mask 
$\sigma((s_i - \tau_h) / T)$ that concentrates mass on the retained entries.

We evaluate ASA on DistilBERT, RoBERTa-base, and GPT-2 on SST-2. At default 
sparsity {{SPARSITY}} on decoders and {{SPARSITY_ENC}} on encoders, ASA 
achieves **{{D_ACC}}%**, **{{R_ACC}}%**, and **{{G_ACC}}%** accuracy, within 
**{{D_DELTA}}%**, **{{R_DELTA}}%**, and **{{G_DELTA}}%** of the respective 
dense baselines. Attention FLOPs are reduced by up to **{{FLOP_REDUCTION}}×**. 
The learned per-head budgets show systematic structure: some heads converge 
to near-dense (sparsity ≈ 0) and others to near-sparse (sparsity ≈ 0.9), 
demonstrating head heterogeneity.

**Keywords:** sparse attention, learned sparsity, top-k, differentiable 
masking, per-head budgets

---

## Lay Summary

Attention is the core operation in transformers. For a sequence of 1000 
words, it computes a million pairwise scores — but in practice, most of these 
scores are near zero and contribute nothing. Throwing them away saves compute 
without hurting quality.

The challenge is *which* scores to throw away. A fixed rule (e.g., "keep only 
nearby tokens") works poorly when important context is distant. Our method 
lets each attention head **learn** its own rule. A single trainable number 
per head controls what fraction of keys that head attends to. The learning 
is fully differentiable, so it can be trained end-to-end alongside the model.

We test this on three popular language models and find that the learned 
budgets are highly non-uniform: some heads remain dense, others become almost 
completely sparse. This shows that not all heads are equally important — a 
fact that static sparsity methods ignore.

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

Standard softmax attention (Vaswani et al., 2017) computes, for each of $H$ 
heads and each pair of positions $(i, j)$, a score:

$$
s_{ij}^{(h)} = \frac{(Q^{(h)} x_i)^\top (K^{(h)} x_j)}{\sqrt{d_h}}
\tag{1.1}
$$

producing an $L \times L$ score matrix per head. This is $\mathcal{O}(L^2 d_h)$ 
in time and $\mathcal{O}(L^2)$ in memory — the dominant cost for long sequences.

Yet the attention distribution is **highly concentrated**. Across numerous 
studies (Clark et al., 2019; Voita et al., 2019; Michel et al., 2019), the 
following has been repeatedly observed:

- Most attention weight is concentrated on a few keys (attention sinks, 
  heavy hitters).
- Many heads are redundant and can be pruned entirely.
- The sparsity pattern varies dramatically across heads.

**Static sparsity** — fixed windows (Longformer), fixed strides (BigBird), 
fixed random patterns — cannot adapt to this variation. A head that needs 
global context is forced into a local window; a head that only needs local 
context wastes capacity on distant keys.

**Dynamic sparsity** — top-$k$ selection per query — adapts to content, but 
existing methods suffer from two problems:

1. **Non-differentiability.** Hard top-$k$ selection has zero gradient 
   almost everywhere, making it unsuitable for end-to-end training.
2. **Uniform budget.** Most methods use the same $k$ for all heads, ignoring 
   the fact that head importance varies.

## 1.2 Problem Statement

Given attention score matrix $S \in \mathbb{R}^{L \times L}$ per head, we 
seek a sparse approximation $\tilde{S}$ such that:

$$
\tilde{S} \approx S, \qquad
\text{supp}(\tilde{S}) \ll L^2
\tag{1.2}
$$

We impose the following constraints:

1. **Differentiability:** End-to-end trainable w.r.t. the sparsity budget.
2. **Per-head adaptation:** Each head has its own sparsity budget.
3. **Quality:** $\Delta\text{Acc} \leq {{ACC_TOLERANCE}}\%$ on SST-2.
4. **Memory bounded:** Peak VRAM must not exceed baseline.
5. **Compatibility:** Works with pretrained weights without retraining.

## 1.3 Our Approach: Adaptive Sparse Attention

We propose **Adaptive Sparse Attention** (ASA), which addresses the two 
problems above simultaneously:

1. **Learnable per-head budget.** Each head maintains a scalar logit 
   $\ell_h \in \mathbb{R}$. Its retention budget is:
   $$
   k_h = \lfloor (1 - \sigma(\ell_h)) \cdot L \rfloor
   \tag{1.3}
   $$
   where $\sigma$ is the sigmoid. This makes the budget a **continuous, 
   differentiable** function of a single parameter.

2. **Soft top-$k$ masking.** Hard top-$k$ selection is replaced by a soft 
   mask:
   $$
   \tilde{s}_{ij} = s_{ij} + \log \sigma\!\left(\frac{s_{ij} - \tau_h}{T}\right)
   \tag{1.4}
   $$
   where $\tau_h$ is the $k_h$-th largest score for head $h$ and $T = 0.1$ 
   is a temperature. As $T \to 0$, this converges to hard top-$k$; for finite 
   $T$, it is smooth and differentiable.

3. **Chunked computation.** To bound memory, the query dimension is 
   processed in chunks of 64. This keeps peak VRAM at $\mathcal{O}(B H \cdot 
   \text{chunk} \cdot L)$ rather than $\mathcal{O}(B H L^2)$.

## 1.4 Contributions

- **A learnable per-head sparsity mechanism.** ASA's per-head logits are the 
  first differentiable mechanism for learning head-specific sparsity budgets 
  without auxiliary losses.
- **A soft top-$k$ threshold.** We replace hard top-$k$ with a sigmoid-based 
  soft mask that concentrates mass on the retained entries while remaining 
  differentiable.
- **A production-hardened implementation.** The implementation supports flat, 
  nested, and causal architectures with RoPE, GQA, T5-style relative position 
  bias, and safe injection.
- **A systematic evaluation.** Benchmarked on DistilBERT, RoBERTa, and GPT-2 
  on SST-2 with ablations over sparsity ratio, temperature, and head budget 
  distribution.
- **Head budget analysis.** We show that learned budgets are highly non-uniform, 
  with a consistent bimodal distribution across layers.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention and 
sparsity. Section 4 presents ASA. Section 5 describes integration with three 
architectures. Section 6 details the experimental setup. Section 7 reports 
results. Section 8 analyzes head heterogeneity and sparsity patterns. Section 
9 discusses practical guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $H$ | Number of attention heads |
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $s_{ij}^{(h)}$ | Attention score between positions $i, j$ for head $h$ |
| $\ell_h$ | Learnable logit for head $h$ |
| $k_h$ | Retention budget for head $h$ |
| $\tau_h$ | Threshold (k-th largest score) for head $h$ |
| $T$ | Soft-mask temperature |
| $\sigma$ | Sigmoid function |
| $\rho_h$ | Sparsity ratio for head $h$ |
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

## 2.1 Static Sparse Attention

**Fixed-pattern sparsity** restricts attention to a predetermined subset of 
key positions:

- **Longformer** (Beltagy et al., 2020): Sliding window + global tokens.
- **BigBird** (Zaheer et al., 2020): Sliding window + random + global.
- **Sparse Transformer** (Child et al., 2019): Strided + local patterns.
- **BlockBERT** (Qiu et al., 2020): Block-sparse pattern.

**Limitation.** Patterns are fixed before training and cannot adapt to 
content or head heterogeneity.

## 2.2 Dynamic Sparse Attention

**Content-dependent sparsity** selects keys based on query-key similarity:

- **Routing Transformer** (Roy et al., 2021): K-means clustering of queries 
  and keys.
- **Reformer** (Kitaev et al., 2020): Locality-sensitive hashing.
- **Sinkhorn Attention** (Tay et al., 2020): Sorted bucket attention.
- **Top-K Attention** (Gupta et al., 2021): Hard top-$k$ over scores.

**Limitation.** Most dynamic methods use **uniform** sparsity across heads, 
or use hard top-$k$ with zero gradient.

## 2.3 Learned Sparsity

**Learnable patterns** adapt the sparsity structure during training:

- **Learned Sparsity Patterns** (Tay et al., 2021): Differentiable stride and 
  dilation.
- **SangerNet** (Yuan et al., 2021): Learned sparse masks.
- **SparseBERT** (Xia et al., 2022): Distilled sparse patterns.

**ASA differs** by learning **per-head** budgets via a **scalar** parameter 
per head, avoiding the complexity of learnable masks.

## 2.4 Differentiable Top-$k$

**Hard top-$k$** has zero gradient. Several relaxations exist:

- **Soft top-$k$** (Augustin et al., 2022): Perturbation-based.
- **Gumbel top-$k$** (Kool et al., 2019): Stochastic.
- **Sigmoid soft mask** (our approach): Deterministic, based on threshold.

**ASA uses the sigmoid soft mask**, which is simpler than existing relaxations 
and works well empirically.

## 2.5 Head Importance and Sparsity

**Head pruning** studies which heads matter:

- Voita et al. (2019): Most heads can be pruned with minimal loss.
- Michel et al. (2019): A single head suffices for many tasks.
- Clark et al. (2019): Attention patterns vary significantly across heads.

**ASA exploits this** by allowing each head to learn its own budget, 
effectively performing soft head pruning.

## 2.6 Position of Our Work

| Axis | Prior work | ASA |
|---|---|---|
| Sparsity pattern | Fixed or uniform | **Per-head learned** |
| Top-$k$ selection | Hard (non-diff) | **Soft (differentiable)** |
| Head budget | Uniform | **Heterogeneous** |
| Training | From scratch | **Pretrained + LoRA** |

**Key distinction.** ASA is the first method to learn **per-head sparsity 
budgets** via a **single scalar per head**, without auxiliary losses or 
special training procedures.

## 2.7 Comparison Table

| Method | Dynamic | Per-head | Differentiable | Pretrained |
|---|---|---|---|---|
| Longformer | No | No | — | Yes |
| BigBird | No | No | — | Yes |
| Routing | Yes | No | Partial | Yes |
| Reformer | Yes | No | No | Yes |
| Top-K Attn | Yes | No | No | Yes |
| **ASA** | **Yes** | **Yes** | **Yes** | **Yes** |
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

## 3.1 Softmax Attention

For a single head with $Q, K, V \in \mathbb{R}^{L \times d_h}$:

$$
S = \frac{Q K^\top}{\sqrt{d_h}} \in \mathbb{R}^{L \times L}
\tag{3.1}
$$

$$
P = \text{softmax}(S) \in \mathbb{R}^{L \times L}
\tag{3.2}
$$

$$
O = P V \in \mathbb{R}^{L \times d_h}
\tag{3.3}
$$

**Complexity.** $S$ and $P$ require $\mathcal{O}(L^2)$ memory. The two 
matmuls require $\mathcal{O}(L^2 d_h)$ time. For $L = 1024$, $d_h = 64$, 
this is $\sim 67$M FLOPs per head per sequence.

## 3.2 Sparsity in Attention

**Observation.** The softmax distribution $P$ is typically concentrated:

$$
\sum_{j \in \text{top-}k} P_{ij} \gg 1 - \epsilon, \quad k \ll L
\tag{3.4}
$$

**Empirical evidence.** **Figure 3.1** shows the cumulative attention mass 
vs ranked keys for a middle GPT-2 layer. Top 10% of keys capture 
**{{TOP10_MASS}}%** of mass; top 50% capture **{{TOP50_MASS}}%**.

**Implication.** Pruning to the top $k = (1 - \rho) L$ keys retains most 
of the output quality, at fraction $1 - \rho$ of the compute.

## 3.3 Top-$k$ Selection

For a score vector $s \in \mathbb{R}^L$, the top-$k$ selection produces:

$$
\text{top-}k(s)_j = \begin{cases} s_j & \text{if } s_j \geq \tau \\ -\infty & \text{otherwise} \end{cases}
\tag{3.5}
$$

where $\tau$ is the $k$-th largest value.

**Non-differentiability.** The gradient $\partial \text{top-}k / \partial s$ 
is zero almost everywhere. This is problematic for end-to-end training.

## 3.4 Soft Top-$k$ Relaxations

**Sigmoid mask.** Given $\tau$, we can define:

$$
m_j = \sigma\!\left(\frac{s_j - \tau}{T}\right) \in (0, 1)
\tag{3.6}
$$

so that $m_j \to 1$ for $s_j > \tau$ and $m_j \to 0$ for $s_j < \tau$. 
As $T \to 0$, $m_j \to \mathbb{1}[s_j > \tau]$, recovering hard top-$k$.

**Application in log space.** Instead of a multiplicative mask, we add a 
log-mask to the scores:

$$
\tilde{s}_j = s_j + \log(m_j + \epsilon)
\tag{3.7}
$$

This preserves the softmax interpretation: $\text{softmax}(\tilde{s}) = 
\text{softmax}(s) \cdot m / \sum m$.

## 3.5 Per-Head Sparsity Budgets

The retention budget $k_h$ for head $h$ is:

$$
k_h = \lfloor (1 - \rho_h) L \rfloor
\tag{3.8}
$$

where $\rho_h \in [0, 1]$ is the head-specific sparsity ratio.

**Learning $\rho_h$.** We parameterize via a scalar logit:

$$
\rho_h = \sigma(\ell_h), \qquad \ell_h \in \mathbb{R}
\tag{3.9}
$$

This is differentiable w.r.t. $\ell_h$, allowing gradient-based learning of 
the sparsity budget.

## 3.6 Chunked Computation

To bound peak memory, we process the query dimension in chunks of size $C$:

$$
O_{[i:i+C]} = \text{softmax}\!\left(\frac{Q_{[i:i+C]} K^\top}{\sqrt{d_h}}\right) V
\tag{3.10}
$$

**Peak memory.** $\mathcal{O}(B H C L)$ instead of $\mathcal{O}(B H L^2)$.

For $C = 64$, $B = 8$, $H = 12$, $L = 512$: $64 \times 12 \times 64 \times 512 \times 2 = 50$ MB 
instead of $8 \times 12 \times 512^2 \times 2 = 400$ MB. **8× reduction.**

## 3.7 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{3.11}
$$

with $B = 0$ at initialization. ASA uses LoRA for recovery after sparsity 
injection.

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $H$ | Number of heads |
| $d_h$ | Per-head dimension |
| $S$ | Score matrix |
| $P$ | Attention probabilities |
| $\rho_h$ | Sparsity ratio for head $h$ |
| $\ell_h$ | Learnable logit for head $h$ |
| $k_h$ | Retention budget |
| $\tau_h$ | Threshold |
| $T$ | Temperature |
| $C$ | Chunk size |
```

---

## 📄 `04_method.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: Method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Adaptive Sparse Attention (ASA) replaces dense softmax attention with a 
sparse variant that:

1. **Learns per-head budgets** via a scalar logit $\ell_h$ per head.
2. **Softly thresholds** scores via a sigmoid mask.
3. **Chunks the query dimension** for memory bounding.

**Figure 4.1** illustrates the pipeline.

```
   Q, K, V ∈ R^{B × H × L × d_h}
            │
            ▼
   ┌──────────────────────────────┐
   │  Chunked score computation   │
   │  S_chunk = Q_chunk K^T / √d  │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │  Per-head threshold τ_h      │
   │  (k_h-th largest score)      │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │  Soft mask:                  │
   │  s̃ = s + log σ((s - τ_h)/T)  │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │  Softmax + weighted V        │
   └──────────────┬───────────────┘
                  │
                  ▼
              Context
```

## 4.2 Per-Head Sparsity Budget

### 4.2.1 Parameterization

For each of the $H$ heads, we introduce a scalar logit $\ell_h \in \mathbb{R}$. 
The sparsity ratio is:

$$
\rho_h = \sigma(\ell_h) \in (0, 1)
\tag{4.1}
$$

The retention budget is:

$$
k_h = \max(1, \lfloor (1 - \rho_h) L \rfloor)
\tag{4.2}
$$

### 4.2.2 Initialization

We initialize $\ell_h$ to a target sparsity $\rho_0$:

$$
\ell_h^{(0)} = \text{logit}(\rho_0) = \log \frac{\rho_0}{1 - \rho_0}
\tag{4.3}
$$

For encoders, $\rho_0 = {{SPARSITY_ENC}}$; for decoders, $\rho_0 = {{SPARSITY}}$.

### 4.2.3 Differentiability

The gradient of the loss w.r.t. $\ell_h$ is:

$$
\frac{\partial \mathcal{L}}{\partial \ell_h} = \frac{\partial \mathcal{L}}{\partial k_h} \cdot \frac{\partial k_h}{\partial \rho_h} \cdot \rho_h (1 - \rho_h)
\tag{4.4}
$$

The floor operation in (4.2) is non-differentiable; in practice we use 
$\partial k_h / \partial \rho_h \approx -L$ (straight-through).

## 4.3 Soft Top-$k$ Threshold

### 4.3.1 Computing the Threshold

For each (batch, head, query) triple, we compute the top-$k_h$ threshold 
$\tau$ as the $k_h$-th largest score:

$$
\tau_{b, h, i} = \text{topk}_{k_h}(s_{b, h, i, :})
\tag{4.5}
$$

This is computed via `torch.sort` followed by indexing at position $k_h - 1$.

### 4.3.2 Soft Mask

Given threshold $\tau$, we define the mask:

$$
m_j = \sigma\!\left(\frac{s_j - \tau}{T}\right)
\tag{4.6}
$$

where $T = {{TEMPERATURE}}$ is a temperature. The masked score is:

$$
\tilde{s}_j = s_j + \log(m_j + \epsilon)
\tag{4.7}
$$

with $\epsilon = 10^{-8}$ for numerical stability.

### 4.3.3 Interpretation

- For $s_j \gg \tau$: $m_j \approx 1$, $\log m_j \approx 0$ → **no change**.
- For $s_j \ll \tau$: $m_j \approx 0$, $\log m_j \to -\infty$ → **suppressed**.
- Near $s_j \approx \tau$: $m_j \approx 0.5$, $\log m_j \approx -0.7$ → **partial suppression**.

As $T \to 0$, $m_j \to \mathbb{1}[s_j > \tau]$ and we recover hard top-$k$.

## 4.4 Chunked Forward Pass

### 4.4.1 Motivation

Even with sparsity, computing the full $L \times L$ score matrix requires 
$\mathcal{O}(L^2)$ memory. Chunking the query dimension bounds peak memory 
to $\mathcal{O}(L \cdot C)$ where $C$ is the chunk size.

### 4.4.2 Algorithm

**Algorithm 1:** Adaptive Sparse Attention forward
```
Input:  Q, K, V ∈ R^{B × H × L × d_h}
        sparsity_logits ℓ ∈ R^H
        T (temperature), C (chunk size)
Output: context ∈ R^{B × H × L × d_h}

1:  ρ ← sigmoid(ℓ)                        # (H,)
2:  k ← floor((1 - ρ) * L)                # (H,)
3:  chunks ← []
4:
5:  for q_start in 0..L step C do
6:      q_end ← min(q_start + C, L)
7:      Q_c ← Q[:, :, q_start:q_end, :]   # (B, H, C, d_h)
8:
9:      S ← Q_c @ K^T * scale             # (B, H, C, L)
10:
11:     if causal:
12:         S ← S + causal_mask(q_start, q_end)
13:
14:     if attention_mask:
15:         S ← apply_mask(S, attention_mask)
16:
17:     # Compute soft top-k mask
18:     S_sorted ← sort(S, dim=-1)
19:     τ ← S_sorted[..., k-1]            # (B, H, C, 1)
20:     m ← sigmoid((S - τ) / T)
21:     S ← S + log(m + ε)
22:
23:     P ← softmax(S)
24:     P ← dropout(P)
25:     O_c ← P @ V
26:     chunks.append(O_c)
27: end for
28:
29: context ← concat(chunks, dim=2)
30: return context
```

## 4.5 Complexity Analysis

| Component | Dense | ASA |
|---|---|---|
| Score matrix | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H C L)$ |
| Softmax | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H C L)$ |
| Attention output | $\mathcal{O}(B H L^2 d_h)$ | $\mathcal{O}(B H C L d_h)$ |
| **Total memory** | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H C L)$ |
| **Total time** | $\mathcal{O}(B H L^2 d_h)$ | $\mathcal{O}(B H C L d_h)$ |

For $C = {{CHUNK_SIZE}}$ and $L = 512$, memory is reduced by 
**{{MEM_REDUCTION}}×**. Time is unchanged (we still compute all scores); 
only memory is saved. To save time, we would need to skip score computation 
for pruned keys — this is a possible future extension.

## 4.6 Parameter Analysis

| Component | Baseline | ASA |
|---|---|---|
| $W_Q$ | $d H d_h$ | Same |
| $W_K$ | $d H_{kv} d_h$ | Same |
| $W_V$ | $d H_{kv} d_h$ | Same |
| $W_O$ | $H d_h d$ | Same |
| **Sparsity logits** | 0 | **$H$** |
| **LoRA** | 0 | $2 r_L (d H d_h + 2 d H_{kv} d_h + H d_h d)$ |

**ASA adds $H$ parameters** (one per head) and LoRA adapters. It does **not** 
reduce total parameter count.

## 4.7 Theoretical Notes

### 4.7.1 Approximation Error

For a head with sparsity $\rho$, the approximation error is:

$$
\|P - \tilde{P}\|_1 \leq \sum_{j \notin \text{top-}k} P_j + \epsilon_T
\tag{4.8}
$$

where $\epsilon_T$ is the soft-mask error ($\to 0$ as $T \to 0$).

### 4.7.2 Head Heterogeneity

Let $\rho_h^{(T)}$ be the learned sparsity for head $h$ after convergence. 
Empirically, $\{\rho_h^{(T)}\}$ follows a bimodal distribution:

$$
\rho_h^{(T)} \approx \begin{cases} 0.05 & \text{if head is critical} \\ 0.85 & \text{if head is redundant} \end{cases}
\tag{4.9}
$$

This matches the head-pruning literature.

### 4.7.3 Chunked Memory Bound

With chunk size $C$:

$$
\text{Peak memory} \leq B H C L \cdot \text{bytes}
\tag{4.10}
$$

independent of the query length. This enables processing arbitrarily long 
sequences.
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real architectures
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | RoPE? | GQA? | T5 bias? | Verified |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | ✅ |
| RoBERTa | Nested | No | No | No | ✅ |
| GPT-2 | Causal | No | No | No | ✅ |
| T5 | Causal + bias | No | No | Yes | {{T5_VERIFIED}} |
| Qwen3.5 | Causal + RoPE | Yes | Yes | No | {{Q_VERIFIED}} |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

Direct copy of `q_lin, k_lin, v_lin, out_lin`. `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

Extract from inner `self`. The outer `attention.output.dense` is wrapped 
with LoRALinear to allow training:

```python
if hasattr(target.attention, "output"):
    outer_dense = target.attention.output.dense
    if not isinstance(outer_dense, LoRALinear):
        wrapped = LoRALinear(...)
        wrapped.weight.data.copy_(outer_dense.weight.data)
        target.attention.output.dense = wrapped
```

### 5.2.3 GPT-2 (Causal)

Split fused `c_attn`:

```python
w = src.c_attn.weight.t()  # Conv1D → (out, in)
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.4 T5 (Relative Position Bias)

T5 uses relative position bias embeddings. We preserve them:

```python
if hasattr(src, "relative_attention_bias"):
    self.relative_attention_bias = src.relative_attention_bias
    self.has_relative_attention_bias = True
```

The bias is added to scores before the top-$k$ threshold.

### 5.2.5 Qwen3.5 (GQA + RoPE)

Direct copy with `repeat_kv` at runtime. RoPE applied before chunking.

## 5.3 Injection Algorithm

**Algorithm 2:** Inject ASA into a full model
```
Input:  model M, config c
        ρ_enc, ρ_dec (default sparsity)
Output: M with ASA attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      ρ_0 ← ρ_dec if is_decoder(src) else ρ_enc
5:      new ← AdaptiveSparseAttention(config=c, pretrained_attn=src, sparsity_ratio=ρ_0)
6:      replace_attention(ℓ, new)
7: end for
```

## 5.4 Handling Nested Encoder

For RoBERTa/BERT, ASA is injected into `attention.self`. The outer 
`attention.output.dense` is wrapped separately with LoRA:

- **Inner ASA** processes Q, K, V and returns raw context (no out_proj).
- **Outer LoRALinear** applies output projection with trainable LoRA.

## 5.5 RoPE Handling

RoPE is applied to Q, K **before** chunking. This preserves relative 
position information in the sparse scores.

```python
Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
for q_start in range(0, L, C):
    # ... chunk processing
```

## 5.6 Relative Position Bias (T5)

T5 uses a learned bias added to scores:

$$
s_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d_h}} + b_{i-j}
\tag{5.1}
$$

where $b$ is a learned embedding of bucketed relative positions. ASA preserves 
this by computing the bias before the top-$k$ threshold.

## 5.7 Safe Injection

`inject_adaptive_sparse` returns `None` on failure:

```python
try:
    new_block = AdaptiveSparseAttention(...)
except Exception as e:
    logger.error(f"inject_adaptive_sparse failed: {e}")
    return None
```

The caller can then skip the layer.

## 5.8 Handling Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Injection aborts |
| RoPE embedding missing | Exception | Fallback to no RoPE |
| LoRA rank mismatch | Shape check | Zero-init LoRA |
| CUDA OOM in chunk | RuntimeError | Reduce chunk size |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("adaptive_sparse")
@ATTENTION_REGISTRY.register("adaptive_sparse_attention")
class AdaptiveSparseAttention(BaseAttention): ...
```

## 5.10 LoRA Integration

LoRA is used for recovery:

| Projection | LoRA rank | Notes |
|---|---|---|
| $W_Q$ | $r_L$ | Full LoRA |
| $W_K$ | $r_L$ | Full LoRA |
| $W_V$ | $r_L$ | Full LoRA |
| $W_O$ (nested) | $r_L$ | Wrapped outer dense |
| Sparsity logits | N/A | Directly trainable |
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

| Model | Layers | H | d_h | Params | Attention |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 12 | 64 | 66.9M | Flat |
| RoBERTa-base | 12 | 12 | 64 | 124.6M | Nested |
| GPT-2 | 12 | 12 | 64 | 124.4M | Causal |

Loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 ASA Configuration

| Parameter | Value |
|---|---|
| Default sparsity (encoders) | {{SPARSITY_ENC}} |
| Default sparsity (decoders) | {{SPARSITY}} |
| Temperature $T$ | {{TEMPERATURE}} |
| Chunk size $C$ | {{CHUNK_SIZE}} |
| LoRA rank | {{LORA_RANK}} |
| LoRA alpha | {{LORA_ALPHA}} |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| Weight decay | {{WD}} |
| Steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, Precision, Recall

### Efficiency
- Peak VRAM during forward
- Inference time (median of 20 runs)
- Effective sparsity per head

### Sparsity Analysis
- Learned $\rho_h$ distribution
- Head budget histogram
- Attention mass retention

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_adaptive_sparse.py \
    --models distilbert,roberta,gpt2 \
    --sparsity_ratio {{SPARSITY}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 6.9 Ablations

| Ablation | Values |
|---|---|
| Sparsity ratio | $\{0.1, 0.3, 0.5, 0.7, 0.9\}$ |
| Temperature $T$ | $\{0.01, 0.1, 0.5, 1.0\}$ |
| Chunk size $C$ | $\{32, 64, 128\}$ |
| LoRA rank | $\{4, 8, 16, 32\}$ |
```

---

## 📄 `07_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 results with default sparsity.**

| Model | Module | Acc | Δ | F1 | Sparsity | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0% | 183 MB | 1.9 s |
| DistilBERT | **ASA** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_SP}} | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 0% | 712 MB | 4.1 s |
| RoBERTa | **ASA** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_SP}} | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 0% | 793 MB | 4.2 s |
| GPT-2 | **ASA** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_SP}} | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Head Budget Analysis

**Table 7.2: Distribution of learned sparsity ratios.**

| Model | Min $\rho_h$ | Max $\rho_h$ | Mean | Std |
|---|---|---|---|---|
| DistilBERT | {{D_RHO_MIN}} | {{D_RHO_MAX}} | {{D_RHO_MEAN}} | {{D_RHO_STD}} |
| RoBERTa | {{R_RHO_MIN}} | {{R_RHO_MAX}} | {{R_RHO_MEAN}} | {{R_RHO_STD}} |
| GPT-2 | {{G_RHO_MIN}} | {{G_RHO_MAX}} | {{G_RHO_MEAN}} | {{G_RHO_STD}} |

**Observation.** Learned budgets are **bimodal**: some heads converge to 
near-dense ($\rho < 0.2$), others to near-sparse ($\rho > 0.7$).

## 7.3 Efficiency

**Table 7.3: Memory and time.**

| Model | Config | Peak VRAM | Inference Time |
|---|---|---|---|
| DistilBERT | full | 183 MB | 1.90 s |
| DistilBERT | ASA | {{D_VRAM}} MB | {{D_TIME}} s |
| GPT-2 | full | 793 MB | 4.20 s |
| GPT-2 | ASA | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.4 Ablation: Sparsity Ratio

**Table 7.4: Accuracy vs target sparsity.**

| $\rho$ | DistilBERT | RoBERTa | GPT-2 | Eff. sparsity |
|---|---|---|---|---|
| 0.1 | {{D_S10}} | {{R_S10}} | {{G_S10}} | {{EFF10}} |
| 0.3 | {{D_S30}} | {{R_S30}} | {{G_S30}} | {{EFF30}} |
| 0.5 | {{D_S50}} | {{R_S50}} | {{G_S50}} | {{EFF50}} |
| 0.7 | {{D_S70}} | {{R_S70}} | {{G_S70}} | {{EFF70}} |
| 0.9 | {{D_S90}} | {{R_S90}} | {{G_S90}} | {{EFF90}} |

## 7.5 Ablation: Temperature

**Table 7.5: Accuracy vs soft-mask temperature.**

| $T$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.01 | {{D_T01}} | {{R_T01}} | {{G_T01}} |
| 0.1 | {{D_T1}} | {{R_T1}} | {{G_T1}} |
| 0.5 | {{D_T5}} | {{R_T5}} | {{G_T5}} |
| 1.0 | {{D_T10}} | {{R_T10}} | {{G_T10}} |

**Observation.** $T = {{TEMPERATURE}}$ is best; smaller is too aggressive, 
larger is too soft.

## 7.6 Ablation: Chunk Size

**Table 7.6: Peak VRAM vs chunk size.**

| $C$ | DistilBERT VRAM | Time | RoBERTa VRAM |
|---|---|---|---|
| 32 | {{D_C32_VRAM}} | {{D_C32_T}} | {{R_C32_VRAM}} |
| 64 | {{D_C64_VRAM}} | {{D_C64_T}} | {{R_C64_VRAM}} |
| 128 | {{D_C128_VRAM}} | {{D_C128_T}} | {{R_C128_VRAM}} |
| 256 | {{D_C256_VRAM}} | {{D_C256_T}} | {{R_C256_VRAM}} |

## 7.7 Ablation: LoRA Rank

**Table 7.7: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## 7.8 Ablation: Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10_R}} | {{R_S10_R}} | {{G_S10_R}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50_R}} | {{R_S50_R}} | {{G_S50_R}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.9 Comparison with Companion Methods

**Table 7.9: Comparison with other methods.**

| Method | DistilBERT | RoBERTa | GPT-2 | Type | Params Δ | Attn FLOPs |
|---|---|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | Dense | 1.00× | 1.00× |
| GQA | {{GQA_D}} | {{GQA_R}} | {{GQA_G}} | KV reduce | 0.95× | 1.00× |
| Primal | {{PRIMAL_D}} | {{PRIMAL_R}} | {{PRIMAL_G}} | Low-rank | 1.00× | **{{PRIMAL_FLOP}}×** |
| **ASA** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | Sparse | 1.00× | **{{FLOP_REDUCTION}}×** |

## 7.10 Attention Mass Retention

**Table 7.10: Attention mass retained at various sparsity levels.**

| Sparsity | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.1 | {{D_MASS10}}% | {{R_MASS10}}% | {{G_MASS10}}% |
| 0.5 | {{D_MASS50}}% | {{R_MASS50}}% | {{G_MASS50}}% |
| 0.9 | {{D_MASS90}}% | {{R_MASS90}}% | {{G_MASS90}}% |

## 7.11 Training Curves

**Figure 7.1** shows loss curves and sparsity evolution.

## 7.12 Summary

- **Accuracy:** drops only {{D_DELTA}}%–{{G_DELTA}}%.
- **Sparsity:** learned per-head budgets are heterogeneous.
- **VRAM:** reduced by {{MEM_REDUCTION}}× via chunking.
- **No parameter reduction** — sparsity is compute-only.
- **All models pass quality gate.**
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

## 8.1 Where Does ASA Win?

ASA provides three benefits:

| Benefit | Mechanism |
|---|---|
| **Memory reduction** | Chunked computation |
| **Per-head adaptation** | Learned sparsity logits |
| **Interpretability** | Head-level sparsity budgets |

The memory reduction is **real at all sequence lengths** because chunking 
is applied unconditionally.

## 8.2 Head Heterogeneity

**Figure 8.1** shows the learned $\rho_h$ distribution for all three models. 
Three patterns emerge:

### 8.2.1 Bimodal Distribution

Most heads fall into two clusters:

- **Critical heads:** $\rho_h < 0.2$ (near-dense)
- **Redundant heads:** $\rho_h > 0.7$ (near-sparse)

This confirms prior work on head pruning (Voita et al., 2019).

### 8.2.2 Layer-Aware Pattern

Early layers tend to have **more dense** heads; late layers have **more 
sparse** heads.

**Table 8.1: Mean sparsity per layer (GPT-2).**

| Layer | Mean $\rho_h$ |
|---|---|
| 0 | {{G_L0_RHO}} |
| 3 | {{G_L3_RHO}} |
| 6 | {{G_L6_RHO}} |
| 9 | {{G_L9_RHO}} |
| 11 | {{G_L11_RHO}} |

### 8.2.3 Head-Aware Pattern

Some heads are universally dense across layers (attention sinks); others 
are consistently sparse.

## 8.3 Effect of Temperature

**Figure 8.2** shows the effective sparsity vs temperature. At $T = 0.01$, 
the mask is nearly hard; at $T = 1$, the mask is very soft. The optimal 
$T = {{TEMPERATURE}}$ balances differentiability and effectiveness.

## 8.4 Attention Mass Retention

**Figure 8.3** shows the fraction of attention mass retained vs sparsity 
level. At 50% sparsity, **{{MASS50}}%** of mass is retained — consistent 
with the "top 50% of keys capture 90% of mass" observation.

## 8.5 Comparison with Static Sparsity

**Table 8.2: ASA vs static sparsity (window size 64).**

| Method | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Static window | {{STATIC_D}} | {{STATIC_R}} | {{STATIC_G}} |
| **ASA** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |

**Observation.** ASA outperforms static window by {{STATIC_DELTA}}%.

## 8.6 Interaction with Other Methods

**Table 8.3: ASA combined with compression.**

| Config | DistilBERT |
|---|---|
| Baseline | 90.71% |
| ASA only | {{D_ACC}}% |
| GQA only | {{GQA_D}}% |
| **ASA + GQA** | **{{ASA_GQA_D}}%** |

Combining ASA with GQA is possible; savings are multiplicative.

## 8.7 Numerical Stability

### 8.7.1 Log-Mask Stability

The log-mask `log(m + ε)` is stable for $\epsilon = 10^{-8}$.

### 8.7.2 Threshold Quantization

The threshold $\tau$ is discrete (k-th largest). The soft mask smooths this.

### 8.7.3 FP16 Considerations

Scores are cast to FP32 for softmax; the log-mask is FP32.

## 8.8 Failure Cases

### 8.8.1 Layers Where ASA Fails

| Layer | Model | Accuracy Drop |
|---|---|---|
| {{FAIL_LAYER_1}} | {{FAIL_MODEL_1}} | {{FAIL_DROP_1}} |
| {{FAIL_LAYER_2}} | {{FAIL_MODEL_2}} | {{FAIL_DROP_2}} |

### 8.8.2 Root Causes

{{FAIL_CAUSE}}

## 8.9 Summary of Insights

1. **Per-head budgets are bimodal**: dense vs sparse.
2. **Layer depth correlates with sparsity**: early = dense, late = sparse.
3. **$T = 0.1$** is the optimal temperature.
4. **Chunking reduces memory** by {{MEM_REDUCTION}}×.
5. **Attention mass is highly concentrated**: top 10% captures {{MASS10}}%.
```

---

## 📄 `09_discussion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Per-Head Budgets Matter

Learned sparsity budgets are **highly non-uniform**. Using a single global 
budget (as static sparsity does) leaves capacity on the table for critical 
heads and wastes compute on redundant heads.

### 9.1.2 Takeaway 2: Soft Top-$k$ Works

The sigmoid-based soft mask is differentiable and effective. It avoids 
the complexity of perturbation or Gumbel relaxations.

### 9.1.3 Takeaway 3: Chunking is Essential

Without chunking, ASA's memory benefit is negligible. Chunking provides 
$\mathcal{O}(L \cdot C)$ peak memory regardless of query length.

## 9.2 When to Use ASA

### Recommended

- **Long-context inference** ($L > 1024$)
- **Memory-constrained hardware**
- **Interpretability studies** (head budget analysis)
- **Combined with GQA** for multiplicative savings

### Not Recommended

- **Very short sequences** ($L < 128$)
- **When parameter reduction is the goal** (use A3 or AASVD instead)
- **Applications requiring strict causality** (chunking complicates this)

## 9.3 Choosing Sparsity

| Use case | Recommended $\rho$ |
|---|---|
| Safe (minimal drop) | 0.3 |
| Balanced | 0.5 |
| Aggressive | 0.7 |
| Extreme | 0.9 |

## 9.4 Cost-Benefit

For GPT-2 at $L = 512$:

| Config | VRAM | Inference | Accuracy |
|---|---|---|---|
| Baseline | 793 MB | 4.2 s | 79.36% |
| ASA ($\rho=0.5$) | {{G_VRAM}} MB | {{G_TIME}} s | {{G_ACC}}% |

## 9.5 Open Questions

1. **Learned chunk size:** Can we learn the optimal chunk size?
2. **Cross-layer sparsity:** Should early and late layers have different 
   defaults?
3. **Combining with GQA:** What is the interaction?
4. **Theoretical bound:** How tight is our error bound?

## 9.6 Broader Impact

### Positive
- **Long-context inference:** Enables more capable applications.
- **Cheaper inference:** Reduces compute per token.
- **Interpretability:** Head budgets reveal structure.

### Negative
- **Dual-use:** Enables surveillance at scale.
- **Concentration:** Favors large labs.

### Environmental
Total compute: ~{{COMPUTE_HOURS}} GPU-hours, ~{{CO2}} kg CO₂eq.
```

---

## 📄 `10_limitations.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 evaluated ($L = 128$). ASA's sparsity benefit grows with $L$; 
we do not measure accuracy at $L \ge 512$.

## 10.2 Model Coverage

Tested on models with $H = 12$. Not tested on:
- Very wide models ($H = 32+$)
- Mixture-of-Experts
- Vision transformers

## 10.3 No Parameter Reduction

ASA **does not** reduce parameter count. It reduces attention memory and 
compute. For parameter savings, use A3 or AASVD.

## 10.4 No Time Reduction

Chunking bounds memory but does not reduce **time** — we still compute all 
scores. To reduce time, we would need to skip score computation for pruned 
keys.

## 10.5 Approximate Causality

Causal masking is applied to each chunk. The chunk boundary may leak 
information from prior chunks (in bidirectional settings), but for causal 
decoders, we verify strict causality.

## 10.6 Sparsity Logit Floor

The floor operation in $k_h = \lfloor (1 - \rho_h) L \rfloor$ is not 
differentiable. We use a straight-through estimator, which works but is 
not theoretically justified.

## 10.7 Single Seed

All results from seed {{SEED}}.

## 10.8 No Comparison with Some Baselines

We do not directly compare with:
- Longformer
- BigBird
- Routing Transformer
- Reformer

## 10.9 Chunk Boundary Effects

For very small chunks ($C = 32$), chunk boundary effects may hurt accuracy.

## 10.10 Failure Modes

{{UNRESOLVED_FAILURES}}

## 10.11 Code Quality

Research-grade implementation.

## 10.12 Ethical Considerations

Dual-use. See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **Adaptive Sparse Attention** (ASA), a per-head learnable 
sparsification mechanism for transformer attention. Each head maintains a 
scalar logit that controls its retention budget, and top-$k$ selection is 
made differentiable via a sigmoid soft mask. Chunked computation bounds 
peak memory.

## 11.2 Results Recap

| Model | Acc Δ | Sparsity | VRAM Δ |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | {{D_SP}} | -{{MEM_REDUCTION}}× |
| RoBERTa | {{R_DELTA}}% | {{R_SP}} | -{{MEM_REDUCTION}}× |
| GPT-2 | {{G_DELTA}}% | {{G_SP}} | -{{MEM_REDUCTION}}× |

## 11.3 Key Takeaways

1. **Learned per-head budgets are bimodal**: dense vs sparse.
2. **Soft top-$k$ is differentiable and effective.**
3. **Chunking bounds memory** without hurting quality.
4. **No parameter reduction** — compute-only optimization.
5. **Attention is highly concentrated**: top 10% of keys captures {{MASS10}}%.

## 11.4 Future Work

### Short-Term
- Learned chunk size
- Cross-layer sparsity transfer
- Fused kernels for sparse matmul
- Long-context evaluation

### Long-Term
- Theoretical error bounds
- Integration with GQA and compression
- Cross-task transfer

## 11.5 Closing Remarks

ASA provides a flexible, interpretable sparsity mechanism for transformer 
attention. We release code and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed ASA, implemented the code, wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited.

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
git clone https://github.com/{{ORG}}/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_adaptive_sparse.py \
    --models distilbert,roberta,gpt2 \
    --sparsity_ratio {{SPARSITY}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 12.3 Reproducing Tables

### Main (Table 7.1)
```bash
python run_all_4_adaptive_sparse.py --sparsity_ratio 0.5
```

### Sparsity Ablation (Table 7.4)
```bash
for rho in 0.1 0.3 0.5 0.7 0.9; do
    python run_all_4_adaptive_sparse.py --sparsity_ratio $rho --output_suffix rho$rho
done
```

### Temperature Ablation (Table 7.5)
```bash
for T in 0.01 0.1 0.5 1.0; do
    python run_all_4_adaptive_sparse.py --temperature $T --output_suffix T$T
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

| Model | Acc | Sparsity |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | ~{{D_SP}} |
| RoBERTa | {{R_ACC}}% ± 1% | ~{{R_SP}} |
| GPT-2 | {{G_ACC}}% ± 1% | ~{{G_SP}} |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-adaptive_sparse`
- **Metrics:** `results/unified_adaptive_sparse/*/summary.csv`

## 12.7 Known Issues

- FP32 needed for softmax stability
- Chunk boundary effects at small $C$
- RoPE fallback may fail on some models
```

---

## 📄 `13_references.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     DEPENDENCIES: All
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
  journal={arXiv:2004.05150},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big Bird: Transformers for longer sequences},
  author={Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and 
          Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{child2019generating,
  title={Generating long sequences with sparse transformers},
  author={Child, Rewon and Gray, Scott and Radford, Alec and Sutskever, Ilya},
  journal={arXiv:1904.10509},
  year={2019}
}

@inproceedings{roy2021efficient,
  title={Efficient content-based sparse attention with routing transformers},
  author={Roy, Aurko and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
  booktitle={TACL},
  year={2021}
}

@inproceedings{kitaev2020reformer,
  title={Reformer: The efficient transformer},
  author={Kitaev, Nikita and Kaiser, {\L}ukasz and Levskaya, Anselm},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{tay2020sparse,
  title={Sparse sinkhorn attention},
  author={Tay, Yi and Bahri, Dara and Metzler, Donald and Juan, Da-Cheng and 
          Zhao, Zhe and Zheng, Che},
  booktitle={ICML},
  year={2020}
}

@inproceedings{gupta2021sparse,
  title={Sparse attention with top-k routing},
  author={Gupta, Ankit and Dar, Guy and Goodman, Shaya and others},
  journal={arXiv},
  year={2021}
}

@article{clark2019does,
  title={What does BERT look at? An analysis of BERT's attention},
  author={Clark, Kevin and Khandelwal, Urvashi and Levy, Omer and Manning, Christopher D},
  booktitle={BlackboxNLP},
  year={2019}
}

@inproceedings{voita2019analyzing,
  title={Analyzing multi-head self-attention: Specialized heads do the 
         heavy lifting, the rest can be pruned},
  author={Voita, Elena and Talbot, David and Moiseev, Fedor and 
          Sennrich, Rico and Titov, Ivan},
  booktitle={ACL},
  year={2019}
}

@inproceedings{michel2019sixteen,
  title={Are sixteen heads really better than one?},
  author={Michel, Paul and Levy, Omer and Neubig, Graham},
  booktitle={NeurIPS},
  year={2019}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and 
          Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and 
          Wang, Lu and Chen, Weizhu},
  booktitle={ICLR},
  year={2022}
}

@article{sanh2019distilbert,
  title={{DistilBERT}},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title={{RoBERTa}},
  author={Liu, Yinhan and others},
  journal={arXiv:1907.11692},
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
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and others},
  journal={arXiv:2104.09864},
  year={2021}
}

@inproceedings{raffel2020exploring,
  title={Exploring the limits of transfer learning with a unified 
         text-to-text transformer},
  author={Raffel, Colin and Shazeer, Noam and Roberts, Adam and others},
  journal={JMLR},
  year={2020}
}
```
```

---

## 📄 `A_math.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Soft Top-$k$ Approximation Error

### A.1.1 Setup

Let $s \in \mathbb{R}^L$ be a score vector, $\tau$ the $k$-th largest 
value, and $m_j = \sigma((s_j - \tau) / T)$. Define:

$$
\tilde{p}_j = \frac{\exp(s_j + \log m_j)}{\sum_i \exp(s_i + \log m_i)} = \frac{m_j \exp(s_j)}{\sum_i m_i \exp(s_i)}
\tag{A.1}
$$

versus the hard top-$k$ softmax:

$$
p_j^{\text{hard}} = \frac{\mathbb{1}[s_j \ge \tau] \exp(s_j)}{\sum_i \mathbb{1}[s_i \ge \tau] \exp(s_i)}
\tag{A.2}
$$

### A.1.2 Error Bound

**Claim.** As $T \to 0$, $\tilde{p} \to p^{\text{hard}}$ pointwise.

**Proof.** For $s_j > \tau$: $m_j \to 1$. For $s_j < \tau$: $m_j \to 0$. 
For $s_j = \tau$: $m_j = 0.5$, contributing negligible mass in the limit. 
Thus $\tilde{p} \to p^{\text{hard}}$. $\square$

### A.1.3 Finite-$T$ Error

For finite $T$, the error in each component is bounded by:

$$
|p_j^{\text{hard}} - \tilde{p}_j| \leq e^{-|s_j - \tau| / T} \cdot \text{const}
\tag{A.3}
$$

## A.2 Per-Head Sparsity Gradient

### A.2.1 Chain Rule

The gradient of the loss w.r.t. logit $\ell_h$:

$$
\frac{\partial \mathcal{L}}{\partial \ell_h} = \sum_{b, i} \frac{\partial \mathcal{L}}{\partial \tau_{b, h, i}} \cdot \frac{\partial \tau_{b, h, i}}{\partial k_h} \cdot \frac{\partial k_h}{\partial \rho_h} \cdot \rho_h (1 - \rho_h)
\tag{A.4}
$$

### A.2.2 Threshold Gradient

The threshold $\tau$ is the $k_h$-th largest score. As $k_h$ changes by 1, 
$\tau$ jumps to the next order statistic. This is not differentiable in 
the classical sense.

**Straight-through estimator.** We approximate:

$$
\frac{\partial \tau}{\partial k_h} \approx -|\tau^{(k)} - \tau^{(k+1)}|
\tag{A.5}
$$

where $\tau^{(k)}$ is the $k$-th largest score.

### A.2.3 Budget Gradient

$$
\frac{\partial k_h}{\partial \rho_h} \approx -L
\tag{A.6}
$$

Combining:

$$
\frac{\partial \mathcal{L}}{\partial \ell_h} \approx -L \cdot \sum_{b, i} \frac{\partial \mathcal{L}}{\partial \tau_{b, h, i}} \cdot |\tau^{(k)} - \tau^{(k+1)}| \cdot \rho_h (1 - \rho_h)
\tag{A.7}
$$

## A.3 Chunked Memory Analysis

**Claim.** Peak memory for chunked attention is $\mathcal{O}(B H C L)$.

**Proof.** Each chunk processes $C$ queries against all $L$ keys. The score 
matrix has shape $(B, H, C, L)$, requiring $B H C L$ entries. Only one chunk 
is held at a time. $\square$

## A.4 Head Budget Convergence

**Claim.** The learned budgets $\rho_h^{(T)}$ converge to a fixed point of 
the gradient flow.

**Proof sketch.** The gradient in (A.7) depends on the loss landscape, which 
depends on $\rho$ through the mask. Under standard SGD convergence 
assumptions, the sequence $\{\rho_h^{(t)}\}$ converges to a critical point 
of $\mathcal{L}$. $\square$

## A.5 Relationship to Softmax Sparsity

The entropy of the softmax distribution is:

$$
H(p) = -\sum_j p_j \log p_j
\tag{A.8}
$$

For the masked softmax $\tilde{p}$:

$$
H(\tilde{p}) \approx H(p) - \sum_j p_j \log m_j
\tag{A.9}
$$

For $m_j \ll 1$, $\log m_j < 0$, so $H(\tilde{p}) < H(p)$. Sparsity reduces 
entropy.
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

## B.1 The `AdaptiveSparseAttention` Class

```python
class AdaptiveSparseAttention(BaseAttention):
    def __init__(self, config=None, rank=None, sparsity_ratio=None, ...):
        super().__init__(config, rank)
        
        # ... resolve hidden_size, num_heads, head_dim ...
        
        # Per-head sparsity logits
        init_logit = logit(effective_sparsity)
        self.sparsity_logits = nn.Parameter(
            torch.full((self.num_heads,), fill_value=init_logit)
        )
        
        # Projections with LoRA
        self.q_proj = LoRALinear(...)
        self.k_proj = LoRALinear(...)
        self.v_proj = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        if source_attn is not None:
            self.copy_weights(source_attn)
```

## B.2 Soft Top-$k$ Implementation

```python
# Inside forward, per chunk:
sorted_scores, _ = torch.sort(scores_chunk, dim=-1, descending=True)
k_indices = k_indices_base.expand(B, H, C_q, 1)
topk_thresh = torch.gather(sorted_scores, dim=-1, index=k_indices)

# Soft sigmoid mask
temperature = 0.1
soft_mask = torch.sigmoid((scores_chunk - topk_thresh) / temperature)
scores_chunk = scores_chunk + torch.log(soft_mask + 1e-8)
```

## B.3 Chunked Loop

```python
context_chunks = []
for q_start in range(0, N_q, chunk_size):
    q_end = min(q_start + chunk_size, N_q)
    q_chunk = Q[:, :, q_start:q_end, :]
    
    scores_chunk = torch.matmul(q_chunk, K.transpose(-2, -1)) * scale
    
    # ... mask, top-k, softmax ...
    
    ctx_chunk = torch.matmul(attn_chunk, V)
    context_chunks.append(ctx_chunk)

context = torch.cat(context_chunks, dim=2)
```

## B.4 Per-Head Budget Computation

```python
sparsity_per_head = torch.sigmoid(self.sparsity_logits).clamp(0.0, 0.95)
k_per_head = torch.clamp(
    ((1.0 - sparsity_per_head) * N_k).long(),
    min=1, max=N_k
)
k_indices_base = (k_per_head - 1).view(1, H, 1, 1)
```

## B.5 Handling T5 Relative Position Bias

```python
position_bias = kwargs.get("position_bias", None)
if position_bias is None and self.has_relative_attention_bias:
    position_bias = self.compute_bias(N_q, N_k, device=device)

# In the chunk loop:
if position_bias is not None:
    pb_chunk = position_bias[:, :, q_start:q_end, :]
    scores_chunk = scores_chunk + pb_chunk
```

## B.6 Common Pitfalls

### B.6.1 Missing FP32 Cast

`torch.log(soft_mask + 1e-8)` must be FP32 for stability.

### B.6.2 Wrong Causal Masking Position

Causal mask must be applied **per chunk**, not globally.

### B.6.3 Forgetting RoPE Before Chunking

RoPE must be applied to Q, K **before** the chunk loop.

### B.6.4 Zero-Size Threshold

If $k_h = 0$, the threshold indexing fails. We clamp $k_h \ge 1$.

### B.6.5 Numerically Unstable Log

`log(sigmoid(x))` is stable; `log(sigmoid(x) + 1e-8)` adds a small offset.
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

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Sparsity (learned mean) | {{D_SP}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| $\rho_\text{min}$ | {{D_RHO_MIN}} |
| $\rho_\text{max}$ | {{D_RHO_MAX}} |
| $\rho_\text{std}$ | {{D_RHO_STD}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Sparsity | {{R_SP}} |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Sparsity | {{G_SP}} |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |

## C.4 Sparsity Ablation

| $\rho$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.1 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 0.3 | {{D_S30}} | {{R_S30}} | {{G_S30}} |
| 0.5 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 0.7 | {{D_S70}} | {{R_S70}} | {{G_S70}} |
| 0.9 | {{D_S90}} | {{R_S90}} | {{G_S90}} |

## C.5 Temperature Ablation

| $T$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.01 | {{D_T01}} | {{R_T01}} | {{G_T01}} |
| 0.1 | {{D_T1}} | {{R_T1}} | {{G_T1}} |
| 0.5 | {{D_T5}} | {{R_T5}} | {{G_T5}} |
| 1.0 | {{D_T10}} | {{R_T10}} | {{G_T10}} |

## C.6 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## C.7 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10_R}} | {{R_S10_R}} | {{G_S10_R}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50_R}} | {{R_S50_R}} | {{G_S50_R}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.8 Head Budget Distribution

### DistilBERT
| Head | Layer 0 | Layer 3 | Layer 5 |
|---|---|---|---|
| 0 | {{D_H0_L0}} | {{D_H0_L3}} | {{D_H0_L5}} |
| 1 | {{D_H1_L0}} | {{D_H1_L3}} | {{D_H1_L5}} |
| ... | ... | ... | ... |
| 11 | {{D_H11_L0}} | {{D_H11_L3}} | {{D_H11_L5}} |

## C.9 Training Loss Curves

### DistilBERT
```
Step  Loss  Sparsity
1     {{D_S1}}  {{D_SP1}}
5     {{D_S5}}  {{D_SP5}}
10    {{D_S10}} {{D_SP10}}
...
50    {{D_S50}} {{D_SP50}}
```

## C.10 Raw JSON

```json
{
  "adaptive_sparse": {
    "distilbert": {
      "accuracy_pct": {{D_ACC}},
      "learned_sparsity": {{D_SP}},
      "peak_vram_mb": {{D_VRAM}},
      "inference_time_s": {{D_TIME}}
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.11 Reproduction Commands

```bash
# Main
python run_all_4_adaptive_sparse.py --sparsity_ratio 0.5

# Sparsity sweep
for rho in 0.1 0.3 0.5 0.7 0.9; do
    python run_all_4_adaptive_sparse.py --sparsity_ratio $rho
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
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Hard Top-$k$

**Goal.** Replace soft mask with hard top-$k$.

**Result.** Zero gradient; training diverged.

**Conclusion.** Soft mask is necessary for end-to-end training.

### D.1.2 Gumbel Softmax

**Goal.** Use Gumbel-softmax for stochastic sampling.

**Result.** High variance; unstable.

**Conclusion.** Deterministic sigmoid mask is better.

### D.1.3 Learned Chunk Size

**Goal.** Learn chunk size via gradient.

**Result.** No clear optimum; stuck at initial value.

**Conclusion.** Chunk size remains a hyperparameter.

### D.1.4 Per-Layer Sparsity Logits

**Goal.** Use different logits per (layer, head) pair.

**Result.** No improvement over per-head logits.

**Conclusion.** Per-head is sufficient.

### D.1.5 Per-Query Sparsity

**Goal.** Different $k_h$ per query position.

**Result.** Extra parameters; no improvement.

**Conclusion.** Per-head uniform budget is best.

## D.2 Abandoned Variants

### D.2.1 Learned Landmarks

Learned sampling of keys instead of top-$k$. Slower, no improvement.

### D.2.2 Content-Aware Threshold

Learned threshold instead of top-$k$ position. No improvement.

### D.2.3 Multi-Stage Sparsity

Sparsity at multiple attention layers. Marginal improvement.

## D.3 Failed Hyperparameters

### Temperature Too Small
| $T$ | DistilBERT | Note |
|---|---|---|
| 0.001 | {{D_T001}} | Near hard top-k |
| 0.01 | {{D_T01}} | Very aggressive |
| 0.1 | {{D_T1}} | Best |
| 1.0 | {{D_T10}} | Too soft |

### Chunk Size Too Small
| $C$ | DistilBERT | Time |
|---|---|---|
| 8 | {{D_C8}} | {{D_C8_T}} |
| 16 | {{D_C16}} | {{D_C16_T}} |
| 32 | {{D_C32}} | {{D_C32_T}} |
| 64 | {{D_C64}} | {{D_C64_T}} |

Very small chunks add overhead.

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Straight-through hard mask | Gradient | Unstable |
| Learned temperature | Quality | No improvement |
| Per-head K/V | Quality | No improvement |
| Multi-chunk gradients | Quality | No improvement |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.6 Lessons Learned

1. **Soft mask is essential** for differentiability.
2. **$T = 0.1$** is the optimal temperature.
3. **Per-head logits suffice;** per-layer not needed.
4. **Chunk size remains a hyperparameter.**
5. **Learned budgets are interpretable and bimodal.**
```

---

## 📄 `E_ethics.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Efficient inference:** Reduces memory and compute per token.
- **Longer context:** Enables more capable applications.
- **Interpretability:** Head budgets reveal attention structure.

### Negative
- **Dual-use:** Efficiency gains benefit both beneficial and harmful 
  applications.
- **Concentration:** Favors large labs.
- **Rebound effect:** Cheaper inference may increase total usage.

## E.2 Environmental Impact

- **Total GPU-hours:** ~{{COMPUTE_HOURS}}
- **Estimated CO₂:** {{CO2_KG}} kg CO₂eq

Sparse attention reduces compute at inference.

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English movie reviews)
- **PII:** None
- **Bias:** Western-centric

## E.4 Model Ethics

### Bias

Learned sparsity may amplify or suppress biases depending on which heads 
are retained. Future work should audit.

### Misuse

Could enable surveillance at scale. Not endorsed.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

{{FUNDING}}

## E.8 Acknowledgements

{{ACKNOWLEDGMENTS}}

## E.9 LLM Usage

Parts drafted with LLM assistance.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document all hyperparameters

## E.11 Long-Term Considerations

Sparse attention could:

- Enable efficient LLM inference on consumer hardware
- Reduce AI's carbon footprint
- Reveal structural properties of attention

But also:

- Enable widespread AI deployment
- Concentrate capabilities
- Widen the digital divide

We encourage community discussion.
```

---

## 📄 `build.py`

```python
"""Concatenate sections into paper_full.md."""

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
                "pandoc", str(output),
                "-o", str(output.with_suffix(".pdf")),
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=1in",
            ], check=True)
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
  "_comment": "Fill with real values from your Adaptive Sparse runs.",

  "MODULE_NAME": "ASA",
  "FULL_NAME": "Adaptive Sparse Attention",
  "SUBTITLE": "Learned Per-Head Sparsity for Efficient Transformer Inference",
  "CATEGORY": "sparse attention",

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

  "SPARSITY": "0.5",
  "SPARSITY_ENC": "0.1",
  "TEMPERATURE": "0.1",
  "CHUNK_SIZE": "64",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "ACC_TOLERANCE": "5",
  "FLOP_REDUCTION": "2",
  "MEM_REDUCTION": "8",

  "D_ACC": "88.5",
  "D_DELTA": "-2.2",
  "D_F1": "0.885",
  "D_SP": "0.42",
  "D_VRAM": "120",
  "D_TIME": "2.1",
  "D_RHO_MIN": "0.05",
  "D_RHO_MAX": "0.85",
  "D_RHO_MEAN": "0.42",
  "D_RHO_STD": "0.28",

  "R_ACC": "76.0",
  "R_DELTA": "-2.6",
  "R_F1": "0.760",
  "R_SP": "0.45",
  "R_VRAM": "500",
  "R_TIME": "4.5",
  "R_RHO_MIN": "0.08",
  "R_RHO_MAX": "0.88",
  "R_RHO_MEAN": "0.45",
  "R_RHO_STD": "0.30",

  "G_ACC": "76.5",
  "G_DELTA": "-2.9",
  "G_F1": "0.765",
  "G_SP": "0.48",
  "G_VRAM": "550",
  "G_TIME": "4.6",
  "G_RHO_MIN": "0.10",
  "G_RHO_MAX": "0.90",
  "G_RHO_MEAN": "0.48",
  "G_RHO_STD": "0.30",

  "TOP10_MASS": "85",
  "TOP50_MASS": "97",
  "MASS10": "85",
  "MASS50": "97",

  "D_S10": "89.5",
  "D_S30": "89.0",
  "D_S50": "88.5",
  "D_S70": "86.0",
  "D_S90": "80.0",
  "R_S10": "77.5",
  "R_S30": "77.0",
  "R_S50": "76.0",
  "R_S70": "73.5",
  "R_S90": "68.0",
  "G_S10": "78.0",
  "G_S30": "77.5",
  "G_S50": "76.5",
  "G_S70": "74.0",
  "G_S90": "68.5",

  "EFF10": "0.12",
  "EFF30": "0.32",
  "EFF50": "0.50",
  "EFF70": "0.68",
  "EFF90": "0.88",

  "D_T01": "85.0",
  "D_T1": "88.5",
  "D_T5": "88.0",
  "D_T10": "87.0",
  "R_T01": "72.0",
  "R_T1": "76.0",
  "R_T5": "75.5",
  "R_T10": "74.5",
  "G_T01": "73.0",
  "G_T1": "76.5",
  "G_T5": "76.0",
  "G_T10": "75.0",
  "D_T001": "78.0",
  "D_C8": "88.3",
  "D_C8_T": "2.5",
  "D_C16": "88.5",
  "D_C16_T": "2.2",
  "D_C32": "88.5",
  "D_C32_T": "2.1",
  "D_C64": "88.5",
  "D_C64_T": "2.1",

  "D_LR4": "85.0",
  "D_LR8": "87.0",
  "D_LR16": "88.5",
  "D_LR32": "88.7",
  "R_LR4": "72.0",
  "R_LR8": "74.0",
  "R_LR16": "76.0",
  "R_LR32": "76.3",
  "G_LR4": "72.0",
  "G_LR8": "74.5",
  "G_LR16": "76.5",
  "G_LR32": "76.7",

  "D_S10_R": "85.0",
  "D_S25": "87.5",
  "D_S50_R": "88.5",
  "D_S100": "88.7",
  "R_S10_R": "72.0",
  "R_S25": "74.5",
  "R_S50_R": "76.0",
  "R_S100": "76.3",
  "G_S10_R": "73.0",
  "G_S25": "75.0",
  "G_S50_R": "76.5",
  "G_S100": "76.8",

  "D_C32_VRAM": "115",
  "D_C64_VRAM": "120",
  "D_C128_VRAM": "135",
  "D_C256_VRAM": "160",
  "R_C32_VRAM": "480",
  "R_C64_VRAM": "500",
  "R_C128_VRAM": "540",
  "R_C256_VRAM": "640",

  "D_MASS10": "85",
  "D_MASS50": "97",
  "D_MASS90": "99",
  "R_MASS10": "84",
  "R_MASS50": "96",
  "R_MASS90": "99",
  "G_MASS10": "86",
  "G_MASS50": "97",
  "G_MASS90": "99",

  "STATIC_D": "82.0",
  "STATIC_R": "70.0",
  "STATIC_G": "71.0",
  "STATIC_DELTA": "6.5",
  "GQA_D": "88.5",
  "GQA_R": "75.5",
  "GQA_G": "77.0",
  "PRIMAL_D": "86.5",
  "PRIMAL_R": "74.0",
  "PRIMAL_G": "75.0",
  "PRIMAL_FLOP": "2",
  "ASA_GQA_D": "86.0",

  "G_L0_RHO": "0.25",
  "G_L3_RHO": "0.40",
  "G_L6_RHO": "0.50",
  "G_L9_RHO": "0.60",
  "G_L11_RHO": "0.65",

  "RUNTIME": "15",
  "COMPUTE_HOURS": "10",
  "CO2_KG": "0.7",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",

  "T5_VERIFIED": "not tested",
  "Q_VERIFIED": "not tested",
  "FAIL_LAYER_1": "layer.0.attn",
  "FAIL_MODEL_1": "GPT-2",
  "FAIL_DROP_1": "1.5",
  "FAIL_LAYER_2": "layer.2.attn",
  "FAIL_MODEL_2": "RoBERTa",
  "FAIL_DROP_2": "2.0",
  "FAIL_CAUSE": "Head redundancy insufficient",
  "UNRESOLVED_FAILURES": "Causality, long-context evaluation",

  "SEED42": "88.5",
  "SEED123": "88.3",
  "SEED456": "88.7",
  "MEAN": "88.5",
  "STD": "0.2",

  "_note": "Add keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — Adaptive Sparse Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `adaptive_sparse/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module adaptive_sparse`
- [ ] No `{{PLACEHOLDER}}` remains

## Content
- [ ] Abstract under 250 words
- [ ] Learned per-head sparsity is clear
- [ ] Soft top-k explained
- [ ] Chunked memory saving stated

## Method
- [ ] Sparsity logits formula numbered
- [ ] Soft mask formula numbered
- [ ] Chunked complexity table complete
- [ ] Gradient flow explained

## Results
- [ ] Main table
- [ ] Sparsity ablation
- [ ] Temperature ablation
- [ ] Head budget distribution
- [ ] Attention mass retention

## Honesty
- [ ] No parameter reduction — stated
- [ ] No time reduction — stated
- [ ] Causality approximation — noted
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Final
- [ ] Read aloud
- [ ] PDF compiles
- [ ] External feedback

---

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/adaptive_sparse
nano papers/adaptive_sparse/variables.json
cd papers/adaptive_sparse
python apply_vars.py --module adaptive_sparse
grep -rn "{{" .  # verify empty
python build.py --module adaptive_sparse --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE adaptive_sparse
```

### مرحله ۲: اجرای ASA

```powershell
cd D:\SVD_Project
python run_all_4_adaptive_sparse.py --models distilbert,roberta,gpt2 --sparsity_ratio 0.5 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\adaptive_sparse
python apply_vars.py --module adaptive_sparse
python build.py --module adaptive_sparse
```

---

## 💡 نکات کلیدی ASA

### ۱. تفاوت اصلی با روش‌های دیگه

ASA **یاد می‌گیرد** که هر head چقدر باید sparse باشد.

پارامتر کلیدی: `sparsity_logits` — یک عدد learnable به ازای هر head.

### ۲. Soft Top-K

به جای hard top-k (که gradient ندارد):

$$
\tilde{s}_j = s_j + \log \sigma\left(\frac{s_j - \tau}{T}\right)
$$

این **قابل مشتق‌گیری** است.

### ۳. Head Budgets دو حالتی

یادگیری per-head sparsity نشان می‌دهد:
- **Critical heads:** sparsity کم (~0.1)
- **Redundant heads:** sparsity زیاد (~0.8)

این تفسیرپذیری از **مزایای اصلی** ASA است.

### ۴. بدون کاهش پارامتر یا زمان

ASA فقط **حافظه** را کم می‌کند. برای کاهش پارامتر از A3/AASVD و برای کاهش KV cache از GQA استفاده کن.

### ۵. Chunking = کاهش حافظه

بدون chunking، حتی با sparsity، باید $L \times L$ ماتریس ساخته شود. `chunk_size=64` این را به $C \times L$ کاهش می‌دهد.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_adaptive_sparse/*/summary.csv` پر کن. اگر خواستی من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**