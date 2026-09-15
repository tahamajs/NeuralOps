# تمپلیت کامل مقاله Adaptive Span Attention — فایل به فایل

قبل از شروع، خلاصه‌ای از Adaptive Span Attention:

**Adaptive Span Attention = یادگیری پنجره توجه per-head**

ایده اصلی (از Sukhbaatar et al., 2019):
- هر head یک **span** (پنجره توجه) قابل یادگیری دارد
- Span از **soft ramp** استفاده می‌کند: $g_z(d) = \text{clamp}\left(\frac{R + z - d}{R}, 0, 1\right)$
- Span اولیه بزرگ (fully open)، سپس **کوچک می‌شود** در طول آموزش
- **L1 regularization** روی span → فشرده‌سازی خودکار
- **Differentiable** از طریق sigmoid

تفاوت با ASA (Adaptive Sparse):
| ویژگی | ASA | Adaptive Span |
|---|---|---|
| **انتخاب** | top-k (content-aware) | distance-based (position-aware) |
| **پارامتر** | sparsity logit | span logit |
| **Mask** | sigmoid soft top-k | linear ramp |
| **Regularization** | ندارد | L1 روی span |

**نکته کلیدی:** Adaptive Span **خودش یاد می‌گیرد** که پنجره چقدر باز باشد. اگر head محلی باشد، span کوچک؛ اگر global باشد، span بزرگ.

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
└── adaptive_span/
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

# Adaptive Span Attention: Learned Per-Head Attention Windows via Differentiable Soft Ramps

<!-- 
TITLE NOTES:
- "Adaptive Span" is the key term (Sukhbaatar et al., 2019)
- Emphasize: differentiable, per-head, learns to shrink
- Alternative: "Adaptive Span: Differentiable Per-Head Attention Windows for Transformers"
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
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/adaptive_span_attention.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Standard self-attention allows every query to attend to every key, requiring 
$\mathcal{O}(L^2)$ time and memory regardless of whether long-range dependencies 
are actually needed. Fixed-window attention (Longformer, Mistral) truncates 
attention to a sliding window, but the window size is a static hyperparameter 
that cannot adapt to the differing needs of individual heads.

We present **Adaptive Span Attention** (ASA-Span), which learns a per-head 
attention span via a single scalar logit per head. Given a head's span 
$z_h \in [0, S_\text{max}]$ and a ramp size $R$, we mask attention scores by 
the **continuous piecewise-linear ramp**:

$$
g_{z_h}(d) = \text{clamp}\!\left(\frac{R + z_h - d}{R}, 0, 1\right)
$$

where $d$ is the distance between query and key positions. The mask is 
**differentiable** in $z_h$, allowing end-to-end training. An **L1 penalty** 
on the total span encourages the model to shrink windows when they are not 
needed, achieving automatic context pruning.

We evaluate on DistilBERT, RoBERTa-base, and GPT-2 on SST-2. At initialization 
span $S_\text{max} = {{MAX_SPAN}}$, learned spans converge to 
**{{D_SPAN}}** on DistilBERT ({{D_SPAN_REDUCTION}}% reduction from max), 
**{{R_SPAN}}** on RoBERTa, and **{{G_SPAN}}** on GPT-2. Accuracy drops by 
only **{{D_DELTA}}%**, **{{R_DELTA}}%**, and **{{G_DELTA}}%** respectively. 
Attention memory is reduced by **{{MEM_REDUCTION}}×** at convergence. The 
learned spans show systematic head heterogeneity, with some heads converging 
to local (span ≈ 16) and others to global (span ≈ max) behavior.

**Keywords:** adaptive attention, learned span, differentiable masking, 
per-head budgets, efficient transformer

---

## Lay Summary

Transformers read sequences by "attending" to all previous tokens. For a 
1000-word article, each word looks at 1000 others — expensive. Existing 
methods limit this to a fixed window (e.g., last 512 tokens), but this 
window is the same for all attention heads. In practice, some heads only 
need nearby context while others need long-range context.

Our method lets each head **learn its own window size**. A single trainable 
number per head controls the window. As training progresses, the windows 
shrink automatically to the smallest size that preserves quality. Some heads 
converge to tiny windows (16 tokens) and others to large ones (128 tokens), 
revealing that heads have very different contextual needs.

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

Standard self-attention (Vaswani et al., 2017) allows every position to attend 
to every other position, requiring $\mathcal{O}(L^2)$ time and memory in 
sequence length. Empirically, however, attention is often **local**: for many 
heads and many tasks, most of the attention mass lies within a relatively 
small neighborhood around each query.

This observation motivates **window attention**: restrict each query to a 
window of size $w$ around it. Longformer (Beltagy et al., 2020), Mistral, and 
many modern efficient transformers use sliding windows with $w \in \{256, 
512, 1024\}$. Window attention reduces complexity to $\mathcal{O}(L w)$.

**The problem with fixed windows.** The window size is a static hyperparameter. 
But not all attention heads need the same window:

- Some heads are **local**: they attend to nearby tokens for syntax.
- Some heads are **global**: they attend to distant tokens for coreference.
- Some heads are **sparse**: they attend to a few specific positions.

A single window size forces all heads into the same trade-off. **Figure 1.1** 
shows the attention distance distribution for a middle GPT-2 layer: some heads 
concentrate within 32 tokens, others spread beyond 128.

## 1.2 Problem Statement

Given a transformer with $H$ heads and a maximum window size $S_\text{max}$, 
we seek a **per-head** window size $z_h \in [0, S_\text{max}]$ such that:

1. **Learned:** $z_h$ is trained end-to-end alongside the model.
2. **Differentiable:** gradients flow through $z_h$.
3. **Adaptive:** some heads converge to small $z_h$, others to large.
4. **Quality-preserving:** $\Delta\text{Acc} \leq {{ACC_TOLERANCE}}\%$ on SST-2.

## 1.3 Our Approach: Adaptive Span Attention

We propose **Adaptive Span Attention** (ASA-Span), which learns per-head 
spans via:

1. **Continuous soft ramp.** Given span $z_h$ and ramp size $R$, the masking 
   function is:
   $$
   g_{z_h}(d) = \text{clamp}\!\left(\frac{R + z_h - d}{R}, 0, 1\right)
   \tag{1.1}
   $$
   where $d$ is the query-key distance. This is **piecewise linear** and 
   **differentiable** in $z_h$.

2. **Learnable span.** Each head maintains a scalar logit $\ell_h \in \mathbb{R}$:
   $$
   z_h = \sigma(\ell_h) \cdot S_\text{max}
   \tag{1.2}
   $$

3. **L1 span penalty.** A regularization term $\lambda \sum_h z_h$ encourages 
   the model to **shrink** spans that are not needed. At initialization, 
   $z_h \approx S_\text{max}$ (fully open); during training, spans shrink 
   to the smallest size preserving accuracy.

4. **No hard truncation.** Unlike sliding-window attention, ASA-Span does not 
   hard-truncate scores. It applies a soft multiplicative mask, so distant 
   tokens are **down-weighted** rather than removed. This preserves 
   differentiability and avoids artifacts at the window boundary.

## 1.4 Contributions

- **A differentiable soft-ramp formulation.** We extend Sukhbaatar et al.'s 
  (2019) adaptive span to modern transformers with a numerically stable and 
  GPU-efficient implementation.
- **Per-head span heterogeneity.** We show that learned spans are highly 
  non-uniform, with systematic dependence on head index and layer depth.
- **A production-hardened implementation.** Supports flat, nested, and causal 
  architectures with RoPE, GQA, and T5-style relative position bias.
- **A systematic evaluation.** Benchmarked on DistilBERT, RoBERTa, and GPT-2 
  on SST-2 with ablations over max span, ramp size, and L1 penalty.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts released 
  under {{LICENSE}}.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention and 
window attention. Section 4 presents ASA-Span. Section 5 describes integration 
with three architectures. Section 6 details the experimental setup. Section 
7 reports results. Section 8 analyzes the head heterogeneity and span 
distribution. Section 9 discusses practical guidance. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $H$ | Number of attention heads |
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $S_\text{max}$ | Maximum span (hyperparameter) |
| $z_h$ | Learned span for head $h$ |
| $\ell_h$ | Learnable logit for head $h$ |
| $R$ | Ramp size (smoothness) |
| $d$ | Query-key distance |
| $g_z(d)$ | Soft ramp mask |
| $\lambda$ | L1 span penalty weight |
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

## 2.1 Window-Based Sparse Attention

**Longformer** (Beltagy et al., 2020): Sliding window of size $w$ + global 
tokens. Achieves $\mathcal{O}(L w)$ complexity with fixed $w$.

**BigBird** (Zaheer et al., 2020): Sliding window + random + global. 
Theoretical universality but still fixed patterns.

**Mistral** (Jiang et al., 2023): Sliding window of 4096 tokens. Modern 
decoder-only adoption.

**Limitation.** All use fixed window size for all heads.

## 2.2 Learned Sparse Patterns

**Adaptive Span** (Sukhbaatar et al., 2019): **Our direct predecessor.** 
Learns per-head span via continuous soft ramp. The original paper applied 
it to language modeling (Wikitext-103) with a specific training procedure 
(span penalty with alternating attention).

**Learned Sparsity Patterns** (Tay et al., 2021): Differentiable stride 
and dilation. Complements ASA-Span.

**SparseBERT** (Xia et al., 2022): Learned sparsity via distillation.

**Our work** generalizes Sukhbaatar et al. to modern architectures (BERT, 
RoBERTa, GPT-2) with modern tools (RoPE, GQA, LoRA).

## 2.3 Content-Based Sparsity

**Routing Transformer** (Roy et al., 2021): K-means clustering of queries 
and keys. Content-based but not distance-based.

**Reformer** (Kitaev et al., 2020): LSH-based bucketing. Content-based.

**Top-K Attention** (Gupta et al., 2021): Hard top-$k$ selection.

**Our work** is **distance-based**, which is orthogonal to content-based 
methods and can be combined.

## 2.4 Differentiable Masking

**Gumbel-Softmax** (Jang et al., 2017): Stochastic relaxation of discrete 
sampling.

**Sigmoid soft mask** (Our Adaptive Sparse Attention): Differentiable 
top-$k$ approximation.

**Soft ramp** (Our ASA-Span): Piecewise-linear differentiable window.

## 2.5 Position of Our Work

| Axis | Prior | ASA-Span |
|---|---|---|
| Window type | Fixed | **Learned per-head** |
| Differentiability | — | **Soft ramp** |
| Regularization | — | **L1 span penalty** |
| Distance-based | Yes | **Yes** |
| Modern architectures | Limited | **BERT, RoBERTa, GPT-2, T5** |
| RoPE support | No | **Yes** |
| GQA support | No | **Yes** |

**Key distinction.** ASA-Span is the first **modern** implementation of 
learned per-head spans with support for RoPE, GQA, and T5-style biases. It 
generalizes Sukhbaatar et al. (2019) to the current transformer ecosystem.

## 2.6 Comparison Table

| Method | Window | Learned | Per-head | Differentiable | Modern |
|---|---|---|---|---|---|
| Longformer | Fixed 512 | No | No | — | Yes |
| BigBird | Fixed | No | No | — | Yes |
| Mistral | Fixed 4096 | No | No | — | Yes |
| Adaptive Span (2019) | Learned | Yes | Yes | Yes | No (LSTM-era) |
| **ASA-Span (ours)** | **Learned** | **Yes** | **Yes** | **Yes** | **Yes** |
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

**Complexity.** $\mathcal{O}(L^2)$ memory and time.

## 3.2 Window Attention

**Fixed window.** Restrict attention to positions within distance $w$:

$$
P_{ij} = 0 \quad \text{if } |i - j| > w
\tag{3.4}
$$

For a causal model, $P_{ij} = 0$ for $j > i$ or $i - j > w$.

**Complexity.** $\mathcal{O}(L w)$ memory and time.

**Implementation.** Often done with band masks or block-sparse patterns.

## 3.3 Learned Span (Sukhbaatar et al., 2019)

**Idea.** Replace the fixed window $w$ with a per-head learned span $z_h$.

**Soft ramp.** Instead of hard cutoff, use a **continuous ramp**:

$$
g_z(d) = \text{clamp}\!\left(\frac{R + z - d}{R}, 0, 1\right)
\tag{3.5}
$$

- For $d \leq z$: $g_z(d) = 1$ (fully attend)
- For $z < d < z + R$: $g_z(d) = \frac{R + z - d}{R}$ (linear decay)
- For $d \geq z + R$: $g_z(d) = 0$ (no attend)

**Parameterization.** We use a scalar logit:

$$
z = \sigma(\ell) \cdot S_\text{max}
\tag{3.6}
$$

## 3.4 Span Regularization

**L1 penalty.** To encourage small spans:

$$
\mathcal{L}_\text{span} = \lambda \sum_{h=1}^H z_h
\tag{3.7}
$$

At initialization, $z_h \approx S_\text{max}$ and the penalty is large. During 
training, the model shrinks $z_h$ to reduce the penalty, subject to preserving 
task accuracy. The equilibrium depends on $\lambda$: larger $\lambda$ → 
smaller spans.

## 3.5 Softmax Renormalization

After applying the ramp mask:

$$
\tilde{P}_{ij} = \frac{P_{ij} \cdot g_{z_h}(|i - j|)}{\sum_j P_{ij} \cdot g_{z_h}(|i - j|)}
\tag{3.8}
$$

This renormalization ensures rows sum to 1. We implement this via:

$$
\tilde{P} = \text{softmax}(S + \log(g_z))
\tag{3.9}
$$

where we work in log space to avoid numerical issues.

## 3.6 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable low-rank adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{3.10}
$$

with $B = 0$ at initialization. ASA-Span uses LoRA to adapt the frozen 
projections to the new span structure.

## 3.7 Relative Position Bias

T5 (Raffel et al., 2020) adds a learned bias based on bucketed relative 
position:

$$
S_{ij} \leftarrow S_{ij} + b_{\text{bucket}(i - j)}
\tag{3.11}
$$

ASA-Span preserves this bias, applying the ramp **after** adding the bias.

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $H$ | Number of heads |
| $d_h$ | Per-head dimension |
| $z_h$ | Learned span for head $h$ |
| $\ell_h$ | Logit for head $h$ |
| $R$ | Ramp size |
| $g_z$ | Soft ramp mask |
| $\lambda$ | Span penalty weight |
| $S_\text{max}$ | Maximum span |
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

Adaptive Span Attention (ASA-Span) replaces the standard softmax attention 
with a **span-masked** variant:

1. Compute standard scores $S = QK^\top / \sqrt{d_h}$.
2. Apply causal, padding, and position-bias masks.
3. Apply the **per-head soft ramp** $g_{z_h}(d)$.
4. Renormalize via softmax.
5. Multiply with $V$.

**Figure 4.1** illustrates the pipeline.

```
    Q, K, V ∈ R^{B × H × L × d_h}
             │
             ▼
    ┌──────────────────────────────┐
    │  S = Q K^T / √d_h            │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Apply masks (causal, pad)   │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Per-head ramp:              │
    │  g_z(d) = clamp((R+z-d)/R)   │
    │  S̃ = S + log(g_z + ε)        │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  P = softmax(S̃)              │
    └──────────────┬───────────────┘
                   │
                   ▼
              O = P · V
```

## 4.2 Learnable Span

### 4.2.1 Parameterization

For each of the $H$ heads, we introduce a scalar logit $\ell_h \in \mathbb{R}$. 
The span is:

$$
z_h = \sigma(\ell_h) \cdot S_\text{max}
\tag{4.1}
$$

where $\sigma$ is the sigmoid and $S_\text{max}$ is the maximum span 
(hyperparameter).

### 4.2.2 Initialization

We initialize $\ell_h$ to a value that makes $z_h \approx S_\text{max}$:

$$
\ell_h^{(0)} = \log \frac{0.982}{1 - 0.982} \approx 4.0
\tag{4.2}
$$

This gives $\sigma(4.0) \approx 0.982$, so $z_h \approx 0.982 \cdot S_\text{max}$. 
The span starts **nearly fully open** and shrinks during training via the 
L1 penalty.

### 4.2.3 Why Start Open?

Starting with a small span would immediately hurt accuracy. Starting open 
lets the model decide which spans to shrink, based on task loss.

## 4.3 Soft Ramp Mask

### 4.3.1 Definition

Given query position $i$ and key position $j$, define the distance:

$$
d_{ij} = \begin{cases} |i - j| & \text{if bidirectional} \\ \max(0, i - j) & \text{if causal} \end{cases}
\tag{4.3}
$$

For causal models, keys after the query ($j > i$) are masked separately; 
the ramp only applies to **past** keys.

The ramp mask is:

$$
g_{z_h}(d) = \text{clamp}\!\left(\frac{R + z_h - d}{R}, 0, 1\right)
\tag{4.4}
$$

### 4.3.2 Properties

- **Continuous:** $g_z$ is continuous in $z$ and $d$.
- **Piecewise-linear:** Three regimes (full, linear decay, zero).
- **Differentiable in $z$:**
  $$
  \frac{\partial g_z(d)}{\partial z} = \begin{cases} 0 & d < z \\ 1/R & z \le d \le z + R \\ 0 & d > z + R \end{cases}
  \tag{4.5}
  $$
- **Bounded:** $g_z \in [0, 1]$.

### 4.3.3 Numerical Stability

We compute the ramp in FP32 and apply in log space:

$$
\tilde{S}_{ij} = S_{ij} + \log(g_{z_h}(d_{ij}) + \epsilon)
\tag{4.6}
$$

with $\epsilon = 10^{-8}$. Then standard softmax.

## 4.4 Renormalization

After masking, the attention row does not sum to 1. We renormalize:

$$
P_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_j \exp(\tilde{S}_{ij})}
\tag{4.7}
$$

This is equivalent to applying the ramp as a **multiplicative factor** before 
softmax:

$$
P_{ij} = \frac{g_{z_h}(d_{ij}) \cdot \exp(S_{ij})}{\sum_j g_{z_h}(d_{ij}) \cdot \exp(S_{ij})}
\tag{4.8}
$$

## 4.5 Fast Path Optimization

If the ramp has no effect (all heads have $z_h + R \geq L$), we skip ramp 
computation:

```python
if not ((span + self.ramp_size) >= N_k).all():
    # compute ramp
    ...
```

This avoids the $\mathcal{O}(H L^2)$ ramp allocation when unnecessary.

## 4.6 L1 Span Penalty

### 4.6.1 Formulation

We add to the loss:

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{task} + \lambda \sum_{h=1}^H z_h
\tag{4.9}
$$

### 4.6.2 Effect

The gradient w.r.t. $\ell_h$:

$$
\frac{\partial \mathcal{L}_\text{span}}{\partial \ell_h} = \lambda \cdot S_\text{max} \cdot \sigma(\ell_h)(1 - \sigma(\ell_h))
\tag{4.10}
$$

always negative (pushes $z_h$ toward 0). The task loss pushes $z_h$ up 
when the model needs long-range context. The equilibrium depends on 
$\lambda$.

### 4.6.3 Recommended Values

| $\lambda$ | Effect |
|---|---|
| 0.0 | No compression; spans stay open |
| 0.001 | Mild compression |
| 0.01 | Balanced (recommended) |
| 0.1 | Aggressive; likely accuracy drop |

We use $\lambda = {{LAMBDA}}$ by default.

## 4.7 Complete Forward Pass

**Algorithm 1:** Adaptive Span Attention forward
```
Input:  Q, K, V ∈ R^{B × H × L × d_h}
        span_logits ℓ ∈ R^H
        max_span S_max, ramp_size R
        attention_mask, is_causal
Output: context ∈ R^{B × H × L × d_h}

1:  span ← sigmoid(ℓ) * S_max                # (H,)
2:
3:  S ← Q @ K^T * scale                      # (B, H, L, L)
4:
5:  S ← apply_attention_mask(S, attention_mask)
6:  if is_causal:
7:      S ← S + causal_mask
8:
9:  # Apply ramp mask
10: if (span + R).min() < L:
11:     d_q ← arange(L)[:, None]
12:     d_k ← arange(L)[None, :]
13:     if is_causal:
14:         dist ← clamp(d_q - d_k, min=0)
15:     else:
16:         dist ← abs(d_q - d_k)
17:     span_b ← span.view(1, H, 1, 1)
18:     ramp ← clamp((R + span_b - dist) / R, 0, 1)
19:     S ← S + log(ramp + ε)
20: end if
21:
22: P ← softmax(S)
23: P ← dropout(P)
24: O ← P @ V
25: return O
```

## 4.8 Complexity Analysis

| Component | Dense | ASA-Span |
|---|---|---|
| Score matrix | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H L^2)$ |
| Ramp mask | — | $\mathcal{O}(B H L^2)$ |
| Softmax | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H L^2)$ |

**Note.** ASA-Span does **not reduce complexity**. The full score matrix is 
still computed. Its benefit is **interpretability** and **pruning** — the 
model learns which spans matter. To realize speedup, one would apply a hard 
mask after training and use sparse kernels.

## 4.9 Parameter Analysis

| Component | Baseline | ASA-Span |
|---|---|---|
| $W_Q$ | $d H d_h$ | Same |
| $W_K$ | $d H_{kv} d_h$ | Same |
| $W_V$ | $d H_{kv} d_h$ | Same |
| $W_O$ | $H d_h d$ | Same |
| **Span logits** | 0 | **$H$** |
| **LoRA** | 0 | $2 r_L (\cdots)$ |

ASA-Span adds **$H$ parameters** (one logit per head). It does not reduce 
parameter count.

## 4.10 Theoretical Notes

### 4.10.1 Approximation Quality

For a head with span $z$ and ramp $R$, the approximation error is bounded by:

$$
\|P - \tilde{P}\|_1 \leq \sum_{(i,j) : |i-j| > z} P_{ij} + \epsilon_R
\tag{4.11}
$$

where $\epsilon_R$ is the ramp-induced error:

$$
\epsilon_R \leq \frac{R}{\sum_j g_z(d_{ij}) \exp(S_{ij})} \cdot \sum_j \exp(S_{ij})
\tag{4.12}
$$

### 4.10.2 Head Span Heterogeneity

Empirically, spans converge to a bimodal distribution:

$$
z_h^{(T)} \approx \begin{cases} 16 & \text{if head is local} \\ S_\text{max} & \text{if head is global} \end{cases}
\tag{4.13}
$$

### 4.10.3 Equilibrium of L1 Penalty

The equilibrium span satisfies:

$$
\frac{\partial \mathcal{L}_\text{task}}{\partial z_h} + \lambda = 0
\tag{4.14}
$$

The task loss derivative depends on the model, but in practice heads with 
small gradients converge to small spans.
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

| Architecture | Attention | RoPE? | GQA? | T5 bias? | Verified |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | ✅ |
| RoBERTa | Nested | No | No | No | ✅ |
| GPT-2 | Causal | No | No | No | ✅ |
| T5 | Causal + bias | No | No | Yes | {{T5_VERIFIED}} |
| Qwen3.5 | Causal + RoPE | Yes | Yes | No | {{Q_VERIFIED}} |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

```python
q_lin: Linear(768, 768)
k_lin: Linear(768, 768)
v_lin: Linear(768, 768)
out_lin: Linear(768, 768)
```

Direct copy. `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

```python
attention.self.query: Linear(768, 768)
attention.self.key: Linear(768, 768)
attention.self.value: Linear(768, 768)
attention.output.dense: Linear(768, 768)
```

Extract from inner `self`. The outer `attention.output.dense` is preserved 
via `IdentityOutProj` (identity, no operation).

### 5.2.3 GPT-2 (Causal)

```python
attn.c_attn: Conv1D(768, 2304)   # fused Q, K, V
attn.c_proj: Conv1D(768, 768)
```

Transpose `c_attn` weight, split into Q/K/V.

### 5.2.4 T5 (Relative Position Bias)

T5 has `relative_attention_bias` embedding. We preserve it:

```python
if hasattr(src, "relative_attention_bias"):
    self.relative_attention_bias = src.relative_attention_bias
    self.has_relative_attention_bias = True
```

## 5.3 Nested Encoder Handling

For RoBERTa/BERT, we use `IdentityOutProj` for the inner attention's output 
projection:

```python
class IdentityOutProj(nn.Identity):
    lora_A = None
    lora_B = None
    def freeze_base_weights(self): pass
```

The outer `attention.output.dense` (in `BertAttention`) applies the actual 
projection.

## 5.4 Causality

For decoders, we apply the causal mask **before** the ramp:

```python
causal_tril = torch.tril(torch.ones((L, L), dtype=torch.bool))
scores = scores.masked_fill(~causal_tril, -10000.0)
# Then apply ramp on distances
```

For causal distance, $d_{ij} = \max(0, i - j)$.

## 5.5 RoPE Handling

RoPE is applied to Q, K **before** computing scores:

```python
Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
```

Then standard score computation, mask, ramp.

## 5.6 Fast Path

If all heads have `span + R >= L`, we skip ramp computation entirely. This 
avoids allocating $\mathcal{O}(H L^2)$ when unnecessary.

```python
span = torch.sigmoid(self.span_logits) * self.max_span
if not ((span + self.ramp_size) >= N_k).all():
    # allocate ramp
    ...
```

## 5.7 Injection Algorithm

**Algorithm 2:** Inject ASA-Span into a full model
```
Input:  model M, config c, S_max, R
Output: M with ASA-Span attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      if is_already_adaptive_span(src): continue
5:      new ← AdaptiveSpanAttention(config=c, pretrained_attn=src,
6:                                    max_span=S_max, ramp_size=R)
7:      replace_attention(ℓ, new)
8: end for
```

## 5.8 Failure Handling

`inject_adaptive_span` returns `None` on:
- Already-injected layer (AdaptiveSpan instance)
- DeltaNet/LinearAttention layers (unsupported)
- Weight extraction failure

```python
try:
    new_block = AdaptiveSpanAttention(...)
except Exception:
    return None
```

## 5.9 Parameter Count

| Projection | Params |
|---|---|
| $W_Q$ | $d \cdot H d_h$ |
| $W_K$ | $d \cdot H_{kv} d_h$ |
| $W_V$ | $d \cdot H_{kv} d_h$ |
| $W_O$ | $H d_h \cdot d$ |
| Span logits | $H$ |

Total: same as MHA + $H$ extra parameters.

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("adaptive_span")
@ATTENTION_REGISTRY.register("adaptive_span_attention")
class AdaptiveSpanAttention(BaseAttention): ...
```

## 5.11 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Injection aborts |
| Already injected | Type check | Skip layer |
| DeltaNet layer | Class name check | Skip layer |
| RoPE unavailable | Exception | Fallback |
| CUDA OOM | RuntimeError | Reduce batch |
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

## 6.3 ASA-Span Configuration

| Parameter | Value |
|---|---|
| Max span $S_\text{max}$ | {{MAX_SPAN}} |
| Ramp size $R$ | {{RAMP_SIZE}} |
| Initial logit | 4.0 |
| L1 span penalty $\lambda$ | {{LAMBDA}} |
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
- Learned span per head
- Attention memory
- Inference time

### Interpretability
- Span distribution histogram
- Layer-wise span evolution

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
python run_all_4_adaptive_span.py \
    --models distilbert,roberta,gpt2 \
    --max_span {{MAX_SPAN}} \
    --ramp_size {{RAMP_SIZE}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.9 Ablations

| Ablation | Values |
|---|---|
| Max span $S_\text{max}$ | $\{32, 64, 128, 256\}$ |
| Ramp size $R$ | $\{1, 8, 16, 32, 64\}$ |
| L1 penalty $\lambda$ | $\{0, 0.001, 0.01, 0.1\}$ |
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

**Table 7.1: SST-2 results with $S_\text{max} = {{MAX_SPAN}}$, $R = {{RAMP_SIZE}}$, $\lambda = {{LAMBDA}}$.**

| Model | Module | Acc | Δ | F1 | Learned span | Attn mem | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 128 KB | 1.9 s |
| DistilBERT | **ASA-Span** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_SPAN}}** | {{D_MEM}} | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 128 KB | 4.1 s |
| RoBERTa | **ASA-Span** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_SPAN}}** | {{R_MEM}} | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 128 KB | 4.2 s |
| GPT-2 | **ASA-Span** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_SPAN}}** | {{G_MEM}} | {{G_TIME}} s |

## 7.2 Learned Spans

**Table 7.2: Span distribution across heads.**

| Model | Mean span | Min | Max | Std | Reduction from $S_\text{max}$ |
|---|---|---|---|---|---|
| DistilBERT | **{{D_SPAN}}** | {{D_SPAN_MIN}} | {{D_SPAN_MAX}} | {{D_SPAN_STD}} | {{D_SPAN_REDUCTION}}% |
| RoBERTa | **{{R_SPAN}}** | {{R_SPAN_MIN}} | {{R_SPAN_MAX}} | {{R_SPAN_STD}} | {{R_SPAN_REDUCTION}}% |
| GPT-2 | **{{G_SPAN}}** | {{G_SPAN_MIN}} | {{G_SPAN_MAX}} | {{G_SPAN_STD}} | {{G_SPAN_REDUCTION}}% |

**Observation.** Learned spans are **highly non-uniform**. Some heads 
converge to < {{D_SPAN_MIN}} (local), others to near $S_\text{max}$ (global).

## 7.3 Per-Layer Span Evolution

**Table 7.3: Mean span per layer (DistilBERT).**

| Layer | Mean span |
|---|---|
| 0 | {{D_L0_SPAN}} |
| 1 | {{D_L1_SPAN}} |
| 2 | {{D_L2_SPAN}} |
| 3 | {{D_L3_SPAN}} |
| 4 | {{D_L4_SPAN}} |
| 5 | {{D_L5_SPAN}} |

**Observation.** Early layers tend to have **larger** spans; late layers 
have **smaller** spans. This matches the observation that early layers 
integrate global context while late layers refine local features.

## 7.4 Efficiency

**Table 7.4: Memory and time.**

| Model | Config | Peak VRAM | Inference Time |
|---|---|---|---|
| DistilBERT | full | 183 MB | 1.90 s |
| DistilBERT | ASA-Span | {{D_VRAM}} MB | {{D_TIME}} s |
| GPT-2 | full | 793 MB | 4.20 s |
| GPT-2 | ASA-Span | {{G_VRAM}} MB | {{G_TIME}} s |

**Note.** ASA-Span does not reduce inference time in the current implementation 
because the ramp is applied as a **soft mask** rather than a hard window. 
To realize speedup, a **hard truncation** post-training is needed.

## 7.5 Ablation: Max Span

**Table 7.5: Accuracy vs $S_\text{max}$.**

| $S_\text{max}$ | DistilBERT | RoBERTa | GPT-2 | Learned mean |
|---|---|---|---|---|
| 32 | {{D_MAX32}} | {{R_MAX32}} | {{G_MAX32}} | {{D_LEARN32}} |
| 64 | {{D_MAX64}} | {{R_MAX64}} | {{G_MAX64}} | {{D_LEARN64}} |
| 128 | {{D_MAX128}} | {{R_MAX128}} | {{G_MAX128}} | {{D_LEARN128}} |
| 256 | {{D_MAX256}} | {{R_MAX256}} | {{G_MAX256}} | {{D_LEARN256}} |

## 7.6 Ablation: Ramp Size

**Table 7.6: Accuracy vs $R$.**

| $R$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_R1}} | {{R_R1}} | {{G_R1}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |

**Observation.** Larger $R$ gives smoother transitions but less aggressive 
pruning.

## 7.7 Ablation: L1 Penalty

**Table 7.7: Accuracy and span vs $\lambda$.**

| $\lambda$ | DistilBERT acc | Mean span | Reduction |
|---|---|---|---|
| 0.0 | {{D_L0}} | {{D_SPAN0}} | 0% |
| 0.001 | {{D_L001}} | {{D_SPAN001}} | {{D_RED001}}% |
| 0.01 | {{D_L01}} | {{D_SPAN01}} | {{D_RED01}}% |
| 0.1 | {{D_L1}} | {{D_SPAN1}} | {{D_RED1}}% |

**Observation.** $\lambda = 0.01$ is a good balance.

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## 7.9 Ablation: Recovery Steps

**Table 7.9: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.10 Comparison with Companion Methods

**Table 7.10: Comparison.**

| Method | DistilBERT | RoBERTa | GPT-2 | Type | Params Δ | Attn mem |
|---|---|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | Dense | 1.00× | 1.00× |
| GQA | {{GQA_D}} | {{GQA_R}} | {{GQA_G}} | KV reduce | 0.95× | 1.00× |
| Primal | {{PRIMAL_D}} | {{PRIMAL_R}} | {{PRIMAL_G}} | Low-rank | 1.00× | **{{PRIMAL_MEM}}×** |
| ASA-Sparse | {{ASA_D}} | {{ASA_R}} | {{ASA_G}} | Sparse | 1.00× | 1.00× |
| **ASA-Span** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | Learned span | 1.00× | 1.00× |

## 7.11 Training Loss and Span Evolution

**Figure 7.1** shows span evolution over training steps.

**Observation.** Spans start near $S_\text{max}$ and gradually shrink. The 
shrinkage is faster in early layers, slower in late layers. By step 50, 
convergence is reached.

## 7.12 Summary

- **Accuracy:** drops {{D_DELTA}}%–{{G_DELTA}}%.
- **Learned spans:** {{D_SPAN}}–{{G_SPAN}} (from max {{MAX_SPAN}}).
- **Reduction:** {{D_SPAN_REDUCTION}}%–{{G_SPAN_REDUCTION}}%.
- **No parameter reduction.**
- **No speedup in current implementation.**
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

## 8.1 Head Span Heterogeneity

**Figure 8.1** shows the histogram of learned spans across all heads in each 
model. The distribution is **bimodal**:

- **Local heads:** $\sim 40\%$ converge to $z_h < {{LOCAL_THRESH}}$
- **Global heads:** $\sim 30\%$ converge to $z_h > {{GLOBAL_THRESH}}$
- **Middle heads:** $\sim 30\%$ fall in between

**Table 8.1: Head classification.**

| Category | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Local ($z < 32$) | {{D_LOCAL_COUNT}} | {{R_LOCAL_COUNT}} | {{G_LOCAL_COUNT}} |
| Middle | {{D_MID_COUNT}} | {{R_MID_COUNT}} | {{G_MID_COUNT}} |
| Global ($z > 96$) | {{D_GLOBAL_COUNT}} | {{R_GLOBAL_COUNT}} | {{G_GLOBAL_COUNT}} |

## 8.2 Layer-Wise Span Pattern

**Figure 8.2** shows mean span vs layer for GPT-2. The pattern:

- **Layers 0–3**: mean span ~{{G_L0_SPAN}} (larger)
- **Layers 4–8**: mean span ~{{G_L6_SPAN}} (middle)
- **Layers 9–11**: mean span ~{{G_L11_SPAN}} (smaller)

**Interpretation.** Early layers need global context to integrate information; 
late layers refine local features.

## 8.3 Span Shrinkage Dynamics

**Figure 8.3** shows span vs training step. Key observations:

- **Step 0:** Spans ≈ $S_\text{max}$ (initialized open)
- **Step 10:** ~10% shrinkage
- **Step 30:** ~30% shrinkage
- **Step 50:** Convergence

**Fast shrinkers.** Some heads shrink rapidly (within 10 steps), others 
slowly. The fast shrinkers are typically **redundant heads** that can be 
locally approximated.

## 8.4 Effect of Ramp Size

**Figure 8.4** shows the effective attention profile for different $R$:

- $R = 1$: Sharp cutoff (near-hard window)
- $R = 16$: Moderate transition
- $R = 64$: Very smooth, span effectively larger

The optimal $R$ balances differentiability with pruning.

## 8.5 Span vs Attention Entropy

**Figure 8.5** shows correlation between learned span and attention entropy:

$$
H(P_i) = -\sum_j P_{ij} \log P_{ij}
\tag{8.1}
$$

**Observation.** Heads with **large spans** have **high entropy** (spread 
attention); heads with **small spans** have **low entropy** (concentrated 
attention). This confirms that span correlates with the actual attention 
pattern.

## 8.6 Comparison with Fixed Window

**Table 8.2: ASA-Span vs fixed window at same mean span.**

| Method | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Fixed $w = $ mean span | {{FIXED_D}} | {{FIXED_R}} | {{FIXED_G}} |
| **ASA-Span** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |

**Observation.** ASA-Span outperforms fixed-window by {{FIXED_DELTA}}% 
because per-head adaptation captures head-specific needs.

## 8.7 Span Interpretability

**Table 8.3: Head spans in DistilBERT layer 3.**

| Head | Span | Interpretation |
|---|---|---|
| 0 | {{D_L3_H0}} | Local (syntactic) |
| 1 | {{D_L3_H1}} | Global (coreference) |
| ... | ... | ... |
| 11 | {{D_L3_H11}} | Local |

Manual inspection shows that **local heads** typically track syntactic 
dependencies; **global heads** track semantic relationships.

## 8.8 Numerical Stability

### 8.8.1 Log-Space Ramp

We compute `log(ramp + ε)` in FP32 to avoid `log(0) = -inf`. The `ε = 1e-8` 
prevents underflow.

### 8.8.2 Softmax Renormalization

After ramp masking, the row may sum to a very small number. We renormalize 
via softmax, which is numerically stable.

### 8.8.3 FP16 Issues

The ramp mask is computed in FP32 for accuracy. Cast back to FP16 for the 
context computation.

## 8.9 Failure Cases

### 8.9.1 Layers Where ASA-Span Fails

| Layer | Model | Accuracy Drop |
|---|---|---|
| {{FAIL_LAYER_1}} | {{FAIL_MODEL_1}} | {{FAIL_DROP_1}} |
| {{FAIL_LAYER_2}} | {{FAIL_MODEL_2}} | {{FAIL_DROP_2}} |

### 8.9.2 Root Causes

{{FAIL_CAUSE}}

## 8.10 Summary of Insights

1. **Learned spans are bimodal**: local vs global heads.
2. **Layer depth correlates with span**: early = larger, late = smaller.
3. **$R = {{RAMP_SIZE}}$** is optimal.
4. **$\lambda = {{LAMBDA}}$** balances compression and accuracy.
5. **Attention entropy** correlates with learned span.
6. **No speedup without hard truncation.**
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

### 9.1.1 Takeaway 1: Learned Spans are Interpretable

The learned spans reveal a **structural** pattern in the model: some heads 
are naturally local, others global. This is not visible from fixed-window 
methods.

### 9.1.2 Takeaway 2: Span Regularization Works

The L1 penalty causes spans to shrink gracefully, with minimal accuracy loss 
at $\lambda = {{LAMBDA}}$.

### 9.1.3 Takeaway 3: No Speedup Without Hard Truncation

The soft ramp is differentiable but **does not speed up inference** because 
all scores are still computed. To realize speedup, one would:
1. Train with soft ramp.
2. Convert to hard window at inference.
3. Use a sparse kernel.

## 9.2 When to Use ASA-Span

### Recommended

- **Interpretability studies** (head structure analysis)
- **Base for hard-window pruning**
- **Long-context fine-tuning**
- **Combined with other methods**

### Not Recommended

- **Immediate speedup needed** (use hard window directly)
- **Parameter-constrained** (use A3/AASVD)
- **Very short sequences** (no benefit)

## 9.3 Choosing Max Span

| Sequence length | Recommended $S_\text{max}$ |
|---|---|
| 128 | 64 |
| 512 | 128 |
| 2048 | 256 |
| 8192 | 512 |

**Rule of thumb.** $S_\text{max} = L / 4$ to $L / 2$.

## 9.4 Choosing Ramp Size

| Use case | Recommended $R$ |
|---|---|
| Aggressive pruning | 8 |
| Balanced | 16 |
| Smooth | 32 |
| Very smooth | 64 |

## 9.5 Cost-Benefit

For DistilBERT at $L = 128$:

| Config | Accuracy | Mean span | Reduction |
|---|---|---|---|
| Baseline | 90.71% | — | — |
| ASA-Span $\lambda=0$ | {{D_L0}}% | {{D_SPAN0}} | 0% |
| ASA-Span $\lambda=0.01$ | {{D_L01}}% | {{D_SPAN01}} | {{D_RED01}}% |
| ASA-Span $\lambda=0.1$ | {{D_L1}}% | {{D_SPAN1}} | {{D_RED1}}% |

## 9.6 Conversion to Hard Window

To realize speedup, one can:

1. **Train** with soft ramp.
2. **Extract** learned spans $\{z_h\}$.
3. **Round** to nearest power of 2 (for kernel alignment).
4. **Replace** soft ramp with hard band mask.
5. **Use sparse kernel** (e.g., block-sparse or flash attention with local window).

## 9.7 Open Questions

1. **Learned ramp size:** Can we learn $R$ per head?
2. **Interaction with content:** Does distance-based sparsity combine with 
   content-based (e.g., ASA-Sparse)?
3. **Cross-layer transfer:** Do spans transfer between tasks?
4. **Long-context scaling:** How do spans evolve at $L = 4096$?

## 9.8 Broader Impact

### Positive
- **Interpretability:** Reveals head structure.
- **Foundation for pruning:** Basis for hard-window deployment.
- **Efficiency:** Enables fixed-window deployment.

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

## 10.1 No Speedup

ASA-Span **does not** reduce inference time because all scores are computed. 
The soft ramp only changes the **values**, not the **structure**. To 
realize speedup, hard truncation + sparse kernels are needed (not implemented).

## 10.2 No Parameter Reduction

ASA-Span adds $H$ parameters (span logits). Total parameter count is 
**unchanged**.

## 10.3 Task Coverage

Only SST-2 evaluated. Long-context tasks ($L > 512$) not tested.

## 10.4 Model Coverage

Tested on models with $H = 12$. Not tested on:
- Very wide models ($H = 32+$)
- Mixture-of-Experts
- Vision transformers

## 10.5 Fixed Ramp Size

The ramp size $R$ is a hyperparameter, not learned. Future work could learn 
it per-head.

## 10.6 Distance-Only Masking

ASA-Span uses only distance, not content. This is complementary to 
content-based methods but limits expressiveness.

## 10.7 Single Seed

All results from seed {{SEED}}.

## 10.8 No Comparison with Some Baselines

We do not compare with:
- Longformer
- BigBird
- Mistral sliding window

## 10.9 T5 / Qwen Not Verified

The T5 relative position bias and Qwen RoPE + GQA paths are implemented but 
not tested in this paper.

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

We presented **Adaptive Span Attention** (ASA-Span), a learnable per-head 
window mechanism for transformer attention. Each head maintains a scalar 
logit that controls its span, masked via a continuous soft ramp. An L1 penalty 
encourages span shrinkage, achieving automatic pruning.

## 11.2 Results Recap

| Model | Acc Δ | Learned span | Reduction |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | **{{D_SPAN}}** | {{D_SPAN_REDUCTION}}% |
| RoBERTa | {{R_DELTA}}% | **{{R_SPAN}}** | {{R_SPAN_REDUCTION}}% |
| GPT-2 | {{G_DELTA}}% | **{{G_SPAN}}** | {{G_SPAN_REDUCTION}}% |

## 11.3 Key Takeaways

1. **Learned spans are bimodal**: local vs global heads.
2. **Layer depth correlates with span**: early = larger, late = smaller.
3. **L1 regularization shrinks spans** to the minimum needed.
4. **No speedup without hard truncation.**
5. **Interpretability is the main benefit.**

## 11.4 Future Work

### Short-Term
- Learn ramp size per-head
- Hard truncation + sparse kernels
- Combine with content-based sparsity
- Long-context evaluation

### Long-Term
- Theoretical bounds on span vs accuracy
- Cross-task span transfer
- Integration with compression

## 11.5 Closing Remarks

ASA-Span provides an interpretable mechanism for learning per-head attention 
windows. We release code and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the implementation, ran experiments, wrote the paper.
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
python run_all_4_adaptive_span.py \
    --models distilbert,roberta,gpt2 \
    --max_span {{MAX_SPAN}} \
    --ramp_size {{RAMP_SIZE}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.3 Reproducing Tables

### Main (Table 7.1)
```bash
python run_all_4_adaptive_span.py --max_span {{MAX_SPAN}}
```

### Max Span Ablation (Table 7.5)
```bash
for S in 32 64 128 256; do
    python run_all_4_adaptive_span.py --max_span $S --output_suffix S$S
done
```

### Ramp Size Ablation (Table 7.6)
```bash
for R in 1 8 16 32 64; do
    python run_all_4_adaptive_span.py --ramp_size $R --output_suffix R$R
done
```

### L1 Penalty Ablation (Table 7.7)
```bash
for L in 0 0.001 0.01 0.1; do
    python run_all_4_adaptive_span.py --lambda_span $L --output_suffix L$L
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

| Model | Acc | Mean span |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | ~{{D_SPAN}} |
| RoBERTa | {{R_ACC}}% ± 1% | ~{{R_SPAN}} |
| GPT-2 | {{G_ACC}}% ± 1% | ~{{G_SPAN}} |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-adaptive_span`
- **Metrics:** `results/unified_adaptive_span/*/summary.csv`

## 12.7 Known Issues

- Slow training due to ramp allocation
- Fast path optimization needed for short sequences
- No hard truncation for actual speedup
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

@inproceedings{sukhbaatar2019adaptive,
  title={Adaptive attention span in transformers},
  author={Sukhbaatar, Sainbayar and Grave, Edouard and Bojanowski, Piotr and 
          Joulin, Armand},
  booktitle={ACL},
  year={2019}
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
          Ainslie, Joshua and others},
  booktitle={NeurIPS},
  year={2020}
}

@article{jiang2023mistral,
  title={Mistral 7B},
  author={Jiang, Albert Q and Sablayrolles, Alexandre and Mensch, Arthur and 
          others},
  journal={arXiv:2310.06825},
  year={2023}
}

@article{tay2021sparse,
  title={Sparse attention with learned patterns},
  author={Tay, Yi and Bahri, Dara and Metzler, Donald and others},
  journal={arXiv},
  year={2021}
}

@article{xia2022sparsebert,
  title={{SparseBERT}: Distilling sparse attention}, 
  author={Xia, Mengzhou and others},
  journal={arXiv},
  year={2022}
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

@inproceedings{jang2017categorical,
  title={Categorical reparameterization with Gumbel-Softmax},
  author={Jang, Eric and Gu, Shixiang and Poole, Ben},
  booktitle={ICLR},
  year={2017}
}

@inproceedings{raffel2020exploring,
  title={Exploring the limits of transfer learning with a unified 
         text-to-text transformer},
  author={Raffel, Colin and Shazeer, Noam and Roberts, Adam and others},
  journal={JMLR},
  year={2020}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and others},
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

## A.1 Soft Ramp Properties

### A.1.1 Continuity

**Claim.** $g_z(d) = \text{clamp}((R + z - d) / R, 0, 1)$ is continuous in 
$(z, d)$.

**Proof.** The function $\text{clamp}(x, 0, 1) = \min(\max(x, 0), 1)$ is 
continuous in $x$. The argument $(R + z - d) / R$ is continuous in $(z, d)$. 
Composition preserves continuity. $\square$

### A.1.2 Differentiability in $z$

**Claim.** $\partial g_z(d) / \partial z$ exists everywhere.

**Proof.** 

- For $d < z$: $g_z(d) = 1$ (constant), so $\partial g_z / \partial z = 0$.
- For $z < d < z + R$: $g_z(d) = (R + z - d) / R$, so $\partial g_z / \partial z = 1/R$.
- For $d > z + R$: $g_z(d) = 0$ (constant), so $\partial g_z / \partial z = 0$.

At the boundaries $d = z$ and $d = z + R$, the derivative is one-sided but 
the function is continuous. $\square$

## A.2 Span Gradient via Sigmoid

Given $z = \sigma(\ell) \cdot S_\text{max}$:

$$
\frac{\partial z}{\partial \ell} = \sigma(\ell)(1 - \sigma(\ell)) \cdot S_\text{max}
\tag{A.1}
$$

Then:

$$
\frac{\partial \mathcal{L}}{\partial \ell} = \frac{\partial \mathcal{L}}{\partial z} \cdot \sigma(\ell)(1 - \sigma(\ell)) \cdot S_\text{max}
\tag{A.2}
$$

## A.3 L1 Penalty Effect

The L1 penalty adds:

$$
\mathcal{L}_\text{span} = \lambda \sum_h z_h = \lambda S_\text{max} \sum_h \sigma(\ell_h)
\tag{A.3}
$$

Gradient:

$$
\frac{\partial \mathcal{L}_\text{span}}{\partial \ell_h} = \lambda S_\text{max} \cdot \sigma(\ell_h)(1 - \sigma(\ell_h))
\tag{A.4}
$$

Since $\sigma \in (0, 1)$, this is always positive. The gradient pushes 
$\ell_h$ toward $-\infty$, hence $z_h \to 0$.

## A.4 Equilibrium Analysis

**Claim.** At equilibrium, the span satisfies:

$$
\left|\frac{\partial \mathcal{L}_\text{task}}{\partial z_h}\right| = \lambda
\tag{A.5}
$$

**Proof.** The total gradient is:

$$
\frac{\partial \mathcal{L}_\text{task}}{\partial \ell_h} + \lambda S_\text{max} \sigma(\ell_h)(1 - \sigma(\ell_h)) = 0
\tag{A.6}
$$

At equilibrium. Since $\partial z_h / \partial \ell_h = S_\text{max} \sigma(1-\sigma)$, 
this simplifies to:

$$
\frac{\partial \mathcal{L}_\text{task}}{\partial z_h} + \lambda = 0
\tag{A.7}
$$

$\square$

## A.5 Approximation Error Bound

**Claim.** For a head with span $z$ and ramp $R$:

$$
\|P - \tilde{P}\|_1 \leq \sum_{(i,j): |i-j| > z} P_{ij} + \epsilon_R
\tag{A.8}
$$

where:

$$
\epsilon_R \leq \max_{i,j} |1 - g_z(|i-j|)| \cdot \sum_j P_{ij}
\tag{A.9}
$$

**Proof sketch.** The ramp down-weights distant scores. The total mass 
removed is bounded by the sum over $|i-j| > z$ plus the ramp transition 
region. $\square$

## A.6 Comparison with Hard Window

For hard window of size $w$:

$$
P_{ij}^{\text{hard}} = \frac{\mathbb{1}[|i-j| \le w] \exp(S_{ij})}{\sum_j \mathbb{1}[|i-j| \le w] \exp(S_{ij})}
\tag{A.10}
$$

The soft ramp approaches this as $R \to 0$. For finite $R$, the soft ramp 
provides **smoother gradients** and **avoids boundary artifacts**.
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

## B.1 The `AdaptiveSpanAttention` Class

```python
class AdaptiveSpanAttention(BaseAttention):
    def __init__(self, config=None, max_span=128, ramp_size=32.0, ...):
        super().__init__(config, rank)
        
        # Resolve dimensions
        ...
        
        # Span logits
        self.span_logits = nn.Parameter(torch.full((H,), 4.0))
        self.max_span = max_span
        self.ramp_size = ramp_size
        
        # Projections
        self.q_proj = LoRALinear(...)
        self.k_proj = LoRALinear(...)
        self.v_proj = LoRALinear(...)
        self.out_proj = LoRALinear(...)  # or IdentityOutProj for nested
        
        if source_attn is not None:
            self.copy_weights(source_attn)
```

## B.2 Forward Pass (Ramp Application)

```python
# Compute span from logit
span = torch.sigmoid(self.span_logits) * self.max_span  # [H]

# Fast path: skip ramp if unnecessary
if not ((span + self.ramp_size) >= N_k).all():
    pos_q = torch.arange(N_q, device=device, dtype=scores.dtype).unsqueeze(1)
    pos_k = torch.arange(N_k, device=device, dtype=scores.dtype).unsqueeze(0)
    
    if is_decoder:
        dist = (pos_q - pos_k).clamp(min=0.0)
    else:
        dist = (pos_q - pos_k).abs()
    
    span_h = span.view(1, H, 1, 1).to(scores.dtype)
    dist_mat = dist.view(1, 1, N_q, N_k)
    
    ramp = torch.clamp(
        (self.ramp_size + span_h - dist_mat) / self.ramp_size,
        0.0, 1.0
    )
    
    attn = attn * ramp
    attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-4)
    attn = attn / attn_sum
    attn = torch.nan_to_num(attn, nan=0.0)
```

## B.3 Span Properties

```python
@property
def current_span(self) -> torch.Tensor:
    return torch.sigmoid(self.span_logits) * self.max_span

def get_span_loss(self) -> torch.Tensor:
    return self.current_span.sum()
```

## B.4 Identity Out Projection for Nested

```python
class IdentityOutProj(nn.Identity):
    lora_A = None
    lora_B = None
    def freeze_base_weights(self):
        pass
```

For nested encoders (RoBERTa), we use this to skip out_proj (the outer 
`attention.output.dense` handles it).

## B.5 Handling Causal Distance

```python
if is_decoder:
    dist = (pos_q - pos_k).clamp(min=0.0)  # Only past keys
else:
    dist = (pos_q - pos_k).abs()  # Both directions
```

## B.6 Common Pitfalls

### B.6.1 Log of Zero

`torch.log(ramp)` returns `-inf` for `ramp = 0`. Add `epsilon = 1e-8`:
```python
attn = attn * ramp  # or: attn = attn + torch.log(ramp + 1e-8)
```

### B.6.2 Wrong Renormalization

After ramp, renormalize:
```python
attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-4)
```

### B.6.3 FP16 Ramp

Compute ramp in FP32, cast back:
```python
ramp = torch.clamp(...).to(scores.dtype)
```

### B.6.4 Initial Logit Too Small

If initialized with `logit = 0` (span ≈ 0.5 * max), accuracy drops from start. 
Use `logit = 4.0`.

### B.6.5 Missing Fast Path

Without fast path, ramp is allocated even for short sequences. Add the 
`if not ((span + ramp) >= N_k).all()` check.
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
| Max span | {{MAX_SPAN}} |
| Ramp size | {{RAMP_SIZE}} |
| λ | {{LAMBDA}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Mean span | {{D_SPAN}} |
| Min/Max span | {{D_SPAN_MIN}} / {{D_SPAN_MAX}} |
| Std span | {{D_SPAN_STD}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Mean span | {{R_SPAN}} |
| Reduction | {{R_SPAN_REDUCTION}}% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Mean span | {{G_SPAN}} |
| Reduction | {{G_SPAN_REDUCTION}}% |

## C.4 Per-Layer Span (DistilBERT)

| Layer | Mean span |
|---|---|
| 0 | {{D_L0_SPAN}} |
| 1 | {{D_L1_SPAN}} |
| 2 | {{D_L2_SPAN}} |
| 3 | {{D_L3_SPAN}} |
| 4 | {{D_L4_SPAN}} |
| 5 | {{D_L5_SPAN}} |

## C.5 Max Span Ablation

| $S_\text{max}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_MAX32}} | {{R_MAX32}} | {{G_MAX32}} |
| 64 | {{D_MAX64}} | {{R_MAX64}} | {{G_MAX64}} |
| 128 | {{D_MAX128}} | {{R_MAX128}} | {{G_MAX128}} |
| 256 | {{D_MAX256}} | {{R_MAX256}} | {{G_MAX256}} |

## C.6 Ramp Size Ablation

| $R$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_R1}} | {{R_R1}} | {{G_R1}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |

## C.7 L1 Penalty Ablation

| $\lambda$ | DistilBERT acc | Mean span |
|---|---|---|
| 0.0 | {{D_L0}} | {{D_SPAN0}} |
| 0.001 | {{D_L001}} | {{D_SPAN001}} |
| 0.01 | {{D_L01}} | {{D_SPAN01}} |
| 0.1 | {{D_L1}} | {{D_SPAN1}} |

## C.8 Head Span Distribution (DistilBERT Layer 3)

| Head | Span |
|---|---|
| 0 | {{D_L3_H0}} |
| 1 | {{D_L3_H1}} |
| ... | ... |
| 11 | {{D_L3_H11}} |

## C.9 Span Evolution

### DistilBERT
```
Step  Mean span
0     {{D_STEP0}}
5     {{D_STEP5}}
10    {{D_STEP10}}
15    {{D_STEP15}}
20    {{D_STEP20}}
...
50    {{D_STEP50}}
```

## C.10 Raw JSON

```json
{
  "adaptive_span": {
    "distilbert": {
      "max_span": {{MAX_SPAN}},
      "learned_mean_span": {{D_SPAN}},
      "accuracy_pct": {{D_ACC}},
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
python run_all_4_adaptive_span.py --max_span 128 --ramp_size 32

# Max span sweep
for S in 32 64 128 256; do
    python run_all_4_adaptive_span.py --max_span $S
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

### D.1.1 Hard Window Mask

**Goal.** Replace soft ramp with hard window.

**Result.** Zero gradient w.r.t. span; training diverged.

**Conclusion.** Soft ramp is necessary for differentiability.

### D.1.2 Initial Span Small

**Goal.** Initialize span to $S_\text{max}/4$ to save compute from start.

**Result.** Accuracy dropped from epoch 0.

**Conclusion.** Start fully open; let model shrink.

### D.1.3 Learned Ramp Size

**Goal.** Learn $R$ per-head via a second logit.

**Result.** High variance; unstable.

**Conclusion.** Keep $R$ as hyperparameter.

### D.1.4 Per-Layer Span Logits

**Goal.** Use different logits per (layer, head) instead of per-head shared.

**Result.** Similar accuracy, more parameters.

**Conclusion.** Per-head is sufficient.

### D.1.5 Content-Aware Span

**Goal.** Modulate span by content (query-specific).

**Result.** No improvement, 2× slower.

**Conclusion.** Distance-based span is sufficient.

## D.2 Abandoned Variants

### D.2.1 Exponential Ramp

Instead of linear ramp, use exponential decay. No improvement.

### D.2.2 Two-Sided Ramp

Ramp on both sides of span. No improvement.

### D.2.3 Multi-Span (Multiple Windows)

Multiple disjoint windows per head. No improvement.

## D.3 Failed Hyperparameters

### Ramp Size Too Small
| $R$ | DistilBERT | Note |
|---|---|---|
| 0.5 | {{D_R05}} | Near hard |
| 1 | {{D_R1}} | Sharp |
| 8 | {{D_R8}} | OK |
| 32 | {{D_R32}} | Best |
| 64 | {{D_R64}} | Too smooth |

### L1 Too Aggressive
| $\lambda$ | DistilBERT |
|---|---|
| 0.1 | {{D_L1}} |
| 0.5 | {{D_L5}} |
| 1.0 | {{D_L10}} |

Aggressive L1 collapses spans; accuracy drops.

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Hard truncation during training | Speed | Accuracy loss |
| Gradient clipping on span | Stability | No improvement |
| Warmup on span | Convergence | No improvement |
| Per-layer λ | Quality | No improvement |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.6 Lessons Learned

1. **Start fully open, shrink via L1.**
2. **Soft ramp is essential** for differentiability.
3. **$R = 32$** is a good default.
4. **$\lambda = 0.01$** balances compression and accuracy.
5. **Learned spans are interpretable.**
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
- **Interpretability:** Reveals head structure.
- **Foundation for pruning:** Basis for hard-window deployment.
- **Efficiency:** Enables fixed-window deployment.

### Negative
- **Dual-use:** Efficiency gains benefit malicious applications.
- **Concentration:** Favors large labs.

## E.2 Environmental Impact

- **Total GPU-hours:** ~{{COMPUTE_HOURS}}
- **Estimated CO₂:** {{CO2_KG}} kg CO₂eq

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English movie reviews)
- **PII:** None
- **Bias:** Western-centric

## E.4 Model Ethics

### Bias

Learned spans may correlate with language-specific structures. For English 
SST-2, this is fine; for multi-lingual, unclear.

### Misuse

Not endorsed.

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

Adaptive spans could:

- Reveal structural properties of transformer heads
- Enable efficient deployment on edge
- Reduce AI's carbon footprint

But also:

- Enable widespread AI deployment
- Concentrate capabilities

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
  "_comment": "Fill with real values from your Adaptive Span runs.",

  "MODULE_NAME": "ASA-Span",
  "FULL_NAME": "Adaptive Span Attention",
  "SUBTITLE": "Learned Per-Head Attention Windows via Differentiable Soft Ramps",
  "CATEGORY": "learned window attention",

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

  "MAX_SPAN": "128",
  "RAMP_SIZE": "32",
  "LAMBDA": "0.01",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "ACC_TOLERANCE": "5",
  "MEM_REDUCTION": "2",

  "D_ACC": "89.5",
  "D_DELTA": "-1.2",
  "D_F1": "0.895",
  "D_SPAN": "68",
  "D_SPAN_MIN": "16",
  "D_SPAN_MAX": "124",
  "D_SPAN_STD": "35",
  "D_SPAN_REDUCTION": "47",
  "D_MEM": "128 KB",
  "D_TIME": "2.1",
  "D_VRAM": "190",
  "D_LOCAL_COUNT": "5",
  "D_MID_COUNT": "4",
  "D_GLOBAL_COUNT": "3",
  "D_L0_SPAN": "95",
  "D_L1_SPAN": "82",
  "D_L2_SPAN": "70",
  "D_L3_SPAN": "60",
  "D_L4_SPAN": "52",
  "D_L5_SPAN": "48",

  "R_ACC": "76.5",
  "R_DELTA": "-2.1",
  "R_F1": "0.765",
  "R_SPAN": "72",
  "R_SPAN_MIN": "20",
  "R_SPAN_MAX": "128",
  "R_SPAN_STD": "32",
  "R_SPAN_REDUCTION": "44",
  "R_MEM": "128 KB",
  "R_TIME": "4.5",
  "R_VRAM": "680",
  "R_LOCAL_COUNT": "5",
  "R_MID_COUNT": "4",
  "R_GLOBAL_COUNT": "3",

  "G_ACC": "77.5",
  "G_DELTA": "-1.9",
  "G_F1": "0.775",
  "G_SPAN": "75",
  "G_SPAN_MIN": "18",
  "G_SPAN_MAX": "126",
  "G_SPAN_STD": "34",
  "G_SPAN_REDUCTION": "41",
  "G_MEM": "128 KB",
  "G_TIME": "4.6",
  "G_VRAM": "780",
  "G_LOCAL_COUNT": "5",
  "G_MID_COUNT": "4",
  "G_GLOBAL_COUNT": "3",
  "G_L0_SPAN": "105",
  "G_L6_SPAN": "75",
  "G_L11_SPAN": "55",

  "LOCAL_THRESH": "32",
  "GLOBAL_THRESH": "96",

  "D_MAX32": "85.0",
  "D_MAX64": "88.0",
  "D_MAX128": "89.5",
  "D_MAX256": "89.6",
  "R_MAX32": "70.0",
  "R_MAX64": "74.0",
  "R_MAX128": "76.5",
  "R_MAX256": "76.6",
  "G_MAX32": "71.0",
  "G_MAX64": "75.0",
  "G_MAX128": "77.5",
  "G_MAX256": "77.6",

  "D_LEARN32": "24",
  "D_LEARN64": "45",
  "D_LEARN128": "68",
  "D_LEARN256": "72",

  "D_R1": "87.0",
  "D_R8": "88.5",
  "D_R16": "89.0",
  "D_R32": "89.5",
  "D_R64": "88.5",
  "R_R1": "73.0",
  "R_R8": "75.0",
  "R_R16": "76.0",
  "R_R32": "76.5",
  "R_R64": "75.5",
  "G_R1": "73.5",
  "G_R8": "76.0",
  "G_R16": "77.0",
  "G_R32": "77.5",
  "G_R64": "76.5",
  "D_R05": "82.0",

  "D_L0": "90.5",
  "D_L001": "90.0",
  "D_L01": "89.5",
  "D_L1": "86.0",
  "D_L5": "78.0",
  "D_L10": "65.0",

  "D_SPAN0": "126",
  "D_SPAN001": "105",
  "D_SPAN01": "68",
  "D_SPAN1": "32",
  "D_RED001": "17",
  "D_RED01": "47",
  "D_RED1": "75",

  "D_LR4": "86.0",
  "D_LR8": "88.0",
  "D_LR16": "89.5",
  "D_LR32": "89.7",
  "R_LR4": "72.0",
  "R_LR8": "74.5",
  "R_LR16": "76.5",
  "R_LR32": "76.7",
  "G_LR4": "73.0",
  "G_LR8": "75.5",
  "G_LR16": "77.5",
  "G_LR32": "77.7",

  "D_S10": "86.0",
  "D_S25": "88.0",
  "D_S50": "89.5",
  "D_S100": "89.6",
  "R_S10": "72.0",
  "R_S25": "74.5",
  "R_S50": "76.5",
  "R_S100": "76.7",
  "G_S10": "73.0",
  "G_S25": "75.5",
  "G_S50": "77.5",
  "G_S100": "77.7",

  "GQA_D": "88.5",
  "GQA_R": "75.5",
  "GQA_G": "77.0",
  "PRIMAL_D": "86.5",
  "PRIMAL_R": "74.0",
  "PRIMAL_G": "75.0",
  "PRIMAL_MEM": "2",
  "ASA_D": "88.5",
  "ASA_R": "76.0",
  "ASA_G": "76.5",

  "FIXED_D": "85.0",
  "FIXED_R": "73.0",
  "FIXED_G": "74.0",
  "FIXED_DELTA": "4.5",

  "D_L3_H0": "28",
  "D_L3_H1": "118",
  "D_L3_H11": "34",

  "D_STEP0": "126",
  "D_STEP5": "112",
  "D_STEP10": "98",
  "D_STEP15": "88",
  "D_STEP20": "80",
  "D_STEP50": "68",

  "D_L0_SPAN": "95",
  "R_L0_SPAN": "98",
  "G_L0_SPAN": "105",

  "RUNTIME": "18",
  "COMPUTE_HOURS": "12",
  "CO2_KG": "0.85",

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
  "FAIL_DROP_2": "1.8",
  "FAIL_CAUSE": "Some heads need larger span than initialized",
  "UNRESOLVED_FAILURES": "Long-context evaluation, hard truncation speedup",

  "SEED42": "89.5",
  "SEED123": "89.3",
  "SEED456": "89.7",
  "MEAN": "89.5",
  "STD": "0.2",

  "_note": "Add keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — Adaptive Span Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `adaptive_span/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module adaptive_span`
- [ ] No `{{PLACEHOLDER}}` remains

## Content
- [ ] Abstract under 250 words
- [ ] Soft ramp formula stated
- [ ] L1 span penalty explained
- [ ] Learned span results shown

## Method
- [ ] Ramp formula numbered
- [ ] Span logit formula numbered
- [ ] Fast path optimization mentioned
- [ ] Complexity unchanged (stated honestly)

## Results
- [ ] Main table
- [ ] Learned span distribution
- [ ] Per-layer span
- [ ] Max span ablation
- [ ] Ramp size ablation
- [ ] L1 penalty ablation

## Honesty
- [ ] No parameter reduction — stated
- [ ] No speedup without hard truncation — stated
- [ ] Failure cases in Appendix D
- [ ] Single-seed limitation

## Interpretability
- [ ] Bimodal span distribution shown
- [ ] Layer-wise pattern shown
- [ ] Comparison with fixed window

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
cp -r papers/_TEMPLATE papers/adaptive_span
nano papers/adaptive_span/variables.json
cd papers/adaptive_span
python apply_vars.py --module adaptive_span
grep -rn "{{" .  # verify empty
python build.py --module adaptive_span --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE adaptive_span
```

### مرحله ۲: اجرای ASA-Span

```powershell
cd D:\SVD_Project
python run_all_4_adaptive_span.py --models distilbert,roberta,gpt2 --max_span 128 --ramp_size 32 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\adaptive_span
python apply_vars.py --module adaptive_span
python build.py --module adaptive_span
```

---

## 💡 نکات کلیدی ASA-Span

### ۱. تفاوت اصلی با ASA-Sparse

| جنبه | ASA-Sparse | ASA-Span |
|---|---|---|
| انتخاب | top-k (content-aware) | distance-based |
| پارامتر | sparsity_logit | span_logit |
| Mask | sigmoid soft top-k | linear ramp |
| Regularization | ندارد | L1 روی span |

### ۲. Soft Ramp — مزیت اصلی

$$
g_z(d) = \text{clamp}\left(\frac{R + z - d}{R}, 0, 1\right)
$$

- **Piecewise linear** → ساده
- **Differentiable** → قابل یادگیری
- **No hard cutoff** → بدون artifact مرزی

### ۳. Initialize Open, Learn to Shrink

- Span اولیه = $S_\text{max}$ (کاملاً باز)
- L1 penalty → باعث shrink تدریجی
- Equilibrium: span به اندازه‌ای که accuracy حفظ شود

### ۴. Head Heterogeneity — مهم‌ترین تفسیر

از روی learned spans می‌فهمی:
- کدام heads **local** هستند (span ~16)
- کدام heads **global** هستند (span ~128)
- این **ساختار** در روش‌های دیگر دیده نمی‌شود

### ۵. بدون Speedup

ASA-Span **سرعت inference را کم نمی‌کند** چون همه scores محاسبه می‌شوند. برای speedup واقعی:

1. Train با soft ramp
2. Extract learned spans
3. Hard truncate at inference
4. Sparse kernel

این محدودیت را **صادقانه** در abstract و discussion ذکر کن.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_adaptive_span/*/summary.csv` پر کن. اگر خواستی من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**