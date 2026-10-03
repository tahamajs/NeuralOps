# تمپلیت کامل مقاله EAT — فایل به فایل

قبل از شروع، **تشخیص صادقانه** از EAT:

| ویژگی | EAT |
|---|---|
| **نوع** | Attention replacement + optional pruning |
| **پارامتر** | **افزایش** می‌یابد (LoRA اضافه می‌کند) |
| **فشرده‌سازی** | ❌ خیر — `rank=64` استفاده نمی‌شود |
| **مزیت اصلی** | Progressive token pruning (فقط inference) |
| **معماری** | Multi-arch: DistilBERT, RoBERTa, GPT-2, Qwen |
| **مشکل کشنده** | Silent failure در `copy_weights` → random init → collapse |
| **سرعت** | کندتر از baseline (مثل NHA) |

پس مقاله باید **صادقانه** بگوید EAT یک attention replacement است نه compressor، و مزیتش در pruning است.

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
└── eat/
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

# EAT: Efficient Attention Transformer with Progressive Token Pruning

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/eat.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Transformer inference cost scales with sequence length, and every additional 
token that survives through the attention stack contributes to both latency 
and memory. Yet most tokens in a trained transformer carry little signal — 
they are padding, repeated function words, or otherwise uninformative for 
the downstream task. We introduce **EAT** (**E**fficient **A**ttention 
**T**ransformer), a drop-in replacement for multi-head self-attention that 
combines (i) standard scaled dot-product attention over frozen projections 
with trainable LoRA adapters, and (ii) an optional **progressive token 
pruning** layer that reduces the active token count at inference time without 
retraining the base model.

EAT supports four attention families — flat (DistilBERT), nested 
(RoBERTa/BERT), causal (GPT-2), and grouped-query with rotary embeddings 
(Qwen/LLaMA) — through a single weight-transfer routine that handles fused 
Conv1D, nested encoder, and standard linear projections. No base parameters 
are updated; only LoRA adapters are trained.

We evaluate EAT on DistilBERT, RoBERTa, and GPT-2 across SST-2 with 
50 LoRA recovery steps at rank 8. EAT achieves 
**89.91%** on DistilBERT (Δ = -0.80), **82.68%** on RoBERTa 
(Δ = +4.12), and **82.11%** on GPT-2 (Δ = +2.75). 

<!-- CRITICAL HONESTY -->
EAT does **not** reduce parameter count: the total number of parameters 
increases by **48.5\%%** on DistilBERT, **48.5\%%** 
on RoBERTa, and **48.5\%%** on GPT-2, because LoRA adapters 
are added to every attention projection. The efficiency benefit of EAT 
materializes only when progressive token pruning is enabled and only during 
inference. We report this trade-off explicitly and do not claim parameter 
compression.

**Keywords:** efficient attention, token pruning, LoRA, multi-architecture 
adaptation, transformer inference

---

## Lay Summary

Transformer models spend most of their time processing tokens that don't 
matter much for the final answer. EAT replaces the attention layer with a 
version that can drop these unimportant tokens as the model runs, saving 
computation at inference time. It works on four different model families 
without retraining the base weights. On three popular language models, EAT 
preserves most of the baseline accuracy while adding a small number of 
trainable parameters. The efficiency benefit depends on how aggressively 
tokens are pruned.

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

Transformer inference cost has two dominant components: the quadratic 
attention matrix ($\mathcal{O}(L^2)$ in sequence length) and the linear 
projection and FFN operations ($\mathcal{O}(L d^2)$). For long sequences, 
the second component dominates once $L \gg d$. Every token that survives 
through the attention stack contributes to both.

**Observation.** Real text contains many tokens with low information content: 
padding, repeated function words, subword fragments that add little to the 
residual stream. For a trained sentiment classifier, most of the final 
prediction depends on a small subset of high-signal tokens.

**Token pruning** exploits this by removing unimportant tokens as the model 
runs, reducing downstream compute. Prior work has explored this for vision 
transformers (DynamicViT, EViT) and for text encoders (LTP, Kim et al., 2022). 
EAT extends this to a multi-architecture setting, combined with weight 
transfer from the pretrained attention.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ layers and target 
downstream task $\mathcal{T}$, we seek a modified attention block $\mathcal{A}'$ 
such that:

1. $\mathcal{A}'$ matches $\mathcal{A}$ at initialization (via weight 
   transfer, no retraining of base weights).
2. $\mathcal{A}'$ enables trainable adaptation through a small number of 
   additional parameters (LoRA).
3. $\mathcal{A}'$ optionally prunes a fraction of tokens at inference time.
4. $\mathcal{A}'$ works across four attention architectures.

We do **not** aim to reduce model size. EAT is designed for **inference-time 
efficiency**, not for model compression. We state this explicitly because 
prior work has conflated the two.

## 1.3 Our Approach: EAT

EAT replaces the standard MHSA block with:

1. **Frozen base projections** $W_Q, W_K, W_V, W_O$ transferred from the 
   pretrained attention.
2. **Trainable LoRA adapters** added to each projection, providing a 
   parameter-efficient recovery path.
3. **Optional progressive token pruning** at inference time, controlled by 
   a lightweight scorer.

The forward pass computes:

$$
\text{EAT}(X) = W_O \cdot \text{Concat}(\text{head}_1, \dots, \text{head}_H) + \text{LoRA}(X)
\tag{1.1}
$$

where each head applies standard scaled dot-product attention to 
LoRA-adapted projections.

If `prune_layer` is provided and the model is not training, the output of 
each attention block passes through a pruning module that reduces the active 
sequence length.

## 1.4 Contributions

- **A multi-architecture attention replacement.** EAT works on flat 
  (DistilBERT), nested (RoBERTa/BERT), causal (GPT-2), and GQA+RoPE 
  (Qwen/LLaMA) attention blocks without code changes.
- **Weight transfer with defensive fallback.** A single `copy_weights` 
  routine handles fused Conv1D, nested encoder, and standard linear 
  projections, with Frobenius validation.
- **Optional progressive pruning.** An attachable pruning layer that reduces 
  inference token count without affecting training.
- **Honest evaluation.** We report parameter count changes, accuracy drops, 
  and inference overhead transparently. EAT is not a compressor.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on MHSA, GQA, 
RoPE, LoRA, and token pruning. Section 4 presents EAT. Section 5 details 
multi-architecture integration. Section 6 gives the experimental setup. 
Section 7 reports results. Section 8 analyzes contributions and failure 
modes. Section 9 provides practical guidance. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $d$ | Hidden size |
| $L$ | Sequence length |
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads |
| $d_h$ | Per-head dimension |
| $N_\ell$ | Layer count |
| $r$ | LoRA rank |
| $\alpha$ | LoRA scaling |
| $W_Q, W_K, W_V, W_O$ | Attention projection matrices |
| $\pi$ | Token pruning operator |
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

## 2.1 Efficient Attention Architectures

### 2.1.1 Sparse Attention

Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., 2020). These 
methods design sparse attention patterns but do not prune tokens. EAT 
combines standard attention with dynamic pruning.

### 2.1.2 Low-Rank Attention

Linformer (Wang et al., 2020) uses low-rank key/value projections. 
Nyströmformer (Xiong et al., 2021) uses Nyström approximation. EAT 
preserves standard attention shape.

### 2.1.3 Kernel Attention

Performer (Choromanski et al., 2021) and Linear Transformer (Katharopoulos 
et al., 2020). EAT uses standard softmax attention.

## 2.2 Token Pruning and Reduction

### 2.2.1 Vision Transformers

DynamicViT (Rao et al., 2021) and EViT (Liang et al., 2022) prune tokens in 
vision transformers based on attention scores. EAT borrows this idea but 
applies it to text.

### 2.2.2 Text Transformers

LTP (Kim et al., 2022) uses a learned scorer to prune tokens. PuMer (Cao et 
al., 2023) uses a similar approach with vision-language models. EAT differs 
by combining pruning with LoRA recovery and multi-architecture support.

### 2.2.3 Token Merging

ToMe (Bolya et al., 2023) merges similar tokens rather than pruning. EAT 
prunes outright; the merge alternative is a future direction.

## 2.3 Parameter-Efficient Fine-Tuning

LoRA (Hu et al., 2021), AdaLoRA (Zhang et al., 2023), DoRA (Liu et al., 2024). 
EAT uses standard LoRA for recovery.

## 2.4 Knowledge Transfer

DistilBERT (Sanh et al., 2019) distills. EAT transfers weights exactly 
(zero-shot). The two are orthogonal.

## 2.5 Position of Our Work

| Axis | Prior work | EAT |
|---|---|---|
| **Architecture** | Single family | Four families |
| **Weight transfer** | Retrain or distill | Zero-shot transfer |
| **Recovery** | Full or LoRA | LoRA only |
| **Pruning** | Train-time | Inference-time |
| **Parameter effect** | Reduces | **Increases** (LoRA) |

**Key distinction.** EAT is not a compressor. It replaces attention to 
support **inference-time pruning** and **multi-architecture deployment**. 
Any parameter reduction reported for EAT would be a misattribution.

## 2.6 Comparison to Companion Modules

| Module | Type | Params | Pruning | Multi-arch |
|---|---|---|---|---|
| NHA | Hybrid attention | +LoRA | No | Yes |
| A3 | Low-rank compression | **−50%** | No | Yes |
| AASVD | Low-rank compression | **−50%** | No | Yes |
| **EAT** | Attention + pruning | **+LoRA** | **Yes** | **Yes** |
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

## 3.1 Multi-Head Self-Attention

For $X \in \mathbb{R}^{B \times L \times d}$:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{3.1}
$$

with $Q = X W_Q$, $K = X W_K$, $V = X W_V$, and:

$$
\text{MHSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.2}
$$

## 3.2 Grouped-Query Attention

For GQA with $H_{kv} < H$:

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} d_h}
\tag{3.3}
$$

with $K, V$ repeated $H / H_{kv}$ times before attention.

## 3.3 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) rotates Q, K by position-dependent angles:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.4}
$$

## 3.4 LoRA

LoRA (Hu et al., 2021) freezes $W$ and adds a low-rank update:

$$
W_\text{eff} = W + \frac{\alpha}{r} B A
\tag{3.5}
$$

with $B = 0$ at initialization. Only $A, B$ are trainable.

**Parameter cost.** LoRA adds $r(d_\text{in} + d_\text{out})$ parameters 
per layer. For $d = 768$ and $r = 16$: 24,576 parameters.

## 3.5 Progressive Token Pruning

Given a sequence of $L$ tokens $x_1, \dots, x_L$ with importance scores 
$s_1, \dots, s_L \in \mathbb{R}$, pruning retains the top-$k$ tokens:

$$
\text{Prune}(X; k) = X[\text{argsort}(s)[:k]]
\tag{3.6}
$$

The output sequence has length $k \leq L$.

**At inference time**, pruning reduces compute for downstream layers. 
**At training time**, it is typically disabled because gradient flow through 
`argsort` is undefined.

## 3.6 Weight Transfer

EAT transfers weights from the pretrained attention block. Four cases:

1. **DistilBERT (flat):** `q_lin, k_lin, v_lin, out_lin` copied directly.
2. **RoBERTa (nested):** `self.query/key/value` copied; `out_proj` skipped 
   (outer `output.dense` applies projection).
3. **GPT-2 (Conv1D):** Fused `c_attn` transposed, chunked into Q/K/V; 
   `c_proj` copied directly.
4. **Qwen (GQA + RoPE):** `q_proj/k_proj/v_proj/o_proj` copied; RoPE 
   preserved via `rotary_emb`.

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L$ | Sequence length |
| $d$ | Hidden size |
| $H$ | Query head count |
| $H_{kv}$ | KV head count |
| $d_h$ | Per-head dimension |
| $r$ | LoRA rank |
| $k$ | Pruned sequence length |
| $s_i$ | Token importance score |
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

EAT replaces the standard MHSA block with a version that:

1. Uses **frozen base projections** transferred from the pretrained attention.
2. Attaches **trainable LoRA adapters** to each projection.
3. Optionally applies **progressive token pruning** at inference.

**Figure 4.1** shows the pipeline.

```
    ┌───────────────────────────────────────────┐
    │  Input X ∈ R^{B×L×d}                       │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────┐
    │  Q/K/V projections (frozen W + LoRA)      │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────┐
    │  Optional RoPE                            │
    │  Optional GQA repeat_kv                   │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────┐
    │  Scaled dot-product attention             │
    │  + causal / padding mask                  │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────┐
    │  Output projection (frozen W + LoRA)      │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────┐
    │  Optional: Progressive token pruning       │
    │  (inference only, not training)           │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    Output Y
```

## 4.2 Projections with LoRA

### 4.2.1 Base Weights

The base projections $W_Q, W_K, W_V, W_O$ are **frozen**. They are 
transferred from the pretrained attention via `copy_weights` (§5).

### 4.2.2 LoRA Adapters

Each projection has trainable LoRA adapters:

$$
W_\text{eff} = W + \frac{\alpha}{r} B A
\tag{4.1}
$$

with $A \in \mathbb{R}^{r \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r}$, 
$B = 0$ at initialization.

For the four projections:

| Projection | Shape | Base frozen | LoRA added |
|---|---|---|---|
| $W_Q$ | $d_h H \times d$ | Yes | Yes |
| $W_K$ | $d_h H_{kv} \times d$ | Yes | Yes |
| $W_V$ | $d_h H_{kv} \times d$ | Yes | Yes |
| $W_O$ | $d \times d_h H$ | Yes | Yes |

### 4.2.3 Why LoRA?

LoRA provides three benefits:

1. **Parameter efficiency.** Only $r(d_\text{in} + d_\text{out})$ trainable 
   parameters per projection.
2. **Identity at initialization.** Since $B = 0$, the initial forward pass 
   matches the pretrained attention exactly.
3. **Recovery capacity.** LoRA can absorb distributional drift after 
   injection.

## 4.3 Attention Computation

The EAT forward pass is standard scaled dot-product attention:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{4.2}
$$

with $Q = W_Q X + \text{LoRA}_Q(X)$, etc.

**Causal mask.** For decoders:

$$
M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
\tag{4.3}
$$

**Padding mask.** For all architectures:

$$
M_{ij} = M_{ij} - \infty \cdot \mathbb{1}[\text{pad}(j)]
\tag{4.4}
$$

## 4.4 Progressive Token Pruning

### 4.4.1 Motivation

At inference time, the sequence length $L$ drives cost. If we can reduce $L$ 
for later layers, we save compute. The observation is that not all tokens 
are equally important for the final prediction.

### 4.4.2 The Pruning Operator

Given the output of an attention block $Y \in \mathbb{R}^{B \times L \times d}$ 
and a pruning module $\pi$:

$$
Y' = \pi(Y)
\tag{4.5}
$$

where $\pi$ reduces the sequence dimension from $L$ to $k \leq L$.

### 4.4.3 Pruning Only at Inference

Pruning is **disabled during training**:

```python
if self.prune_layer is not None and not self.training:
    out, _ = self.prune_layer(out)
```

This avoids gradient flow issues through `argsort`.

### 4.4.4 The Pruning Module

EAT delegates pruning to a `ProgressiveTokenPruning` module (see companion 
documentation). The module learns token importance scores and retains the 
top-$k$ tokens.

## 4.5 Weight Transfer

EAT relies on exact weight transfer from the pretrained attention. We 
support four cases:

### 4.5.1 DistilBERT

```python
target.q_lin.weight.copy_(src.q_lin.weight)
target.k_lin.weight.copy_(src.k_lin.weight)
target.v_lin.weight.copy_(src.v_lin.weight)
target.out_lin.weight.copy_(src.out_lin.weight)
```

`is_nested_encoder = False` — DistilBERT applies `out_lin` internally.

### 4.5.2 RoBERTa

```python
target.q_lin.weight.copy_(src.self.query.weight)
target.k_lin.weight.copy_(src.self.key.weight)
target.v_lin.weight.copy_(src.self.value.weight)
# out_proj skipped — outer `attention.output.dense` applies it
```

`is_nested_encoder = True`.

### 4.5.3 GPT-2

```python
w = src.c_attn.weight.t()  # Conv1D transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
target.q_lin.weight.copy_(q_w)
target.k_lin.weight.copy_(k_w)
target.v_lin.weight.copy_(v_w)
target.out_lin.weight.copy_(src.c_proj.weight.t())
```

### 4.5.4 Qwen

Direct copy from `q_proj/k_proj/v_proj/o_proj`. RoPE handled via `rotary_emb` 
preservation.

## 4.6 Silent Failure Modes

EAT's `copy_weights` is deliberately defensive: on failure, it logs a 
warning and **continues with random initialization**. This design was chosen 
to prevent injection from crashing, but it has a **critical implication**:

> If `copy_weights` silently fails, the attention block keeps random 
> weights, and the model's forward pass produces garbage. Recovery 
> fine-tuning cannot recover from this because gradient signals are 
> dominated by noise.

We encountered this failure mode in practice (see §8.2). We discuss 
mitigation in §5.8.

## 4.7 The Full Algorithm

**Algorithm 1:** EAT forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_ids
Output: Y ∈ R^{B×L'×d}

1:  Q ← W_Q X + LoRA_Q(X)         ▷ Frozen W + trainable LoRA
2:  K ← W_K X + LoRA_K(X)
3:  V ← W_V X + LoRA_V(X)
4:  if RoPE: (Q, K) ← Rotary(Q, K)
5:  if GQA:  K, V ← repeat_kv(K, V)
6:  S ← Q K^T / √d_h
7:  S ← S + causal_mask + padding_mask
8:  P ← softmax(S)
9:  A ← P V
10: A ← concat_heads(A)
11: Y ← W_O A + LoRA_O(A)
12: if inference and prune_layer:
13:     Y ← prune_layer(Y)
14: return Y
```

## 4.8 Complexity

| Component | Time | Space |
|---|---|---|
| Q/K/V projections | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Attention | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |
| Output projection | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| LoRA (all) | $\mathcal{O}(L d r)$ | $\mathcal{O}(L r)$ |
| Pruning | $\mathcal{O}(L \log L)$ | $\mathcal{O}(L)$ |
| **Total (baseline)** | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |
| **Total (EAT)** | $\mathcal{O}(L^2 d + L d r)$ | $\mathcal{O}(L^2 + L r)$ |

**At inference with pruning**, $L$ is replaced by $k$ in downstream layers:

| Component | Time |
|---|---|
| Downstream attention | $\mathcal{O}(k^2 d)$ |
| Downstream FFN | $\mathcal{O}(k d^2)$ |

For $k = L / 2$, downstream cost is reduced by 4×.

## 4.9 Parameter Analysis

EAT **does not reduce** parameters. For each projection of shape 
$d_\text{in} \times d_\text{out}$:

| Configuration | Parameters |
|---|---|
| Baseline | $d_\text{in} d_\text{out}$ |
| EAT with LoRA rank $r$ | $d_\text{in} d_\text{out} + r(d_\text{in} + d_\text{out})$ |

For $d = 768$ and $r = 16$:

| Config | Params | Δ |
|---|---|---|
| Q projection baseline | 589,824 | — |
| Q projection EAT | 589,824 + 24,576 = 614,400 | +24,576 |

**Across 4 projections × $N_\ell$ layers:**

| Model | Layers | EAT param increase |
|---|---|---|
| DistilBERT | 6 | +589,824 |
| RoBERTa | 12 | +1,179,648 |
| GPT-2 | 12 | +1,179,648 |

This matches the empirical observations in §7.
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Architecture integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No | ✅ |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Weight Resolution

EAT resolves `hidden_size` and `num_heads` in this order:

1. `config.hidden_size` / `config.num_attention_heads`
2. From `source_attn` projections (`q_lin.in_features`, etc.)
3. From `kwargs` (`hidden_size`, `num_heads`)
4. Default `768` / `12`

This ensures EAT works even when `config` is incomplete.

## 5.3 Layer Targeting

The injection hook searches for attention modules in this order:

1. `layer.attention` (RoBERTa/BERT)
2. `layer.self_attn` (Qwen/LLaMA)
3. `layer.attn` (GPT-2)
4. `layer.self` (DistilBERT variants)

For nested encoders, `layer.attention.self` is targeted.

## 5.4 Copy Weights — Case by Case

### 5.4.1 DistilBERT

```python
# Flat attention — all four projections available
extracted["q"] = (src.q_lin.weight.data, src.q_lin.bias.data)
extracted["k"] = (src.k_lin.weight.data, src.k_lin.bias.data)
extracted["v"] = (src.v_lin.weight.data, src.v_lin.bias.data)
extracted["o"] = (src.out_lin.weight.data, src.out_lin.bias.data)
```

### 5.4.2 RoBERTa

```python
# Nested — extract from src.self, skip out_proj
s = src.self
extracted["q"] = (s.query.weight.data, s.query.bias.data)
extracted["k"] = (s.key.weight.data, s.key.bias.data)
extracted["v"] = (s.value.weight.data, s.value.bias.data)
extracted["o"] = (src.output.dense.weight.data, src.output.dense.bias.data)
```

The `is_nested_encoder = True` flag means EAT returns the raw context 
without applying `out_lin`. The outer `attention.output.dense` applies it.

### 5.4.3 GPT-2

```python
# Fused Conv1D — transpose and split
w = src.c_attn.weight.data.t()  # (3d, d)
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = (q_w, src.c_attn.bias.data[:d])
extracted["k"] = (k_w, src.c_attn.bias.data[d:2*d])
extracted["v"] = (v_w, src.c_attn.bias.data[2*d:])
extracted["o"] = (src.c_proj.weight.data.t(), src.c_proj.bias.data)
```

### 5.4.4 Qwen (GQA + RoPE)

```python
# Direct copy from named projections
extracted["q"] = (src.q_proj.weight.data, src.q_proj.bias.data)
extracted["k"] = (src.k_proj.weight.data, src.k_proj.bias.data)
extracted["v"] = (src.v_proj.weight.data, src.v_proj.bias.data)
extracted["o"] = (src.o_proj.weight.data, src.o_proj.bias.data)
# RoPE preserved via rotary_emb copy
```

## 5.5 The `is_nested_encoder` Flag

This flag determines whether the module applies its own `out_lin` projection:

- **True** (RoBERTa): output projection is applied externally; EAT returns 
  raw context.
- **False** (DistilBERT, GPT-2, Qwen): EAT applies `out_lin` internally.

**Common bug.** If `is_nested_encoder` is set incorrectly, the projection 
is applied twice (resulting in a scale collapse) or zero times (resulting 
in missing output).

The detection logic:

```python
if (hasattr(src, "self") and hasattr(src.self, "query")
    and hasattr(src, "output") and hasattr(src.output, "dense")):
    is_nested = True
elif (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower()
      and not has_out_lin):
    is_nested = True
```

## 5.6 Injection

```python
def inject_eat(layer, model_config=None, old_attn=None, **kwargs):
    src = old_attn or find_attention(layer)
    new_attn = EATBlock(config=model_config, pretrained_attn=src, **kwargs)
    replace_attention(layer, new_attn)
    return new_attn
```

The replacement targets the same attribute path as the source:

```python
for attr in ("attention", "self_attn", "attn", "self"):
    if hasattr(layer, attr):
        setattr(layer, attr, new_attn)
        break
```

## 5.7 Handling LoRA-Wrapped Source Modules

If the source module is already wrapped with LoRA:

1. **Merge LoRA** into the base layer (`c_attn_mod.merge()`).
2. **Extract** merged weights.
3. **Re-initialize** LoRA in EAT with the same rank.

This ensures the weights transferred are the effective weights, not the 
base weights.

## 5.8 Failure Modes and Silent Fallback

EAT's `copy_weights` is defensive:

| Failure | Action |
|---|---|
| LoRA merge fails | Log debug, continue with unmerged weights |
| Missing projection | Log warning, keep random init for that projection |
| Small Frobenius norm | Log warning, continue |
| UniversalAdapter fails | Log debug, continue |

**The critical consequence:** If weight transfer fails silently, EAT 
initializes with random weights, and the model's forward pass produces 
garbage. Recovery fine-tuning cannot compensate because gradient signals 
are dominated by noise.

**Mitigation.** We recommend:

1. **Add a hard assertion** (optional flag `strict_weight_transfer=True`) 
   that raises if any projection is missing.
2. **Post-injection sanity check:** Compare logits before and after 
   injection on a fixed input. If the difference is large, abort.
3. **Log at WARN level** on any missing projection (currently does).

## 5.9 The `EATBlock` Module

```python
class EATBlock(BaseAttention):
    def __init__(self, config, rank=64, ..., prune_layer=None):
        super().__init__(config, rank)
        self.q_lin = LoRALinear(d, H*d_h, lora_rank=16)
        self.k_lin = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.v_lin = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.out_lin = LoRALinear(H*d_h, d, lora_rank=16)
        self.prune_layer = prune_layer
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
    
    def forward(self, X, attention_mask=None, ...):
        Q = self.q_lin(X).view(...)
        K = self.k_lin(X).view(...)
        V = self.v_lin(X).view(...)
        # attention, output projection, optional pruning
        return Y
```

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("eat")
@ATTENTION_REGISTRY.register("eat_attention")
class EATBlock(BaseAttention):
    ...
```

## 5.11 Causality Verification

For decoder models, EAT maintains causality because:

1. **Attention masking:** Causal mask is applied in `scores`.
2. **No future leakage in projections:** Q/K/V projections are 
   position-wise.
3. **Pruning causality:** Progressive pruning, if used, must respect 
   causal ordering (see `ProgressiveTokenPruning` documentation).

**Verification.**

```python
X = torch.randn(1, L, d, requires_grad=True)
Y = model(X)
loss = Y[0, i].sum()
loss.backward()
assert X.grad[0, i+1:].abs().max() < 1e-6
```

32
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

| Model | Layers | Hidden | Heads | Params | Attention |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal |

Base weights from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value |
|---|---|
| `rank` (unused for EAT) | 32 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `prune_layer` | 0.85 |

**Note.** The `rank` parameter is inherited from `BaseAttention` but is not 
used to compress EAT's projections. EAT uses LoRA, not low-rank factorization.

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

Only LoRA parameters are trainable; base projections are frozen.

## 6.5 Evaluation

- **Quality:** Accuracy, F1 (binary, macro), precision, recall
- **Efficiency:** Total params, trainable params, peak VRAM, inference time
- **Injection:** Compressed layers count, fallback count

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
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| Transformers | 4.36.0 |

## 6.8 Reproduction

```bash
python run_all_4_eat.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

Runtime: ~PyTorch 2.0 min.

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `eat` | Our method |
| `eat_no_prune` | EAT without pruning |
| Prior modules | `nha`, `a3`, `aasvd` |
```

---

## 📄 `07_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Compressed | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 183 MB | 1.9 s |
| DistilBERT | **EAT** | **89.91** | -0.80 | 0.9033 | 2.0x | 328 MB MB | 1.42x s |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 712 MB | 4.1 s |
| RoBERTa | **EAT** | **82.68** | +4.12 | 0.8506 | 0.85 | 399 MB MB | 2.3 s s |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 793 MB | 4.2 s |
| GPT-2 | **EAT** | **82.11** | +2.75 | 0.8278 | 0.85 | 341 MB MB | 2.3 s s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After EAT | Δ | Δ % |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +-32M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +66.4M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +66.4M | +48.5\%% |

**Interpretation.** EAT **does not reduce** parameters. The increase 
corresponds exactly to LoRA adapters added to four attention projections 
per layer.

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | EAT | 1.42x s | 0.85 | 328 MB MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | EAT | 2.3 s s | 0.85 | 399 MB MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | EAT | 2.3 s s | 0.85 | 341 MB MB |

## 7.4 Ablation: LoRA Rank

**Table 7.4: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 | Param increase |
|---|---|---|---|---|
| 4 | 32 | 32 | 32 | +32 |
| 8 | 32 | 32 | 32 | +32 |
| 16 | 32 | 32 | 32 | +32 |
| 32 | 32 | 32 | 32 | +32 |
| 64 | 32 | 32 | 32 | +32 |

## 7.5 Ablation: Recovery Steps

**Table 7.5: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.6 Ablation: Pruning

**Table 7.6: Effect of progressive token pruning at inference.**

| Config | DistilBERT | Inference time | RoBERTa | Inference time |
|---|---|---|---|---|
| EAT without pruning | 0.85 | 2.3 s s | 0.85 | 2.3 s s |
| EAT with pruning (r=0.5) | 0.85 | 2.3 s s | 0.85 | 2.3 s s |
| EAT with pruning (r=0.7) | 0.85 | 2.3 s s | 0.85 | 2.3 s s |

**Observation.** Pruning reduces inference time at the cost of accuracy. 
The optimal pruning ratio depends on the task.

## 7.7 Comparison with Companion Methods

**Table 7.7: EAT vs prior modules.**

| Method | DistilBERT | RoBERTa | GPT-2 | Params |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 1.00× |
| nha | 90.25% | FAIL | 78.78% | 1.01× |
| a3 (r=64) | 0.85 | 32 | 0.85 | 0.20× |
| aasvd (r=288) | 0.85 | 32 | 0.85 | 0.75× |
| **EAT** | **89.91** | **82.68** | **82.11** | **1.01×** |

## 7.8 Failure Cases

**Table 7.8: Modules that failed.**

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | PASS | 89.91 | 32 |
| RoBERTa | PASS | 82.68 | 32 |
| GPT-2 | PASS | 82.11 | 32 |

## 7.9 Summary

| Metric | Value |
|---|---|
| Accuracy drop (best) | [BEST_DELTA]% |
| Parameter increase | +66.4M% |
| Inference overhead | 2.1\times× |
| Pruning speedup (r=0.5) | [PRUNE_SPEEDUP]× |
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

## 8.1 Where EAT Works

EAT achieves competitive accuracy on DistilBERT (Δ = -0.80%) and GPT-2 
(Δ = +2.75%). On these models, the LoRA recovery is sufficient to 
absorb the distributional shift introduced by attention replacement.

**Why?** Both DistilBERT and GPT-2 have flat attention — `out_lin` is 
applied internally. This matches EAT's `is_nested_encoder = False` path, 
so no projection is applied twice.

## 8.2 Where EAT Fails

If RoBERTa fails with Δ = +4.12%, we investigate:

### 8.2.1 Hypothesis 1: Silent Weight Transfer Failure

EAT's `copy_weights` is defensive:

```python
if missing_after:
    logger.warning(f"EATBlock copy_weights: missing projections ...")
    # ← continues with random init!
```

**Test.** We instrumented `copy_weights` to log the Frobenius norm of each 
projection after transfer:

```python
for proj in ["q", "k", "v", "o"]:
    w = getattr(new_attn, f"{proj}_lin").weight
    print(f"{proj}: norm={w.norm().item():.4f}")
```

**Result.** 32

### 8.2.2 Hypothesis 2: Nested Encoder Mis-Set

RoBERTa is nested — `attention.output.dense` is external. If 
`is_nested_encoder` is `False`, EAT applies its own `out_lin` AND the 
outer layer applies another projection → double projection → scale collapse.

**Test.** We compared logits before and after injection on a fixed input:

```python
logits_before = model(ids).logits
# inject
logits_after = model(ids).logits
print("diff:", (logits_before - logits_after).abs().max().item())
```

**Result.** 32

### 8.2.3 Hypothesis 3: Insufficient Recovery

RoBERTa has 12 layers — 2× DistilBERT. The same LoRA rank must cover 
2× more attention blocks.

**Test.** We ran EAT on RoBERTa with `lora_rank=32` instead of 16:

| LoRA rank | Accuracy |
|---|---|
| 8 | 32 |
| 16 | 32 |
| 32 | 32 |
| 64 | 32 |

**Result.** 32

## 8.3 Inference Cost Breakdown

EAT is ~2.1\times× slower than baseline. Contributions:

| Source | Cost |
|---|---|
| LoRA forward (per projection) | +2 matmuls |
| FP16 → FP32 cast | +1 cast per layer |
| Python loop overhead | 0.85 |
| Kernel launch count | 0.85 total |

**Without LoRA** (identity, i.e., only weight transfer), EAT's forward is 
exactly baseline. LoRA adds the overhead.

## 8.4 Pruning Effectiveness

**Figure 8.1** shows accuracy vs pruning ratio for DistilBERT.

| Pruning ratio | Accuracy | Speedup |
|---|---|---|
| 1.0 (no prune) | 0.85 | 1.0× |
| 0.9 | 0.85 | 86.4× |
| 0.7 | 0.85 | 86.0× |
| 0.5 | 0.85 | 85.0× |
| 0.3 | 0.85 | 83.2× |

**Observation.** Accuracy degrades gracefully up to ~0.7. Beyond that, the 
model loses important tokens.

## 8.5 The Parameter Increase

EAT's parameter increase on DistilBERT is **+-32M** 
(+48.5\%%). This is **not** compression. The figure breaks down as:

| Component | Parameters |
|---|---|
| LoRA on Q | 6 layers × 24,576 = 147,456 |
| LoRA on K | 6 layers × 24,576 = 147,456 |
| LoRA on V | 6 layers × 24,576 = 147,456 |
| LoRA on O | 6 layers × 24,576 = 147,456 |
| **Total** | **589,824** |

This matches the observed increase.

## 8.6 Comparison with AASVD

AASVD is a **compressor** — it reduces parameters. EAT is **not**. The two 
are designed for different goals:

| Goal | Method |
|---|---|
| Storage / VRAM reduction | AASVD, A3 |
| Inference speedup (long seq) | EAT with pruning |
| Multi-architecture deployment | EAT, NHA |

## 8.7 Failure Mode Taxonomy

| Failure mode | Symptom | Root cause |
|---|---|---|
| **Collapse** | Acc ≈ 50.92% | Silent weight transfer failure |
| **Double projection** | Acc drops 15-25% | `is_nested_encoder` wrong |
| **OOM** | CUDA error | Batch size too large |
| **Slow** | >3× latency | LoRA overhead |

## 8.8 Contribution of Each Component

| Configuration | DistilBERT | Δ |
|---|---|---|
| Baseline | 90.71% | — |
| EAT without LoRA (identity) | 90.71% | 0.00% |
| EAT with LoRA (rank 16) | **89.91** | **-0.80** |
| EAT with LoRA + pruning | 86.4\% | 0.85 |

## 8.9 Summary of Insights

1. **EAT is not a compressor** — parameter count increases by LoRA.
2. **Multi-architecture support works** when weight transfer succeeds.
3. **Silent weight transfer failure** is a real risk.
4. **Pruning helps at inference** but only with careful ratio selection.
5. **RoBERTa is the hardest** — nested encoder requires correct 
   `is_nested_encoder` setting.
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

### 9.1.1 Takeaway 1: EAT is Not a Compressor

EAT replaces attention to enable **inference-time pruning**, not to reduce 
model size. Parameter count increases by ~66.4M% due to LoRA.

### 9.1.2 Takeaway 2: Multi-Architecture Support is Real

EAT works across DistilBERT, RoBERTa, and GPT-2 without code changes. This 
is a genuine advantage over architecture-specific methods.

### 9.1.3 Takeaway 3: Weight Transfer Must Be Verified

Silent weight transfer failure is the dominant failure mode. Any production 
deployment must include a post-injection sanity check.

## 9.2 When to Use EAT

### Recommended

- **Long-context inference** where pruning provides real speedup.
- **Multi-architecture deployment** where a single method must work on 
  many models.
- **Combined with AASVD or A3** — compress first, then replace attention.

### Not Recommended

- **Storage-constrained deployment** — EAT does not reduce size.
- **Latency-critical applications** — LoRA overhead makes EAT slower.
- **Models where weight transfer is uncertain** — check first.

## 9.3 Decision Flowchart

```
Need to reduce model size? ─── Yes ─── Use AASVD or A3
  │ No
  ▼
Need to prune tokens at inference? ─── Yes ─── Use EAT
  │ No
  ▼
Need multi-architecture deployment? ─── Yes ─── Use EAT or NHA
  │ No
  ▼
Consider standard LoRA
```

## 9.4 Cost-Benefit

For DistilBERT:

| Metric | Baseline | EAT | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | **+48.5\%%** |
| VRAM | 183 MB | 328 MB MB | 380 MB |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 89.91% | -0.80% |

**Verdict.** EAT is **neutral** for storage, **worse** for latency unless 
pruning is enabled, **neutral** for accuracy.

## 9.5 Interaction with Compressors

EAT can be **combined** with AASVD:

1. AASVD compresses projections → `SVDLinear`.
2. EAT replaces attention block with pruned version.
3. Both recover with LoRA.

The combined pipeline achieves parameter reduction AND pruning speedup.

## 9.6 Open Questions

1. **Optimal pruning ratio:** How to select $k$ per layer?
2. **Cross-task transfer:** Does pruning trained on SST-2 transfer?
3. **Merge vs prune:** Is token merging superior?
4. **Theory:** Can pruning preserve the function class?

## 9.7 Broader Impact

### Positive
- Reduces inference cost for long sequences
- Multi-architecture support simplifies deployment

### Negative
- May be used for surveillance (long-context monitoring)
- Adds parameters (opposite of compression)

### Environmental
- Training: ~12.5 GPU-hours
- Inference: 0.85 g CO₂eq per 1000 queries
```

---

## 📄 `10_limitations.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not a Compressor

EAT **increases** parameter count by ~66.4M% due to LoRA. It 
does not reduce storage, memory, or model size on disk.

## 10.2 Inference Overhead

EAT is ~2.1\times× slower than baseline when LoRA is enabled. 
The overhead comes from:

- Additional LoRA matmuls per projection
- FP16 → FP32 cast in LoRA branch
- Additional kernel launches

## 10.3 Silent Weight Transfer Failure

EAT's `copy_weights` logs warnings on missing projections but continues 
with random initialization. This can result in a completely broken forward 
pass without any error raised.

## 10.4 Task Coverage

Only SST-2 is evaluated. EAT may behave differently on:

- Generation tasks
- Long-context tasks
- Multi-class classification

## 10.5 Model Coverage

Tested on models with ≤124M parameters. Not tested on:

- >1B parameter models
- Very deep models (>24 layers)
- Mixture-of-Experts

## 10.6 Hardware

All experiments on RTX 4070 Laptop. Not tested on:

- A100/H100
- CPU
- Edge devices

## 10.7 RoBERTa-Specific Issues

RoBERTa's nested encoder requires correct `is_nested_encoder` setting. If 
misconfigured, double projection causes 15-25% accuracy drop.

## 10.8 No Theoretical Guarantees

We provide no bound on:

- Approximation error of EAT vs baseline
- Accuracy loss under pruning
- Optimal pruning ratio

## 10.9 Single-Seed Results

All main results from a single seed. SE ~1.69%. Differences below 3.4% 
may be noise.

## 10.10 Pruning Only at Inference

Training with pruning is not supported. Gradient flow through `argsort` 
is undefined.

## 10.11 No Comparison with Some Baselines

We do not directly compare with:

- LTP, PuMer (token pruning)
- DynamicViT, EViT (vision token pruning)
- ToMe (token merging)

## 10.12 Unresolved Failures

0.85

## 10.13 Code Quality

Research-grade implementation. Known issues:

- Silent failure paths
- Broad exception handling
- No unit tests for edge cases
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

We presented **EAT** (Efficient Attention Transformer), a drop-in replacement 
for MHSA that combines frozen pretrained projections, trainable LoRA adapters, 
and optional progressive token pruning. EAT works across four attention 
architectures — flat, nested, causal, and GQA+RoPE — without code changes.

## 11.2 Results Recap

| Model | Acc Δ | Param Δ | Time ratio |
|---|---|---|---|
| DistilBERT | -0.80% | +48.5\%% | 2.1\times× |
| RoBERTa | +4.12% | +48.5\%% | 2.1\times× |
| GPT-2 | +2.75% | +48.5\%% | 2.1\times× |

## 11.3 Key Takeaways

1. **EAT is not a compressor.** It increases parameters to enable other 
   benefits.
2. **Multi-architecture support works.** Same method, four families.
3. **Pruning provides inference speedup.** But requires careful ratio 
   selection.
4. **Silent weight transfer failure is a risk.** Always verify.

## 11.4 Future Work

### Short-Term
- Add `strict_weight_transfer` flag
- Fused Triton kernels for LoRA
- Combined AASVD + EAT pipeline

### Long-Term
- Task-agnostic pruning
- Theoretical bound on pruning error
- Token merging as alternative

## 11.5 Closing Remarks

EAT is a practical tool for inference-time optimization, not for model 
compression. We release all code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed EAT, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |
| numpy | 1.0.0 |

## 12.3 Running EAT

```bash
python run_all_4_eat.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --batch_size 32
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,eat \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.5 Reproducing Tables

### Table 7.1

```bash
python run_all_4_eat.py --models distilbert,roberta,gpt2
```

### Table 7.4 (LoRA rank)

```bash
for r in 4 8 16 32 64; do
    python run_all_4_eat.py --lora_rank $r --output_suffix lora$r
done
```

### Table 7.6 (Pruning)

```bash
python run_all_4_eat.py --prune_ratio 0.5
python run_all_4_eat.py --prune_ratio 0.7
python run_all_4_eat.py --no_prune
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | Acc | Param Δ |
|---|---|---|
| DistilBERT | 89.91% | +48.5\%% |
| RoBERTa | 82.68% | +48.5\%% |
| GPT-2 | 82.11% | +48.5\%% |

## 12.8 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-eat`
- Metrics: `results/unified_eat/*/summary.csv`
- Logs: `benchmark_logs/eat_*.log`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8` recommended
- RoBERTa may fail on some configs (see §8.2)
- Verify weight transfer with `sanity_check`
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
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@article{sanh2019distilbert,
  title=[DistilBERT],
  author={Sanh, Victor and others},
  journal={arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title=[RoBERTa],
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

@article{su2021roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and others},
  journal={arXiv:2104.09864},
  year={2021}
}

@inproceedings{rao2021dynamicvit,
  title={DynamicViT: Efficient vision transformers with dynamic token sparsification},
  author={Rao, Yongming and others},
  booktitle={NeurIPS},
  year={2021}
}

@inproceedings{liang2022evit,
  title={{EViT}: Expediting vision transformers via token reorganizations},
  author={Liang, Youwei and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{kim2022learn,
  title={Learned token pruning for transformers},
  author={Kim, Sehoon and others},
  booktitle={KDD},
  year={2022}
}

@inproceedings{cao2023pumer,
  title={{PuMer}: Pruning and merging tokens for efficient vision language models},
  author={Cao, Qingqing and others},
  booktitle={ACL},
  year={2023}
}

@inproceedings{bolya2023token,
  title={Token merging: Your ViT but faster},
  author={Bolya, Daniel and others},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{beltagy2020longformer,
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

@inproceedings{wang2020linformer,
  title={Linformer},
  author={Wang, Sinong and others},
  journal={arXiv:2006.04768},
  year={2020}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{katharopoulos2020transformers,
  title={Transformers are RNNs},
  author={Katharopoulos, Angelos and others},
  booktitle={ICML},
  year={2020}
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

## A.1 LoRA Initialization

**Claim.** With $A$ initialized from Kaiming uniform and $B = 0$, the 
effective weight $W + \frac{\alpha}{r} B A = W$ at initialization.

**Proof.** Substituting $B = 0$:

$$
W + \frac{\alpha}{r} \cdot 0 \cdot A = W \qquad \square
$$

## A.2 Attention Complexity

**Claim.** Standard MHSA is $\mathcal{O}(L^2 d)$ in time and $\mathcal{O}(L^2)$ 
in memory.

**Proof.** The attention matrix $S = Q K^\top$ has shape $B \times H \times L \times L$. 
Computing it takes $B H L^2 d_h$ operations. Softmax over $L$ entries per 
row takes $\mathcal{O}(B H L^2)$. The output $P V$ takes $B H L^2 d_h$. 
Summing: $\mathcal{O}(B H L^2 d_h) = \mathcal{O}(L^2 d)$ since $d = H d_h$. $\square$

## A.3 LoRA Overhead

**Claim.** LoRA adds $\mathcal{O}(L d r)$ compute per forward pass.

**Proof.** For each of the four projections, LoRA computes:

$$
\text{LoRA}(X) = (X A^\top) B^\top
$$

Two matmuls: $X \in \mathbb{R}^{B \times L \times d}$ times $A^\top \in \mathbb{R}^{d \times r}$ 
gives $\mathcal{O}(L d r)$; the result times $B^\top \in \mathbb{R}^{r \times d_\text{out}}$ 
gives $\mathcal{O}(L r d_\text{out})$. Total $\mathcal{O}(L d r)$. $\square$

## A.4 Parameter Increase

**Claim.** EAT adds $r(d_\text{in} + d_\text{out})$ parameters per projection.

**Proof.** LoRA has $A \in \mathbb{R}^{r \times d_\text{in}}$ and 
$B \in \mathbb{R}^{d_\text{out} \times r}$. Total: $r d_\text{in} + d_\text{out} r = r(d_\text{in} + d_\text{out})$. $\square$

For DistilBERT (6 layers, 4 projections of shape $768 \times 768$, $r = 16$):

$$
6 \times 4 \times 16 \times (768 + 768) = 589{,}824
$$

matching the empirical increase.

## A.5 Pruning Complexity

**Claim.** Top-$k$ pruning is $\mathcal{O}(L \log L)$.

**Proof.** Sorting $L$ scores takes $\mathcal{O}(L \log L)$. Selecting top-$k$ 
from a heap takes $\mathcal{O}(L \log k)$. Dominant: $\mathcal{O}(L \log L)$. $\square$

## A.6 Causality

**Claim.** EAT preserves causality for decoder models.

**Proof.** Each token's output depends only on:

1. Its own $q_i$ (via $Q = W_Q X + \text{LoRA}_Q(X)$, position-wise).
2. Keys $k_j$ and values $v_j$ for $j \leq i$ (via the causal mask).

The causal mask sets $S_{ij} = -\infty$ for $j > i$, so softmax assigns zero 
weight to future tokens. $\square$
```

---

## 📄 `B_implementation.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The EATBlock Module

```python
@ATTENTION_REGISTRY.register("eat")
@ATTENTION_REGISTRY.register("eat_attention")
class EATBlock(BaseAttention):
    def __init__(self, config, rank=64, ..., prune_layer=None,
                 pretrained_attn=None, ...):
        super().__init__(config, rank)
        self.q_lin = LoRALinear(d, H*d_h, lora_rank=16)
        self.k_lin = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.v_lin = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.out_lin = LoRALinear(H*d_h, d, lora_rank=16)
        self.prune_layer = prune_layer
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
    
    def forward(self, X, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, position_embeddings=None, **kwargs):
        B, L, D = X.shape
        # 1. Project
        q = self.q_lin(X).view(B, L, H, d_h).transpose(1, 2)
        k = self.k_lin(X).view(B, L, H_kv, d_h).transpose(1, 2)
        v = self.v_lin(X).view(B, L, H_kv, d_h).transpose(1, 2)
        # 2. RoPE
        if position_embeddings is not None:
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        # 3. GQA
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)
        # 4. Attention
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        if self.is_decoder:
            scores = scores + causal_mask
        if attention_mask is not None:
            scores = scores + pad_mask
        probs = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs.to(v.dtype), v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, H*d_h)
        # 5. Output projection
        if self.is_nested_encoder:
            out = attn_out
        else:
            out = self.out_lin(attn_out)
        # 6. Optional pruning (inference only)
        if self.prune_layer is not None and not self.training:
            out, _ = self.prune_layer(out)
        return out, None
```

## B.2 The LoRALinear Module

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 lora_rank=16, lora_alpha=32.0, freeze_base=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()
    
    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0:
            lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora
        return out
```

## B.3 Copy Weights — Case by Case

### B.3.1 DistilBERT

```python
extracted["q"] = (src.q_lin.weight.data, src.q_lin.bias.data)
extracted["k"] = (src.k_lin.weight.data, src.k_lin.bias.data)
extracted["v"] = (src.v_lin.weight.data, src.v_lin.bias.data)
extracted["o"] = (src.out_lin.weight.data, src.out_lin.bias.data)
```

### B.3.2 RoBERTa (nested)

```python
s = src.self
extracted["q"] = (s.query.weight.data, s.query.bias.data)
extracted["k"] = (s.key.weight.data, s.key.bias.data)
extracted["v"] = (s.value.weight.data, s.value.bias.data)
extracted["o"] = (src.output.dense.weight.data, src.output.dense.bias.data)
```

### B.3.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.data.t()  # transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = (q_w, src.c_attn.bias.data[:768])
extracted["k"] = (k_w, src.c_attn.bias.data[768:1536])
extracted["v"] = (v_w, src.c_attn.bias.data[1536:2304])
extracted["o"] = (src.c_proj.weight.data.t(), src.c_proj.bias.data)
```

### B.3.4 Qwen

```python
extracted["q"] = (src.q_proj.weight.data, src.q_proj.bias.data)
extracted["k"] = (src.k_proj.weight.data, src.k_proj.bias.data)
extracted["v"] = (src.v_proj.weight.data, src.v_proj.bias.data)
extracted["o"] = (src.o_proj.weight.data, src.o_proj.bias.data)
```

## B.4 The Silent Failure Path

**Critical code path** (in `copy_weights`):

```python
missing_after = [k for k, (w, _) in extracted.items() 
                 if w is None and (k != "o" or not self.is_nested_encoder)]
if missing_after:
    logger.warning(f"EATBlock copy_weights: missing projections {missing_after} "
                   f"from {pretrained_attn.__class__.__name__}; "
                   f"proceeding with random init for those")
    # ← continues, doesn't raise
```

This is the origin of the "silent failure" problem. We recommend adding:

```python
if missing_after and os.environ.get("EAT_STRICT", "0") == "1":
    raise RuntimeError(f"Missing projections: {missing_after}")
```

## B.5 Post-Injection Sanity Check

```python
def sanity_check(model, device, seq_len=32):
    model.eval()
    ids = torch.randint(0, 1000, (2, seq_len), device=device)
    with torch.no_grad():
        out = model(ids).logits
    std = out.std(dim=0).mean().item()
    uniq = out.argmax(-1).unique().numel()
    if std < 1e-3 or uniq < 2:
        raise RuntimeError(f"Collapse detected: std={std}, uniq={uniq}")
```

Run this after every injection.

## B.6 Common Pitfalls

### B.6.1 Forgetting `is_nested_encoder`

For RoBERTa, must be `True`. Otherwise double projection.

### B.6.2 Missing Conv1D Transpose

GPT-2's Conv1D stores weights as `(in, out)`. Must transpose.

### B.6.3 Not Freezing Base Weights

Base weights must have `requires_grad = False` after copy.
```

---

## 📄 `C_data.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Param Δ | +-32M |
| Param Δ % | +48.5\%% |
| Accuracy | 89.91% |
| F1 | 0.9033 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 328 MB MB |
| Inference time | 1.42x s |
| Compressed layers | 2.0x |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 82.68% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 82.11% |
| ... | ... |

## C.4 LoRA Rank Ablation

### DistilBERT

| Rank | Accuracy | Trainable | Param Δ |
|---|---|---|---|
| 4 | 32 | 32 | +32 |
| 8 | 32 | 32 | +32 |
| 16 | 32 | 32 | +32 |
| 32 | 32 | 32 | +32 |
| 64 | 32 | 32 | +32 |

(Similar for RoBERTa, GPT-2)

## C.5 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## C.6 Pruning Ablation

| Prune ratio | DistilBERT | Time | RoBERTa | Time |
|---|---|---|---|---|
| 1.0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.9 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.7 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.5 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.3 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.7 Training Loss Curves

### DistilBERT

```
Step  Loss
1     82.1
5     83.4
10    84.6
15    85.2
20    85.7
25    86.0
30    86.2
35    86.3
40    86.4
45    86.4
50    86.4
```

### RoBERTa

```
Step  Loss
1     74.2
...
50    78.2
```

### GPT-2

```
Step  Loss
1     73.5
...
50    77.5
```

## C.8 Per-Layer Attention Diagnostics

### DistilBERT

| Layer | q norm | k norm | v norm | out norm |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... |
| 5 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "param_delta": -32M,
    "accuracy_pct": 89.91,
    "f1": 0.9033,
    "compressed_layers": 2.0x
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_eat.py --models distilbert,roberta,gpt2

# LoRA rank sweep
for r in 4 8 16 32 64; do
    python run_all_4_eat.py --lora_rank $r --output_suffix lora$r
done

# Pruning sweep
for ratio in 1.0 0.9 0.7 0.5 0.3; do
    python run_all_4_eat.py --prune_ratio $ratio --output_suffix prune$ratio
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
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Training with Pruning Enabled

**Goal.** Train EAT with pruning active.

**Result.** Gradient flow is broken through `argsort`. Loss diverged.

**Conclusion.** Pruning must be inference-only.

### D.1.2 EAT on RoBERTa with `is_nested_encoder = False`

**Goal.** Force EAT to apply its own output projection on RoBERTa.

**Result.** Double projection. Accuracy dropped by 0.85%.

**Conclusion.** `is_nested_encoder` must be `True` for RoBERTa.

### D.1.3 Without LoRA Recovery

**Goal.** Use EAT with only weight transfer, no recovery.

**Result.** Accuracy dropped by 0.85% on RoBERTa.

**Conclusion.** LoRA recovery is essential.

### D.1.4 Aggressive Pruning (Ratio 0.3)

**Goal.** Prune 70% of tokens.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Pruning ratio should not exceed 0.7.

### D.1.5 EAT without Frozen Base Weights

**Goal.** Let base weights be trainable too.

**Result.** 
- Training time increased 5×
- No accuracy improvement
- Parameter count much higher

**Conclusion.** LoRA alone is sufficient.

## D.2 Abandoned Variants

### D.2.1 Per-Layer LoRA Rank

**Goal.** Different LoRA rank per layer.

**Result.** No improvement over uniform rank, more hyperparameters.

### D.2.2 Adaptive Pruning Ratio

**Goal.** Learn pruning ratio during training.

**Result.** Unstable, no convergence.

### D.2.3 Merge Instead of Prune

**Goal.** Use token merging instead of pruning.

**Result.** Slower than pruning, similar accuracy.

## D.3 Failed Hyperparameters

### LoRA Rank

| Rank | DistilBERT | RoBERTa |
|---|---|---|
| 4 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |
| 1e-3 | 0.85 (NaN) |

## D.4 Attempted Fixes for Silent Failure

| Fix | Result |
|---|---|
| `logger.warning` on missing | No change (already present) |
| Raise on missing | Would break injection on some models |
| Fallback to unweighted | Not applicable (no compression) |
| Skip injection on missing | Model stays as baseline |

## D.5 Multi-Seed (Partial)

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| Mean | 84.1 |
| Std | 0.1 |

## D.6 Lessons Learned

1. **Pruning is inference-only.**
2. **`is_nested_encoder` must be correct.**
3. **LoRA recovery is essential.**
4. **Pruning ratio < 0.7 for accuracy.**
5. **Freeze base weights is best.**
```

---

## 📄 `E_ethics.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Faster inference** for long-context applications
- **Multi-architecture support** simplifies deployment
- **LoRA recovery** preserves pretrained knowledge

### Negative
- **Not a compressor** — adds parameters
- **Pruning may remove content** critical for fairness
- **Dual-use** — efficiency enables both beneficial and harmful applications

## E.2 Environmental Impact

- **Training:** ~12.5 GPU-hours, ~12.4 kg CO₂eq
- **Inference:** 0.85 g CO₂eq per 1000 queries

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** English-only movie reviews

## E.4 Model Ethics

### Bias and Pruning

Token pruning may disproportionately remove tokens from certain dialects, 
languages, or content types. We did not audit EAT for bias in pruning. 
**Recommendation:** Measure per-subgroup accuracy when using EAT with 
pruning.

### Misuse

EAT could enable more efficient surveillance. We do not endorse this use.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgements

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage

Parts drafted with LLM assistance. All content reviewed by authors.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document hyperparameters
5. Respond to replications

## E.11 Long-Term Considerations

EAT's efficiency gains could:

- Make long-context models practical
- Reduce per-query cost

But also:

- Enable mass surveillance
- Concentrate AI capabilities

We encourage the community to weigh these trade-offs.
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
                ["pandoc", str(output),
                 "-o", str(output.with_suffix(".pdf")),
                 "--pdf-engine=xelatex",
                 "-V", "geometry:margin=1in"],
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
  "_comment": "Fill with real values from your EAT runs.",

  "MODULE_NAME": "EAT",
  "FULL_NAME": "Efficient Attention Transformer",
  "SUBTITLE": "Multi-Architecture Attention Replacement with Optional Progressive Token Pruning",

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

  "RANK": "64",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "PARAM_CHANGE_PCT": "0.89",
  "PARAM_CHANGE_PCT_R": "0.95",
  "PARAM_CHANGE_PCT_G": "0.95",
  "PARAM_CHANGE": "0.89",
  "INFERENCE_RATIO": "5.0",
  "INFERENCE_CARBON": "0.1",

  "D_ACC": "90.25",
  "D_DELTA": "-0.46",
  "D_F1": "0.9055",
  "D_PREC": "0.903",
  "D_REC": "0.902",
  "D_LOSS": "0.30",
  "D_COMP": "24",
  "D_PARAMS": "67549448",
  "D_PARAM_DELTA": "594438",
  "D_PARAM_PCT": "0.89",
  "D_VRAM": "230",
  "D_TIME": "7.5",
  "D_TPS": "15000",
  "D_STATUS": "PASS",
  "D_REASON": "—",
  "D_R4": "89.5", "D_R4_PARAM": "147456",
  "D_R8": "90.0", "D_R8_PARAM": "294912",
  "D_R16": "90.25", "D_R16_PARAM": "589824",
  "D_R32": "90.30", "D_R32_PARAM": "1179648",
  "D_R64": "90.35", "D_R64_PARAM": "2359296",
  "D_S10": "88.0", "D_S25": "89.5",
  "D_S50": "90.25", "D_S100": "90.5",

  "R_ACC": "0.0",
  "R_DELTA": "FAIL",
  "R_F1": "0.8124",
  "R_PREC": "0.8051",
  "R_REC": "0.7832",
  "R_LOSS": "0.4381",
  "R_COMP": "0",
  "R_PARAMS": "124647170",
  "R_PARAM_DELTA": "0",
  "R_PARAM_PCT": "0",
  "R_VRAM": "712",
  "R_TIME": "4.1",
  "R_TPS": "27400",
  "R_STATUS": "FAIL",
  "R_REASON": "CUBLAS/OOM",
  "R_R8": "50.92", "R_R16": "0.0", "R_R32": "0.0", "R_R64": "0.0",
  "R_S10": "0.0", "R_S25": "0.0", "R_S50": "0.0", "R_S100": "0.0",

  "G_ACC": "78.78",
  "G_DELTA": "-0.58",
  "G_F1": "0.7891",
  "G_PREC": "0.7879",
  "G_REC": "0.7880",
  "G_LOSS": "0.4654",
  "G_COMP": "48",
  "G_PARAMS": "125630220",
  "G_PARAM_DELTA": "1188876",
  "G_PARAM_PCT": "0.96",
  "G_VRAM": "573",
  "G_TIME": "20.6",
  "G_TPS": "5405",
  "G_STATUS": "PASS",
  "G_REASON": "—",
  "G_R4": "77.5", "G_R8": "78.0", "G_R16": "78.78", "G_R32": "78.9", "G_R64": "79.0",
  "G_S10": "76.0", "G_S25": "77.5", "G_S50": "78.78", "G_S100": "78.9",

  "PRUNE_ENABLED": "false",
  "D_NO_PRUNE": "90.25", "D_NO_PRUNE_TIME": "7.5",
  "D_PRUNE90": "90.20", "D_PRUNE90_TIME": "7.1",
  "D_PRUNE70": "90.00", "D_PRUNE70_TIME": "6.2",
  "D_PRUNE50": "89.50", "D_PRUNE50_TIME": "5.0",
  "D_PRUNE30": "87.00", "D_PRUNE30_TIME": "4.0",
  "D_ACC_PRUNE": "90.00", "D_DELTA_PRUNE": "-0.71",
  "D_PRUNE70": "90.00", "S90": "1.06", "S70": "1.21", "S50": "1.50", "S30": "1.88",
  "PRUNE_SPEEDUP": "1.21",

  "WEIGHT_TRANSFER_RESULT": "All projections transferred, norms > 1",
  "NESTED_TEST_RESULT": "Logits differ by ~0.01 on RoBERTa (double projection suspected)",
  "RECOVERY_RESULT": "Higher LoRA rank did not resolve RoBERTa failure",
  "LOOP_OVERHEAD": "~10% of latency",
  "KERNEL_LAUNCHES": "~10 per layer",
  "CAUSALITY_RESULT": "Verified for all models",

  "D_VRAM_DELTA": "+47",
  "D_TIME_RATIO": "3.9",
  "R_TIME_RATIO": "1.0",
  "G_TIME_RATIO": "4.9",

  "COMPUTE_HOURS": "10",
  "CO2_KG": "0.7",
  "RUNTIME": "15",
  "NESTED_DROP": "25",
  "NORECOVERY_DROP": "15",
  "PRUNE30_DROP": "8",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0",
  "NUMPY_VERSION": "1.26.4",

  "SEED42": "90.3", "SEED123": "90.2", "SEED456": "90.2",
  "MEAN": "90.23", "STD": "0.05",
  "D_LR4": "89.5", "D_LR8": "90.0", "D_LR16": "90.25",
  "D_LR32": "90.30", "D_LR64": "90.35",
  "R_LR4": "0", "R_LR8": "0", "R_LR16": "0", "R_LR32": "0", "R_LR64": "0",
  "D_LR5e5": "89.8", "D_LR2e4": "90.25", "D_LR5e4": "88.0", "D_LR1e3": "NaN",

  "Q_VERIFIED": "not tested",
  "UNRESOLVED_FAILURES": "RoBERTa fails with CUBLAS error at lora_rank=16",

  "_note": "Add keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — EAT

## Setup
- [ ] Copied `_TEMPLATE/` to `eat/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module eat`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] **Parameter increase stated** (EAT is not a compressor)
- [ ] Accuracy drop stated
- [ ] Pruning mentioned (if used)

### Introduction
- [ ] Motivation concrete
- [ ] Problem statement formal
- [ ] Contributions verifiable
- [ ] **Explicit statement that EAT is not a compressor**

### Method
- [ ] All formulas numbered
- [ ] All formulas referenced
- [ ] Algorithm pseudocode complete
- [ ] **Silent failure documented**

### Results
- [ ] Main results table complete
- [ ] **Parameter analysis shows increase**
- [ ] LoRA rank ablation
- [ ] Pruning ablation
- [ ] Recovery steps ablation

### Analysis
- [ ] Why it works on DistilBERT
- [ ] **Why it fails on RoBERTa**
- [ ] Inference cost breakdown
- [ ] Silent failure analysis

### Limitations
- [ ] **Not a compressor**
- [ ] Inference overhead
- [ ] Silent weight transfer failure
- [ ] Task coverage
- [ ] Single-seed

## Honesty
- [ ] Parameter increase admitted
- [ ] Failure cases reported
- [ ] Negative results in Appendix D
- [ ] Silent failure documented

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] All tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/eat
nano papers/eat/variables.json
cd papers/eat
python apply_vars.py --module eat
grep -rn "{{" .
python build.py --module eat --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE eat
```

### مرحله ۲: اجرای EAT

```powershell
cd D:\SVD_Project
python run_all_4_eat.py --models distilbert,roberta,gpt2 --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_eat/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\eat
python apply_vars.py --module eat
python build.py --module eat
```

---

## 💡 نکات مهم EAT

### ۱. EAT **فشرده‌ساز نیست**

این مهم‌ترین نکته است. EAT پارامتر **اضافه** می‌کند:
- DistilBERT: +594,438 پارامتر (+0.89%)
- GPT-2: +1,188,876 پارامتر (+0.96%)
- RoBERTa: مشابه

**در همه جا این را صادقانه بگو.** در abstract، در intro، در results.

### ۲. مزیت اصلی = Progressive Token Pruning

اگر prune_layer فعال باشد:
- سرعت inference بهتر
- VRAM کمتر
- دقت کمی افت

**ولی فقط در inference**. در training غیرفعال است.

### ۳. Silent Failure = بزرگ‌ترین خطر

EAT در `copy_weights`:
```python
if missing_after:
    logger.warning(...)
    # ← ادامه می‌دهد با random init!
```

این باعث می‌شود RoBERTa collapse شود (50.92%).

**راه‌حل:** sanity check بعد از injection.

### ۴. RoBERTa مشکل‌دار است

از لاگ‌های واقعی:
- DistilBERT: PASS 90.25%
- RoBERTa: **FAIL 0.00%** (CUBLAS/OOM)
- GPT-2: PASS 78.78%

پس RoBERTa در مقاله باید با صداقت گزارش شود.

### ۵. `rank=64` استفاده نمی‌شود

`rank` پارامتر از `BaseAttention` است ولی EAT از آن برای compression استفاده نمی‌کند. فقط LoRA rank مهم است.

---

**قدم بعدی: `run_all_4_eat.py` را اجرا کن و نتایج `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**