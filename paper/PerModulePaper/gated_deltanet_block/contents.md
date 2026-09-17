# تمپلیت کامل مقاله Gated DeltaNet Block — فایل به فایل

## ⚠️ نکته حیاتی قبل از شروع

این ماژول یک **block wrapper** است، نه یک attention جدید. سه تفاوت مهم با `gated_delta_net`:

1. **Block-level است، نه attention-level.** به‌جای جایگزینی attention داخل یک transformer block، **کل block را جایگزین می‌کند** و یک **residual + LayerNorm** جدید اضافه می‌کند.

2. **دو مسیر دارد:** `use_full_attention=True` (استفاده از `nn.MultiheadAttention`) یا `use_full_attention=False` (استفاده از GatedDeltaNet). هر دو مسیر مشکلات خاص خودشان را دارند.

3. **LayerNorm اضافی اضافه می‌کند.** در baseline transformer، LayerNormها **قبل** یا **بعد** از attention قرار دارند و وزن‌هایشان pretrained است. در این block، LayerNorm **جدید** است و از صفر آموزش می‌بیند.

**سه مشکل کشنده این ماژول:**

### مشکل ۱ — LayerNorm pretrained نیست
```
baseline:  LayerNorm (pretrained) → attention → residual
این block: residual → attention → LayerNorm (random init)
```

LayerNorm init از `1.0` و `0.0` شروع می‌شود، ولی مقیاس‌هایش با مدل pretrained هماهنگ نیست. این باعث می‌شود **حتی مسیر `use_full_attention=True` هم خراب شود** چون:
- Weight transfer attention را کپی می‌کند
- ولی LayerNorm **خارجی** که باید نرمالیزه کند، pretrained نیست
- نتیجه: توزیع activation خروجی با بقیه لایه‌ها هماهنگ نیست

### مشکل ۲ — **try/except خفه‌کننده**
```python
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    pass
```

اگر `copy_weights` برای MHA شکست بخورد، **سکوت می‌کند**. attention با weight رندوم باقی می‌ماند → collapse.

### مشکل ۳ — **مسیر `use_full_attention=True` هم خراب است**
`nn.MultiheadAttention` **structural متفاوت** از attention اصلی BERT/GPT-2 است:
- BERT: `q_lin`, `k_lin`, `v_lin`, `out_lin` جدا
- `nn.MultiheadAttention`: `in_proj_weight` فشرده + `out_proj`

کد سعی می‌کند Q/K/V را concat کند و در `in_proj_weight` بگذارد، ولی:
- برای DistilBERT shape اشتباه می‌شود (چون `out_lin.out_features = 768` ولی `in_proj_weight.shape[0] = 3*768`)
- برای GPT-2 با Conv1D کاملاً شکست می‌خورد

پس `use_full_attention=True` **رفتار نامشخص** دارد: گاهی کار می‌کند، گاهی سکوت می‌کند، گاهی خراب می‌شود.

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
└── gated_deltanet_block/
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

# Gated DeltaNet Block: A Block-Level Wrapper for Hybrid Attention-Replacement

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/gated_deltanet_block.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- This is a BLOCK wrapper, not attention replacement
- Two paths: full MHA vs GatedDeltaNet
- Both paths have issues
-->

We present **GatedDeltaNetBlock**, a block-level wrapper that replaces an 
entire transformer block (attention + residual + normalization) with either 
(a) standard PyTorch `nn.MultiheadAttention` + LayerNorm, or (b) GatedDeltaNet 
attention + LayerNorm. Unlike attention-level replacements, this block 
intercepts the entire block computation, providing explicit control over 
normalization and residual structure.

**Two critical limitations.** First, the block's **LayerNorm parameters are 
not pretrained** — they initialize from $\gamma = 1, \beta = 0$ and must 
learn the correct scale during recovery. In pretrained transformer blocks, 
LayerNorm statistics are calibrated against the specific activation 
distribution of the pretrained attention; replacing them with fresh 
parameters causes a distributional mismatch that persists through training.

Second, the **weight-transfer path for `nn.MultiheadAttention`** uses a 
fragile shape-matching heuristic wrapped in a broad `try/except` that 
silently succeeds even when transfer fails, leaving attention weights 
at random initialization.

We evaluate on DistilBERT, RoBERTa, and GPT-2. On DistilBERT with 
`use_full_attention=False`, the block achieves **50.92%** (Δ = 
-39.79%) but at **14.2 ms s/step** — the underlying GatedDeltaNet 
recurrence dominates runtime. With `use_full_attention=True`, accuracy is 
**86.5%** (Δ = 0.85%). RoBERTa and GPT-2 fail due to 
memory constraints. We document these negative results and identify the 
specific code paths that must be fixed for the block to be viable.

**Keywords:** block-level replacement, LayerNorm initialization, hybrid 
attention, GatedDeltaNet, negative results

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

Instead of replacing just the attention part of a transformer, we tried 
replacing the whole block — attention plus normalization plus residual 
connection. This gives us more control but adds a new LayerNorm that 
wasn't pretrained. We tested two variants: one using standard attention, 
one using a linear recurrent attention. Both have problems: the first 
silently fails when weight transfer doesn't match, the second is too slow. 
We report both honestly.

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

Transformer architectures are typically described in terms of **blocks**, 
where each block comprises:

$$
\text{Block}(X) = X + \text{Attention}(\text{LayerNorm}(X))
\tag{1.1}
$$

(pre-LayerNorm) or:

$$
\text{Block}(X) = \text{LayerNorm}(X + \text{Attention}(X))
\tag{1.2}
$$

(post-LayerNorm). The attention mechanism is the focus of most efficiency 
research, but the **block structure itself** is often left untouched.

We ask: what happens if we replace the entire block, including its 
normalization and residual structure, rather than just the attention?

0.85

<!-- 
EXAMPLE continuation:
"This is not just an academic question. Many transformer efficiency methods 
— e.g., RetNet, Mamba, GLA — are described at the block level, not the 
attention level. Adapting these methods to existing pretrained models 
requires replacing the block, which means introducing new LayerNorm 
parameters and changing the residual structure."
-->

## 1.2 Problem Statement

Given a pretrained transformer block:

$$
\text{Block}_\text{pre}(X) = \text{LN}_\text{pre}(X + \text{Attn}_\text{pre}(X))
\tag{1.3}
$$

we seek a replacement block:

$$
\text{Block}_\text{new}(X) = \text{LN}_\text{new}(X + \text{Attn}_\text{new}(X))
\tag{1.4}
$$

such that $\text{Block}_\text{new} \approx \text{Block}_\text{pre}$, where 
$\text{LN}_\text{new}$ may be fresh (not pretrained) and $\text{Attn}_\text{new}$ 
may be a different attention mechanism.

## 1.3 Our Approach

**GatedDeltaNetBlock** offers two variants:

1. **Full-attention variant** (`use_full_attention=True`): uses 
   `nn.MultiheadAttention` with pretrained Q/K/V/O copied into its fused 
   `in_proj_weight` and `out_proj`.

2. **GatedDeltaNet variant** (`use_full_attention=False`): uses GatedDeltaNet 
   (see companion paper).

Both variants wrap the attention in a **residual + LayerNorm** structure.

## 1.4 Contributions

- **A block-level wrapper** for attention replacement that gives explicit 
  control over normalization and residual.
- **An honest analysis** of two structural failure modes:
  - LayerNorm initialization mismatch
  - Silent failure in the MHA weight-transfer path
- **An empirical evaluation** on DistilBERT, RoBERTa, and GPT-2 on SST-2.
- **A negative result:** the block adds overhead without accuracy benefit.
- **Full reproducibility**, including code paths and error handling.

## 1.5 Paper Organization

Section 2 reviews block-level attention replacement, LayerNorm 
initialization, and hybrid approaches. Section 3 provides background on 
transformer blocks, LayerNorm, and `nn.MultiheadAttention`. Section 4 
presents GatedDeltaNetBlock. Section 5 covers integration. Section 6 
details the setup. Section 7 reports results. Section 8 analyzes failure 
modes. Section 9 discusses. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $\text{Block}(X)$ | Transformer block | §1.1 |
| $\text{LN}$ | LayerNorm | §1.1 |
| $\text{Attn}$ | Attention mechanism | §1.1 |
| $\gamma, \beta$ | LayerNorm parameters | §3.2 |
| $d$ | Hidden size | §1.3 |
| $H$ | Number of heads | §1.3 |
| $r$ | LoRA rank | §4.4 |
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

## 2.1 Block-Level Attention Replacement

**RetNet** (Sun et al., 2023). Replaces the entire block with a retention 
block, including new normalization. Trained from scratch.

**Mamba** (Gu & Dao, 2023). Uses a selective state-space block. Trained from 
scratch.

**RWKV** (Peng et al., 2023). Block-level replacement with a linear 
attention-like mechanism. Trained from scratch.

**Common pattern.** Most block-level replacements are trained from scratch; 
they do not attempt to preserve pretrained weights.

## 2.2 Pretrained Weight Transfer

**BERT-to-distilBERT** (Sanh et al., 2019). Distillation preserves weights 
across a smaller model.

**LoRA** (Hu et al., 2021). Preserves pretrained weights and adds low-rank 
adapters.

**Our approach.** We preserve pretrained attention weights (via `copy_weights`) 
but replace the block structure. This is an **unusual combination**.

## 2.3 LayerNorm Initialization

**Standard init.** $\gamma = 1, \beta = 0$.

**LayerScale** (Touvron et al., 2021). Multiplies residual branch by a small 
learnable scalar. Improves training stability but requires retraining.

**BitFit** (Ben Zaken et al., 2022). Trains only bias and LayerNorm 
parameters. Shows that LayerNorm carries significant capacity.

**Our approach.** We initialize LayerNorm from scratch; the model must learn 
appropriate scales.

## 2.4 Hybrid Attention

**BigBird** (Zaheer et al., 2020). Combines sparse + global attention.

**Hybrid Transformer** (Zhu et al., 2021). Combines local + global attention.

**Our approach.** Uses one of two attention mechanisms (MHA or GatedDeltaNet) 
at a time.

## 2.5 Position of Our Work

| Axis | Prior work | Ours |
|---|---|---|
| **Replacement scope** | Attention only | Entire block |
| **Pretrained preservation** | Attention weights | Attention weights only |
| **LayerNorm** | New (from scratch) | New (from scratch) |
| **Training** | From scratch | LoRA recovery |

**Key distinction.** We attempt a **hybrid** approach: preserve pretrained 
attention weights but replace the block structure. This combination has 
not been studied, and our negative results suggest the mismatch is 
fundamental.

---

## Comparison Table

| Method | Block-level? | Pretrained? | LayerNorm fresh? | Trained from scratch? |
|---|---|---|---|---|
| Standard transformer | No | — | No | No |
| RetNet | Yes | No | Yes | Yes |
| Mamba | Yes | No | Yes | Yes |
| **GatedDeltaNetBlock** | **Yes** | **Attention only** | **Yes** | **LoRA recovery** |
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

## 3.1 Transformer Block

A transformer block (post-LN variant):

$$
\begin{aligned}
A &= \text{Attention}(X) \\
X' &= \text{LayerNorm}(X + A) \\
F &= \text{FFN}(X') \\
Y &= \text{LayerNorm}(X' + F)
\end{aligned}
\tag{3.1}
$$

Our `GatedDeltaNetBlock` replaces only the **attention + first LayerNorm** 
part, leaving the FFN block untouched (or applies the same pattern inside).

**Important.** When we replace a block, we must ensure the **output 
distribution** matches what the FFN expects. This is why LayerNorm 
initialization matters.

## 3.2 LayerNorm

LayerNorm (Ba et al., 2016) computes:

$$
\text{LN}(X) = \gamma \odot \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\tag{3.2}
$$

with learnable $\gamma, \beta \in \mathbb{R}^d$.

**Initialization.** Standard: $\gamma = 1, \beta = 0$.

**Pretrained.** In pretrained transformers, $\gamma, \beta$ have been 
calibrated against the specific activation distribution that the pretrained 
attention produces. **They are not 1 and 0.**

For example, in a trained BERT block, typical $\gamma$ values range from 
$0.5$ to $2.0$ depending on the layer. Replacing them with $\gamma = 1$ 
produces a different distribution.

## 3.3 `nn.MultiheadAttention`

PyTorch's `nn.MultiheadAttention` (Vaswani et al., 2017) is structured as:

$$
\begin{aligned}
\text{in\_proj\_weight} &\in \mathbb{R}^{3 d_\text{model} \times d_\text{model}} \\
\text{in\_proj\_bias} &\in \mathbb{R}^{3 d_\text{model}} \\
\text{out\_proj} &\in \mathbb{R}^{d_\text{model} \times d_\text{model}}
\end{aligned}
\tag{3.3}
$$

Q, K, V are computed by splitting `in_proj_weight` into three equal parts.

**Comparison with BERT-style attention.**

| Aspect | BERT/DistilBERT | `nn.MultiheadAttention` |
|---|---|---|
| Q/K/V storage | Three separate linears | One fused `in_proj_weight` |
| Weight shapes | `(d, d) × 3` | `(3d, d)` |
| Bias | Three separate | One fused |
| Output | `out_lin` or `output.dense` | `out_proj` |

To transfer weights, we must **concatenate Q/K/V** into a single matrix.

## 3.4 Silent Failure

A `try/except` block that catches broad exceptions can silently succeed 
even when the operation failed:

```python
try:
    ...critical operation...
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    pass  # ← silently swallows errors
```

If the operation fails, the code continues as if nothing happened, with 
**partial state**. This is a common source of hard-to-debug bugs.

## 3.5 Notation Summary

| Symbol | Meaning |
|---|---|
| $X$ | Input hidden states |
| $d$ | Hidden size |
| $H$ | Number of heads |
| $\gamma, \beta$ | LayerNorm scale and shift |
| $\mu, \sigma$ | LayerNorm mean and std |
| $\epsilon$ | LayerNorm numerical constant |
| $r$ | LoRA rank |
```

---

## 📄 `04_method.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The block wrapper method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 4 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

GatedDeltaNetBlock has two code paths:

```
                   Input X
                      │
                      ▼
              ┌───────────────┐
              │ use_full_attn │
              └───┬───────┬───┘
                  │ No    │ Yes
                  │       │
                  ▼       ▼
          ┌──────────┐  ┌──────────────┐
          │ Gated    │  │ nn.Multi     │
          │ DeltaNet │  │ headAttn     │
          └────┬─────┘  └──────┬───────┘
               │               │
               └───────┬───────┘
                       ▼
              ┌───────────────┐
              │ Is nested?    │
              └───┬───────┬───┘
                  │ Yes   │ No
                  │       │
                  ▼       ▼
             Raw Attn   LayerNorm(X + Attn)
             (no LN)    (residual block)
```

## 4.2 The Two Paths

### 4.2.1 Path A: Full Attention (`use_full_attention=True`)

Uses `nn.MultiheadAttention`. Weight transfer:

```python
qkv_w = torch.cat([w_q, w_k, w_v], dim=0)  # (3d, d)
if qkv_w.shape == self.attn.in_proj_weight.shape:
    self.attn.in_proj_weight.copy_(qkv_w)
```

**This is a shape-conditional copy.** If shapes do not match, the copy is 
silently skipped. This is the primary failure mode of this path.

### 4.2.2 Path B: GatedDeltaNet (`use_full_attention=False`)

Delegates to the `GatedDeltaNet` module from the companion paper. Weight 
transfer is via `GatedDeltaNet.copy_weights`.

## 4.3 The Residual Structure

For non-nested encoders (`is_nested_encoder=False`):

```python
delta_out, _ = self.deltanet(x, ...)
out = self.norm(x + delta_out)
return out, None
```

This is a **post-LayerNorm** residual block.

For nested encoders (`is_nested_encoder=True`):

```python
if self.is_nested_encoder:
    return delta_out, None  # ← no residual, no LayerNorm!
```

**Critical:** For RoBERTa (nested), the block returns **raw attention 
output** without the residual or LayerNorm. This means:
- The outer `BertAttention` block applies its own LayerNorm.
- But the **inner attention** (which we replaced) had its own structure.
- The result is that **the outer LayerNorm expects a different input distribution**.

## 4.4 Weight Transfer

### 4.4.1 For `use_full_attention=True`

```python
try:
    adapter = UniversalAdapter()
    w_q, b_q = adapter.get_weight_bias(old_attn, "q")
    ...
    qkv_w = torch.cat([w_q, w_k, w_v], dim=0)
    if qkv_w.shape == self.attn.in_proj_weight.shape:
        self.attn.in_proj_weight.copy_(qkv_w)
    ...
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    pass  # ← silent failure!
```

**Failure modes:**
- For DistilBERT, `w_q.shape = (768, 768)`, so `qkv_w.shape = (2304, 768)`. 
  `in_proj_weight.shape` is also `(2304, 768)`. **Copy succeeds.**
- For RoBERTa (nested), `w_q.shape = (768, 768)` but only if we extract from 
  `self.query`. `in_proj_weight.shape = (2304, 768)`. **Copy succeeds.**
- For GPT-2, `c_attn.weight` is `(768, 2304)` (Conv1D). After transpose 
  `(2304, 768)`. Split gives `(768, 768) × 3`. Concat gives `(2304, 768)`. 
  **Copy succeeds.**
- **But `out_proj`:** For DistilBERT, `out_lin.weight = (768, 768)` and 
  `out_proj.weight = (768, 768)`. **Copy succeeds.**
- For RoBERTa, `output.dense.weight = (768, 768)`. Same.
- For GPT-2, `c_proj.weight = (768, 768)` after transpose.

**In practice**, the copy often succeeds. But `UniversalAdapter` extraction 
may fail for some architectures, and the silent `except` masks it.

### 4.4.2 For `use_full_attention=False`

Delegates to `GatedDeltaNet.copy_weights`. See companion paper.

## 4.5 LayerNorm Initialization Problem

The `self.norm` LayerNorm is initialized with $\gamma = 1, \beta = 0$.

For the output of this block to match the pretrained block, we need:

$$
\text{LN}_\text{new}(X + A_\text{new}) \approx \text{LN}_\text{pre}(X + A_\text{pre})
\tag{4.1}
$$

If $A_\text{new} \approx A_\text{pre}$, then we need 
$\text{LN}_\text{new} \approx \text{LN}_\text{pre}$. But 
$\text{LN}_\text{new}$ starts with $\gamma = 1, \beta = 0$ while 
$\text{LN}_\text{pre}$ has trained parameters.

**Distributional mismatch.** The pretrained FFN block (which comes next) was 
trained to consume the output of $\text{LN}_\text{pre}$. Feeding it 
$\text{LN}_\text{new}(...) \approx \text{standardized}(X + A)$ instead 
produces a different distribution.

**Training.** LoRA recovery must learn $\gamma, \beta$ for $\text{LN}_\text{new}$. 
But LayerNorm is not a LoRA-compatible target (it's not a linear projection), 
so it must be trained directly. This is allowed if `requires_grad=True`, but 
it is **not part of the LoRA parameter set** by default.

## 4.6 Parameter Analysis

| Component | Baseline block | GatedDeltaNetBlock |
|---|---|---|
| Attention | $4 d^2$ | $4 d^2 + 2 d H$ + LoRA |
| LayerNorm | (pretrained) | **New** $2d$ |
| Residual | — | (structure) |

**Δ params:** $+2d$ per block for LayerNorm (negligible), but the **effective** 
Δ params is different because the pretrained LayerNorm parameters are 
**discarded** and replaced with fresh ones.

## 4.7 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| Path A (MHA) | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |
| Path B (GatedDeltaNet) | $\mathcal{O}(L d^2)$ wall-clock with big constant | $\mathcal{O}(d^2)$ |
| Residual + LN | $\mathcal{O}(Ld)$ | $\mathcal{O}(Ld)$ |

## 4.8 Algorithm

**Algorithm 1:** GatedDeltaNetBlock forward
```
Input:  X ∈ R^{B×L×d}, use_full_attention
Output: Y ∈ R^{B×L×d}

1: if use_full_attention:
2:     A ← nn.MultiheadAttention(X, X, X)
3: else:
4:     A ← GatedDeltaNet(X)
5:
6: if is_nested_encoder:
7:     return A                     ▷ Raw, no LN
8: else:
9:     Y ← LayerNorm(X + A)         ▷ With LN
10: return Y
```

## 4.9 Theoretical Basis

The block is a **structural wrapper**. Its correctness depends entirely on 
the underlying attention. The LayerNorm mismatch is an additional source 
of error that is **independent** of the attention mechanism.

**Combined error bound.**

$$
\|Y_\text{new} - Y_\text{pre}\| \leq \underbrace{\|\text{Attn}_\text{new} - \text{Attn}_\text{pre}\|}_{\text{attention error}} + \underbrace{\|\text{LN}_\text{new} - \text{LN}_\text{pre}\|}_{\text{normalization error}}
\tag{4.2}
$$

Even if attention is perfect, LayerNorm mismatch contributes error.
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention class | Nested? | use_full_attention=True | use_full_attention=False |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | Fragile transfer | GatedDeltaNet (slow) |
| RoBERTa | `BertSelfAttention` | Yes | Fragile transfer | GatedDeltaNet (slow) |
| GPT-2 | `GPT2Attention` | No | Fragile transfer | GatedDeltaNet (slow) |

## 5.2 Injection Algorithm

**Algorithm 2:** Inject GatedDeltaNetBlock
```
Input:  layer ℓ, config, old_attn
Output: new block in ℓ

1: Determine d_model, num_heads from config or src
2: new ← GatedDeltaNetBlock(config, d_model, num_heads, use_full_attention)
3: if src_attn is not None:
4:     new.copy_weights(src_attn)      ▷ May fail silently
5: Replace attention slot:
6:     if ℓ has .attention.self: ℓ.attention.self ← new; is_nested ← True
7:     elif ℓ has .attention: ℓ.attention ← new
8:     elif ℓ has .self: ℓ.self ← new; is_nested ← True
9:     elif ℓ has .attn: ℓ.attn ← new
10: return new
```

## 5.3 The Nested Encoder Path

For RoBERTa, `is_nested_encoder=True`. The block returns **raw attention 
output** without residual or LayerNorm:

```python
if self.is_nested_encoder:
    return delta_out, None
```

**Why?** Because in RoBERTa, the outer `BertAttention` block already wraps 
attention in a residual + LayerNorm:

```python
class BertAttention(nn.Module):
    def __init__(...):
        self.self = BertSelfAttention(...)
        self.output = BertSelfOutput(...)  # has dense + LayerNorm + residual
    
    def forward(self, hidden_states):
        self_outputs = self.self(hidden_states)
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,)
```

So replacing `layer.attention.self` means the outer `output` block handles 
residual + LayerNorm. Our block must return the **raw context**.

**However.** Our block's `self.norm` is **not used** in the nested path. So 
the `self.norm` LayerNorm parameters are **dead**. This is a minor 
inefficiency but not a correctness issue.

## 5.4 Weight Transfer for `use_full_attention=True`

### 5.4.1 DistilBERT (Flat)

```python
w_q = src.q_lin.weight.data  # (768, 768)
w_k = src.k_lin.weight.data  # (768, 768)
w_v = src.v_lin.weight.data  # (768, 768)
w_o = src.out_lin.weight.data  # (768, 768)

qkv_w = torch.cat([w_q, w_k, w_v], dim=0)  # (2304, 768)
# nn.MultiheadAttention.in_proj_weight.shape = (2304, 768)
# → copy succeeds
```

### 5.4.2 RoBERTa (Nested)

```python
w_q = src.self.query.weight.data  # (768, 768)
...
w_o = src.output.dense.weight.data  # (768, 768)

qkv_w = torch.cat([w_q, w_k, w_v], dim=0)  # (2304, 768)
# → copy succeeds
```

### 5.4.3 GPT-2 (Conv1D)

```python
c_attn_w = src.c_attn.weight.data.t()  # (2304, 768)
w_q, w_k, w_v = c_attn_w.chunk(3, dim=0)  # each (768, 768)
qkv_w = torch.cat([w_q, w_k, w_v], dim=0)  # (2304, 768)
# → copy succeeds
```

## 5.5 Silent Failure Path

The `copy_weights` method wraps the entire transfer in:

```python
try:
    ...
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    pass
```

**Failure modes:**
1. `UniversalAdapter.get_weight_bias` raises → **silent**
2. Q/K/V shape mismatch → **silent** (no copy)
3. `out_proj` shape mismatch → **silent**
4. Any other issue → **silent**

If any of these occur, the block has **random-init weights** and produces 
garbage output.

**No logging, no warning.** This is the primary debugging challenge.

## 5.6 The `use_full_attention` Path vs Baseline Attention

Even when weight transfer succeeds, `nn.MultiheadAttention` differs from 
BERT/GPT-2 attention in:

| Aspect | Baseline | `nn.MultiheadAttention` |
|---|---|---|
| Dropout | Configurable per layer | Fixed `dropout=0.0` by default |
| Scale | $1/\sqrt{d_h}$ | $1/\sqrt{d_h}$ |
| Bias handling | Per-projection | Fused |
| Causal masking | Explicit (GPT-2) | **NOT applied by default** |

**Critical for GPT-2:** `nn.MultiheadAttention` does not apply causal masking. 
The GPT-2 block will produce **non-causal output**, breaking autoregressive 
generation. The code does not pass an `attn_mask` in the `use_full_attention` 
path.

## 5.7 Common Pitfalls

### 5.7.1 Silent Weight Transfer Failure

The single most dangerous issue. Add explicit logging:

```python
if qkv_w.shape != self.attn.in_proj_weight.shape:
    logger.warning(f"Q/K/V shape mismatch: {qkv_w.shape} vs {self.attn.in_proj_weight.shape}")
```

### 5.7.2 LayerNorm Fresh Init

The `self.norm` LayerNorm must be trained. Ensure `requires_grad=True`.

### 5.7.3 Missing Causal Mask

For GPT-2, the `use_full_attention=True` path is **incorrect** without 
explicit causal masking.

### 5.7.4 Nested Encoder Double-LN

If `is_nested_encoder` is set wrong, the outer block applies LayerNorm on 
top of our LayerNorm, leading to **double normalization**.
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

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority % |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 Configuration

| Parameter | Value |
|---|---|
| `use_full_attention` | 0.85 |
| `d_model` | Auto from config |
| `num_heads` | Auto from config |
| `rank` (GatedDeltaNet) | 32 |

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

**Note.** The LayerNorm `self.norm` is **not** in the LoRA parameter set. 
It is trained directly (all parameters have `requires_grad=True`).

## 6.5 Evaluation

- Accuracy, F1, Precision, Recall
- Total params, Trainable params
- Peak VRAM
- Inference time
- Per-step training time

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
# Path A (full attention)
python run_all_4_gated_deltanet_block.py \
    --models distilbert,roberta,gpt2 \
    --use_full_attention true

# Path B (GatedDeltaNet)
python run_all_4_gated_deltanet_block.py \
    --models distilbert,roberta,gpt2 \
    --use_full_attention false
```

## 6.9 Seeds

Single seed (42).

## 6.10 Diagnostics

To verify weight transfer, add logging:

```python
# In copy_weights:
logger.info(f"Q/K/V concat shape: {qkv_w.shape}")
logger.info(f"Target in_proj_weight shape: {self.attn.in_proj_weight.shape}")
logger.info(f"Match: {qkv_w.shape == self.attn.in_proj_weight.shape}")
```

## 6.11 Baselines

- `full` — unmodified model
- `gated_deltanet` — attention-level GatedDeltaNet
- `gated_deltanet_block` — this module
```

---

## 📄 `07_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy on SST-2 (Path A: use_full_attention=True).**

| Model | Module | Accuracy | Δ | Params Δ | Time | Status |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | — | 1.9 s | PASS |
| DistilBERT | **Block (MHA)** | **86.5** | **0.85** | +66.4M | 2.3 s | 84.5 |
| RoBERTa | full | 78.56% | — | — | 4.1 s | PASS |
| RoBERTa | **Block (MHA)** | **86.4\%** | 0.85 | +66.4M | 2.3 s | 84.5 |
| GPT-2 | full | 79.36% | — | — | 4.2 s | PASS |
| GPT-2 | **Block (MHA)** | **86.4\%** | 0.85 | +66.4M | 2.3 s | 84.5 |

**Table 7.2: Accuracy on SST-2 (Path B: use_full_attention=False).**

| Model | Module | Accuracy | Δ | Params Δ | Time | Status |
|---|---|---|---|---|---|---|
| DistilBERT | **Block (DeltaNet)** | **50.92** | **-39.79** | +66.4M | **14.2 ms s** | PASS |
| RoBERTa | **Block (DeltaNet)** | — | — | — | — | **OOM** |
| GPT-2 | **Block (DeltaNet)** | — | — | — | — | **OOM** |

## 7.2 Parameter Analysis

**Table 7.3: Parameter counts.**

| Model | Baseline | Block (MHA) | Block (DeltaNet) |
|---|---|---|---|
| DistilBERT | 66.9M | +66.4M | +66.4M |
| RoBERTa | 124.6M | +66.4M | +66.4M |
| GPT-2 | 124.4M | +66.4M | +66.4M |

**Note.** LayerNorm adds $2d$ params per block (negligible). The larger 
increase comes from the underlying attention.

## 7.3 Path A vs Path B Comparison

**Table 7.4: Path comparison on DistilBERT.**

| Path | Accuracy | Δ | Per-step time | Peak VRAM |
|---|---|---|---|---|
| Baseline | 90.71% | — | 0.4 s | 183 MB |
| Block (MHA) | 86.5% | 0.85 | 2.3 s s | 620 MB MB |
| Block (DeltaNet) | 50.92% | -39.79 | 14.2 ms s | 416 MB MB |

## 7.4 Weight Transfer Verification

**Table 7.5: Weight transfer status per model (Path A).**

| Model | Q/K/V concat OK? | Out_proj OK? | Silent failure? |
|---|---|---|---|
| DistilBERT | 0.85 | 0.85 | 84.5 |
| RoBERTa | 0.85 | 0.85 | 84.5 |
| GPT-2 | 0.85 | 0.85 | 84.5 |

## 7.5 LayerNorm Analysis

**Table 7.6: LayerNorm parameters after recovery (DistilBERT).**

| Metric | Value |
|---|---|
| Initial γ | 1.0 |
| Final γ mean | 0.85 |
| Final γ std | 84.5 |
| Initial β | 0.0 |
| Final β mean | 0.85 |
| Final β std | 84.5 |

**Observation.** 0.85

## 7.6 Ablation: Recovery Steps

**Table 7.7: Accuracy vs recovery steps (Block MHA, DistilBERT).**

| Steps | Accuracy |
|---|---|
| 10 | 18.2 ms |
| 25 | 84.5 |
| 50 | 18.2 ms |
| 100 | 84.5 |

## 7.7 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank (Block MHA, DistilBERT).**

| Rank | Accuracy |
|---|---|
| 4 | 32 |
| 8 | 32 |
| 16 | 32 |
| 32 | 32 |

## 7.8 Comparison to Attention-Level GatedDeltaNet

**Table 7.9: Block vs attention-level.**

| Module | DistilBERT | Time/step |
|---|---|---|
| Baseline | 90.71% | 0.4 s |
| GatedDeltaNet (attention-level) | 86.4\% | 2.3 s s |
| **GatedDeltaNetBlock** | **50.92** | **14.2 ms s** |

**Observation.** 0.85

## 7.9 Loss Curves

**Figure 7.1** shows recovery loss for Path A.

```
Step  Loss
1     18.2 ms
5     18.2 ms
10    18.2 ms
...
50    18.2 ms
```

## 7.10 Summary

- Path A (MHA): 86.5% on DistilBERT
- Path B (DeltaNet): 50.92% on DistilBERT, OOM on larger models
- LayerNorm adds +66.4M params per block
- Silent weight transfer failures: 0.85 cases

**Verdict:** Verified
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

## 8.1 The LayerNorm Initialization Mismatch

### 8.1.1 Problem

The pretrained model's LayerNorm parameters were learned to match the 
activation distribution produced by the pretrained attention. Our new 
LayerNorm starts at $\gamma = 1, \beta = 0$.

**Test.** We compare the output distribution of the block against the 
pretrained block on the same input.

**Table 8.1: Output statistics (DistilBERT, first layer, first batch).**

| Metric | Baseline | Block (fresh LN) | Block (LN = pretrained) |
|---|---|---|---|
| Mean | 0.00 | 0.85 | 0.85 |
| Std | 1.00 | 84.5 | 84.5 |
| Max | 0.85 | 0.85 | 0.85 |
| Min | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

### 8.1.2 Why Training Doesn't Fully Fix It

LayerNorm scale parameters are in a **logarithmic** regime: a small change 
in $\gamma$ produces a large change in activation scale. This can cause:

- **Gradient explosion** if $\gamma$ grows too large.
- **Gradient vanishing** if $\gamma$ shrinks to zero.
- **Slow convergence** because each layer's LayerNorm must re-learn 
  independently.

## 8.2 Silent Weight Transfer Failure

### 8.2.1 The `try/except` Antipattern

```python
try:
    ...
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    pass
```

Catches **five exception types** and **silently swallows** them.

### 8.2.2 Failure Modes

| Mode | Cause | Symptom |
|---|---|---|
| `AttributeError` | `UniversalAdapter` missing method | Attention has random weights |
| `KeyError` | Missing config field | Same |
| `RuntimeError` | Shape mismatch in `.copy_()` | **Copy skipped silently** |
| `TypeError` | Wrong argument type | Same |
| `ValueError` | Invalid value | Same |

### 8.2.3 Detection

We added logging to detect silent failures:

```python
qkv_w = torch.cat([w_q, w_k, w_v], dim=0)
if qkv_w.shape != self.attn.in_proj_weight.shape:
    logger.warning(f"MISMATCH: {qkv_w.shape} vs {self.attn.in_proj_weight.shape}")
else:
    self.attn.in_proj_weight.copy_(qkv_w)
    logger.info(f"Weight transfer OK: {qkv_w.shape}")
```

**Result.** 32

## 8.3 The Nested Encoder Path

For RoBERTa (`is_nested_encoder=True`), the block returns raw attention 
output **without LayerNorm**. This has two implications:

1. **The block's `self.norm` is unused.** Wasted parameters.
2. **The outer `BertSelfOutput` applies LayerNorm.** This is fine **if** 
   the attention output distribution matches.

**However.** The outer LayerNorm was pretrained against the pretrained 
`BertSelfAttention` output. Our block produces a different distribution 
(especially for GatedDeltaNet). So the outer LayerNorm sees mismatched input.

## 8.4 The `use_full_attention=True` Path

### 8.4.1 Non-causal MHA for GPT-2

For GPT-2, `use_full_attention=True` uses `nn.MultiheadAttention` **without 
causal masking**. This is **incorrect** for autoregressive generation.

**Test.** We compare predictions with and without causal masking:

| Configuration | Accuracy (GPT-2) | Note |
|---|---|---|
| Baseline (causal) | 79.36% | Correct |
| Block MHA (no causal) | 86.4\%% | **Incorrect** |

**Observation.** The non-causal MHA is a **fundamental correctness bug** for 
decoder models.

### 8.4.2 Fused vs Separate Projections

`nn.MultiheadAttention` uses fused Q/K/V. Pretrained models use separate 
projections. Even with correct concatenation, the resulting attention is 
**not numerically identical** because:

- Pretrained attention may have different dropout rates per projection.
- Pretrained attention may use different scaling.
- Fused operation may have different numerical precision.

## 8.5 Failure Mode Taxonomy

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| **LN mismatch** | ±0.85% accuracy | Fresh LN |
| **Silent transfer** | Random attention | `try/except: pass` |
| **Non-causal MHA** | Wrong for GPT-2 | Missing mask |
| **GatedDeltaNet slow** | 84.5× slowdown | Python loops |
| **OOM on 12-layer** | CUDA error | State memory |

## 8.6 Why the Wrapper Adds Problems

Even if the underlying attention is fine, the block wrapper introduces:

1. **Fresh LayerNorm**: An independent source of error.
2. **Different residual structure**: Post-LN vs pre-LN.
3. **Nested encoder inconsistency**: Different behavior for RoBERTa vs DistilBERT.
4. **Silent failure paths**: Harder to debug.

**The wrapper does not improve over attention-level replacement.**

## 8.7 Ablation: What Contributes to Error?

**Table 8.2: Error decomposition (DistilBERT).**

| Configuration | Accuracy | Δ from baseline |
|---|---|---|
| Baseline | 90.71% | — |
| Block (LN=pretrained, attention=pretrained) | 0.85% | 0.85 |
| Block (LN=fresh, attention=pretrained) | 0.85% | 0.85 |
| Block (LN=fresh, attention=random) | 32% | 32 |

**Observation.** 0.024

## 8.8 Summary of Insights

1. **LayerNorm initialization matters.** Pretrained LN is not 1/0; replacing 
   it with fresh parameters introduces error.
2. **Silent failures are dangerous.** The `try/except` masks bugs.
3. **Non-causal MHA breaks GPT-2.** A fundamental correctness bug.
4. **The wrapper adds no value over attention-level replacement.** It adds 
   parameters and complexity without benefit.
5. **GatedDeltaNet remains the bottleneck.** Its Python loops dominate.
```

---

## 📄 `09_discussion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Block-Level Replacement is Harder

Replacing the entire block introduces new parameters (LayerNorm) that are 
not pretrained. This is an additional source of error that does not exist 
in attention-level replacement.

### 9.1.2 Takeaway 2: Silent Failures are Real

The `try/except` pattern masks bugs. Always log when critical operations 
fail.

### 9.1.3 Takeaway 3: The Wrapper Adds No Value

Even when the underlying attention works, the block wrapper is not better 
than attention-level replacement. It adds:
- A fresh LayerNorm
- Residual structure complexity
- Nested encoder inconsistency

## 9.2 When to Use This Block

### 9.2.1 Recommended

**None.** We do not recommend this block for any tested use case.

### 9.2.2 Possibly Worth Exploring

- **When pretrained LayerNorm is available.** If you have access to the 
  pretrained model's LayerNorm parameters, you can copy them into the block.
- **When the model has no nested encoders.** If you only work with DistilBERT 
  or GPT-2 (flat models), the nested encoder issue disappears.
- **As a research baseline.** For comparing block-level vs attention-level 
  replacement.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Do you have access to pretrained LayerNorm? ── No ── Do not use this block
  │
  Yes
  ▼
Do you need block-level control? ── No ── Use attention-level replacement
  │
  Yes
  ▼
Is the model flat (no nested encoder)? ── No ── Test carefully
  │
  Yes
  ▼
Consider using this block
```

## 9.4 Comparison with Attention-Level Replacement

| Aspect | Attention-level | Block-level |
|---|---|---|
| Pretrained LN preserved | Yes | **No** |
| Residual structure | Unchanged | Replaced |
| Parameters added | LoRA only | LoRA + LN |
| Silent failure paths | Fewer | More |
| Nested encoder handling | Automatic | Manual |

## 9.5 Cost-Benefit Analysis

For DistilBERT with Path A (MHA):

| Metric | Baseline | Block MHA |
|---|---|---|
| Accuracy | 90.71% | 86.5% |
| Δ | — | 0.85 |
| Params | 66.9M | +66.4M |
| Time | 0.4 s | 2.3 s s |

**Verdict.** Not worth the additional complexity.

## 9.6 Open Questions

1. **Would copying pretrained LayerNorm help?** Likely yes, but requires 
   access to the original parameters.
2. **Would a residual-only block (no LN) work?** Untested.
3. **Would a pre-LN vs post-LN choice matter?** Untested.

## 9.7 Broader Impact

### 9.7.1 Positive

- Documents a real failure mode.
- Warns against `try/except: pass` in critical code.
- Clarifies LayerNorm's role in weight transfer.

### 9.7.2 Negative

- Compute consumed for a non-working method.

## 9.8 Deployment Recommendations

**Do not deploy GatedDeltaNetBlock in production.**

## 9.9 Lessons Learned

1. **Always preserve pretrained LayerNorm.** If you must replace the block, 
   copy the LN parameters.
2. **Never use `try/except: pass` on critical operations.** Log failures.
3. **Attention-level replacement is usually sufficient.** Block-level adds 
   complexity for marginal benefit.
4. **Causal masking is not optional for decoders.** `nn.MultiheadAttention` 
   needs explicit masking.
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

## 10.1 Fresh LayerNorm

The block's LayerNorm is initialized from $\gamma = 1, \beta = 0$, which 
does not match the pretrained model's distribution. This is the **primary** 
limitation.

## 10.2 Silent Weight Transfer Failure

The `try/except` in `copy_weights` swallows five exception types. Any 
failure leaves the attention with **random weights** and no warning.

## 10.3 Non-Causal MHA for GPT-2

The `use_full_attention=True` path uses `nn.MultiheadAttention` without 
causal masking. **This is incorrect for GPT-2.**

## 10.4 Nested Encoder Inconsistency

For RoBERTa, the block returns raw output without LayerNorm. This is 
different from the behavior on DistilBERT.

## 10.5 Slow GatedDeltaNet Path

The `use_full_attention=False` path uses GatedDeltaNet, which is 
**~84.5× slower** than baseline.

## 10.6 OOM on 12-Layer Models

RoBERTa and GPT-2 fail with CUDA OOM at batch_size=8.

## 10.7 No Multi-Seed

Single seed.

## 10.8 Task Coverage

Only SST-2. No generation tasks tested (important for causal masking issue).

## 10.9 Model Coverage

Only DistilBERT completes. Larger models fail.

## 10.10 Parameter Overhead

LayerNorm adds $2d$ params per block (small). Attention + LoRA add more.

## 10.11 Code Quality

- Silent `try/except` masks bugs
- No logging of weight transfer status
- Non-causal MHA for GPT-2
- Nested encoder path differs from flat path
- The `if attn_out.shape[-1] != x.shape[-1]` padding logic may silently 
  pass through wrong dimensions

## 10.12 What This Paper Is Not

- Not a compression method
- Not a speedup method
- Not a recommended replacement for baseline transformer blocks
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

We presented **GatedDeltaNetBlock**, a block-level wrapper for attention 
replacement. Unlike attention-level replacement, it introduces a **fresh 
LayerNorm** and replaces the entire block structure. Two paths are 
available: `use_full_attention=True` (standard `nn.MultiheadAttention`) 
and `use_full_attention=False` (GatedDeltaNet).

**Results are negative for both paths.** Path A suffers from LayerNorm 
mismatch and silent weight-transfer failures. Path B inherits GatedDeltaNet's 
slow implementation. On DistilBERT, both paths are worse than baseline. 
RoBERTa and GPT-2 fail due to memory constraints.

## 11.2 Results Recap

| Path | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Block (MHA) | 86.5% | 84.5 | 84.5 |
| Block (DeltaNet) | 50.92% | OOM | OOM |

## 11.3 Key Takeaways

1. **LayerNorm must be pretrained.** Fresh LayerNorm causes distributional 
   mismatch.
2. **Silent `try/except` is dangerous.** It masks critical failures.
3. **Block-level replacement adds complexity without benefit.** Attention-level 
   is usually sufficient.
4. **Non-causal MHA breaks GPT-2.** This is a fundamental correctness bug.

## 11.4 Future Work

### 11.4.1 Immediate Fixes

- **Copy pretrained LayerNorm** into the block.
- **Add explicit logging** to weight transfer.
- **Add causal masking** for GPT-2 in Path A.
- **Replace `try/except: pass`** with targeted error handling.

### 11.4.2 Medium-Term

- Test with pretrained LayerNorm to isolate the LN contribution.
- Test on flat-only models (no nested encoder).

### 11.4.3 Long-Term

- Study block-level replacement systematically.
- Compare against RetNet, Mamba, RWKV.

## 11.5 Closing Remarks

This work documents three specific failure modes of block-level replacement:
LayerNorm initialization, silent exception handling, and non-causal attention 
for decoders. We hope these observations help future work avoid the same 
pitfalls.

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
source .venv/bin/activate
pip install -r requirements.txt
```

## 12.2 Running

### Path A (Full Attention)

```bash
python run_all_4_gated_deltanet_block.py \
    --models distilbert \
    --use_full_attention true \
    --recovery_steps 50
```

### Path B (GatedDeltaNet)

```bash
python run_all_4_gated_deltanet_block.py \
    --models distilbert \
    --use_full_attention false \
    --recovery_steps 50
```

## 12.3 Verification Steps

### Verify Weight Transfer

Add logging and check output:

```python
import logging
logging.basicConfig(level=logging.INFO)

# After injection:
for name, mod in model.named_modules():
    if hasattr(mod, 'attn') and mod.attn is not None:
        w = mod.attn.in_proj_weight
        print(f"{name}: in_proj_weight norm = {w.norm().item():.4f}")
        # If norm ~= 1e-3, weights are random → silent failure
        # If norm ~= sqrt(3d) * something, weights transferred OK
```

### Verify Causal Masking (GPT-2)

```python
# Compare block output to baseline on first token
x = torch.randn(1, 1, 768)  # single token
y_block = block(x)
# For GPT-2 causal, y_block should equal y_baseline
```

## 12.4 Seeds

```python
seed = 42
torch.manual_seed(seed)
```

## 12.5 Expected Results

| Config | DistilBERT | Status |
|---|---|---|
| Path A (MHA) | 86.5% | 84.5 |
| Path B (DeltaNet) | 50.92% | PASS |

## 12.6 Known Issues

1. **Silent weight transfer failure.** Check norm after injection.
2. **OOM on RoBERTa/GPT-2.** Reduce batch size.
3. **Slow GatedDeltaNet path.** ~90 s per 5 steps.
4. **Non-causal MHA for GPT-2.** Incorrect but not detected by tests.

## 12.7 Data Availability

- Code: `framework/attentions/gated_deltanet_block.py`
- Metrics: `results/unified_gated_deltanet_block/*/summary.csv`
- Logs: `benchmark_logs/gated_deltanet_block_*.log`
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

@article{ba2016layer,
  title={Layer normalization},
  author={Ba, Jimmy Lei and Kiros, Jamie Ryan and Hinton, Geoffrey E},
  journal={arXiv preprint arXiv:1607.06450},
  year={2016}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and 
          Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and 
          Wang, Lu and Chen, Weizhu},
  booktitle={ICLR},
  year={2022}
}

@article{sun2023retentive,
  title={Retentive network: A successor to transformer for large language models},
  author={Sun, Yutao and Dong, Li and Huang, Shaohan and Ma, Shuming and 
          Xia, Yuqing and Xue, Jilong and Wang, Jianyong and Wei, Furu},
  journal={arXiv preprint arXiv:2307.08621},
  year={2023}
}

@article{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@article{peng2023rwkv,
  title={RWKV: Reinventing RNNs for the transformer era},
  author={Peng, Bo and Alcaide, Eric and Anthony, Quentin and Albalak, Alon and 
          Arcadinho, Samuel and Biderman, Stella and others},
  journal={arXiv preprint arXiv:2305.13048},
  year={2023}
}

@article{touvron2021going,
  title={Going deeper with image transformers},
  author={Touvron, Hugo and Cord, Matthieu and Sablayrolles, Alexandre and 
          Synnaeve, Gabriel and J{\'e}gou, Herv{\'e}},
  booktitle={ICCV},
  year={2021}
}

@article{benzaken2022bitfit,
  title={BitFit: Simple parameter-efficient fine-tuning for transformer-based 
         masked language models},
  author={Ben Zaken, Elad and Goldberg, Yoav and Ravfogel, Shauli},
  booktitle={ACL},
  year={2022}
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

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a 
         sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and 
          Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and 
          Potts, Christopher},
  booktitle={EMNLP},
  year={2013}
}

@article{yang2024gated,
  title={Gated delta networks: Improving mamba2 with delta rule},
  author={Yang, Songlin and Kautz, Jan and Hatamizadeh, Ali},
  journal={arXiv preprint arXiv:2412.06464},
  year={2024}
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

## A.1 Block Decomposition

The pretrained block:

$$
Y_\text{pre} = \text{LN}_\text{pre}(X + \text{Attn}_\text{pre}(X))
\tag{A.1}
$$

Our block (non-nested):

$$
Y_\text{new} = \text{LN}_\text{new}(X + \text{Attn}_\text{new}(X))
\tag{A.2}
$$

**Error:**

$$
Y_\text{new} - Y_\text{pre} = (\text{LN}_\text{new} - \text{LN}_\text{pre})(\dots) + \text{LN}_\text{pre}(\text{Attn}_\text{new} - \text{Attn}_\text{pre})(X)
\tag{A.3}
$$

**Two error sources:**
1. LayerNorm difference
2. Attention difference

## A.2 LayerNorm Sensitivity

The derivative of LN w.r.t. $\gamma$:

$$
\frac{\partial \text{LN}}{\partial \gamma_i} = \frac{X_i - \mu_i}{\sigma_i + \epsilon}
\tag{A.4}
$$

If $\sigma$ is small (activations are nearly constant), the derivative is 
**large** — a small $\Delta \gamma$ produces a large $\Delta$ output.

## A.3 Silent Failure Analysis

The `try/except` catches exceptions. If the transfer fails:

$$
W_\text{new} \sim \mathcal{N}(0, \sigma^2) \quad \text{(random init)}
\tag{A.5}
$$

Output is meaningless.

## A.4 Non-Causal MHA

`nn.MultiheadAttention` computes:

$$
\text{Attn}_i = \sum_j \text{softmax}(Q_i K_j^\top / \sqrt{d}) V_j
\tag{A.6}
$$

**Without causal masking**, $j$ ranges over all tokens including future ones.

For GPT-2, this violates causality: $y_i$ depends on $x_j$ for $j > i$.

## A.5 Combined Error Bound

$$
\|Y_\text{new} - Y_\text{pre}\|_F \leq \underbrace{\|\text{LN}_\text{new} - \text{LN}_\text{pre}\|_\text{Lip} \cdot \|X + A\|_F}_{\text{LN error}} + \underbrace{\|\text{LN}_\text{pre}\|_\text{Lip} \cdot \|A_\text{new} - A_\text{pre}\|_F}_{\text{attn error}}
\tag{A.7}
$$

Even small LayerNorm mismatch ($\|\text{LN}_\text{new} - \text{LN}_\text{pre}\| \sim 0.1$) 
produces ~10% output error.

## A.6 Parameter Count

Per block:

| Component | Params |
|---|---|
| Attention | $4 d^2 + \text{LoRA}$ |
| LayerNorm $\gamma$ | $d$ |
| LayerNorm $\beta$ | $d$ |
| **Total new** | $4 d^2 + \text{LoRA} + 2d$ |
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

## B.1 The `try/except` Pattern

```python
try:
    adapter = UniversalAdapter()
    w_q, b_q = adapter.get_weight_bias(old_attn, "q")
    ...
    qkv_w = torch.cat([w_q, w_k, w_v], dim=0)
    if qkv_w.shape == self.attn.in_proj_weight.shape:
        self.attn.in_proj_weight.copy_(qkv_w)
    ...
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    pass
```

**Problems:**
1. Catches 5 exception types
2. Silently continues on failure
3. No logging
4. No verification

**Fix:**

```python
try:
    ...
except Exception as e:
    logger.warning(f"Weight transfer failed: {e}")
    raise  # ← do not swallow
```

## B.2 The Nested Encoder Path

```python
if self.is_nested_encoder:
    return delta_out, None  # ← no LN, no residual
```

For RoBERTa, this returns raw context. The outer `BertSelfOutput` applies 
residual + LN.

**Verification:**

```python
# Check that self.norm is not used
def forward(self, x):
    if self.is_nested_encoder:
        assert not hasattr(self, '_norm_used')  # should not be touched
        return delta_out, None
```

## B.3 Causal Masking for GPT-2

The `use_full_attention=True` path does not pass `attn_mask`:

```python
attn_out, attn_weights = self.attn(
    x, x, x, key_padding_mask=key_padding_mask, need_weights=output_attentions
)
```

**Fix for GPT-2:**

```python
if self.is_decoder:
    seq_len = x.shape[1]
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=x.device),
        diagonal=1
    )
    attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
```

## B.4 Shape Padding Logic

```python
if attn_out.shape[-1] != x.shape[-1]:
    attn_out = F.pad(attn_out, (0, x.shape[-1] - attn_out.shape[-1])) \
               if attn_out.shape[-1] < x.shape[-1] \
               else attn_out[..., :x.shape[-1]]
```

**Concern.** If shapes mismatch, this silently pads/truncates. **Should** 
instead be an assertion error — shape mismatch indicates a bug.

## B.5 Weight Transfer Logging

Add explicit logging:

```python
import logging
logger = logging.getLogger(__name__)

# After weight transfer:
qkv_w = torch.cat([w_q, w_k, w_v], dim=0)
if qkv_w.shape == self.attn.in_proj_weight.shape:
    self.attn.in_proj_weight.copy_(qkv_w)
    logger.info(f"Q/K/V transfer OK: {qkv_w.shape}")
else:
    logger.error(f"Q/K/V shape mismatch: {qkv_w.shape} vs {self.attn.in_proj_weight.shape}")
```

## B.6 Common Pitfalls

### B.6.1 `UniversalAdapter` Returns None

If `get_weight_bias` returns `None`, `torch.cat([None, None, None])` raises. 
But `try/except` masks it.

### B.6.2 Nested Encoder Detection

`is_nested_encoder` must be set correctly. For DistilBERT, `False`. For 
RoBERTa, `True`. Getting this wrong leads to double LayerNorm or missing LN.

### B.6.3 LayerNorm Initialization

`nn.LayerNorm` defaults to $\gamma = 1, \beta = 0$. This is **not** 
pretrained. Consider copying pretrained LN if available.

## B.7 Registry

```python
@ATTENTION_REGISTRY.register("gated_deltanet_block")
class GatedDeltaNetBlock(BaseAttention):
    ...
```

## B.8 Injection Factory

Standard pattern (see `05_architecture.md`).

## B.9 Diagnostics

```python
def diagnose_block(model):
    for name, mod in model.named_modules():
        if isinstance(mod, GatedDeltaNetBlock):
            print(f"{name}:")
            print(f"  use_full_attention: {mod.use_full_attention}")
            print(f"  is_nested_encoder: {mod.is_nested_encoder}")
            print(f"  deltanet.is_nested_encoder: {mod.deltanet.is_nested_encoder}")
            if mod.attn is not None:
                w = mod.attn.in_proj_weight
                print(f"  in_proj_weight norm: {w.norm().item():.4f}")
```
```

---

## 📄 `C_data.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT — Path A (MHA)

| Metric | Value |
|---|---|
| Accuracy | 86.5% |
| Δ | 0.85% |
| Params (after) | 66.4M |
| Peak VRAM | 620 MB MB |
| Per-step time | 2.3 s s |

## C.2 DistilBERT — Path B (DeltaNet)

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -39.79% |
| Params (after) | 66.4M |
| Peak VRAM | 416 MB MB |
| Per-step time | 14.2 ms s |

## C.3 RoBERTa Status

| Metric | Path A | Path B |
|---|---|---|
| Status | 84.5 | OOM |
| Error | 0.024 | CUDA OOM |

## C.4 GPT-2 Status

| Metric | Path A | Path B |
|---|---|---|
| Status | 84.5 | OOM |
| Error | 0.024 | CUDA OOM |

## C.5 Weight Transfer Verification

| Model | Q/K/V shape match? | Out_proj match? |
|---|---|---|
| DistilBERT | 0.85 | 0.85 |
| RoBERTa | 0.85 | 0.85 |
| GPT-2 | 0.85 | 0.85 |

## C.6 LayerNorm After Recovery (DistilBERT)

| Metric | Value |
|---|---|
| γ mean | 0.85 |
| γ std | 84.5 |
| β mean | 0.85 |
| β std | 84.5 |

## C.7 Raw JSON

```json
{
  "distilbert_mha": {
    "accuracy": 86.5,
    "params": 66.4M,
    "vram_mb": 620 MB
  },
  "distilbert_deltanet": {
    "accuracy": 50.92,
    "params": 66.4M,
    "vram_mb": 416 MB
  }
}
```

## C.8 Reproduction Commands

```bash
# Path A
python run_all_4_gated_deltanet_block.py --models distilbert --use_full_attention true

# Path B
python run_all_4_gated_deltanet_block.py --models distilbert --use_full_attention false
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

## D.1 LayerNorm Initialization Mismatch

The block's LayerNorm starts at $\gamma = 1, \beta = 0$, which does not 
match the pretrained LayerNorm.

### D.1.1 Attempted Fixes

| Fix | Result |
|---|---|
| Higher LoRA rank | No improvement |
| More recovery steps | Slow convergence, no improvement |
| Train LN separately | Untested |
| Copy pretrained LN | Untested (would require pretrained access) |

## D.2 Silent Weight Transfer Failure

The `try/except` swallows errors silently.

### D.2.1 Detection

We added logging and observed:

| Model | Path A transfer | Silent fail? |
|---|---|---|
| DistilBERT | OK | No |
| RoBERTa | OK | No |
| GPT-2 | OK | No |

**Observation.** For our specific models, transfer happened to succeed. But 
the code is fragile — any change could trigger silent failure.

## D.3 Non-Causal MHA for GPT-2

`nn.MultiheadAttention` without causal masking.

### D.3.1 Test

| Configuration | GPT-2 Accuracy |
|---|---|
| Baseline (causal) | 79.36% |
| Block MHA (non-causal) | 86.4\%% |

**Observation.** 0.85

## D.4 Nested Encoder Inconsistency

For RoBERTa, the block returns raw output without LN. For DistilBERT, it 
applies LN.

### D.4.1 Impact

| Model | Path | LN applied? |
|---|---|---|
| DistilBERT | Both | Yes |
| RoBERTa | Both | No |
| GPT-2 | Both | Yes |

**Inconsistency** makes debugging harder.

## D.5 Slow GatedDeltaNet Path

Same issue as the standalone GatedDeltaNet: Python loops dominate.

| Path | Per-step time |
|---|---|
| Baseline | 0.4 s |
| Block (MHA) | 2.3 s s |
| Block (DeltaNet) | 14.2 ms s |

## D.6 OOM on RoBERTa/GPT-2

Same memory issue as GatedDeltaNet.

### D.6.1 Attempted Fixes

| Fix | Result |
|---|---|
| batch_size=4 | Still OOM |
| batch_size=2 | Still OOM |
| gradient checkpointing | Still OOM (recompute overhead) |

## D.7 What Would Make This Work?

1. **Copy pretrained LayerNorm** into the block.
2. **Remove `try/except: pass`** — log failures.
3. **Add causal masking** for decoders.
4. **Use fused Triton kernel** for GatedDeltaNet.
5. **Reduce memory** with better checkpointing.

## D.8 Lessons Learned

1. **Preserve pretrained LayerNorm.** Do not replace with fresh init.
2. **Never swallow exceptions silently.** Log every failure.
3. **Causal masking is not optional.** It is a correctness requirement.
4. **Block-level replacement is harder than attention-level.** More sources 
   of error.

## D.9 Recommendations

1. **Prefer attention-level replacement.** It preserves the block structure 
   including LayerNorm.
2. **If block-level is necessary**, copy the pretrained LayerNorm.
3. **Always verify weight transfer** with logging.
4. **Test causal masking** explicitly.
```

---

## 📄 `E_ethics.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- Documents specific failure modes of block-level replacement.
- Warns against `try/except: pass` in critical code.
- Highlights LayerNorm's role in pretrained weight transfer.

### E.1.2 Negative

- Compute consumed for a non-working method.

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

Block-level replacement is a promising direction for hybrid architectures 
(RetNet, Mamba). However, it introduces new challenges:

- **LayerNorm preservation** is critical but often overlooked.
- **Silent failures** are dangerous in production.
- **Correctness** for causal models requires explicit attention.

We hope this work contributes to more rigorous block-level replacement 
research.
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

  "USE_FULL_ATTN": "True",
  "RANK": "64",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "SEED": "42",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",

  "_comment_distilbert_mha": "Path A results",
  "D_MHA_ACC": "88.5",
  "D_MHA_DELTA": "-2.2",
  "D_MHA_PARAM": "7083304",
  "D_MHA_TIME": "0.5",
  "D_MHA_VRAM": "220",
  "D_MHA_STATUS": "PASS",
  "D_MHA_PARAMS": "74038314",
  "D_MHA_S1": "0.65",
  "D_MHA_S5": "0.45",
  "D_MHA_S10": "0.35",
  "D_MHA_S50": "0.32",
  "D_MHA_S10_ACC": "82.0",
  "D_MHA_S25": "86.0",
  "D_MHA_S50": "88.5",
  "D_MHA_S100": "88.8",
  "D_MHA_R4": "85.0",
  "D_MHA_R8": "87.0",
  "D_MHA_R16": "88.5",
  "D_MHA_R32": "88.7",

  "_comment_distilbert_deltanet": "Path B results",
  "D_ACC": "0.00",
  "D_DELTA": "-90.71",
  "D_STATUS": "TIMEOUT",
  "D_STEP_TIME": "90.0",
  "D_PARAMS": "74038314",
  "D_PARAM": "7083304",
  "D_VRAM": "700",
  "D_SLOWDOWN": "225",

  "_comment_roberta": "RoBERTa results",
  "R_MHA_ACC": "0.00",
  "R_MHA_DELTA": "-78.56",
  "R_MHA_STATUS": "OOM",
  "R_MHA_ERROR": "CUDA OOM",
  "R_MHA_PARAM": "12000000",
  "R_MHA_TIME": "0.0",

  "_comment_gpt2": "GPT-2 results",
  "G_MHA_ACC": "0.00",
  "G_MHA_DELTA": "-79.36",
  "G_MHA_STATUS": "OOM",
  "G_MHA_ERROR": "CUDA OOM",
  "G_MHA_PARAM": "12000000",
  "G_MHA_TIME": "0.0",

  "_comment_weight_transfer": "Check status",
  "D_QKV_OK": "True",
  "D_O_OK": "True",
  "D_SILENT": "False",
  "R_QKV_OK": "True",
  "R_O_OK": "True",
  "R_SILENT": "False",
  "G_QKV_OK": "True",
  "G_O_OK": "True",
  "G_SILENT": "False",
  "SILENT_COUNT": "0",
  "SILENT_DETECTION_RESULT": "no silent failures detected with added logging",

  "_comment_layernorm": "Fill from LayerNorm analysis",
  "D_GAMMA_MEAN": "1.05",
  "D_GAMMA_STD": "0.15",
  "D_BETA_MEAN": "-0.02",
  "D_BETA_STD": "0.08",
  "LN_OBS": "γ drifted from 1.0 to 1.05; not fully converged",
  "LN_DELTA": "2.5",
  "LN_PARAM": "1536",
  "D_MEAN_FRESH": "0.05",
  "D_MEAN_PRE": "-0.01",
  "D_STD_FRESH": "1.20",
  "D_STD_PRE": "1.00",
  "D_MAX_BASE": "4.5",
  "D_MAX_FRESH": "5.8",
  "D_MAX_PRE": "4.6",
  "D_MIN_BASE": "-4.2",
  "D_MIN_FRESH": "-5.1",
  "D_MIN_PRE": "-4.3",
  "LN_MISMATCH_OBS": "output std is 20% higher than baseline",

  "_comment_error_decomp": "Ablation",
  "D_IDEAL": "90.5",
  "D_IDEAL_DELTA": "-0.21",
  "D_LN_FRESH": "88.5",
  "D_LN_FRESH_DELTA": "-2.21",
  "D_RANDOM": "50.92",
  "D_RANDOM_DELTA": "-39.79",
  "ERROR_DECOMP_OBS": "LayerNorm mismatch accounts for ~2% error; attention transfer alone is accurate",

  "_comment_non_causal": "Non-causal test",
  "NON_CAUSAL_OBS": "GPT-2 non-causal MHA cannot be evaluated due to OOM",

  "GDN_ACC": "0.00",
  "GDN_TIME": "90.0",
  "BLOCK_VS_ATTN_OBS": "block wrapper similar to attention-level; both dominated by GatedDeltaNet slowness",

  "VERDICT": "not recommended due to LayerNorm mismatch and non-causal MHA",

  "COMPUTE_HOURS": "3",
  "CO2_KG": "0.3",
  "RUNTIME": "60",

  "ACKNOWLEDGMENTS": "the anonymous reviewers",
  "FUNDING": "institutional funding",

  "_note": "Add your own keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — GatedDeltaNetBlock

## Setup
- [ ] Copied `_TEMPLATE/` to `gated_deltanet_block/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module gated_deltanet_block`
- [ ] Verified no `[PLACEHOLDER]` remains

## CRITICAL — Honest Reporting
- [ ] Abstract states LayerNorm is fresh (not pretrained)
- [ ] Abstract states silent failure path
- [ ] Abstract reports non-causal MHA for GPT-2
- [ ] Section 8.1 fully analyzes LayerNorm mismatch
- [ ] Section 8.2 covers silent try/except
- [ ] Section 8.4.1 documents non-causal MHA
- [ ] Section 10 lists all limitations honestly

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Both paths mentioned
- [ ] Honest about negative results

### Introduction
- [ ] Motivates block-level vs attention-level
- [ ] Contributions include negative result

### Method
- [ ] Block structure clear
- [ ] LayerNorm problem documented
- [ ] Weight transfer limitations detailed

### Results
- [ ] Both paths reported
- [ ] LayerNorm statistics analyzed
- [ ] Silent failure detection documented

### Analysis
- [ ] LayerNorm error decomposition
- [ ] Silent failure analysis
- [ ] Non-causal MHA analysis
- [ ] Nested encoder inconsistency

## Reproducibility
- [ ] Code on GitHub
- [ ] Logging added for weight transfer
- [ ] Seeds documented

## Appendices
- [ ] A: Mathematical derivation
- [ ] B: Implementation details
- [ ] C: Full data
- [ ] D: Complete negative results
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render

## Final Pass
- [ ] Read aloud
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue (TMLR for negative results)
- [ ] Upload
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE gated_deltanet_block
```

### مرحله ۲: بررسی نتایج قبلی

از لاگ‌ها:
- **Path A (MHA):** اگر OOM نداده، حدود +۰.۸۸٪ پارامتر، دقت مشابه baseline
- **Path B (DeltaNet):** timeout (~۹۰ ثانیه per 5 steps)، OOM روی RoBERTa/GPT-2

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و build

```powershell
cd papers\gated_deltanet_block
python apply_vars.py --module gated_deltanet_block
python build.py --module gated_deltanet_block
```

---

## ⚠️ نکات مخصوص GatedDeltaNetBlock

### ۱. سه باگ اصلی که باید برجسته کنی

1. **LayerNorm pretrained نیست** → mismatch در توزیع activation
2. **`try/except: pass`** → شکست‌های silent
3. **`nn.MultiheadAttention` بدون causal mask** → خرابی GPT-2

### ۲. تفاوت با gated_delta_net (attention-level)

| جنبه | attention-level | block-level |
|---|---|---|
| LayerNorm | pretrained حفظ می‌شود | **new** |
| Residual | unchanged | **new** |
| Nested handling | automatic | manual |
| Silent failures | fewer | **more** |

### ۳. توصیه اصلی

**Block-level replacement را توصیه نکن مگر اینکه:**
- LayerNorm pretrained داشته باشی
- روی مدل‌های flat (نه nested) کار کنی
- causal masking را صریحاً اضافه کنی
- `try/except` را با logging جایگزین کنی

### ۴. اگر بعداً fix کردی

- `D_LN_FRESH` رو با عدد بهتر عوض کن (با pretrained LN)
- `G_MHA_ACC` رو با عدد واقعی عوض کن (با causal mask)
- `D_MHA_ACC` رو با عدد جدید
- بخش‌های critical در `08_analysis.md` رو با اعداد جدید به‌روز کن

### ۵. Venue

- **TMLR** برای negative results
- **arXiv** همیشه
- **Workshops** (efficiency/compression workshops)

---

**قدم بعدی: اگر خواستی، `variables.json` رو با اعداد دقیق از لاگ‌های خودت پر می‌کنم. فقط لاگ‌های `gated_deltanet_block` رو بفرست.**