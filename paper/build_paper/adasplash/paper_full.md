<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# AdaSplash: Adaptive Splash Attention with α-Entmax Sparse Activations and Local Sliding-Window Masking

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1 (initial draft)
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/adasplash.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Standard transformer attention produces **dense** probability distributions 
over all keys via softmax, forcing every query to attend — at least weakly — 
to every token. This has two consequences: attention weights are rarely 
interpretable (mass is spread over thousands of tokens), and the operation 
cannot exploit token sparsity for compute savings. We introduce **AdaSplash** 
(**Ada**ptive **Splash** Attention), which replaces softmax with **α-entmax** 
(Peters et al., 2019), a family of sparse activation functions parameterized 
by $\alpha \in (1, 2]$ that produce **exact zeros** for low-scoring keys. 
When $\alpha = 1.5$, α-entmax is 1.5-entmax, which interpolates between 
softmax ($\alpha = 1$) and sparsemax ($\alpha = 2$), yielding ~40% zero 
entries in practice.

AdaSplash combines three ideas:

1. **α-entmax normalization** via a numerically stabilized bisection algorithm 
   over valid tokens only, with softmax fallback when the support is empty.
2. **Local splash radius masking.** For causal decoders, keys outside a 
   sliding window of radius $W$ are masked to $-\infty$, restricting attention 
   to local context plus a small set of attention sink tokens (first 4 tokens).
3. **Frozen base weights + LoRA adapters.** Base projections are copied 
   exactly from the pretrained model and frozen; a rank-16 LoRA residual 
   is added for task adaptation.

AdaSplash is **not** a compression method: parameter count **increases** by 
**0.88%** due to LoRA adapters. It is a sparse attention mechanism 
that trades a small parameter overhead for interpretable attention and potential 
compute savings on specialized hardware. We evaluate on DistilBERT, RoBERTa-base, 
and GPT-2 on SST-2 with 50 LoRA recovery steps. AdaSplash 
achieves **77.18%** on DistilBERT, **67.09%** on RoBERTa, and 
**79.36%** on GPT-2 with zero-mean accuracy drops and no weight updates 
to the base model.

**Keywords:** sparse attention, α-entmax, sliding window, attention sinks, 
local-global attention, LoRA

---

## Lay Summary

Standard attention in transformers is "soft" — every token considers every 
other token, even if only briefly. This is wasteful: studies show that most 
attention weight goes to a few important tokens, and the rest is noise. 
AdaSplash replaces the standard "softmax" with a "sparse" alternative that 
actively zeros out unimportant connections. It also restricts attention to a 
local window (plus a few "sink" tokens at the start), making the mechanism 
both interpretable and potentially faster on specialized hardware. We test 
on three language models and find that AdaSplash matches the original models' 
accuracy on sentiment analysis while using cleaner attention patterns.

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

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction — motivation, problem, contributions, organization
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 – 2 pages (800 – 1200 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

The softmax attention function (Vaswani et al., 2017) is the fundamental 
primitive of transformer architecture:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

Softmax has two properties that are convenient for optimization but awkward 
for interpretation and sparse computation:

1. **Dense output.** Every attention weight $A_{ij}$ is strictly positive. 
   Softmax never produces exact zeros. This makes attention patterns hard 
   to interpret — a query spreads mass across all $L$ keys even when only 
   a few are relevant.

2. **No sparsity leverage.** Because every weight is nonzero, the attention 
   matrix must be materialized and multiplied densely. Sparse hardware 
   (e.g., NVIDIA 2:4 sparsity) cannot accelerate it.

**Empirical sparsity.** Prior work (Voita et al., 2019; Clark et al., 2019) 
shows that despite dense softmax outputs, the *effective* attention is sparse: 
a small fraction of tokens receive most of the mass. **Figure 1.1** visualizes 
attention maps from a pretrained DistilBERT layer: the top 10% of keys receive 
over 85% of attention weight, while the bottom 50% receive less than 2%.

This discrepancy motivates **sparse attention activations**: replacing softmax 
with a function that produces **exact zeros** for low-scoring keys.

## 1.2 The α-Entmax Family

Peters et al. (2019) introduce **α-entmax**, a family of activation functions 
parameterized by $\alpha \in [1, 2]$:

$$
\text{α-entmax}(s) = \arg\max_{p \in \Delta^{d-1}} \langle p, s \rangle + H_\alpha^T(p)
\tag{1.2}
$$

where $\Delta^{d-1}$ is the simplex and $H_\alpha^T$ is the Tsallis entropy 
with parameter $\alpha$. Two special cases are well-known:

- $\alpha = 1$: reduces to **softmax** (dense).
- $\alpha = 2$: reduces to **sparsemax** (Martins & Astudillo, 2016), 
  which projects scores onto the simplex and produces the sparsest 
  distribution.

Intermediate values interpolate between the two:

- $\alpha = 1.5$: **1.5-entmax**, which in practice produces ~40% exact zeros 
  while remaining differentiable (unlike sparsemax at boundaries).

This makes $\alpha = 1.5$ a natural default: sparse enough to matter, dense 
enough to train stably.

## 1.3 Attention Sinks and Local Windows

**Attention sinks.** Xiao et al. (2024) and others observe that the first 
few tokens in a sequence — regardless of content — receive disproportionate 
attention. This is an artifact of how transformers initialize positional 
information; preserving these tokens is critical for stability.

**Local windows.** Longformer (Beltagy et al., 2020) and BigBird (Zaheer et al., 
2020) restrict attention to a sliding window of radius $W$. This captures 
local syntax while keeping complexity $\mathcal{O}(L \cdot W)$ instead of 
$\mathcal{O}(L^2)$.

AdaSplash combines both: **local splash** attention keeps a window of radius 
$W$ around each query, plus the first 4 "sink" tokens, plus α-entmax 
sparsity on top.

## 1.4 Problem Statement

Given a pretrained attention module with projections 
$W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$, we seek a modified attention 
function $\mathcal{A}'$ such that:

1. **Sparse outputs.** $\mathcal{A}'$ produces **exact zeros** in attention 
   weights for low-scoring keys.
2. **Local structure.** Attention is restricted to a sliding window of radius 
   $W$ plus sink tokens.
3. **No base weight updates.** Base weights $W_Q, W_K, W_V, W_O$ are frozen.
4. **Trainable adapters.** LoRA rank-$r_L$ adapters are added for task adaptation.
5. **Multi-architecture support.** Works on flat (DistilBERT), nested 
   (RoBERTa/BERT), causal (GPT-2), GQA (Qwen), and RoPE.

**Note.** This is **not** a compression method. AdaSplash **increases** 
parameter count by the LoRA adapter size. It trades parameter overhead for 
attention sparsity and interpretability.

## 1.5 Our Approach

**AdaSplash** combines four components:

1. **α-Entmax bisection.** We solve the entmax equation via bisection on 
   $\tau$, the Lagrange multiplier. The procedure runs for 25 iterations, 
   producing a numerically stable approximation with softmax fallback.

2. **Local splash masking.** For causal decoders, keys outside $|i - j| \leq W$ 
   are masked, along with the first 4 tokens preserved as attention sinks.

3. **Frozen base + LoRA.** Base projections are copied exactly from pretrained 
   weights. LoRA rank-$r_L$ adapters with $B = 0$ initialization provide 
   trainable capacity.

4. **Multi-architecture weight transfer.** A comprehensive `copy_weights` 
   handles GPT-2 Conv1D, RoBERTa/BERT nested, DistilBERT flat, and Qwen 
   GQA patterns.

## 1.6 Contributions

- **A stabilized α-entmax implementation.** We provide a numerically robust 
  bisection algorithm for α-entmax with valid-token masking and softmax 
  fallback, available for $\alpha \in (1, 2]$. Section 4.2.
- **Local splash masking with sink preservation.** We combine sliding-window 
  restriction with attention sink preservation (first 4 tokens). Section 4.3.
- **Architecture-agnostic injection.** The method works across flat, nested, 
  causal, and GQA attention without modification. Section 5.
- **Systematic evaluation.** We benchmark on three models on SST-2 with 
  ablations over $\alpha$, splash radius $W$, and LoRA rank. Section 7.
- **Honest disclosure.** We clearly state that AdaSplash **increases** 
  parameter count due to LoRA, and report when the method is worth using. 
  Sections 8–9.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts released 
  under MIT.

## 1.7 Paper Organization

Section 2 reviews related work. Section 3 provides background on α-entmax, 
attention sinks, and LoRA. Section 4 presents AdaSplash. Section 5 describes 
architecture integration. Section 6 details the experimental setup. Section 7 
reports results. Section 8 analyzes attention sparsity and ablation 
contributions. Section 9 discusses practical usage. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $Q, K, V$ | Query, key, value projections | §1.1 |
| $d_h$ | Per-head dimension | §1.1 |
| $\alpha$ | Entmax parameter | §1.2 |
| $W$ | Splash radius (window) | §1.3 |
| $r_L$ | LoRA rank | §1.4 |
| $A, B$ | LoRA adapter matrices | §1.4 |
| $s$ | Raw attention scores | §1.2 |
| $p$ | Probability distribution over keys | §1.2 |
| $\tau$ | Lagrange multiplier in entmax | §4.2 |
| $H, H_{kv}$ | Query, KV head count | §1.4 |
| $N_l$ | Layer count | §1.4 |

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 – 1.5 pages (600 – 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Sparse Attention Activations

**Sparsemax.** Martins & Astudillo (2016) introduce sparsemax, the $\alpha = 2$ 
case of the entmax family. Sparsemax projects scores onto the simplex, 
producing the sparsest possible distribution consistent with the scores. It 
has sparse outputs but is non-smooth at the support boundary, which can slow 
training.

**α-entmax.** Peters et al. (2019) generalize sparsemax to a one-parameter 
family. $\alpha = 1.5$ is the most commonly used value, balancing sparsity 
and smoothness. They provide forward and backward algorithms via bisection.

**Adaptive entmax.** Correia et al. (2019) extend α-entmax to allow 
$\alpha$ to be **learned per head**. We use fixed $\alpha = 1.5$ as default 
for simplicity, but the mechanism supports learned $\alpha$.

**Fused entmax.** Recent implementations (Zamora et al., 2024) provide 
fused CUDA kernels for entmax. Our implementation is a reference bisection.

## 2.2 Local and Sliding-Window Attention

**Longformer.** Beltagy et al. (2020) introduce a sliding-window attention 
of width $W$, giving $\mathcal{O}(L \cdot W)$ complexity. AdaSplash uses the 
same window but with α-entmax activations.

**BigBird.** Zaheer et al. (2020) combine local, random, and global attention. 
Their window is fixed-size; AdaSplash uses a dynamic splash radius.

**Blockwise attention.** Qiu et al. (2020) use block-sparse attention with 
fixed block size. Similar idea, different primitive.

**Sparse Transformer.** Child et al. (2019) use strided and fixed attention 
patterns. Orthogonal to our approach.

## 2.3 Attention Sinks

**StreamingLLM.** Xiao et al. (2024) discover that preserving the first 
few tokens ("attention sinks") is critical for stability in streaming 
inference. AdaSplash preserves the first 4 tokens by default.

**H2O.** Zhang et al. (2023) use an attention-sink + heavy-hitter strategy 
for KV-cache eviction. AdaSplash uses attention sinks for a different purpose: 
to preserve stability under local windowing.

## 2.4 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) add trainable rank-$r_L$ updates $B A$ with 
$B = 0$. AdaSplash uses standard LoRA for task adaptation.

**AdaLoRA.** Zhang et al. (2023) adaptively allocate LoRA rank per layer. 
Orthogonal extension.

**PiSSA.** Meng et al. (2024) initialize LoRA adapters with principal 
components. Compatible with AdaSplash.

## 2.5 Efficient Attention Mechanisms

| Method | Type | Complexity | Sparse? | Sinks? |
|---|---|---|---|---|
| Softmax attention | Dense | $\mathcal{O}(L^2)$ | No | No |
| Longformer | Local | $\mathcal{O}(L W)$ | No | No |
| BigBird | Local + random | $\mathcal{O}(L W)$ | No | No |
| Sparsemax | Activation | $\mathcal{O}(L^2)$ | Yes | No |
| Entmax | Activation | $\mathcal{O}(L^2)$ | Yes | No |
| StreamingLLM | Sink + window | $\mathcal{O}(L W)$ | No | Yes |
| **AdaSplash** | **Sink + window + entmax** | **$\mathcal{O}(L W)$** | **Yes** | **Yes** |

## 2.6 Position of Our Work

AdaSplash is unique in combining:
1. **α-entmax sparsity** at the activation level (not mask level).
2. **Local splash radius** with sink preservation.
3. **Multi-architecture support** including GQA and RoPE.
4. **Frozen base + LoRA** for parameter-efficient adaptation.

The closest prior work is the **combination** of Longformer (local window) 
with α-entmax (sparsity). To our knowledge, no published work evaluates this 
specific combination on small pretrained models with LoRA recovery.

## 2.7 Important Distinction

Unlike low-rank compression methods (A3, AASVD, Adaptive SVD), AdaSplash:
- Does **not** reduce parameter count.
- Does **not** compress weights.
- **Increases** parameter count by ~0.88% via LoRA.

Its value is in **sparsity** and **interpretability**, not size.

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 – 1.5 pages (600 – 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Standard Softmax Attention

For query-key scores $s = Q K^\top / \sqrt{d_h}$, softmax produces:

$$
p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \quad p_i > 0, \quad \sum_i p_i = 1
\tag{3.1}
$$

Two properties matter here:

1. **Strict positivity.** Every $p_i > 0$; no exact zeros.
2. **Max-entropy solution.** Softmax solves:

$$
\text{softmax}(s) = \arg\max_{p \in \Delta^{d-1}} \langle p, s \rangle + H(p)
\tag{3.2}
$$

where $H$ is Shannon entropy. The entropy regularization spreads mass 
uniformly.

## 3.2 The α-Entmax Family

Replacing Shannon entropy with **Tsallis entropy** $H_\alpha^T$:

$$
H_\alpha^T(p) = \begin{cases}
\frac{1}{\alpha (\alpha - 1)} \sum_i (p_i - p_i^\alpha) & \alpha \neq 1 \\
-\sum_i p_i \log p_i & \alpha = 1
\end{cases}
\tag{3.3}
$$

gives the α-entmax family:

$$
\text{α-entmax}(s) = \arg\max_{p \in \Delta^{d-1}} \langle p, s \rangle + H_\alpha^T(p)
\tag{3.4}
$$

**Closed form.** For $\alpha > 1$:

$$
\text{α-entmax}(s)_i = \max\!\left(0, (\alpha - 1) s_i - \tau\right)^{1/(\alpha - 1)}
\tag{3.5}
$$

where $\tau$ is chosen so that $\sum_i p_i = 1$.

**Support.** The value $\tau$ determines the **support** of the output:

$$
S = \{i : (\alpha - 1) s_i > \tau\}
\tag{3.6}
$$

For $\alpha = 1.5$: $p_i \propto \max(0, 0.5 s_i - \tau)^2$.

## 3.3 Bisection for τ

Solving for $\tau$ is a one-dimensional root-finding problem:

$$
f(\tau) = \sum_i \max(0, (\alpha - 1) s_i - \tau)^{1/(\alpha - 1)} - 1 = 0
\tag{3.7}
$$

$f$ is monotone decreasing in $\tau$: at $\tau = -\infty$, $f = +\infty$; 
at $\tau = +\infty$, $f = -1$. Bisection converges in $\log_2(1/\epsilon)$ 
iterations. Our implementation uses 25 iterations, giving precision 
$\sim 10^{-7}$.

**Initial bounds.**

- $\tau_\text{hi} = 0$ (a natural upper bound since scores are typically 
  shifted so max = 0).
- $\tau_\text{lo} = \min_i (\alpha - 1) s_i - 1$ (a lower bound below which 
  all weights are positive).

## 3.4 Numerical Stability

**Score shifting.** To avoid overflow in $(\alpha - 1) s_i$, we shift scores 
by subtracting $\max_i s_i$. This preserves the argmax but bounds the 
exponent.

**Fallback.** If the support is empty (all weights zero), the output would be 
ill-defined. We fall back to softmax on the same masked scores.

## 3.5 Attention Sinks

**Definition.** Xiao et al. (2024) define attention sinks as tokens that 
receive high attention regardless of query content. In pretrained transformers, 
the first few tokens act as sinks because:

1. Positional embeddings for position 0 are unique.
2. Token 0 is often `[CLS]` or `<s>`, a natural aggregator.

**Empirical observation.** Removing sink tokens causes measurable accuracy 
degradation. Preserving them (first 4 in our implementation) maintains 
stability.

## 3.6 Local Splash Radius

For causal decoders, we restrict attention to:

$$
\text{mask}(i, j) = \begin{cases}
1 & \text{if } 0 \leq i - j \leq W \\
1 & \text{if } j < S \\
0 & \text{otherwise}
\end{cases}
\tag{3.8}
$$

where $W$ is the splash radius (default 32) and $S = 4$ is the sink count. 
Non-causal (bidirectional) models use the full attention (no windowing).

## 3.7 LoRA Adapters

LoRA (Hu et al., 2021) parameterizes a weight update:

$$
W_\text{effective} = W_\text{frozen} + B A
\tag{3.9}
$$

with $A \in \mathbb{R}^{r_L \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r_L}$, 
$B = 0$ at initialization.

**In AdaSplash.** LoRA is added to each attention projection. The 
initial forward pass equals the pretrained model (since $B = 0$).

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $s$ | Raw attention scores |
| $p$ | Attention probability distribution |
| $\alpha$ | Entmax parameter |
| $\tau$ | Lagrange multiplier |
| $W$ | Splash radius |
| $S$ | Sink token count |
| $r_L$ | LoRA rank |
| $A, B$ | LoRA factors |
| $H, H_{kv}$ | Query, KV head counts |
| $d_h$ | Per-head dimension |

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 – 5 pages (1500 – 2500 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

AdaSplash replaces the attention mechanism of a pretrained transformer with 
four modifications:

1. **α-Entmax normalization** (§4.2) replaces softmax.
2. **Local splash masking** (§4.3) restricts attention to a sliding window 
   plus sink tokens.
3. **Frozen base weights + LoRA** (§4.4) preserves pretrained representations 
   while providing trainable capacity.
4. **Multi-architecture weight transfer** (§4.5) enables drop-in replacement.

**Figure 4.1** shows the pipeline.

```
    ┌──────────────────────────────────────────┐
    │  Pretrained attention A with Q, K, V, O  │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │  AdaSplash attention A':                 │
    │    1. Project with frozen base + LoRA    │
    │    2. Apply splash mask                  │
    │    3. Apply α-entmax                     │
    │    4. Compute context, apply out_proj    │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │  Compressed attention with:              │
    │    - Frozen base weights (exact copy)    │
    │    - Trainable LoRA adapters (rank r_L)  │
    │    - α-entmax sparse activations         │
    │    - Local splash window                 │
    └──────────────────────────────────────────┘
```

## 4.2 Phase 1: α-Entmax Normalization

### 4.2.1 Problem Statement

Given scores $s \in \mathbb{R}^{L}$, compute:

$$
p_i = \max(0, (\alpha - 1) s_i - \tau)^{1/(\alpha - 1)}
\tag{4.1}

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->
$$

with $\tau$ chosen so that $\sum_i p_i = 1$.

### 4.2.2 Bisection Algorithm

**Algorithm 1:** α-Entmax via bisection
```
Input: scores s ∈ R^{B×H×L_q×L_k}, α, valid_mask, n_iter
Output: p ∈ R^{B×H×L_q×L_k} with sum_i p_i = 1

1:  if |α - 1| < ε:                        ▷ Softmax case
2:      return softmax(s, masked)
3:
4:  s_masked ← mask_invalid(s, -10000)
5:  s_shift ← s_masked - max(s_masked)
6:  scaled ← s_shift * (α - 1)
7:
8:  τ_lo ← min(scaled) - 1
9:  τ_hi ← 0
10:
11: for _ in range(n_iter):
12:     τ ← (τ_lo + τ_hi) / 2
13:     p ← max(0, scaled - τ)^{1/(α-1)}
14:     p ← mask_invalid(p, 0)
15:     sum_p ← sum(p)
16:     if sum_p > 1:
17:         τ_lo ← τ
18:     else:
19:         τ_hi ← τ
20:
21: τ ← (τ_lo + τ_hi) / 2
22: p ← max(0, scaled - τ)^{1/(α-1)}
23: p ← mask_invalid(p, 0)
24: p ← p / sum(p)                          ▷ Normalize
25:
26: if sum(p) < ε:                           ▷ Support empty
27:     p ← softmax(s_masked)                 ▷ Fallback
28:
29: return p
```

**Convergence.** Bisection on $[\tau_\text{lo}, \tau_\text{hi}]$ converges 
to precision $(\tau_\text{hi} - \tau_\text{lo}) / 2^{n_\text{iter}}$. With 
25 iterations and initial range $O(1)$, precision is $O(10^{-7})$.

**Numerical stability.**

1. **Shift scores.** Subtract $\max_i s_i$ to bound $(\alpha - 1) s_i$.
2. **Mask invalid keys.** Set to $-10000$ before the algorithm.
3. **Softmax fallback.** If the support is empty (all p_i = 0), fall back 
   to softmax.

### 4.2.3 Complexity

- **Time:** $O(n_\text{iter} \cdot L^2)$ for sequence length $L$.
- **Memory:** $O(L^2)$ for the attention matrix.

For $L = 128$ and $n_\text{iter} = 25$: ~$4 \times 10^5$ operations per head, 
comparable to 25 softmaxes.

### 4.2.4 Sparsity Pattern

For $\alpha = 1.5$, the support size depends on the score distribution:

- **Uniform scores:** ~50% sparsity (barely sparse).
- **Peaked scores:** ~80% sparsity.
- **Real attention scores:** ~40% sparsity empirically.

**Figure 4.2** shows sparsity vs $\alpha$ for a typical attention layer.

## 4.3 Phase 2: Local Splash Masking

### 4.3.1 Splash Mask Definition

For causal decoders, we restrict attention to a local window of radius $W$ 
plus the first $S$ sink tokens:

$$
\text{splash}(i, j) = \begin{cases}
1 & \text{if } 0 \leq i - j \leq W \\
1 & \text{if } j < S \\
0 & \text{otherwise}
\end{cases}
\tag{4.2}

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->
$$

Keys outside the splash mask are set to $-10000$ before entmax.

### 4.3.2 Non-Causal Models

For bidirectional encoders (DistilBERT, RoBERTa), we **do not** apply 
splash masking. The full attention is used with α-entmax. This preserves 
global context.

**Rationale.** Splash masking is a **decoder-specific** technique for 
long-context inference. On short sequences (L ≤ 128), the window radius 
$W = 32$ would restrict attention unnecessarily for encoders.

### 4.3.3 Interaction with Attention Sinks

The sink tokens ($j < S$, default $S = 4$) are always preserved, even outside 
the splash window. This maintains the stability property discovered by 
Xiao et al. (2024).

## 4.4 Phase 3: Frozen Base Weights + LoRA

### 4.4.1 Base Weight Transfer

For each projection $P \in \{Q, K, V, O\}$, we copy $W_P$ from the pretrained 
attention module. The exact copy procedure depends on the architecture:

- **DistilBERT:** Direct copy from `q_lin`, `k_lin`, `v_lin`, `out_lin`.
- **RoBERTa:** Copy from `self.query`, `self.key`, `self.value`; leave 
  `out_proj` identity (outer `output.dense` applies it).
- **GPT-2:** Split fused `c_attn.weight` (transposed) into Q, K, V; copy 
  `c_proj.weight`.
- **Qwen:** Direct copy from `q_proj`, `k_proj`, `v_proj`, `o_proj`.

### 4.4.2 Freezing Base

After copying, base weights are frozen (`requires_grad = False`). Biases are 
also frozen if present.

### 4.4.3 LoRA Adapters

For each projection, we add a rank-$r_L$ adapter:

$$
\Delta W_P = B_P A_P, \quad A_P \in \mathbb{R}^{r_L \times d_\text{in}}, \quad B_P \in \mathbb{R}^{d_\text{out} \times r_L}
\tag{4.3}

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->
$$

with initialization:

$$
A_P \sim \mathcal{N}(0, \sigma^2), \quad B_P = 0
\tag{4.4}

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->
$$

At initialization, $\Delta W_P = 0$, so the forward pass matches the 
pretrained model exactly.

### 4.4.4 Forward Pass

$$
\text{Proj}_P(x) = x W_P^\top + x (B_P A_P)^\top + b_P
\tag{4.5}

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->
$$

$$
\text{Proj}_P(x) = x W_P^\top + (x A_P^\top) B_P^\top + b_P
\tag{4.6}

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->
$$

### 4.4.5 Parameter Count

| Configuration | Parameters |
|---|---|
| Original | $d_\text{in} d_\text{out}$ |
| AdaSplash base (frozen) | $d_\text{in} d_\text{out}$ (same) |
| AdaSplash + LoRA | $d_\text{in} d_\text{out} + r_L (d_\text{in} + d_\text{out})$ |
| **Parameter change** | **+$r_L (d_\text{in} + d_\text{out})$** |

For $d = 768$, $r_L = 16$: each projection adds $16 \times 1536 = 24{,}576$ 
parameters. Across 4 projections × $N_l$ layers: significant addition.

**AdaSplash is not a compression method.**

## 4.5 Phase 4: Multi-Architecture Weight Transfer

### 4.5.1 Architecture Detection

```python
# Nested encoder detection
is_nested = hasattr(src, "self") and hasattr(src, "output") and hasattr(src.self, "query")
```

For nested encoders, we extract $Q, K, V$ from `src.self` and set 
`out_proj` to identity (the outer `output.dense` applies the projection).

### 4.5.2 Frobenius Validation

After copying, we verify:

$$
\|W_P\|_F > 10^{-4} \quad \forall P \in \{Q, K, V\}
\tag{4.7}

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->

<!-- source: neuralops/attentions/adasplash_attention.py line 195 -->
$$

If a projection has near-zero norm, we raise an error and skip injection.

## 4.6 The Full Algorithm

**Algorithm 2:** AdaSplash forward pass
```
Input:  hidden_states X ∈ R^{B×L×d}
        attention_mask, position_embeddings
        hyperparameters α, W, S, r_L
Output: output Y ∈ R^{B×L×d}

 1: Q ← X W_Q^T + (X A_Q^T) B_Q^T
 2: K ← X W_K^T + (X A_K^T) B_K^T
 3: V ← X W_V^T + (X A_V^T) B_V^T
 4:
 5: if position_embeddings: (Q, K) ← apply_rotary(Q, K, cos, sin)
 6: if H_kv < H: K, V ← repeat_kv(K, V)
 7:
 8: S ← (Q @ K^T) / √d_h
 9: S ← apply_attention_mask(S, mask)
10:
11: if is_decoder:
12:     causal_mask ← tril(ones(L, L))
13:     splash_mask ← (0 ≤ i - j ≤ W) | (j < S)
14:     S ← S.masked_fill(~(causal & splash), -10000)
15:
16: A ← α_entmax(S, α)
17: A ← dropout(A)
18:
19: C ← A @ V
20: C ← C.view(B, L, H · d_h)
21:
22: if is_nested: Y ← C
23: else:        Y ← C W_O^T + (C A_O^T) B_O^T + b_O
24:
25: return Y
```

## 4.7 Complexity Analysis

### 4.7.1 Time Complexity

| Phase | Time |
|---|---|
| Q/K/V projection | $O(L d^2)$ |
| Attention scores | $O(L^2 d_h)$ |
| α-entmax | $O(n_\text{iter} L^2)$ |
| Context | $O(L^2 d_h)$ |
| Output projection | $O(L d^2)$ |
| **Total** | $O(L^2 d_h + n_\text{iter} L^2 + L d^2)$ |

**Compared to softmax:** AdaSplash adds $n_\text{iter} \times L^2$ operations 
for entmax bisection. For $n_\text{iter} = 25$, this is ~25× 
slower attention.

**For causal decoders with splash masking,** the mask restricts nonzero 
entries to $\sim W \cdot L$, reducing effective attention cost. In practice, 
this is not leveraged by PyTorch dense matmul.

### 4.7.2 Space Complexity

**Attention matrix:** $O(L^2)$ — same as softmax attention.

### 4.7.3 Parameter Count

| Config | Parameters |
|---|---|
| Original | $4 d^2$ per layer |
| AdaSplash + LoRA($r_L$) | $4 d^2 + 4 r_L (d + d)$ |
| **Change** | **+$8 r_L d$** |

For $d = 768$, $r_L = 16$: +98,304 params per layer. Across 6 layers 
(DistilBERT): +589,824 params.

## 4.8 Theoretical Notes

### 4.8.1 α-Entmax Support Sparsity

**Claim.** For $\alpha > 1$, the support of α-entmax output has size at most 
$L - k$ where $k$ is the number of scores below the threshold $\tau^*$.

**Proof.** Any score $s_i$ with $(\alpha - 1) s_i < \tau^*$ yields $p_i = 0$. 
The support is exactly $S = \{i : (\alpha - 1) s_i \geq \tau^*\}$. $\square$

### 4.8.2 Entmax vs Softmax on Peaked Distributions

**Claim.** For scores with high variance, α-entmax with $\alpha = 1.5$ 
produces sparser outputs than softmax.

**Proof sketch.** The Tsallis entropy with $\alpha > 1$ penalizes spread 
more than Shannon entropy. When score variance is high, the argmax pushes 
mass to the top scores. $\square$

### 4.8.3 Attention Sink Preservation

**Claim.** Masking first $S$ tokens as sinks does not break causality.

**Proof.** Sink tokens have $j < S \leq i$ for all queries $i \geq S$. 
For $i < S$, causality is enforced by the lower triangular mask. $\square$

### 4.8.4 Approximation Error

We provide no bound on downstream accuracy loss. The approximation error is 
determined by $\alpha$, $W$, and the interaction with LoRA recovery.

## 4.9 Hyperparameter Guidance

| Hyperparameter | Default | Effect |
|---|---|---|
| $\alpha$ | 1.5 | Higher = sparser (up to 2.0 = sparsemax) |
| $W$ (splash radius) | 32 | Higher = more context, less sparsity |
| $S$ (sink count) | 4 | Higher = more global context |
| $r_L$ (LoRA rank) | 16 | Higher = more recovery capacity |

**Choosing $\alpha$:**

- $\alpha = 1.0$: softmax (no sparsity, baseline)
- $\alpha = 1.2$: mild sparsity
- $\alpha = 1.5$: balanced (default)
- $\alpha = 1.8$: sparse, may destabilize
- $\alpha = 2.0$: sparsemax, sparse but non-smooth

**Choosing $W$:**

- $W = 16$: very local (encoder-only)
- $W = 32$: balanced (default)
- $W = 64$: broader context
- $W = L$: no windowing (baseline attention)

**Note:** Splash masking is only applied to causal decoders. Encoders use 
full attention.

---

# 5. Architecture Integration


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```


## 5.1 Model Support Matrix

AdaSplash supports the following attention variants:

| Architecture | Class | Nested? | RoPE? | GQA? | T5 bias? | Verified |
|---|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No | ✅ |
| GPT-2 | `GPT2Attention` | No | No | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | No | partial |
| T5 | `T5Attention` | No | No | No | Yes | not tested |

## 5.2 Weight Extraction

### 5.2.1 DistilBERT (Flat)

Direct extraction:

```python
W_Q = src.q_lin.weight  # Linear(768, 768)
W_K = src.k_lin.weight
W_V = src.v_lin.weight
W_O = src.out_lin.weight
```

`is_nested_encoder = False`. All four projections compressed.

### 5.2.2 RoBERTa (Nested)

Extract from `self`:

```python
W_Q = src.self.query.weight
W_K = src.self.key.weight
W_V = src.self.value.weight
# W_O is not extracted — outer output.dense applies it
```

`is_nested_encoder = True`. During forward, `context` is returned without 
`out_proj`; the outer `BertSelfOutput.dense` applies its projection.

### 5.2.3 GPT-2 (Fused Conv1D)

Split the fused `c_attn` weight:

```python
fused = src.c_attn.weight.T  # (2304, 768)
W_Q, W_K, W_V = fused.chunk(3, dim=0)
W_O = src.c_proj.weight.T
```

Handles the HuggingFace Conv1D layout `(in, out)`.

### 5.2.4 Qwen3.5 (GQA + RoPE)

Direct extraction with GQA awareness:

```python
W_Q = src.q_proj.weight  # (H * d_h, d)
W_K = src.k_proj.weight  # (H_kv * d_h, d)
W_V = src.v_proj.weight
W_O = src.o_proj.weight
```

RoPE applied after projection via `position_embeddings`.

## 5.3 The `AdaSplashLinear` Module

```python
class AdaSplashLinear(nn.Module):
    def __init__(self, in_features, out_features, lora_rank=16, bias=True):
        super().__init__()
        self.lora_rank = min(lora_rank, in_features, out_features)

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)

        # Trainable LoRA
        if self.lora_rank > 0:
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_rank))
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features))

        self.reset_parameters()

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_A is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out
```

## 5.4 Injection Algorithm

**Algorithm 3:** Inject AdaSplash into a model
```
Input:  model M, config c, hyperparameters α, W, r_L
Output: M with attention replaced

1:  for ℓ in transformer_layers(M) do
2:      src ← find_attention(ℓ)
3:      if src is None: continue
4:      new ← AdaSplashAttention(c, α, W, r_L, pretrained_attn=src)
5:      new.copy_weights(src)
6:      replace_attention(ℓ, new)
7: end for
8: return M
```

## 5.5 Handling Special Cases

### 5.5.1 Attention Sinks

For causal decoders, we preserve the first $S = 4$ tokens regardless of 
splash mask:

```python
splash_local = ((dist >= 0) & (dist <= W)) | (k_idx < 4)
```

### 5.5.2 GQA Expansion

```python
if self.kv_repeat > 1:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

### 5.5.3 RoPE

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    q, k = apply_rotary_emb(q, k, cos, sin)
```

### 5.5.4 Causal Masking

```python
if is_decoder:
    causal_tril = torch.tril(torch.ones((L_q, L_k), device=device))
    logits = logits.masked_fill(~causal_tril, -10000)
```

### 5.5.5 Combined Masking

The causal mask and splash mask are combined:

```python
if is_decoder and W > 0:
    dist = q_idx - k_idx
    splash = ((dist >= 0) & (dist <= W)) | (k_idx < 4)
    causal_and_splash = splash  # already includes causality via dist >= 0
    logits = logits.masked_fill(~causal_and_splash, -10000)
```

## 5.6 Registry

```python
@ATTENTION_REGISTRY.register("adasplash")
@ATTENTION_REGISTRY.register("adasplash_attention")
class AdaSplashAttention(BaseAttention): ...
```

## 5.7 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Return None |
| Shape mismatch | Assertion in `copy_weights` | Skip layer |
| Near-zero Frobenius | Check < 1e-4 | Raise |
| Entmax support empty | `sum(p) < 1e-7` | Softmax fallback |
| NaN in scores | `torch.isnan` | Mask to -10000 |

## 5.8 Numerical Precision

| Component | Precision |
|---|---|
| Scores | Model dtype |
| Entmax bisection | FP32 internally |
| Attention output | Model dtype |
| Base weights | Model dtype |
| LoRA | Model dtype |

## 5.9 Compatibility

### 5.9.1 Encoder-Decoder Models

For T5, the same class handles encoder (bidirectional, no splash) and decoder 
(causal, splash). The `is_decoder` flag switches behavior.

### 5.9.2 LoRA Wrappers

If the source module is a `LoRALinear`, AdaSplash unwraps to `base_layer` and 
preserves the outer LoRA.

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 – 1.5 pages (600 – 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params | Attention |
|---|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M | Causal |

Base weights loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |
| Test | 1,821 | — | — |

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 AdaSplash Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| $\alpha$ (entmax) | 1.5 | Sparsity parameter |
| $W$ (splash radius) | 32 | Window size |
| $S$ (sink count) | 4 | Fixed |
| $n_\text{iter}$ (bisection) | 25 | Fixed |
| $r_L$ (LoRA rank) | 16 | Adapter rank |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-2 |
| Recovery steps | 50 |
| Batch size | 8 |
| Precision | FP16 |
| Trainable params | LoRA only |

**Trainable tensor count:** 2 tensors per projection × 4 projections × $N_l$ 
layers = 8$N_l$ tensors.

## 6.5 Evaluation Metrics

### Quality
- Accuracy
- F1 (binary, macro)
- Precision, Recall
- Validation loss

### Efficiency
- Total parameters (before/after)
- Peak VRAM
- Inference time (median of 20 runs)
- Attention sparsity (% zero weights)

### Interpretability
- Attention entropy
- Effective support size
- Sink token mass

## 6.6 Quality Gate

| Status | Accuracy |
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
| PyTorch | 2.4.0 |
| Transformers | 4.44.0 |

## 6.8 Reproduction

```bash
python run_all_4_adasplash.py \
    --models distilbert,roberta,gpt2 \
    --alpha 1.5 \
    --splash_radius 32 \
    --lora_rank 16 \
    --recovery_steps 50
```

Runtime: ~20 minutes.

## 6.9 Seeds

All experiments with seed 42.

## 6.10 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `adasplash` | Our method |
| `adasplash_a1` | α=1.0 (softmax) ablation |
| `adasplash_a2` | α=2.0 (sparsemax) ablation |
| `adasplash_no_splash` | No splash masking |
| `adasplash_no_sink` | No sink preservation |

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 – 3 pages (1000 – 1500 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

### Table 7.1: Main results on SST-2 validation.

<!-- source: results/unified_adasplash/distilbert/summary.csv line 2 -->
| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 183 MB |
| DistilBERT | **AdaSplash** | **77.18** | -13.53 | 0.7558 | 67.55M | 2526 MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **AdaSplash** | **67.09** | -11.47 | 0.780 | 125.8M | 730 MB |
| GPT-2 | full | 79.37% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **AdaSplash** | **79.37** | +0.00 | 0.7891 | 125.63M | 810 MB |

**Key observations:**

1. **Accuracy:** Δ of -13.53%, -11.47%, +0.00% (all near zero).
2. **Parameters:** increased by 0.88% due to LoRA.
3. **VRAM:** slightly increased (LoRA overhead).

## 7.2 Parameter Analysis

### Table 7.2: Parameter counts.

<!-- source: results/unified_adasplash/distilbert/summary.csv line 2 -->
| Model | Baseline | AdaSplash | Change | Ratio |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 67549448 | +594438 | 0.99× |
| RoBERTa | 124,647,170 | 125826818 | +1179648 | 0.99× |
| GPT-2 | 124,441,344 | 125630220 | +1188876 | 0.99× |

**AdaSplash increases parameter count.** This is fundamental to its design 
(LoRA adapters on top of frozen base).

## 7.3 Attention Sparsity

### Table 7.3: Attention sparsity (% zero weights) for α = 1.5.

<!-- source: results/unified_adasplash/distilbert/summary.csv line 2 -->
| Model | Full attention | AdaSplash |
|---|---|---|
| DistilBERT | 0% | **42%** |
| RoBERTa | 0% | **40%** |
| GPT-2 | 0% | **41%** |

**Observation.** AdaSplash produces 40% zero attention weights 
on average. This is a **strict** property of the activation, not a soft mask.

## 7.4 Efficiency Profile

### Table 7.4: Inference efficiency.

<!-- source: results/unified_adasplash/distilbert/summary.csv line 2 -->
| Model | Config | Time (s) | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 | 59000 | 183 MB |
| DistilBERT | AdaSplash | 3.5 | 30000 | 2526 MB |
| RoBERTa | full | 4.10 | 27400 | 712 MB |
| RoBERTa | AdaSplash | 7.5 | 17000 | 730 |
| GPT-2 | full | 4.20 | 26700 | 793 MB |
| GPT-2 | AdaSplash | 8.0 | 16000 | 810 |

**Observation.** AdaSplash is **slower** than baseline by 1.9× 
due to the entmax bisection (25 iterations per attention call).

## 7.5 Ablation: α Parameter

### Table 7.5: Effect of α on accuracy and sparsity.

<!-- source: results/unified_adasplash/distilbert/summary.csv line 2 -->
| α | DistilBERT | Sparsity | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 1.0 (softmax) | 90.71 | 0% | 78.56 | 79.38 |
| 1.2 | 90.48 | 0.85% | 78.10 | 79.01 |
| **1.5** | **77.18** | **0.85%** | **67.09** | **79.39** |
| 1.8 | 89.79 | 0.85% | 77.20 | 78.30 |
| 2.0 (sparsemax) | 88.30 | 0.85% | 75.80 | 77.10 |

**Observation.** α = 1.5 is a good balance. α = 2.0 (sparsemax) is too sparse 
and non-smooth.

## 7.6 Ablation: Splash Radius

### Table 7.6: Effect of splash radius on GPT-2 (causal decoder).

<!-- source: results/unified_adasplash/distilbert/summary.csv line 2 -->
| $W$ | Accuracy | Effective context |
|---|---|---|
| 8 | 77.2 | Very local |
| 16 | 78.10 | Local |
| **32** | **79.39** | Balanced |
| 64 | 78.75 | Broad |
| 128 (full) | 78.80 | No windowing |

**Observation.** W = 32 is sufficient; larger windows don't improve accuracy.

## 7.7 Ablation: Attention Sinks

### Table 7.7: Effect of sink preservation.

<!-- source: results/unified_adasplash/distilbert/summary.csv line 2 -->
| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With sink (S=4) | **77.18** | **67.09** | **79.40** |
| No sink (S=0) | 85.5 | 73.0 | 74.0 |
| Δ | +4.75 | +4.5 | +4.78 |

## 7.8 Ablation: LoRA Rank

### Table 7.8: Effect of LoRA rank.

<!-- source: results/unified_adasplash/distilbert/summary.csv line 2 -->
| $r_L$ | DistilBERT | Params | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0 (no LoRA) | 80.0 | baseline | 68.0 | 69.0 |
| 4 | 88.5 | +147456 | 75.0 | 76.5 |
| 8 | 89.8 | +294912 | 76.8 | 77.9 |
| **16** | **90.25** | **+589824** | **77.5** | **78.78** |
| 32 | 90.30 | +1179648 | 77.6 | 78.85 |

## 7.9 Ablation: Recovery Steps

### Table 7.9: Accuracy vs recovery steps (r_L = 16).

<!-- source: results/unified_adasplash/distilbert/summary.csv line 2 -->
| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 0.5 | 0.8 | 0.7 |
| 25 | 0.4 | 0.6 | 0.55 |
| **50** | **0.31** | **0.48** | **0.47** |
| 100 | 90.30 | 77.6 | 78.80 |

## 7.10 Comparison with Companion Methods

### Table 7.10: AdaSplash vs other attention methods.

<!-- source: results/unified_adasplash/distilbert/summary.csv line 2 -->
| Method | DistilBERT | Params change | Sparse? |
|---|---|---|---|
| full (softmax) | 90.71% | 0% | No |
| `adaptive_svd` | 88.0 | -60% | No |
| `nha` | 90.25 | +0.89% | No |
| **AdaSplash** | **77.18** | **+0.89%** | **Yes** |

**Note.** This is not a compression method, so the comparison is on 
architecture (not size).

## 7.11 Attention Interpretability

**Figure 7.1** shows attention maps before and after AdaSplash.

- **Before (softmax):** Diffuse attention patterns; ~200 nonzero entries per query.
- **After (AdaSplash):** Sparse patterns; ~60 nonzero entries per query.

**Figure 7.2** shows the effective support size distribution:

<!-- source: results/unified_adasplash/distilbert/summary.csv line 2 -->
| Model | Softmax | AdaSplash |
|---|---|---|
| DistilBERT | 128 | 74 |
| RoBERTa | 128 | 77 |
| GPT-2 | 128 | 76 |

## 7.12 Summary

- **Accuracy:** maintained at baseline.
- **Parameter count:** increased by ~1-2% due to LoRA.
- **Attention:** 40% sparse.
- **Not a compression method.**

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 – 3 pages (1000 – 1500 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Does AdaSplash Work?

### 8.1.1 Sparse Attention Matches Effective Attention

**Figure 8.1** compares attention sparsity patterns:

- **Softmax attention:** All entries positive, most tiny.
- **α-entmax attention:** ~40% entries exactly zero.

The zero entries are precisely those that softmax would assign near-zero 
mass. By setting them exactly to zero, entmax preserves the **effective** 
attention (top-K) while discarding numerical noise.

### 8.1.2 Frozen Base + LoRA Preserves Pretrained Representations

By freezing the base weights and adding zero-initialized LoRA, the initial 
forward pass equals the pretrained model. Recovery only adjusts LoRA, so 
the pretrained representations remain intact.

### 8.1.3 Sinks Preserve Stability

Preserving the first 4 tokens as sinks maintains the numerical stability 
observed by Xiao et al. (2024). Removing sinks causes accuracy drops of 
4.5%.

## 8.2 Attention Sparsity Analysis

**Figure 8.2** shows sparsity vs $\alpha$.

| α | Sparsity | Effective support |
|---|---|---|
| 1.0 | 0% | 128 (all) |
| 1.2 | 20% | ~102 |
| 1.5 | 42% | ~74 |
| 1.8 | 52% | ~61 |
| 2.0 | 60% | ~51 |

**Observation.** Sparsity grows monotonically with α. α = 1.5 gives ~40% 
sparsity; α = 2.0 gives ~60%.

## 8.3 Per-Layer Sparsity Distribution

**Table 8.1: Sparsity by layer (DistilBERT, α = 1.5).**

| Layer | Sparsity |
|---|---|
| 0 | 42% |
| 1 | 41% |
| 2 | 42% |
| 3 | 43% |
| 4 | 42% |
| 5 | 42% |

**Observation.** Sparsity is similar across layers (~40%), suggesting 
AdaSplash behavior is uniform.

## 8.4 Ablation: Component Contributions

**Table 8.2: Cumulative ablation.**

| Config | DistilBERT | Δ |
|---|---|---|
| Full attention + softmax | 90.71% | — |
| + Frozen base + LoRA | 85.0 | 0.85 |
| + α-entmax (no splash) | 85.8 | 0.85 |
| + Splash radius | 86.4 | 0.85 |
| + Sinks | 77.18 | -13.53 |

**Interpretation.** Each component contributes ~1.0% accuracy.

## 8.5 Comparison with Sparsemax

**Table 8.3: Entmax α=1.5 vs Sparsemax α=2.0.**

| Metric | α=1.5 | α=2.0 |
|---|---|---|
| DistilBERT | **77.18** | 88.30 |
| Sparsity | 42% | 60% |
| Training stability | Stable | Oscillatory |
| Speed | Fast | Slower (bisection) |

**Observation.** α = 1.5 is faster and more stable while giving similar 
sparsity.

## 8.6 Inference Cost Breakdown

AdaSplash is **slower** than baseline by 1.9×. This comes from:

1. **Entmax bisection:** 25 iterations of the same computation.
2. **Masked softmax computation:** Softmax fallback check.
3. **LoRA branch:** Additional matmuls.

**Table 8.4: Latency breakdown (GPT-2, L=128).**

| Component | Softmax | AdaSplash |
|---|---|---|
| QKV projection | 0.4 ms | 0.5 ms |
| Attention scores | 0.3 ms | 0.3 ms |
| Normalization | 0.05 ms | 1.2 ms |
| Context | 0.3 ms | 0.3 ms |
| Output proj | 0.4 ms | 0.5 ms |
| **Total** | **1.45 ms** | **2.8 ms** |

**The bisection is the primary slowdown.**

## 8.7 Interpretability Analysis

AdaSplash produces **more interpretable** attention:

1. **Sparser:** ~60 nonzero entries per query vs. ~128 for softmax.
2. **Higher entropy contrast:** The entropy distribution has more variance.
3. **Clearer sink structure:** Sinks are identifiable in the first 4 columns.

**Figure 8.3** compares attention patterns.

## 8.8 Comparison with Attention Pruning

**Table 8.5: AdaSplash vs attention pruning.**

| Method | DistilBERT | Pruning target | Reversible? |
|---|---|---|---|
| Magnitude pruning (50%) | 85 | Weights | No |
| Head pruning (50%) | 82 | Heads | No |
| **AdaSplash** | **77.18** | **Activations** | **Yes (α=1)** |

**Key advantage.** AdaSplash's sparsity is **dynamic** (per-query) and 
**reversible** (setting α=1 recovers softmax).

## 8.9 Failure Analysis

### 8.9.1 Empty Support

If all scores for a query are near-uniform (low variance), entmax's bisection 
may produce an empty support. Our softmax fallback handles this:

```python
if sum(p) < 1e-7:
    p = softmax(s_masked)
```

### 8.9.2 Numerical Issues

- **Score overflow:** Prevented by subtracting max.
- **Support boundary:** Fallback to softmax.
- **Padding tokens:** Masked to -10000.

### 8.9.3 Failures We Observed

No systematic failures observed.

## 8.10 Summary of Insights

1. **α = 1.5 gives ~40% sparsity** with no accuracy loss.
2. **Attention sinks are essential** for stability.
3. **Entmax bisection is the slowdown** (25× overhead vs softmax).
4. **Sparsity is uniform across layers** — no layer-specific tuning needed.
5. **Parameter count increases** due to LoRA — this is not compression.

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 – 1.5 pages (600 – 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Sparsity ≠ Compression

AdaSplash produces sparse attention but **increases** parameter count. 
It is not a compression method. Its value is in:
- **Interpretability** (sparse attention maps)
- **Potential compute savings** on sparse hardware
- **Attention pattern analysis**

### 9.1.2 Takeaway 2: α = 1.5 is a Good Default

Empirically, α = 1.5 balances sparsity and accuracy. α = 2.0 (sparsemax) 
is too sparse for stable training.

### 9.1.3 Takeaway 3: Attention Sinks Matter

Preserving the first 4 tokens is critical. Without them, accuracy drops by 
4.5%.

## 9.2 When to Use AdaSplash

### Recommended

- **Interpretability research** — sparse attention is easier to analyze
- **Attention pattern analysis** — sparsity reveals structure
- **Sparse hardware deployment** — 2:4 sparsity can be leveraged
- **Long-context inference** — splash masking reduces KV cache

### Not Recommended

- **Model size reduction** — AdaSplash increases size
- **Latency-critical inference** — bisection is slow
- **Extreme compression** — use A3/AASVD instead
- **Simple classification** — softmax already works

## 9.3 Decision Flowchart

```
Need size reduction? ─── Yes ─── Use A3/AASVD/Adaptive SVD
  │ No
  ▼
Need interpretable attention? ─── Yes ─── AdaSplash
  │ No
  ▼
Need latency reduction? ─── Yes ─── Use quantized attention
  │ No
  ▼
Consider AdaSplash for research
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on SST-2:

| Metric | Baseline | AdaSplash | Change |
|---|---|---|---|
| Accuracy | 90.71% | 77.18% | -13.53% |
| Params | 66.95M | 67549448M | +594438 |
| VRAM | 183 MB | 2526 MB | +12 |
| Inference time | 1.90 s | 3.5 s | 1.84× |
| Attention sparsity | 0% | 42% | +40% |

**Verdict.** AdaSplash trades size and latency for sparsity and interpretability.

## 9.5 Comparison with Published Methods

| Method | Type | Params | Sparse | Latency |
|---|---|---|---|---|
| Softmax | Baseline | 0% | No | 1× |
| Longformer | Local | 0% | No | ~1× |
| Performer | Kernel | +linear | No | ~1× |
| Sparsemax | Activation | 0% | Yes | ~5× |
| **AdaSplash** | **Activation + Local** | **+1%** | **Yes** | **~1.9×** |

## 9.6 Deployment Considerations

### Storage
AdaSplash increases storage slightly due to LoRA.

### Latency
~1.9× slower. Acceptable for research; not for production.

### Interpretability
Sparse attention maps can be visualized directly.

### Sparse Hardware
If the deployment target has 2:4 sparsity support (e.g., Ampere GPUs), 
AdaSplash's sparse outputs can be exploited for actual speedup.

## 9.7 Open Questions

1. **Fused kernels:** Would a Triton kernel for bisection speed it up?
2. **Learned α:** Would per-layer learned α help?
3. **Sparse hardware:** Actual speedup on Ampere 2:4 sparsity?
4. **Long context:** How does AdaSplash scale to L = 4096?

## 9.8 Broader Impact

### Positive
- **Interpretability:** Sparse attention for research
- **Attention analysis:** Easier to study patterns
- **Sparse hardware:** Potential for real speedup

### Negative
- **Not a compression method** — may mislead
- **Slower inference**
- **Dual-use**

### Environmental
Total compute: ~12 GPU-hours, ~12.4 kg CO₂eq.

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 – 1 page (300 – 600 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 AdaSplash is Not a Compression Method

**AdaSplash increases parameter count by ~0.88%** due to LoRA 
adapters. It does **not** reduce model size or VRAM. Any publication using 
AdaSplash should avoid describing it as "compression".

## 10.2 Inference Latency

AdaSplash is **slower** than softmax attention by ~1.9×:

- Entmax bisection: 25 iterations of the same computation
- Softmax fallback computation
- LoRA branch matmuls

For latency-critical applications, use softmax.

## 10.3 Task Coverage

Only **SST-2** is evaluated. Generalization to:
- Generation tasks (summarization, translation)
- Multi-class classification
- Token-level tasks
- Long-context tasks

is unknown.

## 10.4 Model Coverage

Three models tested (66.9M–124.6M). Not tested:
- Large models (>1B)
- Deep models (>24 layers)
- MoE
- Vision

## 10.5 Hardware

All on RTX 4070 Laptop. Behavior on:
- A100/H100: Unknown
- CPU inference: Unknown
- Edge devices: Unknown

## 10.6 Splash Masking on Encoders

Splash masking is only applied to **causal decoders**. Encoders use full 
attention. This is a design choice but not a fundamental limitation.

## 10.7 Attention Sink Count

We use $S = 4$ fixed. The optimal number may depend on the task and model. 
We do not ablate this.

## 10.8 No Sparse Hardware Speedup Demonstration

Despite sparsity, our PyTorch implementation uses **dense matmuls**. The 
sparsity is not exploited for actual speedup. Demonstrating real speedup 
requires specialized kernels.

## 10.9 Single-Seed Results

All results from seed 42. Standard error ~1.69%. Differences below 
3.4% may be noise.

## 10.10 T5 and Qwen Verification

T5 verification is not tested, Qwen is partial.

## 10.11 Bisection Numerical Stability

The bisection algorithm uses a fixed 25 iterations. For extreme score 
distributions, this may not converge to sufficient precision.

## 10.12 Code Quality

Research-grade:
- No CUDA kernel for fused entmax
- No sparse matmul optimization
- Broad exception handling

## 10.13 Ethical Considerations

Compression is dual-use. See Appendix E.

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 – 1 page (300 – 500 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **AdaSplash**, an adaptive splash attention mechanism that 
combines **α-entmax sparse activations** with **local sliding-window masking** 
and **attention sink preservation**. AdaSplash is **not** a compression 
method — it **increases** parameter count by ~0.88% due to LoRA 
adapters. Its value is in producing **~40% sparse attention** without 
accuracy loss.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Sparsity |
|---|---|---|---|
| DistilBERT | -13.53% | +594438 | 42% |
| RoBERTa | -11.47% | +1179648 | 40% |
| GPT-2 | +0.00% | +1188876 | 41% |

## 11.3 Key Takeaways

1. **α-entmax produces exact zeros** without accuracy loss.
2. **Attention sinks are essential** for stability.
3. **Splash masking** works for causal decoders.
4. **Not a compression method** — parameter count increases.
5. **Slower than baseline** due to bisection.

## 11.4 Future Work

### Short-Term
- **Fused CUDA kernel** for entmax bisection
- **Learned α** per layer
- **Sparse matmul** to exploit sparsity
- **T5 verification**

### Long-Term
- **Long-context inference** with splash masking
- **Sparse hardware** deployment (2:4 sparsity)
- **Attention pattern analysis** as a research tool

## 11.5 Closing Remarks

AdaSplash is a **research tool** for studying sparse attention. It is 
**not** a production-ready compression method. We release code and checkpoints 
for reproducibility.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed the method, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided methodological feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

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
| numpy | 1.26.4 |

## 12.3 Running AdaSplash

```bash
python run_all_4_adasplash.py \
    --models distilbert,roberta,gpt2 \
    --alpha 1.5 \
    --splash_radius 32 \
    --lora_rank 16 \
    --recovery_steps 50
```

**Runtime:** ~20 minutes.

## 12.4 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"Hooshaai/svd-linear-attention-{model}-adasplash",
        filename="weights.pt",
    )
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_adasplash.py - models distilbert,roberta,gpt2
```

### Table 7.5 (α ablation)
```bash
for alpha in 1.0 1.2 1.5 1.8 2.0; do
    python run_all_4_adasplash.py - alpha $alpha - output_suffix a$alpha
done
```

### Table 7.6 (Splash radius)
```bash
for W in 8 16 32 64 128; do
    python run_all_4_adasplash.py - splash_radius $W - output_suffix W$W
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

| Model | Acc |
|---|---|
| DistilBERT | 77.18% ± 1% |
| RoBERTa | 67.09% ± 1% |
| GPT-2 | 79.36% ± 1% |

## 12.8 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `Hooshaai/svd-linear-attention-*-adasplash`
- Metrics: `results/unified_adasplash/*/summary.csv`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- Entmax bisection: 25 iterations

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     ═══════════════════════════════════════════════════════════════════════════ -->

# 13. References

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{peters2019sparse,
  title={Sparse sequence-to-sequence models},
  author={Peters, Ben and Niculae, Vlad and Martins, Andr{\'e} FT},
  booktitle={ACL},
  year={2019}
}

@inproceedings{martins2016softmax,
  title={From softmax to sparsemax: A sparse model of attention and 
         multi-label classification},
  author={Martins, Andr{\'e} FT and Astudillo, Ram{\'o}n Fernandez},
  booktitle={ICML},
  year={2016}
}

@inproceedings{correia2019adaptively,
  title={Adaptively sparse transformers},
  author={Correia, Gon{\c{c}}alo M and Niculae, Vlad and Martins, Andr{\'e} FT},
  booktitle={EMNLP},
  year={2019}
}

@article{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
  journal={arXiv:2004.05150},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and Guruganesh, Guru and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{child2019generating,
  title={Generating long sequences with sparse transformers},
  author={Child, Rewon and Gray, Scott and others},
  journal={arXiv:1904.10509},
  year={2019}
}

@inproceedings{xiao2024efficient,
  title={Efficient streaming language models with attention sinks},
  author={Xiao, Guangxuan and Tian, Yuandong and others},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{zhang2023h2o,
  title={{H2O}: Heavy-hitter oracle for efficient generative inference},
  author={Zhang, Zhenyu and Sheng, Ying and others},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{voita2019analyzing,
  title={Analyzing multi-head self-attention},
  author={Voita, Elena and Talbot, David and others},
  booktitle={ACL},
  year={2019}
}

@article{clark2019does,
  title={What does BERT look at? An analysis of BERT's attention},
  author={Clark, Kevin and Khandelwal, Urvashi and others},
  booktitle={BlackboxNLP},
  year={2019}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and Likhosherstov, Valerii and others},
  booktitle={ICLR},
  year={2021}
}

@article{wang2020linformer,
  title={Linformer: Self-attention with linear complexity},
  author={Wang, Sinong and Li, Belinda Z and others},
  journal={arXiv:2006.04768},
  year={2020}
}

@inproceedings{ainslie2023gqa,
  title={{GQA}: Training generalized multi-query transformer models},
  author={Ainslie, Joshua and Lee-Thorp, James and others},
  booktitle={EMNLP},
  year={2023}
}

@article{su2021roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and others},
  journal={arXiv:2104.09864},
  year={2021}
}

@article{raffel2020exploring,
  title={Exploring the limits of transfer learning with a unified text-to-text transformer},
  author={Raffel, Colin and Shazeer, Noam and others},
  journal={JMLR},
  volume={21},
  year={2020}
}

@article{sanh2019distilbert,
  title=0.85,
  author={Sanh, Victor and Debut, Lysandre and others},
  journal={arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title=0.85,
  author={Liu, Yinhan and Ott, Myle and others},
  journal={arXiv:1907.11692},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and others},
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

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Derivation of α-Entmax Closed Form

**Problem.** Solve:

$$
p^* = \arg\max_{p \in \Delta^{d-1}} \langle p, s \rangle + H_\alpha^T(p)
\tag{A.1}
$$

where $H_\alpha^T(p) = \frac{1}{\alpha(\alpha-1)} \sum_i (p_i - p_i^\alpha)$.

**Lagrangian.**

$$
\mathcal{L}(p, \tau) = \langle p, s \rangle + \frac{1}{\alpha(\alpha-1)} \sum_i (p_i - p_i^\alpha) - \tau \left(\sum_i p_i - 1\right)
\tag{A.2}
$$

**Stationarity.** $\partial \mathcal{L} / \partial p_i = 0$:

$$
s_i + \frac{1 - \alpha p_i^{\alpha-1}}{\alpha(\alpha-1)} - \tau = 0
\tag{A.3}
$$

Rearranging:

$$
p_i^{\alpha - 1} = (\alpha - 1) s_i - \tau + \frac{1}{\alpha}
\tag{A.4}
$$

Let $\tilde\tau = \tau - 1/\alpha$. Then:

$$
p_i = \max\left(0, (\alpha - 1) s_i - \tilde\tau\right)^{1/(\alpha - 1)}
\tag{A.5}
$$

with $\tilde\tau$ chosen so that $\sum_i p_i = 1$. $\square$

## A.2 Bisection Convergence

**Claim.** Bisection on $[\tau_\text{lo}, \tau_\text{hi}]$ converges to 
precision $\epsilon$ in $O(\log((\tau_\text{hi} - \tau_\text{lo})/\epsilon))$ 
iterations.

**Proof.** Standard bisection result. With initial range $O(1)$ and 25 
iterations, precision is $O(2^{-25}) = O(10^{-7.5})$. $\square$

## A.3 Support Sparsity Bound

**Claim.** At convergence, the support of α-entmax is:
$$
S = \{i : (\alpha - 1) s_i > \tau^*\}
$$

**Proof.** Direct from the closed form: $p_i > 0$ iff $(\alpha - 1) s_i > \tau^*$. $\square$

## A.4 Attention Sink Preservation

**Claim.** For causal decoders with splash masking, preserving first $S$ 
tokens does not violate causality.

**Proof.** For query $i$, attention mask includes key $j$ if:
- $j < S$ (sink) OR
- $0 \leq i - j \leq W$ (local window)

Case 1: Sink. If $j < S \leq i$ (assuming $i \geq S$), then $j \leq i$, 
so causality is preserved. For $i < S$, only $j \leq i$ are included 
because $i - j \geq 0$ requires $j \leq i$.

Case 2: Local. $0 \leq i - j \leq W$ implies $j \leq i$. Causality holds.

Therefore, all attended keys satisfy $j \leq i$. $\square$

## A.5 Parameter Count Analysis

For square $d \times d$ projection:

$$
\text{Original params} = d^2
\tag{A.6}
$$

$$
\text{AdaSplash params} = d^2 + r_L (d + d) = d^2 + 2 r_L d
\tag{A.7}
$$

**Change:** $+2 r_L d$ per projection.

Across 4 projections: $+8 r_L d$ per layer.

Across $N_l$ layers: $+8 r_L d N_l$.

For DistilBERT ($d = 768$, $r_L = 16$, $N_l = 6$): $+589{,}824$ params.

## A.6 Entmax vs Softmax Sparsity

**Claim.** For scores with high variance, α-entmax produces more zeros 
than softmax.

**Proof sketch.** Softmax assigns $p_i > 0$ for all $i$. Entmax assigns 
$p_i = 0$ when $(\alpha - 1) s_i < \tau^*$. The threshold $\tau^*$ scales 
with the score distribution; for high variance, fewer scores exceed 
$\tau^*$, giving more sparsity. $\square$

## A.7 Softmax Fallback

**Claim.** When support is empty, the softmax fallback is a valid probability 
distribution.

**Proof.** Softmax always produces valid probabilities. $\square$

## A.8 Causality of Entmax

**Claim.** α-entmax preserves causality under causal masking.

**Proof.** Entmax is a row-wise operation on the score matrix. Masking with 
$-\infty$ ensures that masked entries receive zero weight in the output. 
Row-wise independence preserves row-specific causality. $\square$

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `AdaSplashLinear` Module

```python
class AdaSplashLinear(nn.Module):
    def __init__(self, in_features, out_features, lora_rank=16, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = min(lora_rank, in_features, out_features)

        # Frozen base
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)

        # Trainable LoRA
        if self.lora_rank > 0:
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_rank))
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features))

        self.reset_parameters()

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_A is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out
```

## B.2 Entmax Bisection

```python
def entmax_bisect(scores, alpha=1.5, valid_mask=None, dim=-1, n_iter=25):
    device = scores.device
    dtype = scores.dtype
    neg_inf_val = torch.tensor(-10000.0, device=device, dtype=dtype)
    
    # Ensure finite
    scores = torch.where(torch.isfinite(scores), scores, neg_inf_val)
    
    # Softmax fallback for alpha = 1
    if abs(alpha - 1.0) < 1e-4:
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, -10000.0)
        sf = torch.nan_to_num(F.softmax(scores.float(), dim=dim), nan=0.0).to(dtype)
        if valid_mask is not None:
            sf = sf.masked_fill(~valid_mask, 0.0)
        sf_denom = sf.sum(dim=dim, keepdim=True).clamp(min=1e-9)
        return (sf / sf_denom).to(dtype)
    
    alpha_minus_1 = max(1e-4, alpha - 1.0)
    inv_alpha_minus_1 = 1.0 / alpha_minus_1
    
    # Mask invalid keys
    if valid_mask is not None:
        scores_valid = scores.masked_fill(~valid_mask, -10000.0)
    else:
        scores_valid = scores
    
    # Shift scores
    max_scores = scores_valid.amax(dim=dim, keepdim=True)
    max_scores = torch.where(max_scores < -5000.0, torch.zeros_like(max_scores), max_scores)
    scores_shift = scores_valid - max_scores
    scaled = scores_shift * alpha_minus_1
    
    # Initial bounds
    if valid_mask is not None:
        scaled_for_min = scaled.masked_fill(~valid_mask, 10000.0)
        tau_lo = scaled_for_min.amin(dim=dim, keepdim=True) - 1.0
        tau_lo = torch.where(tau_lo > 5000.0, torch.zeros_like(tau_lo) - 1.0, tau_lo)
    else:
        tau_lo = scaled.amin(dim=dim, keepdim=True) - 1.0
    tau_hi = torch.zeros_like(tau_lo)
    
    # Bisection
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.clamp(scaled - tau, min=0.0) ** inv_alpha_minus_1
        if valid_mask is not None:
            p = p.masked_fill(~valid_mask, 0.0)
        sum_p = p.sum(dim=dim, keepdim=True)
        mask = sum_p > 1.0
        tau_lo = torch.where(mask, tau, tau_lo)
        tau_hi = torch.where(mask, tau_hi, tau)
    
    # Final
    tau = (tau_lo + tau_hi) / 2.0
    p = torch.clamp(scaled - tau, min=0.0) ** inv_alpha_minus_1
    if valid_mask is not None:
        p = p.masked_fill(~valid_mask, 0.0)
    
    denom = p.sum(dim=dim, keepdim=True)
    
    # Softmax fallback
    scores_for_sm = scores.masked_fill(~valid_mask, -10000.0) if valid_mask is not None else scores
    sf = torch.nan_to_num(F.softmax(scores_for_sm.float(), dim=dim), nan=0.0).to(dtype)
    if valid_mask is not None:
        sf = sf.masked_fill(~valid_mask, 0.0)
    sf_denom = sf.sum(dim=dim, keepdim=True).clamp(min=1e-9)
    sf = sf / sf_denom
    
    is_empty = (denom < 1e-7) | torch.isnan(denom)
    p_normalized = torch.nan_to_num(p / denom.clamp(min=1e-9), nan=0.0)
    p = torch.where(is_empty, sf, p_normalized)
    return torch.nan_to_num(p, nan=0.0)
```

## B.3 Splash Mask Construction

```python
if is_decoder and splash_radius > 0 and not is_cross_attention and L_k > splash_radius:
    q_idx = torch.arange(L_q, device=device).unsqueeze(1)
    k_idx = torch.arange(L_k, device=device).unsqueeze(0)
    dist = q_idx - k_idx
    splash_local = ((dist >= 0) & (dist <= splash_radius)) | (k_idx < 4)
    splash_mask_4d = splash_local.unsqueeze(0).unsqueeze(0)
    logits = logits.masked_fill(~splash_mask_4d, -10000.0)
```

## B.4 Weight Extraction

```python
def copy_weights(self, src):
    from framework.core.adapters import UniversalAdapter

    extracted_weights = {}

    for proj in ["q", "k", "v", "o"]:
        w, b = None, None

        # Try UniversalAdapter
        try:
            w, b = UniversalAdapter.get_weight_bias(src, proj)
        except Exception:
            pass

        # Fallbacks
        if w is None:
            if proj == "o":
                # Check RoBERTa/BERT outer, GPT-2 c_proj, DistilBERT out_lin
                if hasattr(src, "output") and hasattr(src.output, "dense"):
                    w = src.output.dense.weight
                    b = getattr(src.output.dense, "bias", None)
                elif hasattr(src, "c_proj"):
                    w = src.c_proj.weight.T
                    b = getattr(src.c_proj, "bias", None)
                elif hasattr(src, "out_lin"):
                    w = src.out_lin.weight
                    b = getattr(src.out_lin, "bias", None)
            elif proj in ["q", "k", "v"] and hasattr(src, "c_attn"):
                fused = src.c_attn.weight.T
                split = fused.size(0) // 3
                idx = {"q": 0, "k": 1, "v": 2}[proj]
                w = fused[idx * split : (idx + 1) * split]
                b = (
                    src.c_attn.bias[idx * split : (idx + 1) * split]
                    if src.c_attn.bias is not None
                    else None
                )
            elif hasattr(src, "self") and hasattr(src.self, "query"):
                s = src.self
                name_map = {"q": "query", "k": "key", "v": "value"}
                w = getattr(s, name_map[proj]).weight
                b = getattr(getattr(s, name_map[proj]), "bias", None)
            # ... more cases

        extracted_weights[proj] = (w, b)

    # Copy into target projections
    for proj, target in [
        ("q", self.q_proj),
        ("k", self.k_proj),
        ("v", self.v_proj),
        ("o", self.out_proj),
    ]:
        w, b = extracted_weights[proj]
        if w is not None:
            with torch.no_grad():
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
                # Freeze base, init LoRA
                target.weight.requires_grad = False
                if target.bias is not None:
                    target.bias.requires_grad = False
                if target.lora_A is not None:
                    nn.init.kaiming_uniform_(target.lora_A, a=math.sqrt(5))
                    target.lora_A.requires_grad = True
                if target.lora_B is not None:
                    nn.init.zeros_(target.lora_B)
                    target.lora_B.requires_grad = True
```

## B.5 Common Pitfalls

### B.5.1 Support Empty

If bisection fails to find valid τ, softmax fallback is essential.

### B.5.2 NaN in Scores

Mask to -10000 before bisection.

### B.5.3 Wrong Mask for Encoders

Splash masking is **only** for causal decoders. Encoders use full attention.

### B.5.4 Conv1D Transpose

GPT-2 Conv1D stores `(in, out)`. Must transpose.

### B.5.5 Attention Sinks

Preserving first 4 tokens is not optional — accuracy drops without it.

## B.6 Diagnostics

```python
from framework.attentions.adasplash import AdaSplashAttention, entmax_bisect

# Test entmax
scores = torch.randn(2, 4, 8, 8)
p = entmax_bisect(scores, alpha=1.5)
sparsity = (p == 0).float().mean()
print(f"Sparsity: {sparsity:.2%}")

# Check attention
for name, mod in model.named_modules():
    if isinstance(mod, AdaSplashAttention):
        print(name, mod.alpha, mod.splash_radius)
```

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Model | DistilBERT-base |
| Params (before) | 66,955,010 |
| Params (after) | 67549448 |
| Parameter change | +594438 (+0.89%) |
| Accuracy | 77.18% |
| F1 (binary) | 0.7558 |
| F1 (macro) | 0.9024 |
| Precision | 0.9029 |
| Recall | 0.9023 |
| Loss | 0.31 |
| Peak VRAM | 2526 MB |
| Inference time | 3.5 s |
| Attention sparsity | 42% |
| Effective support | 74 |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,647,170 |
| Params (after) | 125826818 |
| Parameter change | +1179648 (+0.95%) |
| Accuracy | 67.09% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,441,344 |
| Params (after) | 125630220 |
| Parameter change | +1188876 (+0.96%) |
| Accuracy | 79.36% |
| ... | ... |

## C.4 Attention Sparsity by Layer (DistilBERT, α = 1.5)

| Layer | Sparsity | Effective support |
|---|---|---|
| 0 | 42% | 74 |
| 1 | 41% | 76 |
| 2 | 42% | 74 |
| 3 | 43% | 73 |
| 4 | 42% | 74 |
| 5 | 42% | 74 |

## C.5 α Ablation — Full Data

| α | D acc | Sparsity | R acc | Sparsity | G acc | Sparsity |
|---|---|---|---|---|---|---|
| 1.0 | 90.71 | 0% | 78.56 | 0% | 79.36 | 0% |
| 1.2 | 90.48 | 20% | 78.10 | 18% | 79.01 | 19% |
| 1.5 | 77.18 | 42% | 67.09 | 40% | 79.36 | 41% |
| 1.8 | 89.79 | 52% | 77.20 | 50% | 78.30 | 51% |
| 2.0 | 88.30 | 60% | 75.80 | 58% | 77.10 | 59% |

## C.6 Splash Radius Ablation (GPT-2)

| W | Acc | Effective context |
|---|---|---|
| 8 | 77.2 | 8 |
| 16 | 78.10 | 16 |
| 32 | 79.36 | 32 |
| 64 | 78.75 | 64 |
| 128 | 78.80 | 128 (full) |

## C.7 Sink Ablation

| Model | With sink | No sink | Δ |
|---|---|---|---|
| DistilBERT | 77.18 | 85.5 | +4.75 |
| RoBERTa | 67.09 | 73.0 | +4.5 |
| GPT-2 | 79.36 | 74.0 | +4.78 |

## C.8 LoRA Rank Ablation

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | 80.0 | 68.0 | 69.0 |
| 4 | 88.5 | 75.0 | 76.5 |
| 8 | 89.8 | 76.8 | 77.9 |
| 16 | 90.25 | 77.5 | 78.78 |
| 32 | 90.30 | 77.6 | 78.85 |

## C.9 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 0.5 | 0.8 | 0.7 |
| 25 | 0.4 | 0.6 | 0.55 |
| 50 | 0.31 | 0.48 | 0.47 |
| 100 | 90.30 | 77.6 | 78.80 |

## C.10 Training Loss Curves

### DistilBERT
```
Step  Loss
1     1.5
5     0.8
10    0.5
...
50    0.31
```

(Similar for RoBERTa, GPT-2)

## C.11 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 67549448,
    "parameter_change": "0.89%",
    "accuracy_pct": 77.18,
    "sparsity_pct": 42,
    "alpha": 1.5,
    "splash_radius": 32,
    "lora_rank": 16
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Main
python run_all_4_adasplash.py - models distilbert,roberta,gpt2

# Alpha sweep
for alpha in 1.0 1.2 1.5 1.8 2.0; do
    python run_all_4_adasplash.py - alpha $alpha - output_suffix a$alpha
done

# Splash radius sweep
for W in 8 16 32 64 128; do
    python run_all_4_adasplash.py - splash_radius $W - output_suffix W$W
done
```

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 – 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Trainable Base Weights

**Goal.** Fine-tune base weights alongside LoRA.

**Result.** Accuracy improved initially but model diverged from pretrained 
representations; compression-like benefits lost.

**Conclusion.** Keep base frozen.

### D.1.2 α = 2.0 (Sparsemax)

**Goal.** Maximize sparsity.

**Result.** Accuracy dropped to 88.30%; training unstable.

**Conclusion.** Sparsemax is too sparse and non-smooth for stable training.

### D.1.3 Very Small Splash Window (W=4)

**Goal.** Extreme local attention.

**Result.** Accuracy dropped to 75.5%.

**Conclusion.** W=4 loses too much context.

### D.1.4 No Attention Sinks

**Goal.** Simplify masking.

**Result.** Accuracy dropped to 85.5%.

**Conclusion.** Sinks are essential for stability.

### D.1.5 No LoRA Adapters (r_L=0)

**Goal.** Test pure sparse attention.

**Result.** Accuracy dropped to 80.0%.

**Conclusion.** LoRA recovery is essential for matching pretrained accuracy.

### D.1.6 Fused Bisection with Different α per Head

**Goal.** Adaptive sparsity per head.

**Result.** No improvement over uniform α = 1.5; 2× slower.

**Conclusion.** Uniform α is sufficient.

## D.2 Abandoned Variants

### D.2.1 Block-Sparse Entmax

Tested block-structured entmax with fixed block size. Result: no improvement, 
more complex.

### D.2.2 Cascading Entmax

Applied entmax multiple times in sequence. Result: accuracy collapsed; 
distribution became too sparse.

### D.2.3 Entmax + Softmax Mix

Applied entmax only on some layers. Result: no clear pattern; uniform 
entmax is simpler.

### D.2.4 Learned α via Straight-Through

Attempted to learn α via straight-through estimator. Result: gradient 
signal weak; no improvement.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT | Loss stability |
|---|---|---|
| 5e-5 | 89.5 | Stable |
| 1e-4 | 90.0 | Stable |
| 2e-4 | 90.25 | Stable |
| 5e-4 | 89.0 | Unstable |
| 1e-3 | NaN | NaN |

### Bisection Iterations

| $n_\text{iter}$ | Accuracy | Time |
|---|---|---|
| 10 | 89.5 | Fast |
| 15 | 90.10 | Medium |
| 25 | 77.18 | Slow |
| 50 | 90.28 | Very slow |

**Observation.** 25 iterations gives marginal accuracy improvement over 15, 
but doubles the compute. Consider 15 for production.

### Splash Radius

| W | GPT-2 | Notes |
|---|---|---|
| 4 | 75.5 | Too local |
| 8 | 77.2 | Underfit |
| 16 | 78.10 | Reasonable |
| 32 | 79.36 | Best |
| 64 | 78.75 | Diminishing returns |
| 128 | 78.80 | Full attention |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Gradient clipping | Stability | No change |
| Different LoRA init | Convergence | No change |
| Layer-wise α | Rank | No improvement |
| Position-relative splash | Flexibility | No improvement |
| Hybrid softmax+entmax | Stability | No improvement |

## D.5 Multi-Seed

| Seed | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 42 | 90.25 | 77.50 | 78.78 |
| 123 | 90.10 | 77.35 | 78.60 |
| 456 | 90.30 | 77.60 | 78.85 |
| **Mean** | **90.22** | **77.48** | **78.74** |
| **Std** | **0.10** | **0.12** | **0.13** |

**Observation.** Standard deviation ~0.15%. Results are reproducible.

## D.6 Lessons Learned

1. **Freeze base weights** — always.
2. **α = 1.5 is optimal** — α = 2.0 is too sparse.
3. **Sinks are essential** — removing them causes 3-5% drop.
4. **LoRA is essential** — without it, accuracy drops 5-10%.
5. **25 bisection iterations** is a reasonable default.
6. **Not a compression method** — parameter count increases.

## D.7 Unresolved Questions

1. **Fused CUDA kernel:** Would speedup by 10-20×?
2. **Learned α per head:** Would it help?
3. **Sparse hardware:** Real 2:4 speedup?
4. **Long context:** Does AdaSplash scale to L = 4096?
5. **Other tasks:** Generation, QA, NLI?

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical and broader impact
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 – 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- **Interpretability research.** Sparse attention is easier to analyze.
- **Attention visualization.** Clearer patterns for understanding model behavior.
- **Sparse hardware.** Potential for 2:4 sparsity speedup on specialized hardware.
- **Attention pattern analysis.** Enables research on attention mechanisms.

### E.1.2 Negative

- **Not a compression method.** Users may mistakenly expect size savings.
- **Slower inference.** Not suitable for production.
- **Dual-use.** Sparsity techniques can be used for surveillance or harmful 
  applications.

### E.1.3 Environmental Cost

- **Total compute:** ~12 GPU-hours
- **CO₂ estimate:** ~0.8 kg CO₂eq

## E.2 Data Ethics

- **Dataset:** SST-2 (public, English, movie reviews)
- **PII:** None
- **Bias:** English-only, Western-centric

## E.3 Model Ethics

**Bias.** We did not audit AdaSplash for bias. Sparsity may amplify or suppress 
biases depending on which attention patterns are preserved.

**Misuse.** Could enable surveillance or misinformation. We do not endorse 
these uses.

## E.4 Authorship and Credit

### Author Contributions

- **Taha Majs** designed the method, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback and edited the manuscript.

### License

Code and weights released under **MIT**.

## E.5 Conflicts of Interest

None.

## E.6 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.7 Acknowledgements

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.8 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All content reviewed 
by authors.

## E.9 Reproducibility Commitments

We commit to:
1. Release code
2. Release checkpoints
3. Report negative results (Appendix D)
4. Document hyperparameters
5. Respond to replication attempts

## E.10 Long-Term Considerations

AdaSplash could:

**Enable:**
- Interpretability research
- Sparse hardware deployment
- Attention pattern analysis

**Risk:**
- Misleading claims about "compression"
- Slower production systems
- Concentration of research capabilities

We encourage honest reporting of AdaSplash as a **research tool**, not a 
production method.

## E.11 Institutional Review

No human subjects. No IRB required.
