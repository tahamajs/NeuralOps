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
4. **Quality-preserving:** $\Delta\text{Acc} \leq 5\%$ on SST-2.

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
  under MIT.

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

---