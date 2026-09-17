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
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/adaptive_span_attention.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

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
span $S_\text{max} = 128$, learned spans converge to 
**68** on DistilBERT (47% reduction from max), 
**72** on RoBERTa, and **75** on GPT-2. Accuracy drops by 
only **-1.2%**, **-2.1%**, and **-1.9%** respectively. 
Attention memory is reduced by **2×** at convergence. The 
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

---