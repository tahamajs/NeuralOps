<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# SVD-Mamba: SVD-Factorized Selective State Space Attention for Transformers

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/svd_mamba.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Attention layers and State Space Models (SSMs) offer complementary inductive 
biases: attention excels at content-based retrieval; SSMs offer linear-time 
recurrence with bounded state. We present **SVD-Mamba**, a drop-in replacement 
for multi-head self-attention that unifies three mechanisms:

1. **SVD-factorized attention projections** — $W_Q, W_K, W_V, W_O$ are 
   stored as $U \Sigma V^\top$ factors, reducing projection parameter count 
   by ~4×.
2. **Selective SSM branch** — a Mamba-style state-space block with causal 
   depthwise convolution, input-dependent gating, and a chunked associative 
   scan.
3. **Global SVD mixing** — a low-rank global aggregation to supplement 
   local attention.

The two main branches (attention, SSM) are fused via a **gating module 
initialized with bias $-7$**, making the SSM branch contribute less than 
0.1% at initialization and preserving the pretrained attention behavior 
during the early recovery phase.

We evaluate SVD-Mamba on DistilBERT, RoBERTa-base, and GPT-2 across SST-2 
with 50 LoRA recovery steps. SVD-Mamba achieves 
**50.92%** on DistilBERT (Δ = -39.79), **50.92%** on RoBERTa 
(Δ = -27.64), and **50.23%** on GPT-2 (Δ = -29.13).

<!-- CRITICAL HONESTY -->
SVD-Mamba **increases** total parameter count by **48.5\%%** 
on DistilBERT. While SVD-factorized projections reduce per-layer parameters, 
the SSM branch, global SVD mixing, gating module, and LoRA adapters add 
more than the factorization saves. The "SVD-factorized" in the name refers 
to the projection storage format, not to overall model compression.

**Keywords:** state space models, Mamba, SVD factorization, hybrid attention, 
selective gating

---

## Lay Summary

Transformers use attention to compare every token with every other token. 
State-space models (SSMs) like Mamba use a different mechanism: they scan 
through the sequence once, maintaining a small internal state. Both have 
strengths. SVD-Mamba combines them — using attention for precise retrieval 
and SSM for cheap long-range memory — and lets the model learn to balance 
the two via a gate that starts favoring attention. The method works on 
three popular language models without retraining the base weights, though 
it adds more parameters than it removes.

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