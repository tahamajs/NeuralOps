<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# ORION: Optimized Rectified Input-aware Orthogonal Network for Hybrid Attention with SVD Factorization

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/orion.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- ORION combines three distinct ideas:
  1. Zero-shot SVD factorization of projection weights
  2. Head-wise bifurcation: 2/3 kernel-linear, 1/3 full softpick
  3. RFF for the kernel heads
- Unlike NHA/NASC/GoldFinch, ORION genuinely reduces parameter count
- But it shares the copy_weights raise bug with AEGIS V4/V5/EAT
-->

We present **ORION** (Optimized Rectified Input-aware Orthogonal Network), 
a hybrid attention mechanism that combines three orthogonal ideas: 
(i) **zero-shot SVD low-rank factorization** of Q/K/V/O projection weights, 
(ii) **head-wise bifurcation** between a linear-attention kernel branch 
(2/3 of heads) and a full softpick branch (1/3 of heads), and (iii) 
**random Fourier features** for the kernel heads. Unlike prior hybrid 
attention modules in our framework (e.g., NHA, NASC, GoldFinch), ORION 
**genuinely reduces parameter count**: each projection $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
is replaced by a product $W_\text{up} W_\text{down}$ with intermediate rank 
$r = 64$, achieving ~6× reduction for $768 \times 768$ projections and ~9.6× 
for $768 \times 3072$ FFN-like projections.

We evaluate ORION on DistilBERT, RoBERTa, and GPT-2 on SST-2 with LoRA 
recovery at rank $r_L = 8$. ORION achieves **55.50%** on 
DistilBERT (baseline 0.85%, Δ = -35.21), **50.92%** on 
RoBERTa (Δ = -27.64), and **50.92%** on GPT-2 (Δ = -28.44). 
Parameter count decreases by **32%** across all models. 
We document one critical implementation bug: `copy_weights` **raises** on 
UniversalAdapter failure instead of falling back, matching the pattern 
observed in other modules and causing catastrophic collapse when triggered.

**Keywords:** SVD factorization, hybrid attention, kernel attention, 
softpick, random Fourier features

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

ORION compresses a transformer by factoring each weight matrix into two 
smaller matrices, saving about 6× in parameter count. It also splits 
attention heads into two groups: two-thirds use a fast "kernel" method 
that approximates attention with math tricks, and one-third use regular 
attention with a sparsity trick. We tested on three pretrained models and 
report both the parameter savings and the accuracy trade-offs. We also 
document a bug in the weight-copying code that can crash on certain 
architectures.

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