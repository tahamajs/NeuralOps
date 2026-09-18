<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# MinimalSVD: Energy-Adaptive Low-Rank Factorization for Efficient Transformer Attention

<!-- 
TITLE NOTES:
- "MinimalSVD" = minimal implementation of SVD-based attention
- "Energy-Adaptive" emphasizes the rank-selection novelty
- Alternative: "MinimalSVD: Spectral Energy Rank Selection for Low-Rank Attention Compression"
- This is the BASELINE variant of a family (NOVA, PRISM, SVD)
-->

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/minimal_svd.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We present **MinimalSVD**, a compact and reproducible method for compressing 
the four attention projections of transformer language models. Each weight 
matrix $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ is factorized as 
$W \approx U V$ via truncated SVD, with $U \in \mathbb{R}^{d_\text{out} \times r}$ 
and $V \in \mathbb{R}^{r \times d_\text{in}}$. The parameter count drops from 
$d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$, achieving 
**50\%×** reduction per projection on DistilBERT at rank $r = 32$.

The core novelty is **spectral energy-adaptive rank selection**: instead of 
fixing $r$ uniformly across layers, MinimalSVD computes the cumulative 
spectral energy $E(r) = \sum_{i \le r} \sigma_i^2 / \sum_i \sigma_i^2$ of 
each weight matrix and selects the smallest rank such that $E(r) \ge \eta$, 
with $\eta = 45.2$ by default. This automatically adapts per-layer 
rank based on spectral decay, avoiding the fixed-rank trade-off between 
under- and over-parameterization.

MinimalSVD is intentionally minimal: the injection protocol targets a single 
transformer layer at a time, delegating whole-model traversal to the caller. 
This makes the implementation easy to audit and extend. We nevertheless 
support four transformer families — DistilBERT (flat), RoBERTa (nested), 
GPT-2 (causal with Conv1D), and Qwen3.5 (GQA + RoPE) — via architecture-aware 
weight extraction, Frobenius norm validation, and LoRA recovery.

On DistilBERT, RoBERTa-base, and GPT-2 with rank cap $r = 32$ and 
energy $\eta = 45.2$, MinimalSVD achieves **50\%×–2.1\times×** 
parameter reduction with accuracy drops of only **-1.83%–+1.49%** 
after 50 LoRA recovery steps. Peak inference VRAM decreases 
by **380 MB%**.

**Keywords:** low-rank factorization, SVD, energy-adaptive rank, attention 
compression, LoRA recovery, multi-architecture transformers

---

## Lay Summary

Language models spend most of their parameters on four large matrices inside 
each attention layer. MinimalSVD replaces each matrix with a smaller pair 
whose product approximates the original. Instead of fixing the size of the 
smaller pair, MinimalSVD measures how concentrated the matrix is — if a few 
directions dominate, those are kept; if the matrix is uniform, more 
directions are preserved. On three popular models (DistilBERT, RoBERTa, 
GPT-2), MinimalSVD removes 60–70% of the attention parameters with only 
1–3% loss in accuracy.

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