<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# MinimalNOVA: Energy-Adaptive SVD Factorization for Efficient Transformer Attention

<!-- 
TITLE NOTES:
- "MinimalNOVA" = Minimal Native Optimized V-Low-rank Attention
- "Energy-Adaptive" emphasizes the rank-selection novelty
- Alternative: "MinimalNOVA: SVD-Factorized Projections with Spectral Energy Thresholding"
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
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/minimal_nova.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Transformer attention projections dominate the parameter budget of modern 
language models. Standard multi-head attention stores four dense matrices 
$W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$, resulting in $4d^2$ parameters 
per layer. We introduce **MinimalNOVA** (**Min**imal **N**ative **O**ptimized 
**V**-low-rank **A**ttention), a drop-in replacement for attention projections 
that factorizes each matrix as $W \approx U V$ with $U \in \mathbb{R}^{d \times r}$ 
and $V \in \mathbb{R}^{r \times d}$, achieving parameter count 
$r(d + d) = 2rd$ instead of $d^2$. For $d = 768$ and $r = 64$, this yields a 
**6× parameter reduction per projection** while maintaining accuracy within 
**-2.06%** on SST-2.

Our key novelty is **energy-adaptive rank selection**: instead of fixing 
$r$ uniformly across layers, MinimalNOVA computes the cumulative spectral 
energy of each weight matrix $W = \sum_i \sigma_i u_i v_i^\top$ and selects 
the smallest $r$ such that $\sum_{i \le r} \sigma_i^2 \geq \eta \cdot \sum_i \sigma_i^2$ 
for threshold $\eta = 0.95$. This adapts rank per layer based on spectral 
decay, concentrating compression where the spectrum is concentrated.

We provide a production-hardened implementation supporting four transformer 
families (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with:
(i) nested encoder bypass for RoBERTa's `BertAttention`,
(ii) Conv1D handling for GPT-2's fused `c_attn`,
(iii) GQA support via `repeat_kv`,
(iv) RoPE position embeddings,
(v) LoRA adapters initialized to zero for recovery fine-tuning.

On DistilBERT, RoBERTa, and GPT-2 with rank cap $r = 32$ and energy 
$\eta = 45.2$, MinimalNOVA achieves **50\%×–2.1\times×** 
parameter reduction with accuracy drops of only **-2.06%–+0.00%** 
after 50 LoRA recovery steps. Peak inference VRAM decreases 
by **380 MB%**.

**Keywords:** low-rank factorization, energy-adaptive rank, SVD, attention 
compression, LoRA recovery, multi-architecture transformers

---

## Lay Summary

Transformer language models spend most of their parameters on four large 
"projection" matrices inside each attention layer. MinimalNOVA replaces each 
of these with a smaller pair of matrices whose product approximates the 
original. Instead of choosing a fixed size, MinimalNOVA measures how 
concentrated the matrix is — if a few directions dominate, they are kept; 
if the matrix is uniform, more directions are preserved. On three popular 
models (DistilBERT, RoBERTa, GPT-2), MinimalNOVA removes 60–80% of the 
attention parameters with only 1–3% loss in accuracy. This makes models 
easier to store and cheaper to run.

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