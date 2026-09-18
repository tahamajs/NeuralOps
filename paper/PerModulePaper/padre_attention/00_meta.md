# PADRe: Polynomial Attention with Dropout Regularization for Multi-Architecture Transformers

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/padre.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We introduce **PADRe** (**P**olynomial **A**ttention with **D**ropout **R**egularization), 
a multi-architecture attention module that replaces the softmax normalization 
of standard attention with a **polynomial kernel**:

$$
\text{attn}(q_i, k_j) = \frac{(1 + s_{ij}/\text{order})^\text{order}}{\sum_{j'} (1 + s_{ij'}/\text{order})^\text{order}}
$$

where $s_{ij}$ is the scaled, centered dot-product score. This polynomial 
kernel is a **Taylor approximation** of $\exp(s)$ and, unlike softmax, is 
**kernelizable** — it can be expressed as $\phi(q) \cdot \phi(k)$ where $\phi$ 
is a finite-dimensional feature map. This property has the potential to enable 
$\mathcal{O}(N)$ attention.

**However, the current implementation is not kernelized.** The attention 
weights are materialized as an $[N \times N]$ matrix, so the runtime is 
$\mathcal{O}(N^2 d_h)$ — identical to standard softmax attention. The 
polynomial kernel does not, by itself, provide speedup without the 
corresponding kernelization step.

**What PADRe does provide:**

1. **Numerical stability via max-centering** of scores over valid positions, 
   preventing overflow in the polynomial expansion.
2. **Dropout regularization** on the polynomial attention weights.
3. **Multi-architecture support** across DistilBERT, RoBERTa, GPT-2, and Qwen.
4. **Frobenius-norm weight-copy validation** with output-equivalence check.
5. **Nested encoder bypass** for RoBERTa-style architectures.

On SST-2, PADRe achieves **90.37%** on DistilBERT (baseline 0.85%, 
Δ = -0.34), **64.56%** on RoBERTa (baseline 32%, 
Δ = -14.00), and **50.92%** on GPT-2 (baseline 0.85%, 
Δ = -28.44), using 50 LoRA recovery steps. We 
position PADRe as a **research module** for exploring polynomial attention 
in pretrained transformers, with the kernelization step explicitly left to 
future work.

**Keywords:** polynomial attention, kernelized attention, Taylor approximation, 
softmax replacement, multi-architecture

---

## Lay Summary

Standard attention uses a "softmax" function to convert similarity scores into 
weights. Softmax is hard to approximate efficiently, which is why standard 
attention is quadratic in the sequence length. We've built an alternative 
that uses a **polynomial** instead of softmax — polynomials are much easier 
to expand into a kernel feature map, which could enable linear-time attention. 
**Caveat:** in this implementation we still materialize the full attention 
matrix, so there's no speedup yet. What we demonstrate is that the polynomial 
substitution can be done without breaking pretrained models (DistilBERT, 
RoBERTa, GPT-2), and that it can be made numerically stable with a simple 
centering trick.

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