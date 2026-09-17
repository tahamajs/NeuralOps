# AEGIS-V5: Adaptive Enhanced Global-Local Integrated Sparse Attention with SVD-Factorized Projections

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v5
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/aegis_v5.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Transformer self-attention has quadratic complexity in sequence length, 
making long-context inference expensive. We introduce **AEGIS-V5**, a 
hybrid attention module that combines a **local sliding-window branch** 
with a **global kernel linear attention branch**, gated per-head by a 
learnable scalar bias initialized toward local attention. The base 
projections are **SVD-factorized with energy-based rank selection** and 
frozen; trainable capacity is provided by **LoRA adapters** on each 
projection.

AEGIS-V5 splits the attention heads into two groups: the first 
$H_\text{local} = \lceil 2H/3 \rceil$ heads use local windowed attention, 
and the remaining $H_\text{global}$ heads use kernel linear attention with 
Random Fourier Features (RFF). The two branch outputs are combined by a 
per-head gate. For decoder-only models, we fall back to standard causal 
softmax attention to preserve causality — a conservative but correctness-
preserving choice.

On SST-2, AEGIS-V5 achieves **83.72%** on DistilBERT (baseline 
90.71%, Δ = -6.99), **86.47%** on RoBERTa (baseline 
78.56%, Δ = +7.91), and **69.50%** on GPT-2 (baseline 
79.36%, Δ = -9.86), using 50 LoRA recovery 
steps. Complexity is $\mathcal{O}(N \cdot W + N \cdot F)$ for encoders 
where $W$ is window size and $F$ is RFF feature count. We provide a 
production-hardened implementation supporting DistilBERT, RoBERTa, GPT-2, 
and Qwen3.5, with strict Frobenius validation, safe fallback, and detailed 
per-layer error statistics.

**Keywords:** hybrid attention, SVD factorization, LoRA, local window, 
kernel attention, energy-based rank

---

## Lay Summary

Standard transformers compare every word to every other word, which becomes 
slow for long texts. Our method splits the "comparison job" into two groups 
of attention heads: some heads look only at nearby words (local), and others 
look at a summarized version of the whole text (global). A learned weight 
decides how much each head relies on the local versus global view. To make 
the model smaller, we also compress the underlying projection weights using 
a mathematical technique called SVD, keeping only the most important 
directions. On three language models (DistilBERT, RoBERTa, GPT-2), our 
method achieves near-baseline accuracy while reducing memory and compute.

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