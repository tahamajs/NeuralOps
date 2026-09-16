# AEGIS-V5: Adaptive Enhanced Global-Local Integrated Sparse Attention with SVD-Factorized Projections

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/aegis_v5.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

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

On SST-2, AEGIS-V5 achieves **{{D_ACC}}%** on DistilBERT (baseline 
{{BASE_D}}%, Δ = {{D_DELTA}}), **{{R_ACC}}%** on RoBERTa (baseline 
{{BASE_R}}%, Δ = {{R_DELTA}}), and **{{G_ACC}}%** on GPT-2 (baseline 
{{BASE_G}}%, Δ = {{G_DELTA}}), using {{RECOVERY_STEPS}} LoRA recovery 
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
```

---