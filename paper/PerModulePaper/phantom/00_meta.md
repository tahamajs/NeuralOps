<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# PHANTOM: Parameter-Hashed Attention with Token-wise Orthogonal Mapping

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/phantom_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Standard transformer attention couples two independent problems: how to 
project hidden states into query/key/value representations (a **dense linear 
algebra** problem), and how to compute the attention weights themselves (a 
**quadratic matching** problem). We introduce **PHANTOM** 
(**P**arameter-**H**ashed **A**ttentio**N** with **T**oken-wise 
**O**rthogonal **M**apping), which addresses both problems with a unified 
architecture. Projections are compressed via **leverage-score CUR matrix 
decomposition**, replacing each $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
with a low-rank product $C U^+ R$ that preserves statistically important rows 
and columns. Attention is computed via **multi-round angular LSH**, which 
partitions tokens into hash buckets by random projection signs and computes 
attention only within buckets, reducing effective cost from $\mathcal{O}(L^2)$ 
to $\mathcal{O}(L \cdot B)$ where $B$ is the average bucket size. We use a 
**rectified softpick** activation instead of softmax, which drops far-tail 
attention weights while falling back to softmax when no candidate survives.

PHANTOM compresses DistilBERT by **50\%×**, RoBERTa by **2.1\times×**, 
and GPT-2 by **2.1\times×**, with SST-2 accuracy drops of only 
**-18.12%**, **-27.64%**, and **-28.44%** after 
50 LoRA recovery steps. Peak inference VRAM decreases by 
**380 MB%**. Unlike attention-replacement methods that preserve 
parameter count, PHANTOM achieves genuine parameter reduction from 
$d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$ via CUR 
factorization. We provide a production-hardened implementation with strict 
Frobenius validation, safe per-layer fallback, and multi-architecture support.

**Keywords:** CUR decomposition, leverage scores, locality-sensitive hashing, 
sparse attention, transformer compression, softpick

---

## Lay Summary

Every transformer layer has two bottlenecks: the projection matrices that 
transform hidden states (large and dense), and the attention computation 
that matches every token against every other token (quadratic in sequence 
length). PHANTOM attacks both at once. It replaces dense projections with 
a low-rank CUR approximation that picks the most statistically important 
rows and columns of each weight matrix, and it replaces full attention with 
a hashing scheme that only compares tokens sharing a random hash bucket. 
The result is a model that is ~3× smaller and runs on much less memory while 
preserving most of the original accuracy.

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