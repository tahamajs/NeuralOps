<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# SALT: Singular Value Adaptation with Low-Rank Tuning for Parameter-Efficient Attention

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/salt_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- SALT is a REAL parameter reduction method (unlike NHA/RALA/routing)
- BUT forward reconstructs full W → no inference speedup
- Honest about the reconstruction overhead
-->

Low-rank adaptation methods such as LoRA freeze pretrained weights and add a 
trainable rank-$r$ residual. While LoRA achieves large reductions in trainable 
parameters, it does not reduce **total** parameters — the base weight matrix 
remains at full size. We introduce **SALT** (**S**ingular Value 
**A**daptation with **L**ow-Rank **T**uning), which replaces each weight matrix 
$W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ with a three-component 
representation: (i) a truncated SVD $(U_{:,:r}, \Sigma_{:r}, V_{:r,:})$ stored 
as **non-trainable buffers**; (ii) a trainable per-singular-value scale 
$\gamma \in \mathbb{R}^r$ that modulates the retained singular values; and (iii) 
a trainable low-rank residual $AB$ initialized to zero.

For $d_\text{out} = d_\text{in} = 768$ and $r = 16$, this reduces stored 
parameters from 589,824 to **49,936** — a **11.8× reduction** — while 
maintaining the full expressiveness of a rank-$2r$ effective weight matrix. 
We evaluate on DistilBERT, RoBERTa, and GPT-2 on SST-2. Accuracy is preserved 
within **-10.43%** (DistilBERT), **-27.64%** (RoBERTa), and 
**-28.44%** (GPT-2) after 50 recovery steps.

We honestly report a key limitation: because our forward pass reconstructs 
the full weight matrix $W = U \text{diag}(\Sigma \odot \gamma) V^\top + AB$ 
before calling `F.linear`, **inference memory and latency are not reduced**. 
Storage footprint is reduced by **32×**, but runtime 
efficiency is unchanged (or slightly worse due to reconstruction overhead). 
We identify the necessary change — performing the forward as three sequential 
matmuls — to realize the inference-time benefits.

**Keywords:** singular value decomposition, low-rank adaptation, parameter 
efficiency, fine-tuning, transformer compression

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

Pretrained language models are large because they store big matrices of 
numbers. We propose to store only the top-16 "directions" of each matrix 
(instead of all 768), plus two small learned corrections. This shrinks the 
model by about 12×. We test our method on three small models. Accuracy is 
preserved, and the models occupy much less disk space. However, our current 
code rebuilds the full matrix every time it runs, so the models do not run 
faster and are not smaller in GPU memory during inference. We explain this 
limitation and outline how to fix it.

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