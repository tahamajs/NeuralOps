<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# A3: Activation-Aware Factorized SVD for Transformer Compression

<!-- 
TITLE NOTES:
- Keep under 15 words
- "A3" is the acronym (Activation-Aware ...)
- Avoid catchy phrases; be descriptive
- Alternative: "A3 Compressor: Fisher-Weighted Low-Rank Factorization with LayerNorm Recalibration"
-->

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1 (initial draft)
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/compression_engine/a3_compressor.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Structure: problem → method → results → limitations
- Emphasize: this is REAL compression (unlike attention-replacement methods)
- Every number must match Section 7
-->

Low-rank factorization is a classical approach to compressing the linear layers 
of transformer models, replacing each weight matrix $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
with a product $U V$ where $U \in \mathbb{R}^{d_\text{out} \times r}$, 
$V \in \mathbb{R}^{r \times d_\text{in}}$, and $r \ll \min(d_\text{in}, d_\text{out})$. 
Standard SVD applies this factorization uniformly across all input dimensions, 
ignoring the fact that transformer activations have highly non-uniform variance 
across channels. We introduce **A3** (**A**ctivation-**A**ware **A**daptive 
factorization), which addresses this by (i) collecting a streaming empirical 
Fisher diagonal during calibration, (ii) weighting the SVD input space by this 
energy estimate before factorization, (iii) adaptively increasing rank when 
reconstruction error exceeds a threshold, and (iv) recalibrating LayerNorm 
affine parameters after factorization to absorb distributional shift.

A3 achieves **5.0%** parameter reduction on DistilBERT, 
**4.5%** on RoBERTa, and **4.5%** on GPT-2 
at rank $r = 64$. On SST-2, accuracy drops by only 
**-12.73%**, **+7.11%**, and **-4.02%** respectively, 
after 50 LoRA recovery steps. Peak VRAM decreases by 
**40%**. We provide a fully reproducible implementation with 
strict Frobenius-norm validation, safe fallback to original layers, and 
streaming O($d^2$) memory calibration.

**Keywords:** low-rank factorization, activation-aware compression, Fisher 
information, LayerNorm recalibration, transformer efficiency

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. No jargon. Plain English.
-->

Large language models contain millions of numbers. Many of these numbers are 
redundant. A3 finds redundancy by looking at which input features actually 
matter — those that get "activated" most during real examples. It then 
compresses the model along those directions, using a mathematical technique 
called SVD, but weighted by importance. We test A3 on three pretrained models 
and find that it removes about half the parameters while keeping accuracy 
within 1–2%. This makes models smaller, faster to load, and cheaper to 
deploy.

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
