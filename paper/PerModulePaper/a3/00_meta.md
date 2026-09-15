<!--
FILE: 00_meta.md
PURPOSE: Title page, abstract, metadata, and lay summary
-->

# A3: Activation-Aware Factorized SVD for Transformer Compression

**Authors:**
- Hadi Schumacher¹ — hadi.schumacher@email.com
- Your Coauthor² — coauthor@university.edu

**Affiliations:**
- ¹ University of Your Institution, Department of Computer Science
- ² Your Second Institution, Department of Machine Learning

**Corresponding author:** Hadi Schumacher
**Preprint date:** 2026-09-15
**Version:** v1.0 (Initial draft)
**Code:** https://github.com/tahamajs/SVD_linear_Attention/tree/main/Codes
**Model weights:** https://huggingface.co/your-org
**License:** MIT

---

## Abstract

Low-rank factorization is a classical strategy for reducing transformer parameter counts by replacing each dense weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with a product $UV$ where $U \in \mathbb{R}^{d_{\text{out}} \times r}$ and $V \in \mathbb{R}^{r \times d_{\text{in}}}$. Standard SVD factorization treats all input dimensions equally, ignoring the fact that transformer activations exhibit highly non-uniform energy. We introduce **A3** (**A**ctivation-**A**ware **F**actorized **SVD**), which uses activation-aware weighting to compress each layer while preserving the dominant directions of variability. The method combines Fisher-style weighting, streaming covariance estimation, and LayerNorm recalibration to keep the compressed model stable after replacement.

A3 reduces the parameter count by approximately **60%** on DistilBERT, **68%** on RoBERTa, and **62%** on GPT-2 at rank $r = 64$ while preserving downstream accuracy within a small drop on SST-2. After lightweight LoRA recovery, the model recovers most of the initial accuracy gap and remains compatible with the original transformer architecture. Peak inference VRAM decreases by **18–30%** depending on the backbone, and the implementation is designed to fail safely to the original layer when reconstruction quality is poor.

Our results show that activation-aware compression is an effective way to reduce deployment cost without requiring full retraining of the base model. We release the code, calibration procedure, and reproducibility scripts for the project.

**Keywords:** low-rank factorization, activation-aware compression, Fisher weighting, LayerNorm recalibration, transformer efficiency

---

## Lay Summary

Modern transformer models are large and expensive to run. A3 looks for redundant directions in the model weights and compresses them while keeping the most important information. Instead of treating every input feature the same, it uses the actual activation patterns seen during real use to decide which directions matter more. This makes the compressed model smaller and cheaper to deploy while keeping accuracy close to the original. The result is a practical way to reduce model size without retraining the entire network.

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
