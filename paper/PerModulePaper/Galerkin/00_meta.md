<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# Galerkin Attention: Petrov-Galerkin Projection onto Orthonormal Bases for Transformer Self-Attention

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1 (initial draft)
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/galerkin.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- BE HONEST: This is a negative result for encoders
- The Galerkin path only applies to bidirectional (encoder) models
- GPT-2 uses standard causal attention (no-op)
- RoBERTa collapses to 48.97% (chance level)
- DistilBERT drops -24.77%
-->

We reinterpret transformer self-attention as a **discretization of a 
continuous non-local integral operator** and approximate it via 
**Petrov-Galerkin projection** onto an orthonormal basis (discrete cosine 
transform). The method projects queries, keys, and values onto a truncated 
$r$-dimensional basis $\Phi \in \mathbb{R}^{N \times r}$ before computing 
attention, reducing the quadratic interaction matrix from $N \times N$ to 
$r \times r$. For bidirectional (encoder) models, this yields a data-agnostic 
low-rank attention approximation; for causal (decoder) models, we fall back 
to standard scaled dot-product attention to preserve causality.

We evaluate on three pretrained models — DistilBERT, RoBERTa, and GPT-2 — 
on SST-2 with LoRA recovery. Our results are **negative for encoders**: 
DistilBERT drops from 90.71% to **50.92%** (Δ = -39.79) and RoBERTa 
collapses from 78.56% to **50.92%** (Δ = -27.64), the latter at 
chance level. GPT-2 shows no change (Δ = 0.00%) because the causal fallback 
path leaves attention unmodified. We attribute encoder failures to the 
**data-independent, fixed DCT basis**, which cannot capture the 
content-dependent, position-specific attention patterns learned by 
pretraining. We document this negative result to inform future work on 
basis-projected attention.

**Keywords:** Galerkin projection, transformer attention, orthonormal basis, 
integral operator, negative results

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

We tried to compute attention using a fixed mathematical basis (cosine waves) 
instead of learning it from data. This makes attention much cheaper to 
compute — instead of comparing every pair of words, we compare their 
compressed representations. We tested on three pretrained models and found 
that this **does not work for encoder models** like DistilBERT and RoBERTa, 
which drop by 25–30%. It also **does nothing** for GPT-2, because our code 
uses standard attention for that model. We report this negative result 
honestly to warn others away from fixed-basis projection.

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
