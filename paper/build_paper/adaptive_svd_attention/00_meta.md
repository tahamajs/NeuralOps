<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords, lay summary, ToC
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# Adaptive SVD Attention: Energy-Thresholded Low-Rank Factorization with Residual Adapters for Multi-Architecture Transformers

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1 (initial draft)
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/adaptive_svd_attn.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Low-rank factorization is a widely used primitive for compressing transformer 
attention, but the standard approach — truncated SVD at a fixed global rank — 
treats all layers identically. This is suboptimal because attention projection 
matrices exhibit highly heterogeneous spectral decay: some are well-approximated 
by only 32 singular directions, while others require 200 or more. We introduce 
**Adaptive SVD Attention** (**Adaptive SVD**), a compression method that selects 
a **per-projection rank** via an **energy threshold** on the cumulative squared 
singular values:

$$
r^{(\ell)} = \min\left\{ r : \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_\text{in}, d_\text{out})} \sigma_j^2} \geq \tau \right\}, \quad \tau = 0.92
$$

Unlike activation-aware methods (A3, AASVD), Adaptive SVD does **not** require 
calibration data, activation statistics, or forward passes. Rank is determined 
purely from the weight's own spectrum, making the method self-contained and 
deployable in zero-data settings.

Each attention projection $W$ is replaced with a frozen symmetric factorized 
base $U_r V_r$ plus a **zero-initialized** residual adapter $B A$:

$$
W \approx U_r V_r + B A, \quad U_r = U_{:, :r} \Sigma^{1/2}, \quad V_r = \Sigma^{1/2} V_{:r, :}^\top
$$

The zero initialization ensures the initial forward pass of the compressed model 
matches the pretrained model up to the truncation residual, giving a clean 
starting point for LoRA recovery.

We evaluate on DistilBERT (6 layers, flat attention), RoBERTa-base (12 layers, 
nested attention), and GPT-2 (12 layers, causal attention) on SST-2 with 
50 LoRA recovery steps at residual rank 16. 
Adaptive SVD achieves parameter reductions of **3.0×–2.5×** 
with accuracy drops of only **-2.7%–-3.9%**. The method 
additionally supports **T5 relative attention bias**, **GQA**, and **RoPE**, 
making it applicable across encoder-only, decoder-only, and encoder-decoder 
architectures. Per-layer ranks range from 64 to 150, 
confirming that the energy criterion adapts to each weight's spectrum.

**Keywords:** adaptive rank selection, cumulative energy thresholding, low-rank 
attention, residual adapters, transformer compression

---

## Lay Summary

Every attention weight matrix in a transformer is a grid of numbers. Some 
grids have a "flat" structure — their information concentrates in a few 
directions, so they can be compressed heavily. Others have "sharp" structure 
and need more directions to represent accurately. Standard compression uses 
the same rank for every matrix, wasting space on some layers and losing 
accuracy on others. Adaptive SVD looks at each matrix individually and asks: 
"How many directions do I need to capture 92% of this matrix's energy?" It 
keeps exactly that many, plus a small trainable correction. On three 
popular language models, Adaptive SVD removes over half the attention 
parameters with under 3% accuracy loss — and needs no example data to 
calibrate.

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
