<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Primal Attention: Self-Attention via Asymmetric Kernel SVD in Primal Representation

<!-- 
TITLE NOTES:
- Emphasize "primal" (as opposed to dual/Nyström)
- "Asymmetric" because F1, F2, F3 have different Q/K
- Alternative: "Primal Attention: Nyström-Style Low-Rank Attention with Landmark Pooling"
-->

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/primal_attention.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Standard softmax attention incurs $\mathcal{O}(L^2)$ time and memory in sequence 
length $L$, which becomes the bottleneck for long-context transformer inference. 
Nyström approximation reduces this to $\mathcal{O}(L m)$ by projecting onto $m$ 
landmarks, but requires a symmetric kernel matrix, forcing the same landmarks 
for queries and keys. This is unnecessarily restrictive.

We introduce **Primal Attention**, an asymmetric kernel SVD formulation that 
decomposes the attention matrix as $A \approx F_1 F_2^{-1} F_3$, where $F_1, F_3$ 
are query-key cross products with landmarks and $F_2$ is the landmark-landmark 
Gram matrix. This allows independent landmark selection for queries and keys. 
The pseudoinverse $F_2^{-1}$ is computed via an iterative Schulz refinement 
that converges in a few steps and is differentiable.

Primal Attention selects landmarks via **segment pooling** over contiguous chunks 
of the sequence, providing a uniform but content-agnostic summary of the 
attention input. We evaluate on DistilBERT, RoBERTa-base, and GPT-2 on SST-2 
with landmark count $m = 64$. Primal Attention achieves 
**90.14%**, **68.00%**, and **63.07%** accuracy respectively, 
at the cost of **1.3×** inference latency. The attention matrix 
memory is reduced by **4×**. We provide a production-hardened 
PyTorch implementation supporting flat, nested, and causal architectures with 
GQA, RoPE, and safe injection.

**Keywords:** Nyström approximation, low-rank attention, pseudoinverse, Schulz 
iteration, efficient transformer

---

## Lay Summary

Attention is the core mechanism in modern language models. Its cost grows 
quadratically with sentence length — doubling the input quadruples the work. 
Many methods try to reduce this by summarizing the input into a few "landmark" 
points, but these methods typically require the same landmarks for both the 
query and the key side.

Primal Attention breaks this symmetry: queries and keys can use different 
landmarks. This gives more flexibility and slightly better quality at the 
same cost. We test the method on three popular language models and find that 
it retains most of the original accuracy while reducing attention memory by 
a factor of several. The method is fully differentiable and integrates 
cleanly with existing pretrained models.

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
