<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# RALA: Rank-Augmented Linear Attention with a Softplus Kernel

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/rala_linear_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Be honest: RALA is a replacement, not compression
- Parameter count increases (LoRA + U, V, alpha)
- Causal scan is sequential (slow)
- The rank augmentation uses frozen random U, V; only alpha is trainable
-->

Linear attention reduces the quadratic complexity of softmax attention to 
$\mathcal{O}(L)$ in sequence length by factorizing the attention matrix as 
a product of kernel feature maps. However, standard linear attention loses 
the head-specific interaction structure that gives softmax attention much 
of its expressiveness. We introduce **RALA** (**R**ank-**A**ugmented 
**L**inear **A**ttention), which augments the standard linear attention 
output with a low-rank per-head interaction term learned through a scalar 
gate.

RALA combines three components: (i) a **softplus kernel** $\phi(x) = 
\mathrm{softplus}(x)$ that is smooth and strictly positive, in contrast to 
the more common $\mathrm{elu}(x) + 1$; (ii) a **rank-$r$ head interaction 
matrix** $M_h = U_h V_h^\top \in \mathbb{R}^{d \times d}$ per head, with 
$r \ll d$; and (iii) a **scalar gate** $\alpha$ that blends the linear and 
augmented outputs. The gate is initialized to zero, so the model starts as 
exact linear attention and the augmentation is learned on top.

We evaluate RALA on DistilBERT, RoBERTa, and GPT-2 on SST-2 with LoRA recovery. 
On DistilBERT, RALA achieves **60.09%** (baseline 0.85%, 
Δ = -30.62). Parameter count increases by **48.5\%%** due to 
LoRA adapters and the augmentation matrices; RALA is **not** a compression 
method. Wall-clock inference increases by **2.3 s×** because the 
causal prefix scan is implemented as a Python loop. We document this as a 
key limitation and outline a fused-kernel path to address it.

**Keywords:** linear attention, low-rank augmentation, softplus kernel, 
efficient transformers

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

Standard attention is slow for long texts. One family of solutions — "linear 
attention" — makes it fast but loses some precision. We add a small learned 
correction term to a standard linear attention layer: a tiny matrix per head 
that captures how different parts of the sequence interact. A single learned 
number, starting at zero, controls how much this correction contributes. We 
test the result on three small pretrained models. Our method does not make 
the model smaller; it makes it a bit slower because of a loop in the code, but 
it lets us explore whether the added expressiveness improves accuracy.

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