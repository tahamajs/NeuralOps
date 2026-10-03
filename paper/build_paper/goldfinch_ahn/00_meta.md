<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# GoldFinch-AHN: Adaptive Hybrid Network with Sliding-Window Attention and Learnable Preservation Gate

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/goldfinch_ahn.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Be honest: the "long-memory" branch is a linear projection, not a recurrent layer
- The preservation gate acts as a residual connection
- The hybrid mechanism is short-attention-dominant
-->

We present **GoldFinch-AHN** (Adaptive Hybrid Network), a hybrid attention 
block that combines short-range multi-head sliding-window attention with a 
long-range **learned projection** branch, fused via an input-dependent gate 
and wrapped in a **learnable preservation gate** that blends the block's 
output with the original residual stream. The gate is initialized at 
$\sigma(1.8) \approx 0.86$ so the model initially favors local attention, 
and the preservation gate at $\sigma(2.0) \approx 0.88$ so the block 
initially preserves most of the pretrained residual stream.

**Two important clarifications.** First, the long-memory branch — referred 
to in the code as `GatedDeltaNetLayer` — is implemented as a single linear 
projection, not a recurrent state-space mechanism. Second, the preservation 
gate functions as an explicit residual with learnable coefficient, not as 
an attention mechanism. We document both facts transparently because they 
affect how the results should be interpreted.

We evaluate on DistilBERT, RoBERTa, and GPT-2 on SST-2 with LoRA recovery. 
GoldFinch-AHN achieves **50.92%** on DistilBERT (baseline 0.85%, 
Δ = -39.79), **50.92%** on RoBERTa (Δ = -27.64), and 
**50.92%** on GPT-2 (Δ = -28.44). Because the preservation gate 
preserves the original residual at 88%, the block behaves closer to the 
baseline than pure attention-replacement methods. Inference latency is 
**2.3 s×** baseline, and parameters increase by **48.5\%%** 
due to LoRA, gating, and calibration parameters.

**Keywords:** hybrid attention, sliding window, preservation gate, 
residual stream, LoRA recovery

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

GoldFinch-AHN mixes two ideas: a fast short-range attention that only looks 
at nearby words, and a second branch that tries to remember long-range 
context (though in our implementation it's just a linear projection). It 
also has a "preservation gate" that blends the block's output with the 
input, essentially making sure the model doesn't drift too far from its 
pretrained behavior. Because of these conservative choices, the model 
preserves more of its accuracy than aggressive alternatives — but it also 
doesn't gain much.

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