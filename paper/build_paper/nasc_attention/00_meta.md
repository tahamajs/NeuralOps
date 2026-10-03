<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# NASC: NEXUS-AEGIS Spectral Cascade Attention for Hybrid Transformer Inference

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/nasc.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Three-branch cascade with batch-level gating
- softpick is a hardened softmax alternative
- Encoder path may default to dense (config-dependent)
- Sparse block has a Python loop
- inject_nasc raises on failure (unlike other modules)
-->

We present **NASC** (NEXUS-AEGIS Spectral Cascade Attention), a hybrid 
attention block that combines three complementary mechanisms: (i) a 
**local sliding-window attention** with strict causal masking, (ii) a 
**sparse block attention** that selects the top-$k$ key blocks by query-key 
mean alignment, and (iii) a **global spectral RFF attention** that projects 
queries and keys onto random Fourier features for linear-time global 
context. The three branches are fused with a learned softmax gate 
conditioned on the mean hidden state per batch:

$$
\text{Output} = g_0 \cdot \text{Local} + g_1 \cdot \text{Sparse} + g_2 \cdot \text{Global}
$$

where $g_0 + g_1 + g_2 = 1$.

We evaluate NASC on DistilBERT, RoBERTa, and GPT-2 on SST-2 with LoRA 
recovery. NASC achieves **90.48%** on DistilBERT (baseline 0.85%, 
Δ = -0.23), **89.22%** on RoBERTa (Δ = +10.66), and 
**85.32%** on GPT-2 (Δ = +5.96). We document three implementation 
caveats honestly: (a) the gate is **batch-level**, so all tokens in a 
batch share the same branch mixture; (b) `softpick` falls back to standard 
softmax when all rows in a batch have zero denominator, which is a coarse 
fix compared to per-row handling; and (c) `inject_nasc` **raises** on 
failure, unlike other modules in our framework.

**Keywords:** hybrid attention, sparse block, random Fourier features, 
sliding window, gating

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

NASC is like having three specialists vote on each attention decision. One 
specialist looks at nearby words, one looks at a few important distant 
blocks, and one looks at the whole sequence via a mathematical shortcut. A 
learned gate decides how much to trust each specialist. The gate is 
computed per batch, so all examples in a batch get the same mixture. We 
tested on three pretrained models and report the results along with 
several implementation caveats.

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