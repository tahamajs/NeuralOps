<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# Gated DeltaNet Block: A Block-Level Wrapper for Hybrid Attention-Replacement

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/gated_deltanet_block.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- This is a BLOCK wrapper, not attention replacement
- Two paths: full MHA vs GatedDeltaNet
- Both paths have issues
-->

We present **GatedDeltaNetBlock**, a block-level wrapper that replaces an 
entire transformer block (attention + residual + normalization) with either 
(a) standard PyTorch `nn.MultiheadAttention` + LayerNorm, or (b) GatedDeltaNet 
attention + LayerNorm. Unlike attention-level replacements, this block 
intercepts the entire block computation, providing explicit control over 
normalization and residual structure.

**Two critical limitations.** First, the block's **LayerNorm parameters are 
not pretrained** — they initialize from $\gamma = 1, \beta = 0$ and must 
learn the correct scale during recovery. In pretrained transformer blocks, 
LayerNorm statistics are calibrated against the specific activation 
distribution of the pretrained attention; replacing them with fresh 
parameters causes a distributional mismatch that persists through training.

Second, the **weight-transfer path for `nn.MultiheadAttention`** uses a 
fragile shape-matching heuristic wrapped in a broad `try/except` that 
silently succeeds even when transfer fails, leaving attention weights 
at random initialization.

We evaluate on DistilBERT, RoBERTa, and GPT-2. On DistilBERT with 
`use_full_attention=False`, the block achieves **50.92%** (Δ = 
-39.79%) but at **14.2 ms s/step** — the underlying GatedDeltaNet 
recurrence dominates runtime. With `use_full_attention=True`, accuracy is 
**86.5%** (Δ = 0.85%). RoBERTa and GPT-2 fail due to 
memory constraints. We document these negative results and identify the 
specific code paths that must be fixed for the block to be viable.

**Keywords:** block-level replacement, LayerNorm initialization, hybrid 
attention, GatedDeltaNet, negative results

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

Instead of replacing just the attention part of a transformer, we tried 
replacing the whole block — attention plus normalization plus residual 
connection. This gives us more control but adds a new LayerNorm that 
wasn't pretrained. We tested two variants: one using standard attention, 
one using a linear recurrent attention. Both have problems: the first 
silently fails when weight transfer doesn't match, the second is too slow. 
We report both honestly.

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