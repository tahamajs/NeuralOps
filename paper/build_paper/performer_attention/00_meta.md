<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Performer: Fast Attention via Positive Orthogonal Random Features (FAVOR+) for Pretrained Transformers

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/performer.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Performer is a PUBLISHED method (Choromanski et al., ICLR 2021)
- Our contribution is the PRODUCTION-HARDENED PORT to pretrained transformers
- Uses FAVOR+ (Positive Orthogonal Random Features)
- Exact causal prefix scan for decoders
- BUG: inject_performer raises instead of returning None (contrary to docstring)
-->

The Transformer's self-attention has $\mathcal{O}(L^2)$ time and memory 
complexity, which is prohibitive for long sequences. **Performer** 
(Choromanski et al., ICLR 2021) addresses this by approximating the softmax 
kernel with **positive orthogonal random features** (FAVOR+), achieving 
$\mathcal{O}(L)$ complexity while preserving attention's expressiveness. We 
present a **production-hardened port** of Performer to pretrained 
transformers, supporting four architecture families — DistilBERT, 
RoBERTa/BERT, GPT-2, and Qwen/LLaMA — with exact weight transfer, GQA, 
RoPE, and two computational paths: a **bidirectional** path via 
associative accumulation and a **causal prefix-scan** path with 
chunk_size = 64 for decoders. The feature map includes numerical 
stabilization via max-subtraction and norm-subtraction, avoiding overflow 
in FP16.

We evaluate Performer on DistilBERT, RoBERTa, and GPT-2 on SST-2 with LoRA 
recovery at rank $r_L = 8$ and $F = 64$ features. 
Performer achieves **50.92%** on DistilBERT (baseline 0.85%, 
Δ = -39.79), **50.92%** on RoBERTa (Δ = -27.64), and 
**77.5%** on GPT-2 (Δ = 0.85). We document two important 
discrepancies between the code and its documentation: (a) `inject_performer` 
**raises** on failure rather than returning `None` as the docstring claims, 
and (b) `LoRALinear` contains an extra `output_scale` parameter not present 
in the baseline checkpoint.

**Keywords:** Performer, FAVOR+, random features, kernel attention, linear 
attention, production port

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

Performer replaces the expensive softmax attention with a mathematical 
approximation based on random features. This makes attention linear in 
sequence length instead of quadratic. We ported this to three pretrained 
models and tested how much accuracy they lose. The results show moderate 
degradation. We also document a bug where the injection code raises 
exceptions instead of returning `None` as documented, which could cause 
unexpected crashes in production.

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