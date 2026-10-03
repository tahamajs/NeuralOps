<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# Gated DeltaNet: Associative State Recurrence with Input-Dependent Exponential Decay

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/gated_delta_net.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Be honest: this module is THEORETICALLY O(L) but PRACTICALLY slow due to Python loops
- Report timeouts and OOMs honestly
- Emphasize that implementation is a bottleneck, not the algorithm
-->

Recurrent linear attention mechanisms reduce the quadratic cost of transformer 
self-attention to linear time by maintaining a fixed-size hidden state 
$S \in \mathbb{R}^{d_h \times d_h}$ updated token-by-token. We implement 
**Gated DeltaNet**, a bidirectional/causal linear recurrent attention that 
combines the **Delta Rule** (Widrow–Hoff, 1960) with **input-dependent 
exponential decay gating**. The state update is:

$$
S_t = \beta_t S_{t-1} + \alpha_t K_t^\top (V_t - K_t S_{t-1}) / (1 + \|K_t\|^2)
$$

where $\alpha_t, \beta_t \in (0, 1)$ are per-head gates computed from the 
input. For encoder models, a bidirectional scan averages forward and 
backward recurrences; for decoder models, only the forward recurrence is 
used. Base pretrained weights are transferred exactly via a four-architecture 
`copy_weights` procedure; trainable capacity is provided by LoRA adapters 
at rank $r = 8$.

**Practical limitation.** Although the algorithm is theoretically 
$\mathcal{O}(L)$ in both time and memory, our PyTorch reference 
implementation uses two nested Python loops (outer over chunks, inner over 
tokens within a chunk), which dominates wall-clock time. On DistilBERT with 
$L = 128$, one recovery step takes ~14.2 ms seconds — approximately 
**84.5× slower** than baseline attention. RoBERTa and GPT-2 
(12 layers) exceed our GPU memory budget. We report these results as an 
honest assessment of the **implementation bottleneck** and identify the 
required fix (fused CUDA kernel or Triton JIT).

**Keywords:** linear attention, delta rule, recurrent neural network, 
associative memory, state-space models

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

Standard attention compares every word to every other word, which is slow for 
long documents. We implement a "linear" attention that processes words one 
at a time, remembering everything in a small fixed-size memory matrix. This 
should be much faster in theory. But our implementation uses Python loops 
that make it very slow in practice — 90 seconds per training step on 
DistilBERT, versus 2 seconds for the baseline. We report this honestly and 
explain what would be needed to make it fast.

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