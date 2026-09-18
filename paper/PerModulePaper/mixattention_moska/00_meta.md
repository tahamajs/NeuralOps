<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# FASA: Frequency/Variance-Aware Sparse Attention for Bidirectional Encoders

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/fasa.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- FASA is a TOKEN PRUNING method (not attention replacement)
- Only applies to bidirectional encoders (decoder path skips pruning)
- Uses variance of K as importance proxy (heuristic)
- Protects CLS token always
-->

We present **FASA** (Frequency/Variance-Aware Sparse Attention), a 
token-pruning method that reduces the effective KV sequence length in 
bidirectional transformers by retaining a fixed fraction of tokens based 
on their key-variance. Given attention keys $K \in \mathbb{R}^{B \times H \times L \times d_h}$, 
FASA computes a per-token importance score:

$$
\text{imp}_i = \text{mean}_h \text{Var}_d(K_{i,h,d})
$$

and retains the top-$r \cdot L$ tokens along with a pinned **[CLS]** token 
at index 0. The pruned attention is then computed over a shortened sequence 
of length $r \cdot L$, reducing attention FLOPs by $r^2$ for the score 
matrix and $r$ for the value matmul.

**Two important constraints.** First, FASA is a **test-time pruning** 
method: no parameters are trained to select tokens; the selection is 
entirely heuristic. Second, **FASA is not applied to causal decoders**: 
the code path for causal and cross-attention uses full attention to 
preserve strict lower-triangular causality, so on GPT-2 the method acts 
as a LoRA-only adapter.

We evaluate on DistilBERT, RoBERTa, and GPT-2 on SST-2 with LoRA recovery. 
With `fasa_keep_ratio = 0.5`, FASA achieves **90.94%** on DistilBERT 
(baseline 0.85%, Δ = +0.23), **86.24%** on RoBERTa 
(Δ = +7.68), and **87.27%** on GPT-2 (Δ = +7.91). GPT-2's 
near-zero delta confirms the decoder path is not pruned.

**Keywords:** sparse attention, token pruning, KV compression, variance 
importance, attention sinks

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

FASA looks at each word's "key" representation and measures how variable 
it is across its dimensions. Words with high variance are kept; words with 
low variance are pruned away. This reduces the number of comparisons 
attention has to make. The first word (CLS) is always kept — this follows 
the "attention sink" insight that the first token matters disproportionately. 
We tested on three pretrained models and found that it works on encoders 
(which do two-way attention) but does nothing on causal models like GPT-2 
because those need strict causality.

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