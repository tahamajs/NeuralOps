<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# Routing Attention: Content-Based Clustering for Sparse Attention

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/routing_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Honest: routing is a replacement, not compression
- k-means is non-differentiable
- Loop over clusters is inefficient
- Complexity is O(K·L²) in this naive implementation
-->

Sparse attention methods aim to reduce the quadratic cost of self-attention by 
restricting each query to a subset of keys. We introduce **Routing Attention**, 
which partitions the sequence into content-based clusters via online k-means 
on the joint query-key representation, then restricts each query to keys within 
its own cluster (plus a global context anchor at position 0).

The method has three components: (i) **content-based clustering** via 
$k$-means on $\ell_2$-normalized $(Q + K)$ features; (ii) **cluster-restricted 
attention** where each query attends only to keys in its assigned cluster; 
and (iii) **token-0 anchoring** which preserves the first token as a universal 
context key. Base weights are transferred from the pretrained attention via 
a multi-architecture `copy_weights` procedure, and trainable capacity is 
provided by LoRA adapters at rank $r = 8$.

We evaluate on DistilBERT, RoBERTa, and GPT-2 on SST-2. Accuracy is preserved 
within **-26.95%** (DistilBERT), **-27.64%** (RoBERTa), and 
**-28.44%** (GPT-2). Parameter count increases by **48.5\%%** 
due to LoRA adapters; this is **not** a compression method. We honestly report 
that the current implementation does **not** achieve real sparsity speedups: 
the loop over $K$ clusters recomputes $K$ full attention matrices, making the 
naive complexity $\mathcal{O}(K \cdot BHL^2)$ — worse than dense attention. 
We identify the necessary changes (sparse kernels, dense clustering with 
differentiable assignment) to make Routing Attention practical.

**Keywords:** sparse attention, k-means clustering, content-based routing, 
efficient transformers

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

Standard attention is slow for long sequences because every word looks at 
every other word. We propose to group similar words into clusters, then let 
each word look only at words in its own cluster. We also let every word look 
at the first word, which usually contains important context. The grouping is 
done automatically based on the words' hidden representations. We test our 
method on three small pretrained models. Our code works but does not yet run 
faster than normal attention — the way we implemented the loop over clusters 
recomputes the full attention matrix many times, offsetting the savings. We 
document this honestly and outline how to fix it.

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