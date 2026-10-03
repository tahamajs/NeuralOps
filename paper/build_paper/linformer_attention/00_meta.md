# Linformer Attention: Production-Hardened Low-Rank Sequence Projection for Transformers

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/linformer.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Standard transformer attention is quadratic in sequence length $L$, making 
long-context inference expensive. Linformer (Wang et al., 2020) reduces 
this to linear by projecting keys and values to a fixed dimension $k$ via 
two learned matrices $E, F \in \mathbb{R}^{k \times L}$. We present a 
**production-hardened implementation** of Linformer that (i) initializes 
$E$ and $F$ with a diagonal identity block plus small Gaussian noise to 
accelerate convergence, (ii) adaptively selects the appropriate slice of 
$E$ and $F$ based on $L$ vs $k$ vs the pre-allocated maximum sequence 
length, and (iii) falls back to exact dense causal attention when the 
projection would break causality.

On SST-2, our Linformer implementation achieves **50.92%** on DistilBERT 
(baseline 0.85%, Δ = -39.79), **48.17%** on RoBERTa (baseline 
32%, Δ = -30.39), and **77.5%** on GPT-2 (baseline 
0.85%, Δ = 0.85), using 50 LoRA recovery 
steps. Complexity is $\mathcal{O}(L \cdot k \cdot d_h)$ — linear in $L$ for 
fixed $k$.

The implementation increases parameter count by approximately 
**66.4M%** per layer due to the $E, F$ matrices, LoRA adapters, and 
output scale. We are explicit that this is an **inference-compute** 
optimization, not a parameter reduction.

**Keywords:** linformer, low-rank attention, linear complexity, 
efficient transformers, sequence projection

---

## Lay Summary

Standard attention compares every word to every other word, which becomes 
slow for long texts. Linformer solves this by projecting all the keys and 
values into a fixed-size summary (say 64 slots) using two learned matrices. 
We've built a hardened version of Linformer that works across four different 
model architectures (DistilBERT, RoBERTa, GPT-2, Qwen) and gracefully falls 
back to standard attention when the low-rank projection would break causal 
masking in decoder models. Our implementation is production-ready and 
includes LoRA adapters for cheap fine-tuning.

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