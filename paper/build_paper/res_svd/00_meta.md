# Res-SVD Attention: Spectral Residual Low-Rank Factorization for Attention Projections

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/res_svd_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We present **Res-SVD Attention**, a drop-in attention replacement that stores 
each projection as a **two-level SVD decomposition**:

$$
W = U_\text{base} V_\text{base}^\top + \alpha B A
\tag{1}
$$

where $U_\text{base} V_\text{base}^\top$ is a **frozen** rank-$r_b$ SVD 
approximation, and $B A$ is a **trainable** rank-$r_r$ residual adapter 
whose columns are **initialized from the top-$r_r$ singular directions of 
the residual $W - U_\text{base} V_\text{base}^\top$**.

This combines two ideas that are usually separate: (i) low-rank SVD 
compression of attention projections, and (ii) parameter-efficient 
fine-tuning of the compressed model. The **spectral initialization** of the 
LoRA adapter ensures that at initialization the combined approximation 
matches the best rank-$(r_b + r_r)$ SVD of the original weight — the 
residual is not random but populated with the "second-order" spectrum.

Res-SVD Attention supports four attention families — flat (DistilBERT), 
nested (RoBERTa/BERT), causal Conv1D (GPT-2), and grouped-query with rotary 
embeddings (Qwen/LLaMA) — with weight transfer, GQA, RoPE, and causal 
masking.

On SST-2 with 50 LoRA recovery steps and $(r_b, r_r) = 
(32, 16)$, Res-SVD Attention achieves 
**89.56%** on DistilBERT (Δ = -1.15), **76.61%** on RoBERTa 
(Δ = -1.95), and **71.33%** on GPT-2 (Δ = -8.03). Total 
parameter count decreases by **50\%×–2.1\times×**.

<!-- CRITICAL HONESTY -->
Res-SVD Attention **reduces** parameter count. The residual adapter adds 
$r_r(d_\text{in} + d_\text{out})$ trainable parameters, but this is 
offset by the reduction from SVD factorization of the projections. The 
"residual" in the name refers to the **residual weight matrix after base 
SVD**, not to residual connections.

**Keywords:** spectral initialization, residual SVD, low-rank factorization, 
parameter-efficient fine-tuning, attention compression

---

## Lay Summary

Transformer attention layers contain large weight matrices. Res-SVD stores 
each matrix in two parts: a "coarse" part computed by SVD that stays fixed, 
and a "fine" part that is small and trainable. The fine part is not random — 
it is initialized from the second-order singular directions of the original 
matrix, so the combined approximation is as close to the true weight as 
possible. This allows a small, compressed model to be fine-tuned efficiently 
while preserving most of the original information.

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