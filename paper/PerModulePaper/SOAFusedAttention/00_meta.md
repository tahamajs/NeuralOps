# SOA Fused Attention: Zero-Shot Low-Rank Factorization of Multi-Head Attention Projections

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/soa_fused_attention.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Transformer attention projections $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ 
account for 75% of the parameters in a standard attention block. We introduce 
**SOA Fused Attention**, a parameter-efficient replacement that factorizes 
each projection into a low-rank bottleneck pair:

$$
W_Q \approx U_Q V_Q, \quad U_Q \in \mathbb{R}^{d \times k}, \quad V_Q \in \mathbb{R}^{k \times d}
$$

where $k \ll d$ is the bottleneck rank. Factor initialization is **zero-shot**: 
we take the truncated SVD of the pretrained attention weight and split the 
singular values symmetrically between the two factors. This requires **no 
calibration data** and achieves near-optimal reconstruction at the target 
rank.

We replace $d \times d$ with $k(d + d) = 2kd$ parameters per projection, 
yielding a compression ratio of $\frac{d}{2k}$. For $d = 768$ and $k = 128$, 
this is a **3× reduction**. LoRA adapters attached to the up-projection 
provide recovery capacity; base factor weights remain frozen to preserve 
compression.

On SST-2, SOA Fused Attention achieves **{{D_ACC}}%** on DistilBERT 
(baseline {{BASE_D}}%), **{{R_ACC}}%** on RoBERTa (baseline {{BASE_R}}%), 
and **{{G_ACC}}%** on GPT-2 (baseline {{BASE_G}}%), with **{{D_RATIO}}×–{{G_RATIO}}×** 
parameter reduction on attention projections. Peak VRAM decreases by 
**{{VRAM_REDUCTION}}%**. We support GQA, RoPE, and both causal and 
bidirectional attention. The implementation is production-hardened with 
per-projection Frobenius validation, safe fallback, and a nested-encoder 
bypass for RoBERTa.

**Keywords:** attention compression, low-rank factorization, zero-shot 
initialization, SVD, parameter-efficient transformers

---

## Lay Summary

A transformer's attention block contains four large matrices (Q, K, V, O) 
that each contain ~590,000 numbers for a typical 768-dim model. We show that 
three of these (Q, K, V) can be compressed to about 200,000 numbers each 
without any additional training data — by analyzing the matrices themselves. 
We split each into a "down" and "up" pair, initialized from the matrix's own 
singular value decomposition. This gives a 3× reduction in the Q/K/V 
parameters of every attention block, at a small accuracy cost that we recover 
with lightweight adapters.

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
