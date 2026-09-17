<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Dobi-SVD: Differentiable Low-Rank Factorization with Learnable Spectral Gating

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/compression_engine/dobi_svd.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Low-rank factorization compresses transformer attention by replacing each 
weight matrix $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ with a 
product $U V$ where $U \in \mathbb{R}^{d_\text{out} \times r}$ and 
$V \in \mathbb{R}^{r \times d_\text{in}}$. The **rank $r$ is a discrete 
hyperparameter**: too small loses accuracy, too large defeats the purpose. 
Existing methods fix $r$ a-priori or select it via post-hoc search, treating 
rank selection as a non-differentiable problem.

We introduce **Dobi-SVD**, which makes rank selection **differentiable** via 
a learnable spectral gate $\alpha \in \mathbb{R}^r$. Each rank component is 
scaled by $\sigma(\alpha_i)$:

$$
y = (x (V \odot \sigma(\alpha))^\top) U^\top + b
\tag{1}
$$

Gates are initialized at $\alpha_i = 0.5$ so that 
$\sigma(\alpha_i) \approx 1.0$ at step 0. A **gate-compensation 
scale** $1/\sigma(\alpha_i)$ is pre-applied to $V$ so that the initial 
forward pass exactly reproduces the truncated SVD — the gating is a 
**differentiable refinement**, not a perturbation.

At the end of training, gates that have drifted toward zero identify **rank 
components that can be pruned** post-hoc, yielding a smaller effective rank 
without retraining. We provide a production-hardened implementation with 
multi-architecture support (DistilBERT, RoBERTa, GPT-2, Qwen3.5), Conv1D 
handling, LoRALinear wrapping, adaptive rank escalation, and safe fallback.

On SST-2, Dobi-SVD achieves **90.48%** on DistilBERT (baseline 
0.85%, Δ = -0.23), **74.54%** on RoBERTa (Δ = -4.02), 
and **58.14%** on GPT-2 (Δ = -21.22), with effective parameter 
ratio **2.1\times** (DistilBERT).

**Keywords:** differentiable SVD, spectral gating, learnable rank selection, 
low-rank factorization, attention compression, multi-architecture transformers

---

## Lay Summary

Standard low-rank compression replaces each large matrix with a smaller 
pair, but the size of the pair (rank) must be chosen in advance. Dobi-SVD 
makes this choice **learnable**: each rank component has a numeric "gate" 
that multiplies its contribution. Gates start near 1 (so the model starts 
unchanged) and are trained alongside the recovery adapters. At the end of 
training, gates that learned to be near 0 identify components that can be 
safely removed. The result is a self-pruning compression scheme.

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