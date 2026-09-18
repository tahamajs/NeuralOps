<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# RPC: Random Projection Compressed Attention for Transformers

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/rpc_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Self-attention computes an $L \times L$ score matrix from $Q, K \in \mathbb{R}^{L \times d_h}$, 
dominated by the $\mathcal{O}(L^2 d_h)$ cost of the matrix product $Q K^\top$. 
We introduce **RPC** (**R**andom **P**rojection **C**ompressed) attention, 
which sketches $Q$ and $K$ into a lower-dimensional subspace before score 
computation:

$$
\text{score}_{ij} = \frac{(Q_i \Omega)(K_j \Omega)^\top}{\sqrt{r}}, \qquad
\Omega \in \mathbb{R}^{d_h \times r}, \; r < d_h
$$

where $\Omega$ is a **fixed head-wise Gaussian random matrix**, non-trainable. 
By the Johnson–Lindenstrauss lemma, inner products are approximately preserved 
under this projection with high probability, so the sketched attention 
approximates the full attention while reducing score FLOPs from 
$\mathcal{O}(L^2 d_h)$ to $\mathcal{O}(L d_h r + L^2 r)$.

RPC supports four attention families through a single weight-transfer 
routine: flat (DistilBERT), nested (RoBERTa/BERT), causal Conv1D (GPT-2), 
and grouped-query with rotary embeddings (Qwen/LLaMA). Base projections 
are frozen; only LoRA adapters are trained.

We evaluate RPC on DistilBERT, RoBERTa, and GPT-2 across SST-2 with 
50 LoRA recovery steps. RPC achieves **87.84%** on 
DistilBERT (Δ = -2.87), **66.97%** on RoBERTa (Δ = -11.59), 
and **57.34%** on GPT-2 (Δ = -22.02).

<!-- CRITICAL HONESTY -->
RPC **increases** parameter count by **48.5\%%** due to LoRA 
adapters on four projections. RPC **does not reduce memory**: the attention 
score matrix remains $L \times L$, only its content is computed in a lower 
dimension. The "compressed" in the name refers to the Q/K representations, 
not to model or memory size.

**Keywords:** random projection, Johnson-Lindenstrauss, attention sketching, 
low-dimensional attention, transformer efficiency

---

## Lay Summary

Attention in transformers compares every token against every other token, 
which is expensive. RPC speeds this up by first projecting each token into 
a smaller space using a fixed random matrix, then doing the comparison 
there. Because random projections approximately preserve distances (a 
mathematical fact called the Johnson-Lindenstrauss lemma), the result is 
close to the original attention but cheaper to compute. RPC works on four 
popular model families without retraining the base weights. It does not 
make the model smaller or use less memory — only less compute.

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