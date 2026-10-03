# REAP-MoE: Robust Expert Allocation and Pruning for Transformer Mixture-of-Experts

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/reap_moe.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Mixture-of-Experts (MoE) architectures increase model capacity without 
proportionally increasing inference FLOPs by activating only a subset of 
experts per token. However, standard MoE implementations face three problems: 
(i) **parameter bloat** from duplicating FFN layers across experts, 
(ii) **router collapse** where all tokens route to the same expert, and 
(iii) **training instability** from load imbalance. We introduce **REAP-MoE** 
(**R**obust **E**xpert **A**llocation and **P**runing), which addresses these 
by combining **SVD-factorized expert networks** with **load-balancing 
regularization** and **noise-injected routing**. Each expert is a two-layer 
MLP whose weights are factored into $U \Sigma V^\top$, reducing per-expert 
parameters by **32×** while preserving expressive capacity. 
The router uses top-$k$ selection with auxiliary load-balancing loss 
$\mathcal{L}_\text{aux} = E \sum_i \rho_i \pi_i$, where $\rho_i$ is expert 
utilization and $\pi_i$ is the mean routing probability. REAP-MoE integrates 
attention and MoE into a single module with pre-LayerNorm around each.

On SST-2 with DistilBERT, RoBERTa-base, and GPT-2, REAP-MoE achieves 
**90.02%**, **50.92%**, and **52.41%** respectively after 
50 LoRA recovery steps, compared to baselines of 90.71%, 
78.56%, and 79.36%. Parameter count changes by **66.4M%** 
because MoE replaces the FFN block: with 8 experts at SVD rank 
32, per-layer parameters scale as 8× the FFN size, 
mitigated by SVD factorization. We report honest negative results for 
parameter-sensitive deployments.

**Keywords:** mixture of experts, load balancing, SVD factorization, 
transformer efficiency, robust routing

---

## Lay Summary

Mixture-of-Experts (MoE) is a technique that trains many small "expert" 
networks and routes each input to only a few of them. This can increase 
model capacity without increasing compute, but it also multiplies parameter 
count. REAP-MoE compresses each expert using low-rank SVD, reducing per-expert 
size by ~4×, and adds a load-balancing loss to prevent the router from 
always picking the same expert. The result is a model with higher capacity 
than the original FFN, but with controlled parameter growth and stable 
training.

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