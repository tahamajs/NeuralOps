# CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel and Adaptive Gating

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** Your Name
**Preprint date:** 2026-09-16
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/chela_slay.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Linear attention replaces the quadratic softmax over $L^2$ attention pairs 
with an associative kernel feature map, achieving $\mathcal{O}(L)$ complexity. 
However, naive linear attention is often unstable and fails to recover 
pretrained attention accuracy. We introduce **CHELA-SLAY**, a hybrid attention 
that fuses a **dense softmax stream** with a **spherically normalized linear 
stream** via a learnable per-head gate.

The linear stream combines three components:

1. **Spherical normalization.** We project $Q, K$ onto the unit hypersphere:
   
   $$
   \hat{Q} = \frac{Q}{\|Q\|_2 + \epsilon}, \quad
   \hat{K} = \frac{K}{\|K\|_2 + \epsilon}
   $$
   
2. **Yat-Kernel feature map.** We apply a trigonometric random Fourier 
   mapping:
   
   $$
   \phi(x) = \frac{1}{F^{1/4}} [\cos(x \Omega) \,\|\, \sin(x \Omega)]
   $$
   
   producing a $2F$-dimensional positive feature vector.
3. **Adaptive gating.** A per-head, per-token gate blends the dense and 
   linear streams. The gate is initialized with a bias of 
   $b_g = 5.0$ so that $\sigma(b_g) \approx 0.993$ 
   — biasing the output toward the dense path at step 0.

We evaluate on DistilBERT, RoBERTa-base, and GPT-2 on SST-2. CHELA-SLAY 
achieves **89.91%** on DistilBERT (baseline 90.71%), 
**83.83%** on RoBERTa (baseline 78.56%), and **85.89%** on 
GPT-2 (baseline 79.36%) using 50 LoRA recovery steps.

**We report two limitations honestly.**

1. **Memory.** The causal cumsum path materializes a 
   $(\text{B}, \text{H}, L, 2F, d_h)$ intermediate, which for 
   $L = 128$, $F = 128$, $d_h = 64$ is roughly 
   **8192 MB** per layer at FP32.
2. **Speed.** On SST-2 ($L = 128$), CHELA-SLAY is **2.2× slower** 
   than baseline because both dense and linear streams are computed 
   simultaneously, and the cumsum has overhead.

The linear stream would dominate only at longer sequences 
($L \geq 1024$), where the quadratic term of the dense stream 
becomes the bottleneck.

**Keywords:** linear attention, random Fourier features, spherical 
normalization, hybrid attention, adaptive gating

---

## Lay Summary

Standard attention compares every word with every other word — a calculation 
that grows quadratically with text length. Linear attention replaces this 
with a compressed feature map, giving a linear cost. But linear attention 
is often unstable and loses accuracy. We combine the best of both: we keep 
the standard attention alongside a "linear" version, and let the model 
learn how much to trust each. To make the linear version more stable, we 
project the query and key vectors onto a unit sphere and use a trigonometric 
feature map. The model starts by trusting the standard path (initialized 
gate ≈ 0.993 toward dense) and gradually learns to blend.

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