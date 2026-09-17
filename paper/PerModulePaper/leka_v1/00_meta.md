# LEKAv1: Linear-Time Efficient Kernel Attention with Random Fourier Features

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/leka_v1.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Standard transformer attention scales quadratically with sequence length, 
making long-context inference prohibitively expensive. **Linear attention** 
reduces this to $\mathcal{O}(L)$ by replacing softmax with a kernel 
feature map $\phi$ such that $\exp(q^\top k) \approx \phi(q)^\top \phi(k)$. 
The **Performer** (Choromanski et al., 2021) uses positive orthogonal 
random features (PORF); the **Linear Transformer** (Katharopoulos et al., 
2020) uses $\phi(x) = \text{elu}(x) + 1$.

We introduce **LEKAv1**, a production-hardened linear-time attention module 
that approximates the **Gaussian RBF kernel** $\exp(-\|q - k\|^2 / 2)$ 
using Random Fourier Features (RFF):

$$
\phi(x) = \frac{1}{\sqrt{F}} \left[ \cos(x W) \,\|\, \sin(x W) \right], 
\quad W \in \mathbb{R}^{d_h \times F}
$$

where $W$ is **QR-orthogonalized** for stability. The feature dimension 
$F = 64$ yields $\phi(x) \in \mathbb{R}^{2F}$. We apply a 
**head-wise LayerNorm** before the projection, then compute linear attention 
via a causal cumulative sum (decoders) or a global KV sum (encoders). A 
**preservation gate** blends the attention output with the residual 
input, controlled by a scalar $\alpha = \sigma(w)$:

$$
\text{out} = \alpha \odot A + (1 - \alpha) \odot x
$$

with $w_0 = 2$ (initial $\alpha \approx 0.88$).

On SST-2, LEKAv1 achieves **51.03%** on DistilBERT (baseline 
0.85%), **50.92%** on RoBERTa (baseline 32%), and 
**49.31%** on GPT-2 (baseline 0.85%). Inference scales linearly 
in sequence length: at $L = [LONG_L]$, LEKAv1 is **0.85×** 
faster than standard attention.

**We are explicit that LEKAv1 does not reduce parameter count.** It adds 
approximately 66.4M% parameters due to LoRA, the projection 
matrix, and the preservation gate. Its benefit is **$\mathcal{O}(L)$ 
inference** and **kernel-based attention** that composes with GQA and 
RoPE.

**Keywords:** linear attention, random Fourier features, kernel methods, 
long-context efficiency, Gaussian RBF

---

## Lay Summary

Standard transformer attention compares every word with every other word, 
which scales quadratically. We replace this with a **kernel approximation**: 
each word's "query" and "key" are mapped to a fixed-size random feature 
vector, and attention is computed via a running sum. This makes the 
computation scale linearly with text length. We use a Gaussian kernel 
approximation with orthogonalized random features, and add a small 
"preservation gate" that mixes the attention output with the original 
residual signal. On three standard models, we match baseline accuracy 
while enabling efficient long-context inference.

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