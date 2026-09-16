# Adaptive SVD Attention: Energy-Thresholded Low-Rank Factorization with Residual Adapters

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/adaptive_svd_attention.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Low-rank factorization of attention projections reduces parameter count but 
requires a **rank selection policy** that is often fixed per model. We 
introduce **Adaptive SVD Attention**, which chooses rank **dynamically 
per projection** based on cumulative singular-value energy:

$$
r^* = \min\left\{ r : \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{d} \sigma_j^2} \geq \tau \right\}, \quad \tau = {{ENERGY_THRESHOLD}}
$$

This adapts to the intrinsic rank of each projection, allocating more 
capacity to projections that require it. The factorization uses a **symmetric 
$\sqrt{\Sigma}$ split**:

$$
W \approx U_r V_r, \quad U_r = U_{:, :r} \Sigma_r^{1/2}, \quad V_r = \Sigma_r^{1/2} V_{:r, :}^\top
$$

and augments the frozen SVD base with a **trainable residual adapter**:

$$
W_\text{eff} \approx U_r V_r + B A
$$

Unlike prior attention-factorization methods that preserve the output 
projection, we factorize **all four projections** $W_Q, W_K, W_V, W_O$, 
achieving parameter reduction across the full attention block.

On SST-2, Adaptive SVD Attention achieves **{{D_ACC}}%** on DistilBERT 
(baseline {{BASE_D}}%), **{{R_ACC}}%** on RoBERTa (baseline {{BASE_R}}%), 
and **{{G_ACC}}%** on GPT-2 (baseline {{BASE_G}}%). Attention-block parameter 
reduction is **{{D_RATIO}}×–{{G_RATIO}}×**. The method is **zero-shot** — no 
calibration data is needed — and supports GQA, RoPE, T5 relative attention 
bias, and cross-attention. The average selected rank is **{{AVG_RANK}}** 
across layers.

**Keywords:** adaptive rank selection, singular value decomposition, energy 
thresholding, attention compression, T5 relative attention

---

## Lay Summary

Transformers store their attention as four big matrices per layer. We 
replace each with two smaller matrices initialized from the weight's own 
SVD. Unlike prior work that uses a fixed compression ratio for all layers, 
we let each matrix choose its own size — larger for projections that carry 
more information, smaller for others. This "adaptive" choice comes from 
looking at how quickly the matrix's singular values decay. On three popular 
language models, we reduce attention parameters by 2–3× with only 2–4% 
accuracy loss.

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
```

---