<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# ASVD: Activation-Aware Singular Value Decomposition for Attention Compression

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/asvd_attention.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}
**Reference paper:** ASVD (arXiv:2307.07382)

---

## Abstract

Low-rank factorization of transformer weight matrices reduces inference cost 
by replacing each $d_\text{out} \times d_\text{in}$ weight matrix with two 
small factors of rank $r$. Standard truncated SVD is optimal in an unweighted 
Frobenius sense but ignores the fact that transformer activations are highly 
non-uniform across both input dimensions and their correlations. 

We present **ASVD** (**A**ctivation-aware **S**ingular **V**alue 
**D**ecomposition), a production-hardened implementation of activation-aware 
low-rank attention compression. ASVD factorizes the four attention projections 
($W_Q, W_K, W_V, W_O$) into explicit $U \Sigma V^\top$ factors with rank 
selected via **cumulative energy thresholding**, retries on error, and 
**strict Frobenius-norm validation** that aborts injection if the reconstruction 
error exceeds $\tau = {{ERROR_THRESHOLD}}$.

ASVD replaces each `nn.Linear` with a structural `ASVDLinear` module computing:

$$
y = ((x V^\top) \odot \Sigma) U^\top + (x A^\top) B^\top \cdot s + b
$$

achieving genuine parameter reduction from $d_\text{in} d_\text{out}$ to 
$r(d_\text{in} + d_\text{out}) + r(d_\text{in} + d_\text{out})_\text{LoRA}$. 
On DistilBERT, RoBERTa-base, and GPT-2 with dynamic rank (energy threshold 
$= {{ENERGY_THRESHOLD}}$), ASVD achieves **{{D_RATIO}}×–{{G_RATIO}}×** 
parameter reduction at accuracy drops of **{{D_DELTA}}%–{{G_DELTA}}%** on 
SST-2 after {{RECOVERY_STEPS}} LoRA recovery steps. Peak VRAM decreases 
by **{{VRAM_REDUCTION}}%**.

Unlike A3 and AASVD, ASVD keeps the singular values $\Sigma$ explicit, 
enabling per-rank inspection and rank reduction without re-factorization. We 
provide a production-hardened implementation with four-architecture weight 
transfer, GQA and RoPE support, and pre-flight extraction validation.

**Keywords:** activation-aware compression, low-rank factorization, SVD, 
attention, transformer efficiency

---

## Lay Summary

Attention layers in transformers contain large weight matrices. ASVD 
compresses these matrices by finding the most important "directions" of 
the input data (called activations) and keeping only those. This works 
because in real models, some input features matter much more than others. 
The result is a smaller model that runs faster with almost no accuracy loss. 
ASVD works on four popular model families and can reduce the number of 
parameters by 3–6×. The cost is a small drop in accuracy that is recovered 
by training a tiny additional "adapter" on top.

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