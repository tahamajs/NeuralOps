# Galerkin Attention: Petrov-Galerkin Projection for Linear-Complexity Transformer Self-Attention

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/galerkin.py
**License:** {{LICENSE}}

---

## Abstract

Self-attention is fundamentally a **continuous non-local integral operator** 
that maps query signals to context signals via a kernel:

$$
(\mathcal{K}Q)(s) = \int \kappa(s, t) V(t) \, dt
\tag{1}
$$

Standard implementations discretize this operator on the raw token grid, 
requiring $\mathcal{O}(N^2 d)$ computation for $N$ tokens. We propose 
**Galerkin Attention**, which formulates self-attention as a **Petrov-Galerkin 
projection** onto an orthonormal basis (a discrete cosine transform variant) 
of dimension $r \ll N$. Queries, keys, and values are projected into this 
basis, attention is computed at rank $r$ in basis space, and the output is 
reconstructed to the original sequence grid via basis synthesis. Complexity 
reduces from $\mathcal{O}(N^2 d)$ to $\mathcal{O}(N r d)$ — **linear in 
sequence length** — for encoder models.

For **decoder models** (GPT-2, Qwen), the projection-based formulation is 
incompatible with causal masking. We therefore fall back to **standard 
causal softmax attention**, sacrificing the linear-complexity benefit in 
exchange for exactness.

Base Q/K/V/O projections are frozen and copied from the pretrained model; 
LoRA rank-16 adapters provide trainable capacity. On DistilBERT and 
RoBERTa-base with SST-2, Galerkin Attention achieves **{{D_ACC}}%** and 
**{{R_ACC}}%** respectively, within **{{D_DELTA}}%** and **{{R_DELTA}}%** 
of the SDPA baseline. On GPT-2, it matches the baseline exactly 
(**{{G_ACC}}%**) because it uses the standard attention fallback.

**Important disclosure.** Galerkin Attention is **not** a compression 
method — parameter count increases by ~{{PARAM_CHANGE}}% via LoRA. It is 
a **compute-efficiency** method for encoder attention.

**Keywords:** Petrov-Galerkin projection, integral operator attention, 
linear-complexity attention, orthonormal basis, DCT

---

## Lay Summary

Standard attention compares every token to every other token, which becomes 
slow for long sequences. Galerkin Attention uses a different strategy: 
instead of working with individual tokens, it projects them onto a small 
set of "smooth basis functions" (similar to how audio is compressed as a 
sum of cosine waves). Attention is then computed between these smooth 
components, which is much faster. The result is projected back to the 
original sequence. This works well for encoders (like BERT) but not for 
decoders (like GPT-2), where we fall back to standard attention. The 
trade-off: faster compute for encoder models, no benefit for decoder models.

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
