# MoE-DeltaKV-Paged: A Hybrid Block-Sparse Attention with Differentiable Temporal Delta Compression

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/moe_deltakv_paged.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We present **MoE-DeltaKV-Paged**, a hybrid attention module that combines 
three orthogonal efficiency techniques:

1. **Mixture-of-Block-Attention (MoBA) routing:** A learned router selects 
   the top-$k$ most relevant pages of keys/values per query page, yielding 
   block-sparse attention.
2. **Delta-KV temporal compression:** Adjacent key/value pairs are differenced 
   into temporal deltas, factorized via low-rank SVD, and quantized to 
   4-bit using a Straight-Through Estimator.
3. **Paged attention:** The sequence is partitioned into fixed-size pages 
   (default 64 tokens), enabling efficient block-wise computation and 
   page-level routing.

We are explicit about two critical caveats:

**Caveat 1.** The Delta-KV and 4-bit quantization steps are **simulations** 
implemented with the Straight-Through Estimator (STE). They change the 
forward-pass behavior of the model (making attention robust to low-precision 
reconstruction), but they do **not** reduce the actual KV cache memory 
footprint at inference time. The reconstructed $K_\text{hat}, V_\text{hat}$ 
have the same shape as the original.

**Caveat 2.** The block-sparse routing provides sub-quadratic attention **only 
when $k_\text{experts} \ll \text{num\_pages}$**. For sequences with fewer than 
$k_\text{experts} + 1$ pages, the module falls back to dense attention.

On SST-2, MoE-DeltaKV-Paged achieves **86.47%** on DistilBERT 
(baseline 0.85%, Δ = -4.24), **59.75%** on RoBERTa (baseline 
32%, Δ = -18.81), and **50.92%** on GPT-2 (baseline 
0.85%, Δ = -28.44), using 50 LoRA recovery steps. 
The module includes CLS-token preservation via page-0 logit boosting, an 
auxiliary load-balancing loss to prevent router collapse, and full multi-architecture weight transfer.

**Keywords:** mixture of experts, block-sparse attention, delta compression, 
paged attention, differentiable quantization, straight-through estimator

---

## Lay Summary

We've built an attention module that combines three ideas to make transformers 
faster. First, it groups tokens into "pages" (blocks) and lets each query 
page attend only to the top-k most relevant key pages — a form of sparse 
attention. Second, it stores keys and values as *differences* between 
consecutive positions (temporal deltas), which are more compressible than 
raw values. Third, it simulates 4-bit quantization of these deltas using a 
technique called a Straight-Through Estimator, so the model learns to be 
robust to low-precision reconstruction. **Important:** the compression 
simulation does not actually reduce memory — it's a training-time technique 
that could enable future real compression. We also make sure the first page 
(containing the `[CLS]` token) is always selected by the router.

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