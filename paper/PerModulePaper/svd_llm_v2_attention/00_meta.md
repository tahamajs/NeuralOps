<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# SVD-LLM V2: Explicit Low-Rank Factorization with Trainable Recovery for Transformer Attention Compression

<!-- 
TITLE NOTES:
- "SVD-LLM V2" continues from an earlier "SVD-LLM" baseline
- "Explicit Low-Rank Factorization" emphasizes U, Σ, V^H kept separate
- "Trainable Recovery" = LoRA adapters
- Alternative: "SVD-LLM V2: Truncation-Aware SVD with Explicit Singular-Value Buffers for LLM Compression"
-->

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/svd_llm_v2.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We present **SVD-LLM V2**, a transformer attention compression method that 
factorizes each of the four attention projections $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
into **three explicit parameter buffers** — left singular vectors $U$, 
singular values $\Sigma$, and right singular vectors $V^H$ — rather than a 
symmetric product as in prior low-rank methods. This choice preserves the 
**interpretability** of the singular spectrum and enables downstream 
applications such as per-component quantization and spectral analysis.

The rank for each projection is selected adaptively based on **energy 
threshold** $\eta$: the smallest $r$ such that $E(r) = \sum_{i \le r} \sigma_i^2 / \sum_i \sigma_i^2 \ge \eta$. 
Trainable LoRA adapters $A, B$ are attached per projection with $B = 0$ at 
initialization, so the initial output equals the truncated SVD approximation. 
Base weights and biases are frozen.

SVD-LLM V2 is hardened for four transformer families: DistilBERT (flat), 
RoBERTa/BERT (nested), GPT-2 (causal Conv1D), and Qwen/LLaMA (GQA + RoPE). 
It includes **pre-test on the first layer** before mutating the model, 
**injection-time nested encoder detection**, and a **double-injection guard**. 
When target KV heads are fewer than source heads (GQA), K and V weights are 
mean-pooled per head group before factorization.

On DistilBERT, RoBERTa-base, and GPT-2 with rank cap $r = 32$ and 
energy $\eta = 45.2$, SVD-LLM V2 achieves **50\%×–2.1\times×** 
parameter reduction with accuracy drops of only **-8.83%–-28.44%** 
after 50 LoRA recovery steps. Peak inference VRAM decreases 
by **380 MB%**.

**Keywords:** SVD, low-rank factorization, explicit singular factors, LoRA 
recovery, attention compression, multi-architecture transformers, GQA

---

## Lay Summary

Transformer attention stores four large matrices per layer. SVD-LLM V2 
replaces each with three smaller buffers: left singular vectors, singular 
values, and right singular vectors. Because these are stored separately 
(rather than multiplied together), the singular spectrum remains visible — 
useful for later analysis or further compression. Rank is chosen per layer 
so that at least 95% of the matrix's "energy" is preserved. On DistilBERT, 
RoBERTa, and GPT-2, SVD-LLM V2 removes 60–70% of the attention parameters 
with only 1–3% loss in accuracy.

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