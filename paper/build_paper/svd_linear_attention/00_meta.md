<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# SVD-Linear Attention: Factorized Low-Rank Linear Attention with Exponential Decay Gating

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/svd_linear_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We introduce **SVD-Linear Attention**, a compression-and-efficiency module that 
simultaneously (i) reduces attention projection parameters via SVD-factorization 
and (ii) reduces attention complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ 
via a kernelized linear attention approximation. Our method combines four 
mechanisms within a single drop-in replacement for multi-head self-attention:

1. **SVD-factorized projections:** $W_Q, W_K, W_V, W_O$ stored as $U V^\top$ 
   with rank $r$, reducing projection parameters by factor ~50\%× at 
   $r = 32$.
2. **Kernelized linear attention:** $\phi(x) = \text{ELU}(x) + 1$ feature map, 
   enabling the identity 
   $\text{attn} = \phi(Q)(\phi(K)^\top V) / (\phi(Q) \phi(K)^\top \mathbf{1})$, 
   computable in $\mathcal{O}(N)$ via an associative prefix-sum.
3. **Exponential decay gating:** a data-dependent retention factor 
   $\lambda_t = \exp(-\text{softplus}(W_g x_t)) \in (0, 1)$, initialised with 
   log-spaced retention times from 1 to 1024 tokens.
4. **Chunked parallel scan:** a chunk size of 64 gives $\mathcal{O}(N)$ 
   complexity with $\mathcal{O}(\log C)$ parallel depth within each chunk.

SVD-Linear Attention supports four attention families — flat (DistilBERT), 
nested (RoBERTa/BERT), causal Conv1D (GPT-2), and grouped-query with rotary 
embeddings (Qwen/LLaMA) — and provides both a **batch mode** and a 
**step-recurrent mode** (`step_recurrent`) for incremental streaming inference 
with $\mathcal{O}(1)$ per-step cost.

On SST-2 with 50 LoRA recovery steps, SVD-Linear Attention 
achieves **80.73%** on DistilBERT (Δ = -9.98), **50.92%** on 
RoBERTa (Δ = -27.64), and **77.5%** on GPT-2 (Δ = 0.85), 
with peak VRAM reductions of **380 MB%**.

<!-- CRITICAL HONESTY -->
Unlike attention-replacement methods that add parameters (NHA, EAT, OHA, 
Infini-Attention, SVD-Mamba), SVD-Linear Attention **reduces** parameter count 
by factor 50\%×–2.1\times×. LoRA adapters add a small number of 
recovery parameters on top.

**Keywords:** linear attention, SVD factorization, exponential decay, 
associative memory, streaming inference

---

## Lay Summary

Standard attention compares every token with every other token, which costs 
$O(N^2)$ time. SVD-Linear Attention replaces this with a cheaper approach 
that runs in $O(N)$ time, while also compressing the model itself via low-rank 
factorization. The method works by:
1. Reducing each weight matrix to a smaller factorization (like ASVD).
2. Using a mathematical trick to convert attention to a linear-time operation.
3. Adding a "memory decay" mechanism that lets the model forget old information.

The result is a smaller, faster attention module that works on four popular 
model families. The method supports both batch training and streaming 
inference (one token at a time).

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