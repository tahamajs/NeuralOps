# FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention with Block Gating and Short-Sequence Fast Path

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- 0.85² — 0.85

**Affiliations:**
- ¹ 0.85, 0.85
- ² 0.85, 0.85

**Corresponding author:** 0.85
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/fusion_attn.py
**License:** Apache 2.0

---

## Abstract

Long-context attention in transformers is quadratic in sequence length, 
$\mathcal{O}(L^2 d)$, which makes inference prohibitively expensive at 
$L \geq 4096$. Existing sparse methods — Longformer, BigBird, MoBA — reduce 
this to $\mathcal{O}(L \cdot W \cdot d)$ or $\mathcal{O}(K \cdot L \cdot d)$ 
but often apply the same sparsity pattern regardless of sequence length. 
We introduce **FUSIONATTN**, a **multi-paradigm hybrid attention** that 
dispatches between two paths based on sequence length:

1. **Short-sequence path ($L \leq 128$):** Standard softmax attention, no 
   sparsity overhead.
2. **Long-sequence path ($L > 128$):** Block-sparse attention with a 
   **learnable gating MLP** that scores query-key block pairs, selects the 
   top-$K$ blocks per query block, and computes attention only within the 
   selected blocks (**MoBA-style**).

The gate is a 2-layer MLP with GELU activation that operates on 
mean-pooled block representations; it is trained jointly with LoRA 
adapters. Base Q/K/V/O projections are frozen and copied exactly from the 
pretrained model, ensuring step-0 preservation.

**Important disclosure.** FUSIONATTN is **not** a compression method. It 
**increases** parameter count via LoRA adapters and the gate MLP. Its value 
lies in **long-context compute efficiency**, not size reduction. We 
additionally note that at $L \leq 128$ (our test sequence length), FUSIONATTN 
takes the **short path** and produces output identical to the pretrained 
model with no sparsity benefit.

We evaluate on DistilBERT, RoBERTa-base, and GPT-2 on SST-2 with 
50 LoRA recovery steps. FUSIONATTN achieves 
**86.4%** on DistilBERT, **78.2%** on RoBERTa, and **77.5%** 
on GPT-2, matching the SDPA baseline within noise.

**Keywords:** hybrid sparse attention, block gating, MoBA, short-sequence 
fast path, learnable sparsity, LoRA

---

## Lay Summary

Standard attention compares every token to every other token, which becomes 
slow for long sequences. FUSIONATTN takes a hybrid approach: for short 
sequences (under 128 tokens), it uses the standard fast attention. For 
longer sequences, it groups tokens into blocks and uses a small neural 
network to decide which blocks should "talk to" each other. Only the 
selected blocks compute attention. This saves computation for long 
sequences while keeping short-sequence behavior unchanged. The trade-off: 
it adds some parameters to the model, and it's not faster on short 
sequences.

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