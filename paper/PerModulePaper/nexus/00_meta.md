# NEXUS: Neural Efficient X-scoring Unified Sparse Attention

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/nexus.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We present **NEXUS** (**N**eural **E**fficient **X**-scoring **U**nified **S**parse 
attention), a modular attention framework that unifies six recent advances in 
sparse attention: (1) XAttention antidiagonal block scoring (training-free), 
(2) SeerAttention learnable gating, (3) DuSA dual-stage intra/inter-block 
sparsity, (4) AnchorAttention stripe band masking, (5) Spectraformer-style 
Random Fourier Feature global rectification, and (6) softpick rectified softmax.

The core mechanism partitions the sequence into blocks of size $B_s$, scores 
block-to-block interaction with coarse subsampled correlations, retains the 
top-$k$ key blocks per query block, and computes exact intra-block attention 
with optional stripe masking. A learnable Seer gating network provides a 
data-dependent override, and periodic RFF layers inject a global receptive 
field.

On SST-2, NEXUS achieves **50.92%** on DistilBERT (baseline 0.85%, 
Δ = -39.79), **50.92%** on RoBERTa (baseline 32%, Δ = -27.64), 
and **51.61%** on GPT-2 (baseline 0.85%, Δ = -27.75), using 
50 LoRA recovery steps.

**Important honest caveats:**

- **Encoder-only sparsity.** The block-sparse path is used only for 
  non-causal models. Causal decoders (GPT-2) always fall back to dense 
  attention, so NEXUS provides **no compute savings on GPT-2**.
- **Stripe masking is gated at $N > 256$.** For SST-2 ($N = 128$), the stripe 
  component is inactive.
- **RFF rectification is disabled by default** when pretrained weights are 
  provided. It activates only when `enable_spectraformer=True`.
- **Softpick activation** occasionally produces near-uniform distributions 
  when all logits are negative; this is mitigated by our temperature 
  parameter.

We present NEXUS as a **modular framework** for studying sparse attention 
combinations, not as a single breakthrough method.

**Keywords:** sparse attention, block routing, mixture of block attention, 
softpick, random Fourier features, multi-scale sparsity

---

## Lay Summary

Long sequences make attention slow because every token has to "look at" every 
other token. We've built a modular attention system that combines several 
modern sparse-attention ideas: it groups tokens into blocks, uses a coarse 
similarity score to decide which blocks are important, learns a data-dependent 
gate to adjust this selection, and periodically applies a global linear 
approximation to keep overall context. **Caveats:** for causal decoders like 
GPT-2, the module falls back to standard dense attention — so it's mostly 
useful for encoder models. And the stripe-based fine-grained sparsity only 
activates for sequences longer than 256 tokens, which is not the case for 
our SST-2 experiments. We present NEXUS as a **framework** to study how these 
techniques combine, not as a single "beat the state-of-the-art" system.

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