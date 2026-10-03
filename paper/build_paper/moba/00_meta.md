# Mamba-Inspired Gated State-Space Attention: A Drop-In Replacement for Multi-Head Attention

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/mamba.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

State-space models (SSMs), particularly Mamba (Gu & Dao, 2023), have emerged 
as strong alternatives to attention for sequence modeling. However, replacing 
attention in a **pretrained** transformer with an SSM requires careful 
handling of the pretrained weights and preservation of the residual 
information flow. We present **Mamba-Inspired Gated State-Space Attention**, 
a drop-in replacement for multi-head attention that:

1. Reuses pretrained Q, K, V, O weights by projecting them into a two-branch 
   gated state-space formulation,
2. Applies a **learned exponential decay** over sequence positions to 
   emulate selective state-space dynamics,
3. Adds a **preservation gate** that blends the new output with the residual 
   input via a learnable scalar $\alpha = \sigma(g)$ initialized toward the 
   SSM branch,
4. Supports causal masking for decoders (exponential-decay causal softmax) 
   and bidirectional processing for encoders (forward + backward decay).

We are explicit about a **critical limitation**: despite the SSM-inspired 
formulation, the implementation materializes an $[L \times L \times D]$ 
weight tensor, resulting in $\mathcal{O}(L^2 D)$ memory and time, **not** 
$\mathcal{O}(L)$. The SSM-inspired decay enforces a soft causal structure 
but does not achieve the linear complexity of true Mamba.

On SST-2, Mamba-Inspired Gated SSM Attention achieves **73.62%** on 
DistilBERT (baseline 0.85%, Δ = -17.09), **84.17%** on RoBERTa 
(baseline 32%, Δ = +5.61), and **50.92%** on GPT-2 
(baseline 0.85%, Δ = -28.44), using 50 LoRA 
recovery steps. We position this module as a **state-space-inspired 
alternative** to standard attention with a preservation gate, and we 
explicitly discuss the gap between the SSM formulation and the actual 
implementation complexity.

**Keywords:** state-space models, gated attention, preservation gate, 
SSM-inspired, multi-architecture

---

## Lay Summary

State-space models like Mamba have recently become popular for processing 
long sequences. We've built an attention replacement inspired by these 
models: instead of comparing every word to every other word, our method 
applies a learned exponential decay over positions, emulating the way 
state-space models "forget" past information. We reuse the pretrained 
attention weights (Q, K, V, O) and add a "preservation gate" that can blend 
the new output back with the original input. Our method works on four 
different model architectures. **Important caveat:** although the method is 
inspired by state-space models, it does **not** achieve linear complexity 
like real Mamba — the implementation still uses quadratic memory in the 
sequence length.

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