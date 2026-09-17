<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# AdaSplash: Adaptive Splash Attention with α-Entmax Sparse Activations and Local Sliding-Window Masking

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1 (initial draft)
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/adasplash.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Standard transformer attention produces **dense** probability distributions 
over all keys via softmax, forcing every query to attend — at least weakly — 
to every token. This has two consequences: attention weights are rarely 
interpretable (mass is spread over thousands of tokens), and the operation 
cannot exploit token sparsity for compute savings. We introduce **AdaSplash** 
(**Ada**ptive **Splash** Attention), which replaces softmax with **α-entmax** 
(Peters et al., 2019), a family of sparse activation functions parameterized 
by $\alpha \in (1, 2]$ that produce **exact zeros** for low-scoring keys. 
When $\alpha = 1.5$, α-entmax is 1.5-entmax, which interpolates between 
softmax ($\alpha = 1$) and sparsemax ($\alpha = 2$), yielding ~40% zero 
entries in practice.

AdaSplash combines three ideas:

1. **α-entmax normalization** via a numerically stabilized bisection algorithm 
   over valid tokens only, with softmax fallback when the support is empty.
2. **Local splash radius masking.** For causal decoders, keys outside a 
   sliding window of radius $W$ are masked to $-\infty$, restricting attention 
   to local context plus a small set of attention sink tokens (first 4 tokens).
3. **Frozen base weights + LoRA adapters.** Base projections are copied 
   exactly from the pretrained model and frozen; a rank-16 LoRA residual 
   is added for task adaptation.

AdaSplash is **not** a compression method: parameter count **increases** by 
**0.88%** due to LoRA adapters. It is a sparse attention mechanism 
that trades a small parameter overhead for interpretable attention and potential 
compute savings on specialized hardware. We evaluate on DistilBERT, RoBERTa-base, 
and GPT-2 on SST-2 with 50 LoRA recovery steps. AdaSplash 
achieves **77.18%** on DistilBERT, **67.09%** on RoBERTa, and 
**79.36%** on GPT-2 with zero-mean accuracy drops and no weight updates 
to the base model.

**Keywords:** sparse attention, α-entmax, sliding window, attention sinks, 
local-global attention, LoRA

---

## Lay Summary

Standard attention in transformers is "soft" — every token considers every 
other token, even if only briefly. This is wasteful: studies show that most 
attention weight goes to a few important tokens, and the rest is noise. 
AdaSplash replaces the standard "softmax" with a "sparse" alternative that 
actively zeros out unimportant connections. It also restricts attention to a 
local window (plus a few "sink" tokens at the start), making the mechanism 
both interpretable and potentially faster on specialized hardware. We test 
on three language models and find that AdaSplash matches the original models' 
accuracy on sentiment analysis while using cleaner attention patterns.

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
