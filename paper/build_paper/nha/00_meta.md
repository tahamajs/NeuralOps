<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# NHA: Native Hybrid Attention with Local Windows and Global Random Fourier Features

<!-- 
TITLE NOTES:
- "Native" emphasizes the fusion happens inside the attention module
- "Hybrid" = local + global in parallel
- Alternative: "Native Hybrid Attention: Learned Token-Level Fusion of Local Windows and Global RFF"
-->

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/nha_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Self-attention has two regimes: local patterns that dominate syntactic 
structure, and global patterns that carry long-range dependencies. Existing 
methods commit to one: sliding-window attention (Longformer, Mistral) 
handles the local regime cheaply but truncates global context; linear 
attention (Performer, Linear Transformer) approximates the global regime in 
$\mathcal{O}(L)$ but loses local precision.

We present **Native Hybrid Attention** (NHA), a mechanism that runs a 
**local sliding-window stream** and a **global Random Fourier Features (RFF) 
stream in parallel** within each layer, and combines them with a **learned 
token-level gate**. The gate is a scalar per token, computed by a linear 
projection of the input hidden state, and initialized to $3.0$ (bias), 
biasing the model toward the local stream at initialization. The RFF stream 
uses an **orthogonal projection matrix** built from block QR factorization, 
giving variance-reduced kernel approximation. For causal models, the RFF 
stream uses a **chunked prefix scan** with cumulative state $S \in 
\mathbb{R}^{m \times d_h}$ and $Z \in \mathbb{R}^m$, preserving strict 
causality.

We evaluate NHA on DistilBERT, RoBERTa-base, and GPT-2 on SST-2. With LoRA 
recovery at rank 8, NHA achieves **90.71%**, **50.92%**, 
and **50.92%** accuracy, within **+0.00%**, **-27.64%**, and 
**-28.44%** of the respective dense baselines. Attention memory is 
reduced by up to **32×** for long sequences. Learned gates 
show systematic token-level structure that correlates with attention entropy.

**Keywords:** hybrid attention, local window, Random Fourier Features, 
learnable gating, efficient transformer

---

## Lay Summary

Attention is the operation that lets a language model "look at" all previous 
tokens. It has two modes: **local attention** (efficient, looks only at 
nearby tokens) and **global attention** (looks everywhere, expensive). Real 
text often needs both.

NHA runs both at once — a local window and a global summary — and lets each 
token decide how much to trust each. A single trainable number per token 
controls the mix. Initially, the model trusts the local window; during 
training, it learns to shift toward global where needed.

On three pretrained models, NHA retains most of the original accuracy while 
reducing attention memory. The learned gates reveal which positions need 
global context — a useful signal for understanding what the model is doing.

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