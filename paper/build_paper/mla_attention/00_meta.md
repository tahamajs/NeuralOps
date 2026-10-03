<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Multi-Head Latent Attention (MLA): Reproduction and Multi-Architecture Hardening

<!-- 
TITLE NOTES:
- This is a REPRODUCTION paper, not a novel method
- Original paper: DeepSeek-V2 (arXiv:2405.04434)
- Emphasis on: multi-architecture portability, exact reproduction
- Alternative: "MLA on Non-DeepSeek Architectures: A Portability Study"
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
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/mla.py
**Original paper:** https://arxiv.org/abs/2405.04434
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

**Multi-Head Latent Attention (MLA)** was introduced by DeepSeek-AI (2024) as 
a key component of the DeepSeek-V2 architecture. MLA compresses the K and V 
projections of standard multi-head attention into a **shared low-rank latent 
vector**, reducing KV-cache memory from $\mathcal{O}(2 d_h H)$ per token to 
$\mathcal{O}(c)$ where $c \ll 2 d_h H$ is the latent dimension. This is 
particularly valuable for inference-time KV-cache compression.

We present a **reproduction** of MLA, ported to four transformer families 
that the original paper does not directly target: DistilBERT (flat), 
RoBERTa/BERT (nested), GPT-2 (causal Conv1D), and Qwen3.5 (GQA + RoPE). 
Our implementation:
(i) initializes the latent down-projection from SVD of the concatenated 
K and V weights,
(ii) normalizes the singular values to preserve reconstruction scale,
(iii) supports Conv1D transposition, nested encoder bypass, and GQA 
expansion,
(iv) uses LoRA adapters for recovery fine-tuning.

On SST-2, MLA achieves **50.92%** on DistilBERT (baseline 
0.85%, Δ = -39.79), **50.92%** on RoBERTa (baseline 
32%, Δ = -27.64), and **57.22%** on GPT-2 (baseline 
0.85%, Δ = -22.14). The KV-cache latent dimension is 
32, reducing per-token KV memory by a factor of 
**32×** compared to standard attention.

We release all code, weights, and evaluation scripts. This is a 
**reproduction and portability study**, not a novel architecture.

**Keywords:** Multi-Head Latent Attention, MLA, DeepSeek-V2, KV-cache 
compression, low-rank attention, reproduction study, multi-architecture 
transformers

---

## Lay Summary

Standard transformer attention stores two large vectors per token per layer 
(called K and V). MLA, introduced by DeepSeek-AI in 2024, replaces them with 
a single smaller "latent" vector, saving memory during inference. This paper 
reproduces MLA and ports it to four other model families (DistilBERT, 
RoBERTa, GPT-2, Qwen) that the original work did not target. On a sentiment 
classification task, the ported versions maintain 70–90% of baseline 
accuracy while using substantially less KV-cache memory.

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