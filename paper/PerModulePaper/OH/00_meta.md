<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# OHA: Optimized Hybrid Attention with Sequence-Compressed Global Context

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/oha_attention.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Standard self-attention is $\mathcal{O}(L^2)$ in sequence length, which is 
prohibitive for long-context inference. Sliding-window attention reduces 
this to $\mathcal{O}(L \cdot W)$, but cannot capture long-range dependencies. 
We introduce **OHA** (**O**ptimized **H**ybrid **A**ttention), a drop-in 
replacement for multi-head self-attention that combines (i) local sliding-window 
attention with (ii) **sequence-compressed global attention**, where the 
global branch attends not to all $L$ tokens but to $d_c \ll L$ compressed 
"summary" slots produced by learnable soft-attention weights. The two 
branches are combined through a **head-wise gating** scalar initialized to 
favor the local branch at start.

OHA supports four attention families through a single weight-transfer routine: 
flat (DistilBERT), nested (RoBERTa), causal Conv1D (GPT-2), and grouped-query 
with rotary embeddings (Qwen/LLaMA). Base projections are frozen; only LoRA 
adapters, the compression projections, and the gate are trained.

We evaluate OHA on DistilBERT, RoBERTa, and GPT-2 across SST-2 with 
50 LoRA recovery steps. OHA achieves **89.2%** on 
DistilBERT (Δ = -1.51), **76.5%** on RoBERTa (Δ = -2.06), 
and **77.8%** on GPT-2 (Δ = -1.56).

<!-- CRITICAL HONESTY -->
OHA **increases** parameter count by **1.40%** on DistilBERT 
due to three additions: (i) the compression projections 
$W_k^\text{comp}, W_v^\text{comp} \in \mathbb{R}^{d \times d_c}$ 
(ii) the gate $W_g \in \mathbb{R}^{d \times H}$, and (iii) LoRA adapters on 
four projections. The "compression" in the name refers to the **sequence 
length** of the global branch, not to model size reduction.

**Keywords:** hybrid attention, sliding-window, sequence compression, 
gated fusion, transformer efficiency

---

## Lay Summary

Attention in transformers has a quadratic cost in sequence length, which 
makes long contexts expensive. OHA splits attention into two parts: a 
**local** part that looks only at nearby tokens (cheap), and a **global** 
part that looks at a **compressed summary** of the entire sequence (also 
cheap). The model learns to balance the two using a per-head gate that 
initially favors the local branch. This works on four different model 
families without retraining the base weights. The cost: OHA adds some 
learned parameters that must be trained during fine-tuning.

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
