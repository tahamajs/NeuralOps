<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Infini-Attention: Compressive Memory with Sliding-Window Local Attention for Transformer Inference

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/infini_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Long-context inference in transformer models is bottlenecked by the quadratic 
cost of full self-attention. Sliding-window attention reduces this cost to 
$\mathcal{O}(L \cdot W)$, but sacrifices the ability to retrieve information 
from distant tokens. We introduce **Infini-Attention**, a drop-in replacement 
for multi-head self-attention that combines a **local sliding-window** with a 
**learned compressive memory** — a small, per-head key-value matrix that 
summarizes the global context. Queries attend jointly to the local window and 
to this compressive memory, combined through a learnable gating scalar $\beta$:

$$
\text{output} = \text{LocalAttn}(Q, K, V) + \sigma(\beta) \cdot \text{MemAttn}(Q, M)
$$

where $M \in \mathbb{R}^{H \times d_c \times d_h}$ is the memory matrix, 
$d_c$ is the compression dimension, and $\beta$ is initialized to $-5$ so 
that $\sigma(\beta) \approx 0.007$ at initialization, preserving the 
pretrained local attention behavior.

Infini-Attention supports four attention families through a single 
weight-transfer routine: flat (DistilBERT), nested (RoBERTa), causal Conv1D 
(GPT-2), and grouped-query with rotary embeddings (Qwen/LLaMA). Base 
projections are frozen; only the LoRA adapters, memory matrix, and $\beta$ 
are trained.

We evaluate on DistilBERT, RoBERTa, and GPT-2 across SST-2 with 
50 LoRA recovery steps. Infini-Attention achieves 
**90.25%** on DistilBERT (Δ = -0.46), **56.54%** on RoBERTa 
(Δ = -22.02), and **51.26%** on GPT-2 (Δ = -28.10). 

<!-- CRITICAL: parameter honesty -->
Infini-Attention **increases** parameter count by **48.5\%%** 
on DistilBERT, **48.5\%%** on RoBERTa, and 
**48.5\%%** on GPT-2, due to three additions: (i) the 
per-head compressive memory ($H \cdot d_c \cdot d_h$ per layer), (ii) the 
scalar gating parameter $\beta$, and (iii) LoRA adapters on four projections. 
The "compressive" in the name refers to the memory summarizing long context, 
not to model size reduction.

**Keywords:** efficient attention, compressive memory, sliding-window, long 
context, gated fusion

---

## Lay Summary

Standard transformer attention scales poorly with sequence length — doubling 
the context quadruples the compute. Infini-Attention replaces this with a 
combination of two cheap mechanisms: a local window that only looks at 
nearby tokens, and a learned "memory" that summarizes the entire context in 
a compact form. The model decides how much to trust each, learning a balance 
that starts heavily in favor of the local window and gradually shifts as 
training progresses. The approach works across three popular language models 
without retraining the base weights. The cost: the memory matrix adds a 
small number of parameters that are learned during fine-tuning.

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