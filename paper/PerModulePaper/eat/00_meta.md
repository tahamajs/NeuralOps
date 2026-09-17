<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# EAT: Efficient Attention Transformer with Progressive Token Pruning

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/eat.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Transformer inference cost scales with sequence length, and every additional 
token that survives through the attention stack contributes to both latency 
and memory. Yet most tokens in a trained transformer carry little signal — 
they are padding, repeated function words, or otherwise uninformative for 
the downstream task. We introduce **EAT** (**E**fficient **A**ttention 
**T**ransformer), a drop-in replacement for multi-head self-attention that 
combines (i) standard scaled dot-product attention over frozen projections 
with trainable LoRA adapters, and (ii) an optional **progressive token 
pruning** layer that reduces the active token count at inference time without 
retraining the base model.

EAT supports four attention families — flat (DistilBERT), nested 
(RoBERTa/BERT), causal (GPT-2), and grouped-query with rotary embeddings 
(Qwen/LLaMA) — through a single weight-transfer routine that handles fused 
Conv1D, nested encoder, and standard linear projections. No base parameters 
are updated; only LoRA adapters are trained.

We evaluate EAT on DistilBERT, RoBERTa, and GPT-2 across SST-2 with 
50 LoRA recovery steps at rank 8. EAT achieves 
**89.91%** on DistilBERT (Δ = -0.80), **82.68%** on RoBERTa 
(Δ = +4.12), and **82.11%** on GPT-2 (Δ = +2.75). 

<!-- CRITICAL HONESTY -->
EAT does **not** reduce parameter count: the total number of parameters 
increases by **48.5\%%** on DistilBERT, **48.5\%%** 
on RoBERTa, and **48.5\%%** on GPT-2, because LoRA adapters 
are added to every attention projection. The efficiency benefit of EAT 
materializes only when progressive token pruning is enabled and only during 
inference. We report this trade-off explicitly and do not claim parameter 
compression.

**Keywords:** efficient attention, token pruning, LoRA, multi-architecture 
adaptation, transformer inference

---

## Lay Summary

Transformer models spend most of their time processing tokens that don't 
matter much for the final answer. EAT replaces the attention layer with a 
version that can drop these unimportant tokens as the model runs, saving 
computation at inference time. It works on four different model families 
without retraining the base weights. On three popular language models, EAT 
preserves most of the baseline accuracy while adding a small number of 
trainable parameters. The efficiency benefit depends on how aggressively 
tokens are pruned.

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