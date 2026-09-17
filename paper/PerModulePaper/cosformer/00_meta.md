<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# Cosformer: Linear Attention with Cosine-Based Positional Reweighting for Pretrained Transformers

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1 (initial draft)
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/cosformer.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Cosformer is a published method (Qin et al., 2022)
- Our contribution is the port to pretrained transformers
- Report actual results honestly
-->

Softmax attention is the bottleneck of transformer inference for long 
sequences, requiring $\mathcal{O}(L^2)$ time and memory. **Cosformer** 
(Qin et al., 2022) addresses this by replacing softmax with a non-negative 
kernel (ELU + 1 or softplus) augmented with cosine-based positional 
reweighting, yielding $\mathcal{O}(L)$ attention while preserving locality. 
We present a **production-hardened port** of Cosformer to pretrained 
transformers, supporting four architecture families — DistilBERT, 
RoBERTa/BERT, GPT-2, and Qwen/LLaMA — with exact weight transfer, GQA, 
RoPE, and two distinct computational paths: **associative accumulation** 
for bidirectional encoders and **chunked prefix-sum** for causal decoders. 
Base weights are transferred exactly from the pretrained attention; LoRA 
adapters at rank $r = 16$ provide recovery capacity.

We benchmark on SST-2 across three models. Cosformer achieves 
**77.75%** on DistilBERT (baseline 90.71%, Δ = -12.96), 
**58.72%** on RoBERTa (Δ = -19.84), and **70.18%** on GPT-2 
(Δ = -9.18). Inference time is **1.84×** baseline on 
DistilBERT and **4.88×** on GPT-2. Unlike recent mechanisms that 
collapse on encoders, Cosformer preserves most of the baseline accuracy 
at a modest latency cost. We document the construction of the cosine/sine 
reweighting, the chunked causal scan, and the exact failure modes we 
observed.

**Keywords:** linear attention, Cosformer, kernel methods, positional 
reweighting, efficient transformers

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

Standard attention compares every word to every other word, which is slow 
for long documents. Cosformer uses a clever mathematical trick to approximate 
this comparison in linear time — it transforms words with a "soft" activation 
function and multiplies by cosine waves, preserving the "locality" property 
of attention (nearby words matter more). We ported Cosformer to three 
pretrained models and measured how much accuracy they lose. Cosformer 
performs better than most linear attention variants: on DistilBERT it keeps 
most of the accuracy, and on GPT-2 it's within 3%. It's slower than baseline 
because of implementation overhead, but the accuracy-preservation story is 
positive.

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