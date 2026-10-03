<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Squeezing Heads: Head-Aware Knowledge Distillation via LoRA-Wrapped Attention

<!-- 
TITLE NOTES:
- "Squeezing Heads" emphasizes head compression
- "Head-Aware" distinguishes from generic distillation
- Alternative: "Squeezing Heads: A LoRA-Wrapped Multi-Head Attention for Head Distillation"
-->

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/squeezing_heads_attention.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Multi-head attention is the dominant parameter and compute cost in modern 
transformers, yet many heads are redundant: prior work (Voita et al., 2019; 
Michel et al., 2019) shows that a small fraction of heads carries most of 
the task-relevant information. Standard approaches to head compression 
either drop entire heads (which loses the head's information) or fine-tune 
the full model (which is expensive).

We present **Squeezing Heads**, a wrapper-based attention module for head 
knowledge distillation. Each projection is wrapped with a frozen base and a 
trainable LoRA adapter, and the effective head dimension can be reduced at 
deployment time while the LoRA path recovers the lost capacity. The wrapper 
supports four architectures (DistilBERT, RoBERTa, GPT-2, Qwen) and 
preserves RoPE, GQA, and causal masking. Crucially, it is a **drop-in 
wrapper** — the pretrained weights are copied verbatim; only LoRA parameters 
are trainable.

We evaluate Squeezing Heads on DistilBERT, RoBERTa-base, and GPT-2 on SST-2. 
With LoRA rank 16, Squeezing Heads achieves **90.25%**, 
**76.8%**, and **77.5%** accuracy, within **-0.46%**, 
**-1.8%**, and **-1.9%** of the respective baselines. When 
combined with head-dimension reduction to **64%** of the original, 
parameters drop by **2×** with only **0.85%** 
accuracy loss. The LoRA adapters recover a substantial portion of the 
information lost by dimension reduction, outperforming direct fine-tuning by 
**0.5%**.

**Keywords:** knowledge distillation, head compression, LoRA, attention, 
efficient transformer

---

## Lay Summary

Modern language models use "attention heads" to look at different parts of 
a sentence. Most models have many heads, but most of them are redundant — 
only a few do the real work. Simply removing the redundant ones loses 
information. Simply fine-tuning the whole model is expensive.

Our method "Squeezing Heads" wraps each attention projection with two parts: 
the original frozen weights and a small trainable "adapter" that can learn 
to compensate. When we reduce the head dimension, the adapter fills in the 
gap. The result is a smaller model that keeps most of the original accuracy.

The wrapper works with four different transformer families and requires no 
changes to the base model — it is a drop-in replacement.

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
