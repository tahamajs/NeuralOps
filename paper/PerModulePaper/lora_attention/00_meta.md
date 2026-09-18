# LoRA Attention: Production-Hardened Parameter-Efficient Attention Fine-Tuning Across Four Transformer Architectures

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/lora_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

**LoRA** (Low-Rank Adaptation) has become the dominant paradigm for 
parameter-efficient fine-tuning of large language models. However, most 
existing LoRA implementations target either (a) a single transformer 
architecture, or (b) the FFN projections only, leaving attention projections 
under-adapted. We present **LoRA Attention**, a production-hardened 
implementation of LoRA on the four attention projections ($W_Q, W_K, W_V, W_O$) 
that supports four distinct transformer architectures: DistilBERT (flat), 
RoBERTa (nested), GPT-2 (Conv1D), and Qwen3.5 (GQA + RoPE).

We are explicit that **this module does not reduce parameter count** — LoRA 
adapters add $r_L (2d + 2H d_h)$ parameters per layer on top of the frozen 
$4 d^2$ base projection parameters. What it does reduce is:
(i) **trainable parameters** by a factor of roughly **$[TRAIN_REDUCTION]×$**,
(ii) **training memory** by a factor of roughly **$[MEM_REDUCTION]×$**, and
(iii) **training time**.

On SST-2, LoRA Attention achieves **89.91%** on DistilBERT 
(baseline 0.85%, Δ = -0.80), **58.03%** on RoBERTa (baseline 
32%, Δ = -20.53), and **51.26%** on GPT-2 (baseline 
0.85%, Δ = -28.10), using only 50 steps and 
66.4M trainable parameters.

We position LoRA Attention as a **recovery mechanism** for compression 
methods (SVD, pruning, quantization) — any compressed model can be equipped 
with LoRA Attention for cheap post-compression fine-tuning.

**Keywords:** LoRA, parameter-efficient fine-tuning, attention, multi-architecture, 
recovery, PEFT

---

## Lay Summary

Fine-tuning a large language model usually means updating all of its 
parameters — expensive in time, memory, and compute. LoRA is a technique 
that freezes the original weights and adds tiny trainable "adapters" on top, 
reducing the number of trainable parameters by orders of magnitude. We've 
built a hardened version of LoRA that specifically targets the four attention 
projection matrices, and works across four different model architectures. It 
doesn't make the model smaller — it makes the *training* cheaper. We also 
show how it can be used to recover accuracy after a model has been 
compressed.

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