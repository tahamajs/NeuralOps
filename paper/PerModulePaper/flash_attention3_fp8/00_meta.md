<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# FlashAttention-3 FP8: Low-Precision 8-bit Attention with Hardware-Aware Fallback

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/flash_attention3_fp8.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- CRITICAL: Be honest about hardware requirements
- FP8 tensor cores require Hopper or Ada GPUs
- On RTX 4070 (this study's hardware), the FP8 path is NOT taken
- The fallback path is PyTorch SDPA
- LoRA adds parameters (not a compression method)
-->

Recent attention implementations — FlashAttention, FlashAttention-2, and 
FlashAttention-3 — have reduced attention memory from $\mathcal{O}(L^2)$ to 
$\mathcal{O}(L)$ by exploiting tiling and, in the third generation, 8-bit 
floating-point (FP8) tensor cores on Hopper architectures. However, these 
methods assume specific hardware that many practitioners do not have access 
to. We present a **hardened PyTorch implementation** of FlashAttention-3 FP8 
that automatically degrades to PyTorch's native Scaled Dot-Product Attention 
(SDPA) when FP8 tensor cores are unavailable, ensuring portability across 
all CUDA GPUs.

Our implementation supports four transformer families — DistilBERT, 
RoBERTa/BERT, GPT-2, and Qwen/LLaMA — with correct handling of grouped-query 
attention (GQA), rotary position embeddings (RoPE), causal masking, and 
nested encoder architectures. Base weights are transferred exactly from the 
pretrained attention via a weight-copy procedure that handles fused Conv1D, 
nested self-attention, and GQA-based projections; trainable capacity is 
provided by LoRA adapters at rank $r = 8$.

We benchmark on SST-2 across DistilBERT, RoBERTa, and GPT-2. Because our 
test hardware (RTX 4070 Laptop) does not expose FP8 tensor cores, the FP8 
path is unavailable and all models run through the SDPA fallback. We report 
this transparently and identify the exact hardware requirements for 
reproducing the FP8 speedups claimed in the original FlashAttention-3 paper. 
On our hardware, accuracy is preserved within **-0.92%** (DistilBERT), 
**0.85%** (RoBERTa), and **0.85%** (GPT-2), but parameter 
count increases by **66.4M%** due to LoRA adapters.

**Keywords:** FlashAttention-3, FP8 quantization, Hopper tensor cores, 
attention implementation, hardware-aware fallback

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

New GPUs (H100, H200) have special "FP8" arithmetic units that make attention 
twice as fast. Our code uses these units when available but falls back to 
standard attention when they are not — for example on consumer GPUs. We 
tested the code on three small pretrained models and found that it works 
correctly on all of them. On our test GPU (RTX 4070 Laptop), the fast FP8 
path was not available, so the code ran the fallback path. We report this 
honestly so that others know what to expect.

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