# FlashAttention-3: Exact Attention with IO-Aware Memory Management

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/flash_attention3.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Standard scaled dot-product attention requires $\mathcal{O}(L^2)$ memory to 
materialize the attention matrix. FlashAttention-3 (Dao et al., 2024) 
eliminates this bottleneck via **IO-aware tiling** and **asynchronous 
hardware overlap** on Hopper GPUs, achieving near-optimal utilization of 
TMA (Tensor Memory Accelerator) and warp specialization. It produces 
**exact** attention — no approximation.

We present a **production-hardened FlashAttention-3 integration** for 
pretrained transformer attention. Our implementation:

1. **Dispatches to FlashAttention-3** when the `flash_attn` library is 
   available and the input is on CUDA.
2. **Falls back gracefully to PyTorch SDPA** when FA3 is unavailable, 
   preserving exact attention semantics.
3. **Optionally applies FP8 precision** (`e4m3fn`) when requested.
4. **Transfers pretrained weights exactly** from DistilBERT, RoBERTa, 
   GPT-2, and Qwen3.5 via a robust `copy_weights` procedure.
5. **Adds LoRA adapters** for parameter-efficient recovery fine-tuning.
6. **Supports cross-attention** for encoder-decoder models.

**We report two important caveats.**

1. **No parameter reduction.** FlashAttention is an **algorithm**, not a 
   compression method. Parameter count changes only by the LoRA adapters 
   (+{{D_PARAM_PCT}}%). We make no compressive claims.
2. **No speedup without the FA3 kernel.** When `flash_attn` is not installed 
   (e.g., on non-Hopper GPUs or with an incompatible CUDA version), our 
   implementation falls back to PyTorch SDPA. This fallback is functionally 
   identical but numerically slower.

On SST-2, the module produces baseline-identical accuracy (since attention 
is exact) with only LoRA-induced variance of **{{D_DELTA}}%**. The primary 
contribution is a **drop-in integration** of FlashAttention-3 into existing 
pretrained models with correct weight transfer.

**Keywords:** FlashAttention, IO-aware attention, Hopper TMA, FP8, exact 
attention, inference optimization

---

## Lay Summary

Standard attention builds a big $L \times L$ table of "which words look at 
which other words". For long texts, this table does not fit in a GPU's fast 
memory, so attention is slow. FlashAttention-3 avoids building the table: 
it processes the attention in small blocks, keeping intermediate data in 
fast SRAM. The result is the **same mathematically exact** attention, but 
much faster and using less memory. Our work integrates FlashAttention-3 
into existing pretrained models — copying weights correctly, providing 
LoRA adapters for fine-tuning, and offering a safe fallback to standard 
attention when the specialized kernel is unavailable.

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
