<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# UnifiedRoPE: A Portable Rotary Position Embedding Attention Module for Multi-Architecture Transformers

<!-- 
TITLE NOTES:
- This is a PORTABILITY / ENGINEERING paper, not a novel method
- Emphasis: "Unified" (one module for 4 architectures), "Portable"
- Alternative: "One RoPE to Rule Them All: A Unified Attention Module for DistilBERT, RoBERTa, GPT-2, and Qwen"
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
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/unified_rope.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Rotary Position Embedding (RoPE; Su et al., 2021) has become the default 
position encoding for modern decoder-only transformers. However, porting 
RoPE-based attention to encoder-only models such as DistilBERT and 
RoBERTa, or to fused-convolution architectures such as GPT-2, requires 
architecture-specific handling: nested encoder bypass for BERT-style 
models, Conv1D transposition for GPT-2, grouped-query attention (GQA) 
expansion for Qwen, and correct partial-dimension rotation for models with 
non-standard head dimensions.

We present **UnifiedRoPE**, a single attention module that implements 
RoPE-based scaled dot-product attention and works across **four 
transformer families** without modification:

1. **DistilBERT** (flat attention)
2. **RoBERTa / BERT** (nested encoder self-attention)
3. **GPT-2** (causal decoder with fused Conv1D)
4. **Qwen3.5 / LLaMA** (GQA with rotary position embeddings)

**UnifiedRoPE is a replacement module, not a compression method.** It 
preserves all base parameters, produces identical forward outputs to the 
original attention (up to numerical precision), and can be used as a 
substrate for downstream compression (e.g., LoRA recovery, SVD factorization). 
The `LoRALinear` projections allow trainable fine-tuning while freezing 
base weights.

We provide a production-hardened implementation with:
(i) precomputed `cos`/`sin` frequency caches (non-persistent),
(ii) robust multi-architecture weight transfer with LoRA preservation,
(iii) nested encoder bypass for RoBERTa-style models,
(iv) GQA expansion via `repeat_kv`,
(v) safe injection returning `None` on failure.

On SST-2, UnifiedRoPE matches baseline accuracy to within **-2.75%** 
on DistilBERT, **-14.23%** on RoBERTa, and **-21.91%** on GPT-2.

**Keywords:** RoPE, rotary position embeddings, portable attention, 
multi-architecture transformers, nested encoder, Conv1D, GQA, production 
hardening

---

## Lay Summary

Rotary Position Embedding (RoPE) is a technique for encoding position 
information in transformer attention. It works well in modern decoders 
(GPT, Qwen), but adapting it to older architectures (BERT, RoBERTa) or 
unusual ones (GPT-2's fused convolution) is error-prone. This paper 
describes a single attention module that works correctly across four 
different families without modification. The module is a drop-in 
replacement: it preserves the pretrained weights and produces the same 
output. It does not compress the model — it is intended as a foundation 
for other compression techniques.

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