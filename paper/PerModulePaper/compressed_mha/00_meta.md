# Compressed MHA: Joint Truncated SVD for Multi-Architecture Attention Compression

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/compressed_mha.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

We present **Compressed MHA**, a low-rank compression module for transformer 
attention that performs **joint truncated SVD** on the stacked query, key, and 
value projection matrices $[W_Q; W_K; W_V]$, and separate truncated SVD on the 
output projection $W_O$. The joint decomposition shares the input-space basis 
across Q, K, V, exploiting the strong correlation between these projections 
in pretrained transformers. Base SVD factors are frozen; trainable LoRA 
adapters on both the QKV and output branches provide recovery capacity.

For a typical transformer with $d = 768$ and head dimension $d_h = 64$, the 
default configuration achieves **{{D_RATIO}}×** parameter reduction on 
attention projections (from $4d^2$ to $2r(d + d_h)$ where $r$ is the SVD 
rank). On SST-2, Compressed MHA achieves **{{D_ACC}}%** on DistilBERT 
(baseline {{BASE_D}}%, Δ = {{D_DELTA}}), **{{R_ACC}}%** on RoBERTa (baseline 
{{BASE_R}}%, Δ = {{R_DELTA}}), and **{{G_ACC}}%** on GPT-2 (baseline 
{{BASE_G}}%, Δ = {{G_DELTA}}), using {{RECOVERY_STEPS}} LoRA recovery steps.

The implementation is production-hardened across four architectures 
(DistilBERT flat, RoBERTa nested, GPT-2 Conv1D, Qwen GQA+RoPE) with:
(i) **correct nested-encoder detection** that checks for `out_lin` before 
nested structure, avoiding a subtle double-projection bug, (ii) **safe failure 
handling** using `logger.warning` instead of `raise`, and (iii) **per-layer 
fallback** preserving original layers on weight extraction failure.

**Keywords:** joint SVD, low-rank attention, LoRA recovery, multi-architecture 
compression, nested encoder

---

## Lay Summary

Large language models store most of their parameters in the attention 
projection matrices. We compress these matrices using a mathematical technique 
called SVD, which keeps only the most important directions. Unlike previous 
methods that compress Q, K, and V separately, we compress them **together** — 
a trick that shares information across them and yields smaller compressed 
matrices. We also compress the output projection separately. The compressed 
weights are frozen, and we add small trainable "adapters" (LoRA) on top to 
recover any lost accuracy. On three popular models (DistilBERT, RoBERTa, 
GPT-2), we achieve near-baseline accuracy while reducing the attention 
parameter count significantly.

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