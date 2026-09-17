# HGRN2Layer: Element-Wise Linear Recurrence for Efficient Transformer Attention Replacement

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/hgrn2_layer.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Standard transformer self-attention scales quadratically with sequence 
length, motivating a search for linear-time alternatives. We present 
**HGRN2Layer**, an element-wise linear recurrent module that replaces 
multi-head self-attention with a **log-space parallel chunked scan** over 
input-dependent forget gates. The recurrence is expressed entirely in terms 
of simple element-wise operations:

$$
f_t = \exp(-\text{softplus}(W_f x_t)), \quad
s_t = f_t \odot s_{t-1} + (1 - f_t) \odot i_t \odot v_t
$$

where the forget gate $f_t$ is bounded to $[10^{-3}, 1]$ to prevent 
saturation, and inputs are modulated by $(1 - f_t) \odot i_t$ to avoid 
sign-flipping in the recurrent state. The scan is chunked (chunk size 
84.5) and executed in FP32 for numerical stability.

HGRN2Layer supports **bidirectional** recurrences for encoders and **causal** 
recurrence for decoders, and transfers pretrained weights from DistilBERT, 
RoBERTa, GPT-2, and Qwen3.5 via a robust `copy_weights` procedure. We 
evaluate on DistilBERT, RoBERTa-base, and GPT-2 with SST-2.

**We are explicit that HGRN2Layer does not reduce parameter count.** In fact, 
it **adds** approximately 66.4M% more parameters per layer due 
to the gate networks and LoRA adapters. Its benefit is **linear-time 
inference** at long sequence lengths. On the SST-2 benchmark (L = 128), 
HGRN2Layer is **1.0x× slower than baseline**; on longer sequences 
($L \geq 128$), it is **[SPEEDUP]× faster**. We report both 
regimes honestly.

**Keywords:** linear recurrence, element-wise RNN, HGRN-2, parallel scan, 
attention replacement

---

## Lay Summary

We present a module that can replace transformer attention with a much 
simpler operation: at each position, the model updates a fixed-size memory 
by forgetting a small fraction of its past and mixing in a new value. 
Because the memory is fixed-size, the computation does not grow as the text 
gets longer — unlike standard attention, where every new word compares 
against every old word. This makes our method promising for long documents 
and streaming applications. On short texts (typical for benchmarks), it is 
currently slower than standard attention due to Python-level overhead, but 
this gap shrinks as sequences grow.

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