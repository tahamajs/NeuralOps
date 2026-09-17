# LEKAv2: Linear-Time Efficient Kernel Attention with Landmark Distillation

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/leka_v2.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Linear attention scales as $\mathcal{O}(L)$ in sequence length by replacing 
softmax with a kernel feature map, but achieves this at the cost of 
**uniform aggregation**: every query attends to the full aggregated 
KV state, losing the ability to selectively focus on important 
positions. We introduce **LEKAv2**, which extends the linear-attention 
primitive with **landmark distillation** and **learned local-global 
fusion**:

1. **Dual-scale random features.** We map Q/K through two independent 
   RFF projections ($F$ and $F/2$ dimensions) and concatenate the 
   resulting cos/sin features, enriching the kernel representation.
2. **Landmark distillation.** A learned scorer selects $L_\text{max} = \rho L$ 
   landmark tokens whose features are used in the linear-attention 
   aggregation. Position 0 (or `[CLS]`) is always retained.
3. **Softpick.** We replace softmax with a ReLU-damped variant that 
   stabilizes masked attention:
   
   $$
   \text{softpick}(x)_i = \text{ReLU}\!\left(\frac{e^{(x_i - m)/T} - e^{-m/T}}{\sum_j \text{ReLU}(e^{(x_j - m)/T} - e^{-m/T})}\right)
   $$
4. **Local-global fusion.** A learned per-head gate blends a **local 
   sliding-window attention** with the **global landmark attention**.

On SST-2, LEKAv2 achieves **86.01%** on DistilBERT (baseline 
0.85%), **50.92%** on RoBERTa (baseline 32%), and 
**50.92%** on GPT-2 (baseline 0.85%). Inference is linear at 
long sequences ($\geq 128$) and preserves pretrained weights 
via SVD-free copying with LoRA adapters.

**We are explicit that LEKAv2 does not reduce parameter count.** It adds 
approximately 66.4M% parameters due to LoRA, projections, 
landmark scorer, and the correction matrices. Its benefit is 
**linear-time long-context inference** with **learned landmark 
selection**.

**Keywords:** linear attention, random Fourier features, landmark 
distillation, local-global attention, kernel approximation

---

## Lay Summary

Standard attention is expensive for long texts because it compares every 
word against every other word. Linear attention fixes this by aggregating 
a running summary, but it loses the ability to focus on specific 
important words. Our LEKAv2 solves this by: (1) mapping queries and keys 
into two different random feature spaces, (2) selecting a set of 
"landmark" tokens via a small learned network, (3) using a robust 
softmax-like function that handles masks gracefully, and (4) mixing a 
local sliding window with global landmark attention. The result is a 
hybrid mechanism that scales linearly and retains the ability to focus.

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